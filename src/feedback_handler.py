"""
Feedback Loop Handler
Manages low-confidence predictions and user corrections
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class FeedbackHandler:
    """Handle uncertain predictions and collect user feedback"""

    def __init__(self, confidence_thresholds: Optional[Dict[str, float]] = None,
                 feedback_log_path: Optional[str] = None):
        """
        Initialize feedback handler

        Args:
            confidence_thresholds: Dict with 'high', 'medium', 'low' thresholds
            feedback_log_path: Path to feedback CSV log
        """
        self.confidence_thresholds = confidence_thresholds or {
            'high': 0.85,
            'medium': 0.70,
            'low': 0.60
        }

        self.feedback_log_path = feedback_log_path or 'data/feedback_log.csv'
        self._initialize_feedback_log()

    def _initialize_feedback_log(self):
        """Initialize feedback log CSV if it doesn't exist"""
        log_path = Path(self.feedback_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if not log_path.exists():
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'transaction',
                    'user_selected_category',
                    'system_predicted_category',
                    'confidence',
                    'timestamp',
                    'alternatives',
                    'model_votes'
                ])

    def handle_prediction(self, prediction_result: Dict) -> Dict:
        """
        Process prediction based on confidence level

        Args:
            prediction_result: Dictionary with prediction and explanation

        Returns:
            Processed result with recommended action
        """
        confidence = prediction_result['confidence']
        predicted_category = prediction_result['predicted_category']
        alternatives = prediction_result.get('alternatives', [])

        # Determine action based on confidence
        if confidence >= self.confidence_thresholds['high']:
            status = "ACCEPTED"
            action = "no_review_needed"
            message = "High confidence prediction - accepted automatically"

        elif confidence >= self.confidence_thresholds['medium']:
            status = "REVIEW_RECOMMENDED"
            action = "suggest_alternatives"
            message = f"Medium confidence - consider reviewing. Top alternatives: {[a['category'] for a in alternatives[:2]]}"

        elif confidence >= self.confidence_thresholds['low']:
            status = "MANUAL_REVIEW"
            action = "require_user_selection"
            message = "Low confidence - manual review required"

        else:
            status = "VERY_LOW_CONFIDENCE"
            action = "flag_for_expert_review"
            message = "Very low confidence - expert review recommended"

        result = {
            'transaction': prediction_result['transaction'],
            'predicted_category': predicted_category,
            'confidence': confidence,
            'status': status,
            'action': action,
            'message': message,
            'alternatives': alternatives,
            'requires_user_input': status in ['REVIEW_RECOMMENDED', 'MANUAL_REVIEW', 'VERY_LOW_CONFIDENCE']
        }

        return result

    def log_correction(self, transaction: str, user_selected: str,
                      system_predicted: str, confidence: float,
                      alternatives: Optional[List[Dict]] = None,
                      model_votes: Optional[Dict] = None):
        """
        Log user correction to feedback file

        Args:
            transaction: Original transaction string
            user_selected: User's selected category
            system_predicted: System's predicted category
            confidence: System's confidence score
            alternatives: List of alternative predictions
            model_votes: Individual model votes
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(self.feedback_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                transaction,
                user_selected,
                system_predicted,
                confidence,
                timestamp,
                json.dumps(alternatives) if alternatives else '',
                json.dumps(model_votes) if model_votes else ''
            ])

    def get_feedback_statistics(self) -> Dict:
        """
        Analyze feedback log and generate statistics

        Returns:
            Dictionary with feedback statistics
        """
        log_path = Path(self.feedback_log_path)

        if not log_path.exists() or log_path.stat().st_size == 0:
            return {
                'total_corrections': 0,
                'message': 'No feedback data available'
            }

        df = pd.read_csv(log_path)

        if len(df) == 0:
            return {
                'total_corrections': 0,
                'message': 'Feedback log is empty'
            }

        stats = {
            'total_corrections': len(df),
            'correction_rate': len(df[df['user_selected_category'] != df['system_predicted_category']]) / len(df),

            # By confidence level
            'corrections_by_confidence': {
                'high': len(df[df['confidence'] >= self.confidence_thresholds['high']]),
                'medium': len(df[(df['confidence'] >= self.confidence_thresholds['medium']) &
                                 (df['confidence'] < self.confidence_thresholds['high'])]),
                'low': len(df[df['confidence'] < self.confidence_thresholds['medium']])
            },

            # Most corrected categories
            'most_corrected_from': df['system_predicted_category'].value_counts().head(5).to_dict(),
            'most_corrected_to': df['user_selected_category'].value_counts().head(5).to_dict(),

            # Average confidence of corrections
            'avg_confidence_of_corrections': float(df['confidence'].mean()),

            # Temporal trends
            'first_correction': df['timestamp'].min() if 'timestamp' in df.columns else None,
            'last_correction': df['timestamp'].max() if 'timestamp' in df.columns else None,
        }

        return stats

    def get_retraining_data(self) -> Tuple[List[str], List[str]]:
        """
        Extract corrected data for model retraining

        Returns:
            Tuple of (transactions, corrected_labels)
        """
        log_path = Path(self.feedback_log_path)

        if not log_path.exists():
            return [], []

        df = pd.read_csv(log_path)

        # Only include cases where user corrected the prediction
        corrections = df[df['user_selected_category'] != df['system_predicted_category']]

        transactions = corrections['transaction'].tolist()
        labels = corrections['user_selected_category'].tolist()

        return transactions, labels

    def get_uncertain_patterns(self, min_occurrences: int = 3) -> List[Dict]:
        """
        Identify patterns in uncertain/corrected predictions

        Args:
            min_occurrences: Minimum occurrences to consider a pattern

        Returns:
            List of pattern dictionaries
        """
        log_path = Path(self.feedback_log_path)

        if not log_path.exists():
            return []

        df = pd.read_csv(log_path)

        # Find frequently corrected transaction patterns
        corrections = df[df['user_selected_category'] != df['system_predicted_category']]

        if len(corrections) == 0:
            return []

        # Group by common words/patterns
        patterns = []

        for category in corrections['system_predicted_category'].unique():
            cat_corrections = corrections[corrections['system_predicted_category'] == category]

            if len(cat_corrections) >= min_occurrences:
                patterns.append({
                    'predicted_category': category,
                    'occurrences': len(cat_corrections),
                    'common_actual_categories': cat_corrections['user_selected_category'].value_counts().to_dict(),
                    'avg_confidence': float(cat_corrections['confidence'].mean()),
                    'sample_transactions': cat_corrections['transaction'].head(3).tolist()
                })

        # Sort by occurrences
        patterns.sort(key=lambda x: x['occurrences'], reverse=True)

        return patterns

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable Python types"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def export_feedback_report(self, output_path: str):
        """
        Generate and export comprehensive feedback report

        Args:
            output_path: Path to save report
        """
        stats = self.get_feedback_statistics()
        patterns = self.get_uncertain_patterns()

        report = {
            'statistics': stats,
            'patterns': patterns,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Convert to JSON-serializable format
        report = self._convert_to_json_serializable(report)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Also create markdown version
        md_path = Path(output_path).with_suffix('.md')
        with open(md_path, 'w') as f:
            f.write("# Feedback Analysis Report\n\n")
            f.write(f"Generated: {report['generated_at']}\n\n")

            f.write("## Statistics\n\n")
            f.write(f"- **Total Corrections**: {stats.get('total_corrections', 0)}\n")
            f.write(f"- **Correction Rate**: {stats.get('correction_rate', 0):.2%}\n")
            f.write(f"- **Avg Confidence of Corrections**: {stats.get('avg_confidence_of_corrections', 0):.4f}\n\n")

            if 'corrections_by_confidence' in stats:
                f.write("### Corrections by Confidence Level\n\n")
                for level, count in stats['corrections_by_confidence'].items():
                    f.write(f"- **{level.upper()}**: {count}\n")
                f.write("\n")

            if patterns:
                f.write("## Common Error Patterns\n\n")
                for i, pattern in enumerate(patterns, 1):
                    f.write(f"### Pattern {i}\n\n")
                    f.write(f"- **Predicted Category**: {pattern['predicted_category']}\n")
                    f.write(f"- **Occurrences**: {pattern['occurrences']}\n")
                    f.write(f"- **Avg Confidence**: {pattern['avg_confidence']:.4f}\n")
                    f.write(f"- **Common Actual Categories**: {pattern['common_actual_categories']}\n")
                    f.write(f"- **Sample Transactions**:\n")
                    for trans in pattern['sample_transactions']:
                        f.write(f"  - {trans}\n")
                    f.write("\n")


if __name__ == '__main__':
    # Test feedback handler
    handler = FeedbackHandler()

    # Test prediction handling
    test_prediction = {
        'transaction': 'Starbucks Coffee',
        'predicted_category': 'Food & Dining',
        'confidence': 0.72,
        'alternatives': [
            {'category': 'Shopping', 'confidence': 0.15},
            {'category': 'Entertainment', 'confidence': 0.08}
        ]
    }

    result = handler.handle_prediction(test_prediction)
    print("Feedback handler test result:")
    print(json.dumps(result, indent=2))
