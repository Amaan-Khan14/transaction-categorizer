"""
Bias Auditor for Ethical AI
Monitors and reports potential biases in transaction categorization
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict


class BiasAuditor:
    """Audit model predictions for potential biases"""

    def __init__(self, taxonomy_loader, bias_settings: Optional[Dict] = None):
        """
        Initialize bias auditor

        Args:
            taxonomy_loader: TaxonomyLoader instance
            bias_settings: Bias mitigation settings
        """
        self.taxonomy_loader = taxonomy_loader
        self.bias_settings = bias_settings or {
            'monitor_amount': True,
            'monitor_region': True,
            'enforce_balanced_accuracy': True
        }

        self.audit_results = {}

    def audit_per_category_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze per-category performance to identify biased categories

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Per-category performance metrics
        """
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Extract per-category metrics
        categories = self.taxonomy_loader.get_categories()
        per_category_metrics = {}

        for category in categories:
            if category in report:
                metrics = report[category]
                per_category_metrics[category] = {
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1-score'],
                    'support': int(metrics['support'])
                }

        # Calculate variance in performance
        f1_scores = [m['f1_score'] for m in per_category_metrics.values()]
        f1_variance = np.var(f1_scores) if f1_scores else 0
        f1_std = np.std(f1_scores) if f1_scores else 0

        # Identify underperforming categories (F1 < mean - 1 std)
        mean_f1 = np.mean(f1_scores) if f1_scores else 0
        threshold = mean_f1 - f1_std

        underperforming = [
            cat for cat, metrics in per_category_metrics.items()
            if metrics['f1_score'] < threshold
        ]

        results = {
            'per_category_metrics': per_category_metrics,
            'mean_f1': mean_f1,
            'f1_variance': f1_variance,
            'f1_std': f1_std,
            'underperforming_categories': underperforming,
            'bias_detected': len(underperforming) > 0,
            'bias_level': 'high' if f1_std > 0.15 else 'medium' if f1_std > 0.10 else 'low'
        }

        self.audit_results['per_category'] = results
        return results

    def audit_confusion_patterns(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze confusion matrix for systematic misclassification patterns

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion pattern analysis
        """
        categories = self.taxonomy_loader.get_categories()

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=categories)

        # Find most confused pairs
        confused_pairs = []

        for i, true_cat in enumerate(categories):
            for j, pred_cat in enumerate(categories):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'true_category': true_cat,
                        'predicted_category': pred_cat,
                        'count': int(cm[i, j]),
                        'percentage': float(cm[i, j] / cm[i].sum() * 100) if cm[i].sum() > 0 else 0
                    })

        # Sort by count
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)

        # Identify systematic confusions (>10% of true category predictions)
        systematic_confusions = [
            pair for pair in confused_pairs
            if pair['percentage'] > 10
        ]

        results = {
            'confusion_matrix': cm.tolist(),
            'category_labels': categories,
            'top_confused_pairs': confused_pairs[:10],
            'systematic_confusions': systematic_confusions,
            'total_confusions': int(np.sum(cm) - np.trace(cm)),
            'bias_detected': len(systematic_confusions) > 0
        }

        self.audit_results['confusion_patterns'] = results
        return results

    def audit_amount_based_bias(self, transactions: pd.DataFrame,
                                y_true: np.ndarray, y_pred: np.ndarray,
                                amount_column: str = 'amount') -> Dict:
        """
        Analyze if performance varies based on transaction amount

        Args:
            transactions: DataFrame with transaction data
            y_true: True labels
            y_pred: Predicted labels
            amount_column: Name of amount column

        Returns:
            Amount-based bias analysis
        """
        if amount_column not in transactions.columns:
            return {
                'bias_detected': False,
                'message': f'Column {amount_column} not found - skipping amount-based audit'
            }

        # Split into quartiles
        transactions['amount_quartile'] = pd.qcut(
            transactions[amount_column],
            q=4,
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'],
            duplicates='drop'
        )

        # Calculate F1 score for each quartile
        quartile_performance = {}

        for quartile in transactions['amount_quartile'].unique():
            mask = transactions['amount_quartile'] == quartile
            if mask.sum() > 0:
                y_true_q = y_true[mask]
                y_pred_q = y_pred[mask]

                f1 = f1_score(y_true_q, y_pred_q, average='macro', zero_division=0)

                quartile_performance[str(quartile)] = {
                    'f1_score': float(f1),
                    'sample_count': int(mask.sum())
                }

        # Calculate variance across quartiles
        f1_scores = [metrics['f1_score'] for metrics in quartile_performance.values()]
        f1_variance = np.var(f1_scores) if len(f1_scores) > 1 else 0

        # Detect bias if variance is significant (>0.05)
        bias_detected = f1_variance > 0.05

        results = {
            'quartile_performance': quartile_performance,
            'f1_variance': float(f1_variance),
            'bias_detected': bias_detected,
            'bias_level': 'high' if f1_variance > 0.10 else 'medium' if f1_variance > 0.05 else 'low'
        }

        self.audit_results['amount_based'] = results
        return results

    def audit_text_length_bias(self, transactions: List[str],
                               y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze if performance varies based on transaction text length

        Args:
            transactions: List of transaction strings
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Text length bias analysis
        """
        # Calculate text lengths
        lengths = np.array([len(t) for t in transactions])

        # Split into categories
        short_mask = lengths < np.percentile(lengths, 33)
        medium_mask = (lengths >= np.percentile(lengths, 33)) & (lengths < np.percentile(lengths, 67))
        long_mask = lengths >= np.percentile(lengths, 67)

        length_performance = {}

        for mask, label in [(short_mask, 'short'), (medium_mask, 'medium'), (long_mask, 'long')]:
            if mask.sum() > 0:
                y_true_l = y_true[mask]
                y_pred_l = y_pred[mask]

                f1 = f1_score(y_true_l, y_pred_l, average='macro', zero_division=0)

                length_performance[label] = {
                    'f1_score': float(f1),
                    'sample_count': int(mask.sum()),
                    'avg_length': float(lengths[mask].mean())
                }

        # Calculate variance
        f1_scores = [metrics['f1_score'] for metrics in length_performance.values()]
        f1_variance = np.var(f1_scores) if len(f1_scores) > 1 else 0

        bias_detected = f1_variance > 0.05

        results = {
            'length_performance': length_performance,
            'f1_variance': float(f1_variance),
            'bias_detected': bias_detected,
            'bias_level': 'high' if f1_variance > 0.10 else 'medium' if f1_variance > 0.05 else 'low'
        }

        self.audit_results['text_length'] = results
        return results

    def audit_confidence_calibration(self, y_true: np.ndarray,
                                    y_pred_proba: np.ndarray) -> Dict:
        """
        Analyze confidence calibration across categories

        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities

        Returns:
            Confidence calibration analysis
        """
        # Get predicted confidences
        confidences = np.max(y_pred_proba, axis=1)
        predictions = np.argmax(y_pred_proba, axis=1)

        # Encode true labels if needed
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(self.taxonomy_loader.get_categories())
        y_true_encoded = le.transform(y_true)

        # Split by confidence bins
        bins = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_labels = ['<0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']

        calibration_results = {}

        for i in range(len(bins) - 1):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])

            if mask.sum() > 0:
                accuracy = (predictions[mask] == y_true_encoded[mask]).mean()
                avg_confidence = confidences[mask].mean()

                calibration_results[bin_labels[i]] = {
                    'count': int(mask.sum()),
                    'accuracy': float(accuracy),
                    'avg_confidence': float(avg_confidence),
                    'calibration_gap': float(abs(avg_confidence - accuracy))
                }

        # Overall calibration error
        calibration_errors = [r['calibration_gap'] for r in calibration_results.values()]
        mean_calibration_error = np.mean(calibration_errors) if calibration_errors else 0

        results = {
            'calibration_by_bin': calibration_results,
            'mean_calibration_error': float(mean_calibration_error),
            'is_well_calibrated': mean_calibration_error < 0.10
        }

        self.audit_results['confidence_calibration'] = results
        return results

    def generate_bias_report(self) -> Dict:
        """
        Generate comprehensive bias audit report

        Returns:
            Complete bias audit results
        """
        # Aggregate all audit results
        report = {
            'audit_timestamp': pd.Timestamp.now().isoformat(),
            'bias_settings': self.bias_settings,
            'audits_performed': list(self.audit_results.keys()),
            'results': self.audit_results,

            # Overall bias assessment
            'overall_bias_detected': any(
                result.get('bias_detected', False)
                for result in self.audit_results.values()
            ),

            'recommendations': self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on audit results"""
        recommendations = []

        # Per-category performance
        if 'per_category' in self.audit_results:
            per_cat = self.audit_results['per_category']
            if per_cat.get('bias_detected'):
                underperforming = per_cat.get('underperforming_categories', [])
                recommendations.append(
                    f"Collect more training data for underperforming categories: {', '.join(underperforming)}"
                )

        # Confusion patterns
        if 'confusion_patterns' in self.audit_results:
            conf = self.audit_results['confusion_patterns']
            if conf.get('systematic_confusions'):
                recommendations.append(
                    "Add disambiguating features to separate frequently confused categories"
                )

        # Amount-based bias
        if 'amount_based' in self.audit_results:
            amt = self.audit_results['amount_based']
            if amt.get('bias_detected'):
                recommendations.append(
                    "Ensure balanced representation across transaction amount ranges in training data"
                )

        # Text length bias
        if 'text_length' in self.audit_results:
            txt = self.audit_results['text_length']
            if txt.get('bias_detected'):
                recommendations.append(
                    "Improve handling of short/long transaction strings through data augmentation"
                )

        # Confidence calibration
        if 'confidence_calibration' in self.audit_results:
            cal = self.audit_results['confidence_calibration']
            if not cal.get('is_well_calibrated'):
                recommendations.append(
                    "Consider applying calibration techniques (e.g., Platt scaling, isotonic regression)"
                )

        if not recommendations:
            recommendations.append("No significant biases detected - model performs consistently")

        return recommendations

    def _convert_to_json_serializable(self, obj):
        """
        Convert numpy types to JSON-serializable Python types

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
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

    def export_report(self, output_path: str):
        """
        Export bias audit report to file

        Args:
            output_path: Path to save report
        """
        report = self.generate_bias_report()

        # Convert all numpy types to JSON-serializable types
        report = self._convert_to_json_serializable(report)

        # JSON version
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Markdown version
        from pathlib import Path
        md_path = Path(output_path).with_suffix('.md')

        with open(md_path, 'w') as f:
            f.write("# Bias Audit Report\n\n")
            f.write(f"**Generated**: {report['audit_timestamp']}\n\n")

            f.write("## Overall Assessment\n\n")
            f.write(f"- **Bias Detected**: {report['overall_bias_detected']}\n")
            f.write(f"- **Audits Performed**: {', '.join(report['audits_performed'])}\n\n")

            f.write("## Recommendations\n\n")
            for rec in report['recommendations']:
                f.write(f"- {rec}\n")

            f.write("\n## Detailed Results\n\n")

            # Per-category performance
            if 'per_category' in report['results']:
                f.write("### Per-Category Performance\n\n")
                per_cat = report['results']['per_category']
                f.write(f"- Mean F1 Score: {per_cat['mean_f1']:.4f}\n")
                f.write(f"- F1 Standard Deviation: {per_cat['f1_std']:.4f}\n")
                f.write(f"- Bias Level: {per_cat['bias_level']}\n\n")

                if per_cat['underperforming_categories']:
                    f.write("**Underperforming Categories**:\n")
                    for cat in per_cat['underperforming_categories']:
                        f.write(f"- {cat}\n")
                    f.write("\n")

            # Add other audit results...
            for audit_name, audit_results in report['results'].items():
                if audit_name != 'per_category':
                    f.write(f"### {audit_name.replace('_', ' ').title()}\n\n")
                    f.write(f"```json\n{json.dumps(audit_results, indent=2)}\n```\n\n")


if __name__ == '__main__':
    print("Bias auditor module loaded successfully!")
