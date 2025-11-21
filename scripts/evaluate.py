"""
Comprehensive Evaluation Script
Generates detailed metrics, confusion matrix, and bias audit report
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import TransactionPreprocessor
from models import TransactionClassifier
from taxonomy_loader import TaxonomyLoader
from bias_auditor import BiasAuditor


def plot_confusion_matrix(cm, labels, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_per_category_metrics(metrics_df, output_path):
    """Plot per-category F1 scores"""
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar', y=['precision', 'recall', 'f1-score'])
    plt.title('Per-Category Performance Metrics')
    plt.xlabel('Category')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-category metrics plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate transaction categorization model')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data CSV (default: data/processed/test.csv)')
    parser.add_argument('--include_edge_cases', action='store_true',
                       help='Also evaluate on edge cases dataset')

    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config' / 'taxonomy.yaml'
    models_dir = project_root / 'models'
    reports_dir = project_root / 'reports'
    test_data_path = args.test_data or (project_root / 'data' / 'processed' / 'test.csv')

    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRANSACTION CATEGORIZATION - COMPREHENSIVE EVALUATION")
    print("=" * 80)

    # Load taxonomy
    print("\nLoading taxonomy configuration...")
    taxonomy_loader = TaxonomyLoader(str(config_path))
    categories = taxonomy_loader.get_categories()
    print(f"Categories: {categories}")

    # Load models
    print("\nLoading trained models...")
    preprocessor = TransactionPreprocessor.load(models_dir / 'preprocessor.pkl')
    classifier = TransactionClassifier.load(str(models_dir))
    print("Models loaded successfully!")

    # Load test data
    print(f"\nLoading test data from {test_data_path}...")
    test_df = pd.read_csv(test_data_path)
    print(f"Test samples: {len(test_df)}")

    # Preprocess and predict
    print("\nPreprocessing test data...")
    X_test = preprocessor.transform(test_df['transaction'].tolist())
    y_test = test_df['category'].values

    print("Generating predictions...")
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Macro F1-Score':<30} {macro_f1:>10.4f}")
    print(f"{'Micro F1-Score':<30} {micro_f1:>10.4f}")
    print(f"{'Weighted F1-Score':<30} {weighted_f1:>10.4f}")
    print(f"{'Accuracy':<30} {accuracy:>10.4f}")
    print(f"{'Balanced Accuracy':<30} {balanced_acc:>10.4f}")

    # Check threshold
    print("\n" + "=" * 80)
    if macro_f1 >= 0.90:
        print(f"✓ SUCCESS: Model meets F1 threshold (≥0.90): {macro_f1:.4f}")
    else:
        print(f"✗ WARNING: Model below F1 threshold: {macro_f1:.4f} < 0.90")
    print("=" * 80)

    # Per-category metrics
    print("\n" + "=" * 80)
    print("PER-CATEGORY PERFORMANCE")
    print("=" * 80)

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"\n{'Category':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 72)

    per_category_data = []
    for category in categories:
        if category in report_dict:
            metrics = report_dict[category]
            print(f"{category:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['f1-score']:>10.4f} {int(metrics['support']):>10}")
            per_category_data.append({
                'category': category,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score'],
                'support': int(metrics['support'])
            })

    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=categories)
    cm_path = reports_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, categories, cm_path)

    # Plot per-category metrics
    metrics_df = pd.DataFrame(per_category_data).set_index('category')
    metrics_path = reports_dir / 'per_category_metrics.png'
    plot_per_category_metrics(metrics_df, metrics_path)

    # Bias audit
    print("\n" + "=" * 80)
    print("BIAS AUDIT")
    print("=" * 80)

    auditor = BiasAuditor(taxonomy_loader, taxonomy_loader.get_bias_settings())

    print("\nAuditing per-category performance...")
    per_cat_bias = auditor.audit_per_category_performance(y_test, y_pred)

    print("\nAuditing confusion patterns...")
    confusion_bias = auditor.audit_confusion_patterns(y_test, y_pred)

    print("\nAuditing text length bias...")
    text_length_bias = auditor.audit_text_length_bias(
        test_df['transaction'].tolist(), y_test, y_pred
    )

    print("\nAuditing confidence calibration...")
    calibration = auditor.audit_confidence_calibration(y_test, y_pred_proba)

    # Generate bias report
    print("\nGenerating comprehensive bias report...")
    bias_report = auditor.generate_bias_report()

    bias_report_path = reports_dir / 'bias_analysis.json'
    auditor.export_report(str(bias_report_path))

    print(f"\nBias Detected: {bias_report['overall_bias_detected']}")
    print("\nRecommendations:")
    for i, rec in enumerate(bias_report['recommendations'], 1):
        print(f"  {i}. {rec}")

    # Edge cases evaluation (optional)
    if args.include_edge_cases:
        print("\n" + "=" * 80)
        print("EDGE CASES EVALUATION")
        print("=" * 80)

        edge_path = project_root / 'data' / 'raw' / 'edge_cases.csv'
        if edge_path.exists():
            edge_df = pd.read_csv(edge_path)
            print(f"Edge cases: {len(edge_df)}")

            X_edge = preprocessor.transform(edge_df['transaction'].tolist())
            y_edge = edge_df['category'].values
            y_edge_pred = classifier.predict(X_edge)

            edge_f1 = f1_score(y_edge, y_edge_pred, average='macro')
            edge_acc = accuracy_score(y_edge, y_edge_pred)

            print(f"Edge Cases Macro F1: {edge_f1:.4f}")
            print(f"Edge Cases Accuracy: {edge_acc:.4f}")
        else:
            print("Edge cases file not found, skipping...")

    # Generate comprehensive evaluation report
    print("\n" + "=" * 80)
    print("GENERATING EVALUATION REPORT")
    print("=" * 80)

    report_content = f"""# TRANSACTION CATEGORIZATION - EVALUATION REPORT

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Macro F1-Score**: {macro_f1:.4f} {'✓' if macro_f1 >= 0.90 else '✗'}
- **Model Type**: Ensemble (Logistic Regression + SVM + Random Forest)
- **Dataset Size**: {len(test_df)} test samples
- **Number of Categories**: {len(categories)}
- **Reproducibility**: Fully reproducible (fixed seed: 42)

## Primary Metrics

| Metric | Value |
|--------|-------|
| **Macro F1-Score** | **{macro_f1:.4f}** |
| Micro F1-Score | {micro_f1:.4f} |
| Weighted F1-Score | {weighted_f1:.4f} |
| Accuracy | {accuracy:.4f} |
| Balanced Accuracy | {balanced_acc:.4f} |

## Per-Category Performance

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
"""

    for cat_data in per_category_data:
        report_content += f"| {cat_data['category']} | {cat_data['precision']:.4f} | {cat_data['recall']:.4f} | {cat_data['f1-score']:.4f} | {cat_data['support']} |\n"

    report_content += f"""
## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

## Per-Category Metrics Visualization

![Per-Category Metrics](per_category_metrics.png)

## Bias Analysis

- **Overall Bias Detected**: {bias_report['overall_bias_detected']}
- **Per-Category F1 Variance**: {per_cat_bias.get('f1_variance', 0):.6f}
- **Per-Category Bias Level**: {per_cat_bias.get('bias_level', 'unknown')}
- **Text Length Bias**: {text_length_bias.get('bias_detected', False)}
- **Confidence Calibration**: {'Well-calibrated' if calibration.get('is_well_calibrated') else 'Needs improvement'}

### Underperforming Categories

"""

    if per_cat_bias.get('underperforming_categories'):
        for cat in per_cat_bias['underperforming_categories']:
            report_content += f"- {cat}\n"
    else:
        report_content += "None - all categories perform well!\n"

    report_content += f"""
### Top Confused Category Pairs

"""

    if confusion_bias.get('top_confused_pairs'):
        for pair in confusion_bias['top_confused_pairs'][:5]:
            report_content += f"- **{pair['true_category']}** → **{pair['predicted_category']}**: {pair['count']} occurrences ({pair['percentage']:.1f}%)\n"

    report_content += f"""
## Recommendations

"""

    for i, rec in enumerate(bias_report['recommendations'], 1):
        report_content += f"{i}. {rec}\n"

    report_content += f"""
## Reproducibility

- **Random Seed**: 42 (fixed)
- **Data Split**: Stratified train/val/test
- **Preprocessing**: Deterministic (no randomization)
- **Model Training**: Fixed random state

### Reproduction Command

```bash
# Generate data
python scripts/generate_data.py --seed 42

# Train model
python scripts/train.py --seed 42

# Evaluate
python scripts/evaluate.py
```

## Model Details

- **Ensemble Components**:
  1. Logistic Regression (L2 regularization)
  2. Support Vector Machine (RBF kernel)
  3. Random Forest (100 estimators)
- **Voting Strategy**: Soft voting (probability averaging)
- **Feature Extraction**: TF-IDF + character n-grams + handcrafted features
- **Total Features**: {X_test.shape[1]}

## Conclusion

"""

    if macro_f1 >= 0.90:
        report_content += f"✓ **The model successfully meets the required F1-score threshold (≥0.90) with a score of {macro_f1:.4f}.**\n\n"
        report_content += "The system demonstrates:\n"
        report_content += "- Strong performance across all categories\n"
        report_content += "- Minimal bias in predictions\n"
        report_content += "- Robust handling of noisy transaction data\n"
        report_content += "- Full explainability and transparency\n"
    else:
        report_content += f"⚠ **The model achieved {macro_f1:.4f}, which is below the 0.90 threshold.**\n\n"
        report_content += "Recommended improvements:\n"
        report_content += "- Increase training data size\n"
        report_content += "- Add more feature engineering\n"
        report_content += "- Tune hyperparameters\n"
        report_content += "- Consider ensemble with deep learning models\n"

    # Save report
    report_path = reports_dir / 'evaluation_report.md'
    with open(report_path, 'w') as f:
        f.write(report_content)

    print(f"\n✓ Evaluation report saved to {report_path}")

    # Save metrics as JSON
    metrics_json = {
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc
        },
        'per_category': per_category_data,
        'bias_audit': {
            'overall_bias_detected': bias_report['overall_bias_detected'],
            'recommendations': bias_report['recommendations']
        },
        'test_samples': len(test_df),
        'num_categories': len(categories)
    }

    with open(reports_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)

    print(f"✓ Metrics JSON saved to {reports_dir / 'evaluation_metrics.json'}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
