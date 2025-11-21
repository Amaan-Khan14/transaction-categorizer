"""
Prediction Script for Inference
Provides transaction categorization with explanations
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import TransactionPreprocessor
from models import TransactionClassifier
from taxonomy_loader import TaxonomyLoader
from explainer import TransactionExplainer
from feedback_handler import FeedbackHandler


def convert_to_json_serializable(obj):
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
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description='Predict transaction categories')
    parser.add_argument('--transaction', type=str, help='Single transaction to categorize')
    parser.add_argument('--file', type=str, help='CSV file with transactions')
    parser.add_argument('--explain', action='store_true', help='Include detailed explanations')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--output', type=str, help='Output file path (JSON)')

    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config' / 'taxonomy.yaml'
    models_dir = project_root / 'models'

    # Check if models exist
    if not (models_dir / 'preprocessor.pkl').exists():
        print("Error: Models not found. Please train the model first:")
        print("  python scripts/train.py")
        return

    # Load components
    print("Loading models and configuration...")
    taxonomy_loader = TaxonomyLoader(str(config_path))
    preprocessor = TransactionPreprocessor.load(models_dir / 'preprocessor.pkl')
    classifier = TransactionClassifier.load(str(models_dir))

    confidence_thresholds = taxonomy_loader.get_confidence_thresholds()
    feedback_handler = FeedbackHandler(confidence_thresholds)

    # Initialize explainer if requested
    explainer = None
    if args.explain:
        explainer = TransactionExplainer(
            classifier,
            preprocessor,
            taxonomy_loader.get_categories()
        )

    print("Models loaded successfully!\n")

    # Interactive mode
    if args.interactive:
        print("=" * 80)
        print("INTERACTIVE TRANSACTION CATEGORIZATION")
        print("=" * 80)
        print("\nEnter transactions to categorize (type 'quit' to exit)\n")

        while True:
            transaction = input("Transaction: ").strip()

            if transaction.lower() in ['quit', 'exit', 'q']:
                break

            if not transaction:
                continue

            # Make prediction
            if args.explain:
                result = explainer.explain_prediction(transaction)
                print("\n" + "-" * 80)
                print(f"Transaction: {result['transaction']}")
                print(f"Predicted Category: {result['predicted_category']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Confidence Flag: {result['confidence_flag']}")

                print("\nTop Features:")
                for feat in result['explanation']['top_features'][:5]:
                    print(f"  - {feat['feature']}: {feat['weight']:.4f} ({feat['impact']})")

                print("\nModel Votes:")
                for model, vote in result['explanation']['model_votes'].items():
                    print(f"  - {model}: {vote['category']} ({vote['confidence']:.2%})")

                if result['alternatives']:
                    print("\nAlternatives:")
                    for alt in result['alternatives']:
                        print(f"  - {alt['category']}: {alt['confidence']:.2%}")

                # Feedback handling
                handled = feedback_handler.handle_prediction(result)
                print(f"\nStatus: {handled['status']}")
                print(f"Message: {handled['message']}")

            else:
                # Simple prediction
                features = preprocessor.transform([transaction])
                prediction = classifier.predict_with_confidence(features, top_k=3)[0]

                print("\n" + "-" * 80)
                print(f"Transaction: {transaction}")
                print(f"Predicted Category: {prediction['predicted_category']}")
                print(f"Confidence: {prediction['confidence']:.2%}")

                if prediction['alternatives']:
                    print("\nAlternatives:")
                    for alt in prediction['alternatives']:
                        print(f"  - {alt['category']}: {alt['confidence']:.2%}")

            print("-" * 80 + "\n")

    # Single transaction
    elif args.transaction:
        transaction = args.transaction

        if args.explain:
            result = explainer.explain_prediction(transaction)
            result = convert_to_json_serializable(result)

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Explanation saved to {args.output}")
            else:
                print(json.dumps(result, indent=2))

        else:
            features = preprocessor.transform([transaction])
            prediction = classifier.predict_with_confidence(features, top_k=3)[0]

            output = {
                'transaction': transaction,
                'predicted_category': prediction['predicted_category'],
                'confidence': prediction['confidence'],
                'alternatives': prediction['alternatives']
            }
            output = convert_to_json_serializable(output)

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(output, f, indent=2)
                print(f"Prediction saved to {args.output}")
            else:
                print(json.dumps(output, indent=2))

    # Batch file
    elif args.file:
        import pandas as pd

        print(f"Processing transactions from {args.file}...")
        df = pd.read_csv(args.file)

        if 'transaction' not in df.columns:
            print("Error: CSV file must have a 'transaction' column")
            return

        transactions = df['transaction'].tolist()
        print(f"Found {len(transactions)} transactions")

        if args.explain:
            print("Generating explanations (this may take a while)...")
            results = explainer.explain_batch(transactions)

            # Generate summary
            summary = explainer.generate_summary_report(results)
            print(f"\n{'='*80}")
            print("BATCH PREDICTION SUMMARY")
            print(f"{'='*80}")
            print(f"Total Predictions: {summary['total_predictions']}")
            print(f"Average Confidence: {summary['overall_avg_confidence']:.4f}")
            print(f"\nConfidence Distribution:")
            for level, count in summary['confidence_distribution'].items():
                pct = summary['confidence_distribution_pct'][level]
                print(f"  {level}: {count} ({pct}%)")
            print(f"\nRequires Review: {summary['requires_review']} ({summary['requires_review_pct']}%)")

        else:
            print("Generating predictions...")
            features = preprocessor.transform(transactions)
            predictions = classifier.predict_with_confidence(features, top_k=3)

            results = []
            for trans, pred in zip(transactions, predictions):
                results.append({
                    'transaction': trans,
                    'predicted_category': pred['predicted_category'],
                    'confidence': pred['confidence'],
                    'alternatives': pred['alternatives']
                })

        # Save results
        if args.output:
            output_path = args.output
        else:
            output_path = Path(args.file).stem + '_predictions.json'

        # Convert to JSON-serializable format
        results = convert_to_json_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nPredictions saved to {output_path}")

    else:
        print("Error: Please provide --transaction, --file, or --interactive")
        print("\nExamples:")
        print("  python scripts/predict.py --transaction 'Starbucks Coffee'")
        print("  python scripts/predict.py --transaction 'Amazon.com' --explain")
        print("  python scripts/predict.py --file transactions.csv --output predictions.json")
        print("  python scripts/predict.py --interactive --explain")


if __name__ == '__main__':
    main()
