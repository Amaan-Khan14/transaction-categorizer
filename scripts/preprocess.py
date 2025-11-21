"""
Data Preprocessing Script
Prepares raw data for training
"""

import sys
from pathlib import Path
import argparse
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import TransactionPreprocessor, RobustnessEvaluator


def main():
    parser = argparse.ArgumentParser(description='Preprocess transaction data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--input', type=str, help='Input CSV file')
    parser.add_argument('--output', type=str, help='Output CSV file')
    parser.add_argument('--test_robustness', action='store_true',
                       help='Test preprocessing robustness')

    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    input_path = args.input or (project_root / 'data' / 'raw' / 'transactions.csv')
    output_path = args.output or (project_root / 'data' / 'processed' / 'preprocessed.csv')

    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} transactions")

    # Initialize preprocessor
    print("\nInitializing preprocessor...")
    preprocessor = TransactionPreprocessor(seed=args.seed)

    # Normalize transactions
    print("Normalizing transactions...")
    df['normalized'] = df['transaction'].apply(preprocessor.normalize_text)

    # Extract features
    print("Extracting hand-crafted features...")
    features_list = []
    for text in df['normalized']:
        features = preprocessor.extract_features(text)
        features_list.append(features)

    # Add features to DataFrame
    features_df = pd.DataFrame(features_list)
    result_df = pd.concat([df, features_df], axis=1)

    # Save preprocessed data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\nPreprocessed data saved to {output_path}")

    # Display sample
    print("\nSample preprocessed transactions:")
    print(result_df[['transaction', 'normalized', 'text_length', 'word_count']].head(10))

    # Robustness testing
    if args.test_robustness:
        print("\n" + "=" * 80)
        print("ROBUSTNESS TESTING")
        print("=" * 80)

        # Fit preprocessor for robustness testing
        preprocessor.fit(df['transaction'].tolist())
        evaluator = RobustnessEvaluator(preprocessor)

        test_transactions = [
            "Starbucks Coffee",
            "Amazon.com",
            "Shell Gas Station"
        ]

        for transaction in test_transactions:
            print(f"\n\nOriginal: {transaction}")
            print("-" * 40)

            # Test case variations
            print("\nCase Variations:")
            case_results = evaluator.test_case_variations(transaction)
            for var, _ in case_results:
                normalized = preprocessor.normalize_text(var)
                print(f"  {var:30} → {normalized}")

            # Test special characters
            print("\nSpecial Character Handling:")
            char_results = evaluator.test_special_chars(transaction)
            for var, _ in char_results:
                normalized = preprocessor.normalize_text(var)
                print(f"  {var:30} → {normalized}")

        print("\n" + "=" * 80)
        print("Robustness testing completed!")
        print("=" * 80)


if __name__ == '__main__':
    main()
