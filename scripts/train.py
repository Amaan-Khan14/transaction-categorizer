"""
Training Script with Reproducibility Guarantees
Trains the ensemble classifier on transaction data
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import TransactionPreprocessor
from models import TransactionClassifier
from taxonomy_loader import TaxonomyLoader


def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # sklearn uses numpy's random state


def load_and_split_data(data_path, test_size=0.15, val_size=0.15, seed=42):
    """
    Load data and create stratified train/val/test splits

    Args:
        data_path: Path to transactions CSV
        test_size: Proportion for test set
        val_size: Proportion for validation set
        seed: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Total samples: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print("\nCategory distribution:")
    print(df['category'].value_counts())

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df['category']
    )

    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=seed,
        stratify=train_val_df['category']
    )

    print(f"\nData split:")
    print(f"  Training: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Train transaction categorization model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to transaction data CSV')
    parser.add_argument('--test_size', type=float, default=0.15,
                       help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.15,
                       help='Validation set proportion')
    parser.add_argument('--max_features', type=int, default=5000,
                       help='Maximum TF-IDF features')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    print(f"Setting random seed to {args.seed} for reproducibility...")
    set_random_seeds(args.seed)

    # Paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config' / 'taxonomy.yaml'
    data_path = args.data_path or (project_root / 'data' / 'raw' / 'transactions.csv')
    models_dir = project_root / 'models'
    processed_dir = project_root / 'data' / 'processed'

    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load taxonomy
    print("\nLoading taxonomy configuration...")
    taxonomy_loader = TaxonomyLoader(str(config_path))
    print(f"Categories loaded: {taxonomy_loader.get_categories()}")

    # Load and split data
    train_df, val_df, test_df = load_and_split_data(
        data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed
    )

    # Save splits
    print("\nSaving data splits...")
    train_df.to_csv(processed_dir / 'train.csv', index=False)
    val_df.to_csv(processed_dir / 'validation.csv', index=False)
    test_df.to_csv(processed_dir / 'test.csv', index=False)

    # Initialize preprocessor
    print("\nInitializing preprocessor...")
    preprocessor = TransactionPreprocessor(
        max_features=args.max_features,
        ngram_range=(1, 3),
        seed=args.seed
    )

    # Fit preprocessor on training data
    print("Fitting preprocessor on training data...")
    X_train = preprocessor.fit_transform(train_df['transaction'].tolist())
    y_train = train_df['category'].values

    print(f"Training feature matrix shape: {X_train.shape}")

    # Transform validation and test sets
    print("Transforming validation and test sets...")
    X_val = preprocessor.transform(val_df['transaction'].tolist())
    y_val = val_df['category'].values

    X_test = preprocessor.transform(test_df['transaction'].tolist())
    y_test = test_df['category'].values

    # Save preprocessor
    print("\nSaving preprocessor...")
    preprocessor.save(models_dir / 'preprocessor.pkl')

    # Initialize and train classifier
    print("\nInitializing ensemble classifier...")
    classifier = TransactionClassifier(seed=args.seed)

    print("Training ensemble (this may take a few minutes)...")
    classifier.fit(X_train, y_train)

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_predictions = classifier.predict(X_val)

    from sklearn.metrics import f1_score, accuracy_score
    val_f1_macro = f1_score(y_val, val_predictions, average='macro')
    val_f1_micro = f1_score(y_val, val_predictions, average='micro')
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"Validation Results:")
    print(f"  Macro F1: {val_f1_macro:.4f}")
    print(f"  Micro F1: {val_f1_micro:.4f}")
    print(f"  Accuracy: {val_accuracy:.4f}")

    # Check if meets threshold
    if val_f1_macro >= 0.90:
        print(f"\n✓ Model meets F1 threshold (≥0.90): {val_f1_macro:.4f}")
    else:
        print(f"\n⚠ Model below F1 threshold: {val_f1_macro:.4f} < 0.90")
        print("  Consider:")
        print("  - Generating more training data")
        print("  - Adjusting model hyperparameters")
        print("  - Adding more features")

    # Save models
    print("\nSaving trained models...")
    classifier.save(models_dir)
    print(f"Models saved to {models_dir}")

    # Save training metadata
    metadata = {
        'seed': args.seed,
        'training_samples': len(train_df),
        'validation_samples': len(val_df),
        'test_samples': len(test_df),
        'num_categories': len(taxonomy_loader.get_categories()),
        'categories': taxonomy_loader.get_categories(),
        'feature_dim': X_train.shape[1],
        'max_features': args.max_features,
        'val_f1_macro': val_f1_macro,
        'val_f1_micro': val_f1_micro,
        'val_accuracy': val_accuracy
    }

    import json
    with open(models_dir / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n✓ Training completed successfully!")
    print(f"\nTo evaluate the model, run:")
    print(f"  python scripts/evaluate.py")


if __name__ == '__main__':
    main()
