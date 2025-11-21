"""
Synthetic Transaction Data Generator
Generates realistic financial transaction strings with variations
"""

import pandas as pd
import numpy as np
import random
import yaml
import argparse
from pathlib import Path


class TransactionGenerator:
    def __init__(self, config_path, seed=42):
        """Initialize the generator with taxonomy configuration"""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.categories = config['categories']
        self.merchants = self._build_merchant_list()

    def _build_merchant_list(self):
        """Extract merchants from taxonomy"""
        merchants = {}
        for category, details in self.categories.items():
            merchants[category] = details['keywords']
        return merchants

    def _generate_variations(self, merchant):
        """Generate variations of a merchant name"""
        variations = [merchant]

        # Original case
        variations.append(merchant)

        # Uppercase
        variations.append(merchant.upper())

        # Title case
        variations.append(merchant.title())

        # Random case
        variations.append(''.join(random.choice([c.upper(), c.lower()]) for c in merchant))

        # With extra spaces
        variations.append(f"  {merchant}  ")

        # With special characters
        variations.append(f"{merchant}*")
        variations.append(f"{merchant}.com")
        variations.append(f"{merchant}-store")
        variations.append(f"{merchant}/shop")

        # With location/numbers
        variations.append(f"{merchant} #{random.randint(100, 9999)}")
        variations.append(f"{merchant} {random.randint(1, 999)} main st")
        variations.append(f"{merchant} store {random.randint(1, 500)}")

        # Typos (Levenshtein distance 1-2)
        if len(merchant) > 4:
            # Character deletion
            idx = random.randint(1, len(merchant) - 2)
            variations.append(merchant[:idx] + merchant[idx+1:])

            # Character substitution
            idx = random.randint(0, len(merchant) - 1)
            char_list = list(merchant)
            char_list[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
            variations.append(''.join(char_list))

        # Abbreviations
        if len(merchant.split()) > 1:
            abbrev = ''.join([word[0] for word in merchant.split()])
            variations.append(abbrev.upper())

        return variations

    def generate_dataset(self, n_samples=10000, distribution='balanced'):
        """
        Generate synthetic transaction dataset

        Args:
            n_samples: Total number of transactions
            distribution: 'balanced' or 'realistic' (imbalanced)
        """
        transactions = []
        labels = []

        categories_list = list(self.merchants.keys())

        if distribution == 'balanced':
            # Equal samples per category
            samples_per_category = n_samples // len(categories_list)
            category_counts = {cat: samples_per_category for cat in categories_list}
        else:
            # Realistic distribution (some categories more common)
            weights = [0.20, 0.18, 0.15, 0.15, 0.12, 0.10, 0.07, 0.03]
            category_counts = {cat: int(n_samples * w) for cat, w in zip(categories_list, weights)}

        for category, count in category_counts.items():
            merchants = self.merchants[category]

            for _ in range(count):
                # Select random merchant
                merchant = random.choice(merchants)

                # Generate variation
                variations = self._generate_variations(merchant)
                transaction = random.choice(variations)

                transactions.append(transaction)
                labels.append(category)

        # Create DataFrame
        df = pd.DataFrame({
            'transaction': transactions,
            'category': labels
        })

        # Shuffle
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        return df

    def generate_edge_cases(self, n_samples=100):
        """Generate challenging edge cases for robustness testing"""
        edge_cases = []
        labels = []

        categories_list = list(self.merchants.keys())
        samples_per_category = n_samples // len(categories_list)

        for category in categories_list:
            merchants = self.merchants[category]

            for _ in range(samples_per_category):
                merchant = random.choice(merchants)

                # Generate challenging variations
                variations = [
                    # Minimal text
                    merchant[:3] if len(merchant) > 3 else merchant,

                    # Extreme noise
                    f"***{merchant}***",
                    f"@@@{merchant}@@@",

                    # Multiple typos
                    self._add_multiple_typos(merchant, 2),

                    # Mixed with random text
                    f"{merchant} {random.choice(['payment', 'transaction', 'purchase', 'sale'])}",

                    # Abbreviation + noise
                    ''.join([word[0] for word in merchant.split()]).upper() + str(random.randint(1, 999)),
                ]

                edge_case = random.choice(variations)
                edge_cases.append(edge_case)
                labels.append(category)

        df = pd.DataFrame({
            'transaction': edge_cases,
            'category': labels
        })

        return df

    def _add_multiple_typos(self, text, n_typos):
        """Add multiple typos to text"""
        if len(text) < 4:
            return text

        text_list = list(text)
        for _ in range(min(n_typos, len(text) // 2)):
            idx = random.randint(0, len(text_list) - 1)
            text_list[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')

        return ''.join(text_list)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic transaction data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_samples', type=int, default=10000, help='Total samples')
    parser.add_argument('--distribution', type=str, default='balanced',
                        choices=['balanced', 'realistic'], help='Distribution type')

    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config' / 'taxonomy.yaml'
    output_dir = project_root / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    print(f"Generating {args.n_samples} transactions with seed={args.seed}...")
    generator = TransactionGenerator(config_path, seed=args.seed)

    # Main dataset
    df = generator.generate_dataset(args.n_samples, args.distribution)
    output_path = output_dir / 'transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved main dataset to {output_path}")

    # Edge cases
    edge_df = generator.generate_edge_cases(n_samples=200)
    edge_output_path = output_dir / 'edge_cases.csv'
    edge_df.to_csv(edge_output_path, index=False)
    print(f"Saved edge cases to {edge_output_path}")

    # Print statistics
    print("\n--- Dataset Statistics ---")
    print(f"Total samples: {len(df)}")
    print("\nCategory distribution:")
    print(df['category'].value_counts().sort_index())
    print(f"\nSample transactions:")
    print(df.head(10))

    # Create data documentation
    doc_path = project_root / 'data' / 'source_documentation.md'
    with open(doc_path, 'w') as f:
        f.write("# Data Source Documentation\n\n")
        f.write("## Data Generation Method\n\n")
        f.write("**Type**: Synthetic data generated programmatically\n\n")
        f.write(f"**Total Samples**: {len(df)}\n\n")
        f.write(f"**Random Seed**: {args.seed}\n\n")
        f.write(f"**Distribution**: {args.distribution}\n\n")
        f.write("## Generation Strategy\n\n")
        f.write("1. **Base Merchants**: Extracted from taxonomy configuration (config/taxonomy.yaml)\n")
        f.write("2. **Variations Generated**:\n")
        f.write("   - Case variations (lowercase, UPPERCASE, Title Case, rAnDoM)\n")
        f.write("   - Special characters (.com, -store, /shop, *)\n")
        f.write("   - Location/number suffixes (#1234, 123 Main St, Store 456)\n")
        f.write("   - Typos (character deletion, substitution)\n")
        f.write("   - Abbreviations (multi-word merchants)\n")
        f.write("   - Whitespace noise\n\n")
        f.write("## Category Distribution\n\n")
        f.write(df['category'].value_counts().sort_index().to_markdown())
        f.write("\n\n## Sample Transactions\n\n")
        f.write(df.head(20).to_markdown(index=False))
        f.write("\n\n## Known Limitations\n\n")
        f.write("- Synthetic data may not capture all real-world edge cases\n")
        f.write("- Merchant list is based on common US merchants (cultural bias)\n")
        f.write("- No temporal or seasonal patterns\n")
        f.write("- Transaction amounts not included in this version\n")

    print(f"\nDocumentation saved to {doc_path}")


if __name__ == '__main__':
    main()
