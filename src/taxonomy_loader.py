"""
Dynamic Taxonomy Configuration Loader
Supports runtime category updates without code changes
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
import json


class TaxonomyLoader:
    """Load and manage category taxonomy from configuration files"""

    def __init__(self, config_path: str):
        """
        Initialize taxonomy loader

        Args:
            config_path: Path to taxonomy YAML configuration
        """
        self.config_path = Path(config_path)
        self.taxonomy = None
        self.categories = None
        self.keywords_map = None
        self.confidence_thresholds = None
        self.bias_settings = None

        self.load()

    def load(self) -> Dict:
        """
        Load taxonomy from YAML file

        Returns:
            Taxonomy dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Taxonomy config not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.taxonomy = yaml.safe_load(f)

        # Validate schema
        self._validate_schema()

        # Extract components
        self.categories = list(self.taxonomy['categories'].keys())
        self.keywords_map = {
            cat: details['keywords']
            for cat, details in self.taxonomy['categories'].items()
        }

        self.confidence_thresholds = self.taxonomy.get('confidence_thresholds', {
            'high': 0.85,
            'medium': 0.70,
            'low': 0.60
        })

        self.bias_settings = self.taxonomy.get('bias_mitigation', {})

        return self.taxonomy

    def _validate_schema(self):
        """Validate taxonomy schema"""
        required_keys = ['categories', 'version']

        for key in required_keys:
            if key not in self.taxonomy:
                raise ValueError(f"Missing required key in taxonomy: {key}")

        # Validate each category
        for cat_name, cat_details in self.taxonomy['categories'].items():
            if 'keywords' not in cat_details:
                raise ValueError(f"Category '{cat_name}' missing 'keywords' field")

            if not isinstance(cat_details['keywords'], list):
                raise ValueError(f"Category '{cat_name}' keywords must be a list")

    def get_categories(self) -> List[str]:
        """Get list of all categories"""
        return self.categories

    def get_keywords(self, category: str) -> List[str]:
        """Get keywords for a specific category"""
        if category not in self.keywords_map:
            raise ValueError(f"Category not found: {category}")
        return self.keywords_map[category]

    def get_all_keywords(self) -> Dict[str, List[str]]:
        """Get all keywords mapped to categories"""
        return self.keywords_map

    def get_confidence_boost(self, category: str) -> float:
        """Get confidence boost for a category"""
        return self.taxonomy['categories'].get(category, {}).get('confidence_boost', 0.0)

    def get_confidence_thresholds(self) -> Dict[str, float]:
        """Get confidence threshold settings"""
        return self.confidence_thresholds

    def get_bias_settings(self) -> Dict:
        """Get bias mitigation settings"""
        return self.bias_settings

    def update_category(self, category: str, keywords: List[str],
                       aliases: Optional[List[str]] = None,
                       confidence_boost: float = 0.0):
        """
        Update or add a category at runtime

        Args:
            category: Category name
            keywords: List of keywords
            aliases: Optional list of aliases
            confidence_boost: Confidence boost value
        """
        if category not in self.taxonomy['categories']:
            self.taxonomy['categories'][category] = {}

        self.taxonomy['categories'][category]['keywords'] = keywords

        if aliases:
            self.taxonomy['categories'][category]['aliases'] = aliases

        self.taxonomy['categories'][category]['confidence_boost'] = confidence_boost

        # Update internal state
        self.categories = list(self.taxonomy['categories'].keys())
        self.keywords_map[category] = keywords

    def remove_category(self, category: str):
        """
        Remove a category from taxonomy

        Args:
            category: Category name to remove
        """
        if category in self.taxonomy['categories']:
            del self.taxonomy['categories'][category]
            self.categories = list(self.taxonomy['categories'].keys())
            if category in self.keywords_map:
                del self.keywords_map[category]

    def save(self, output_path: Optional[str] = None):
        """
        Save taxonomy to YAML file

        Args:
            output_path: Optional custom output path
        """
        save_path = Path(output_path) if output_path else self.config_path

        with open(save_path, 'w') as f:
            yaml.dump(self.taxonomy, f, default_flow_style=False, sort_keys=False)

    def export_to_json(self, output_path: str):
        """
        Export taxonomy to JSON format

        Args:
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self.taxonomy, f, indent=2)

    def get_category_count(self) -> int:
        """Get total number of categories"""
        return len(self.categories)

    def keyword_to_category_map(self) -> Dict[str, str]:
        """
        Create reverse mapping from keywords to categories

        Returns:
            Dictionary mapping keywords to category names
        """
        keyword_map = {}
        for category, keywords in self.keywords_map.items():
            for keyword in keywords:
                if keyword in keyword_map:
                    # Handle keyword appearing in multiple categories
                    if isinstance(keyword_map[keyword], list):
                        keyword_map[keyword].append(category)
                    else:
                        keyword_map[keyword] = [keyword_map[keyword], category]
                else:
                    keyword_map[keyword] = category

        return keyword_map

    def get_statistics(self) -> Dict:
        """
        Get taxonomy statistics

        Returns:
            Dictionary with statistics
        """
        stats = {
            'version': self.taxonomy.get('version'),
            'total_categories': len(self.categories),
            'categories': {},
            'total_keywords': 0
        }

        for category in self.categories:
            keywords = self.keywords_map[category]
            stats['categories'][category] = {
                'keyword_count': len(keywords),
                'has_aliases': 'aliases' in self.taxonomy['categories'][category],
                'confidence_boost': self.get_confidence_boost(category)
            }
            stats['total_keywords'] += len(keywords)

        stats['avg_keywords_per_category'] = stats['total_keywords'] / len(self.categories)

        return stats

    def __repr__(self):
        return f"TaxonomyLoader(categories={len(self.categories)}, version={self.taxonomy.get('version')})"


if __name__ == '__main__':
    # Test taxonomy loader
    config_path = Path(__file__).parent.parent / 'config' / 'taxonomy.yaml'

    if config_path.exists():
        loader = TaxonomyLoader(str(config_path))
        print("Taxonomy loaded successfully!")
        print(f"\nCategories: {loader.get_categories()}")
        print(f"\nStatistics:")

        stats = loader.get_statistics()
        for key, value in stats.items():
            if key != 'categories':
                print(f"  {key}: {value}")

        print("\nConfidence Thresholds:")
        for level, threshold in loader.get_confidence_thresholds().items():
            print(f"  {level}: {threshold}")
    else:
        print(f"Config file not found: {config_path}")
