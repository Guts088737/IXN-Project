"""
Focus on the collection of country data
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from .base_collector import ConfigurableCollector
except ImportError:
    from src.data_collection.base_collector import ConfigurableCollector


class CountryCollector(ConfigurableCollector):
    """
    Collect basic information of all countries
    """

    def get_data_directory(self) -> Path:
        return self.config.paths.countries_dir

    def get_all_countries_info(self, silent: bool = False) -> List[Dict]:
        """Obtain all countries information"""
        # Check cache
        cached_data = self._load_from_cache('countries/all_countries')
        if cached_data:
            # Filter deprecated countries from cache
            active_countries = self._filter_active_countries(cached_data)
            if not silent:
                self.logger.info(f"Load {len(active_countries)} active countries from cache.")
            return active_countries

        # Obtain from API
        if not silent:
            self.logger.info("Obtain country data from the API...")

        countries_response = self.make_api_request(
            'countries',
            params={'limit': 1000, 'ordering': 'name'},
            use_cache=False
        )

        if isinstance(countries_response, dict):
            countries = countries_response.get('results', [])
        elif isinstance(countries_response, list):
            countries = countries_response
        else:
            self.logger.error(f"Unknown API response format: {type(countries_response)}")
            return []

        # Filter deprecated countries before saving
        active_countries = self._filter_active_countries(countries)
        
        # save to cache (only active countries)
        if active_countries:
            self._save_to_cache(active_countries, 'countries/all_countries')
            if not silent:
                filtered_count = len(countries) - len(active_countries)
                self.logger.info(f"Successfully retrieved {len(active_countries)} countries (filtered {filtered_count} regions/clusters/deprecated).")

        return active_countries

    def get_all_country_ids(self, silent: bool = False) -> List[int]:
        countries = self.get_all_countries_info(silent=silent)
        return [c['id'] for c in countries if c.get('id')]

    def collect_data(self, silent: bool = False) -> Dict[str, Any]:
        """Data collection"""
        countries = self.get_all_countries_info(silent=silent)

        # save data
        if not silent:
            self.save_data(countries, "all_countries")

        return {
            'data': countries,
            'collection_metadata': {
                'total_active_countries': len(countries),
                'active_countries_with_ids': len([c for c in countries if c.get('id')]),
                'collection_timestamp': datetime.now().isoformat(),
                'api_endpoint': 'country'
            }
        }

    def get_country_by_id(self, country_id: int) -> Optional[Dict]:
        countries = self.get_all_countries_info(silent=True)
        return next((c for c in countries if c.get('id') == country_id), None)

    def get_country_by_name(self, country_name: str) -> Optional[Dict]:
        countries = self.get_all_countries_info(silent=True)
        return next((c for c in countries if c.get('name', '').lower() == country_name.lower()), None)

    def get_countries_by_region(self, region_id: int) -> List[Dict]:
        countries = self.get_all_countries_info(silent=True)
        return [c for c in countries if c.get('region') == region_id]

    def _filter_active_countries(self, countries: List[Dict]) -> List[Dict]:
        """Filter out regions, clusters, deprecated countries and invalid records"""
        real_countries = []
        for country in countries:
            if not self._is_deprecated_country(country):
                real_countries.append(country)
        return real_countries

    def _is_deprecated_country(self, country: Dict) -> bool:
        """Check if a country is deprecated or not a real country"""
        # Check is_deprecated field from API
        if country.get('is_deprecated'):
            return True
        
        # Filter out non-country records (只保留真正的国家)
        record_type_display = country.get('record_type_display', '')
        if record_type_display != 'Country':
            return True
        
        # Additional validation: check if required fields are missing
        if not country.get('id') or not country.get('name'):
            return True
            
        return False

    def _get_required_fields(self) -> List[str]:
        return ['id', 'name']


if __name__ == "__main__":
    from src.config import load_config

    config = load_config()
    collector = CountryCollector(config)

    # Run collection (this will show red log messages)
    result = collector.run_collection()

    # Show results after all logging is complete
    print("\n" + "="*50)
    print("COLLECTION RESULTS")
    print("="*50)
    metadata = result.get('collection_metadata', {})
    print(f"Number of active countries collected: {metadata.get('total_active_countries', 0)}")
    print(f"Number of active countries with valid IDs: {metadata.get('active_countries_with_ids', 0)}")
    print("="*50)