"""Collects local units configuration and options data"""

from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from .base_collector import ConfigurableCollector
except ImportError:
    from src.data_collection.base_collector import ConfigurableCollector


class LocalUnitsOptionsCollector(ConfigurableCollector):

    def get_data_directory(self) -> Path:
        options_dir = self.config.paths.collection_raw_dir / "local_units_options"
        options_dir.mkdir(parents=True, exist_ok=True)
        return options_dir

    def collect_data(self) -> Dict[str, Any]:
        self.logger.info("Starting local units options data collection...")
        options_data = self._collect_options_data()
        if options_data:
            self.save_data(options_data, "local_units_options_data")

        record_count = self._count_options_records(options_data)
        
        return {
            'data': options_data,
            'collection_metadata': {
                'local_units_options_records': record_count,
                'collection_timestamp': datetime.now().isoformat(),
            }
        }

    def _collect_options_data(self) -> Optional[Any]:
        """Collect local units options data from API or cache"""
        cached_data = self._load_from_cache('local_units_options/local_options')
        if cached_data:
            record_count = self._count_options_records(cached_data)
            self.logger.info(f"Loaded {record_count} local units options records from cache")
            return cached_data

        try:
            self.logger.info("Getting local units options data from API...")
            options_data = self.make_api_request(
                'local_units_options',
                params={'format': 'json'},
                use_cache=True,
                cache_key='local_units_options/local_options'
            )

            if options_data:
                record_count = self._count_options_records(options_data)
                self.logger.info(f"Retrieved {record_count} local units options records from API")
                return options_data
            else:
                self.logger.warning("No local units options data retrieved")
                return None

        except Exception as e:
            self.logger.error(f"Failed to collect local units options data: {e}")
            return None

    def _count_options_records(self, data: Any) -> int:
        """Count total local units options records"""
        if not data:
            return 0
            
        record_count = 0
        if isinstance(data, dict):
            # For options API, count all option categories
            for key, value in data.items():
                if isinstance(value, list):
                    record_count += len(value)
                elif isinstance(value, dict):
                    # If nested dict, count nested lists
                    for nested_value in value.values():
                        if isinstance(nested_value, list):
                            record_count += len(nested_value)
        elif isinstance(data, list):
            record_count = len(data)
        
        return record_count


if __name__ == "__main__":
    from src.config import load_config

    config = load_config()
    collector = LocalUnitsOptionsCollector(config)
    result = collector.run_collection()

    print("\n" + "="*50)
    print("LOCAL UNITS OPTIONS DATA COLLECTION RESULTS")
    print("="*50)
    
    metadata = result.get('collection_metadata', {})
    print(f"Local units options records: {metadata.get('local_units_options_records', 0)}")
    print(f"Collection timestamp: {metadata.get('collection_timestamp', 'N/A')}")
    print("="*50)