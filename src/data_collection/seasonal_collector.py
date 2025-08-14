"""Collects seasonal disaster data """

from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from .base_collector import ConfigurableCollector
except ImportError:
    from src.data_collection.base_collector import ConfigurableCollector


class SeasonalCollector(ConfigurableCollector):
    """Collects  global seasonal disaster data """

    def get_data_directory(self) -> Path:
        seasonal_dir = self.config.paths.collection_raw_dir / "seasonal"
        seasonal_dir.mkdir(parents=True, exist_ok=True)
        return seasonal_dir

    def collect_data(self) -> Dict[str, Any]:
        self.logger.info("Starting global seasonal data collection...")
        global_seasonal = self._collect_global_seasonal_data()
        self.save_data(global_seasonal, "global_seasonal_data")

        record_count = self._count_seasonal_records(global_seasonal)
        result = {
            'data': global_seasonal,
            'collection_metadata': {
                'global_seasonal_records': record_count,
                'collection_timestamp': datetime.now().isoformat(),
            }
        }

        return result

    def _collect_global_seasonal_data(self) -> Optional[Dict]:
        """Collect global seasonal disaster """
        self.logger.info("Collecting global seasonal data...")

        # Check cache first
        cached_data = self._load_from_cache('seasonal/global_seasonal_data')
        if cached_data:
            record_count = self._count_seasonal_records(cached_data)
            self.logger.info(f"Loaded {record_count} global seasonal records from cache")
            return cached_data

        try:
            # Get from API if not in cache
            self.logger.info("Getting global seasonal data from API...")
            seasonal_data = self.make_api_request(
                'seasonal',
                params={'limit': 10000, 'format': 'json'},  # Increased limit
                use_cache=True,
                cache_key='seasonal/global_seasonal_data'
            )

            if seasonal_data:
                record_count = self._count_seasonal_records(seasonal_data)
                self.logger.info(f"Retrieved {record_count} global seasonal records from API")
                
                return seasonal_data
            else:
                self.logger.warning("No global seasonal data retrieved")
                return None

        except Exception as e:
            self.logger.error(f"Failed to collect global seasonal data: {e}")
            return None

    def _count_seasonal_records(self, data: Any) -> int:
        """Count total seasonal records in nested data structure"""
        if not data:
            return 0
            
        record_count = 0
        if isinstance(data, dict):
            record_count = len(data.get('results', []))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'idmc' in item:
                    idmc_data = item.get('idmc', [])
                    if isinstance(idmc_data, list):
                        record_count += len(idmc_data)
            if record_count == 0:
                record_count = len(data)
        
        return record_count



if __name__ == "__main__":
    from src.config import load_config

    config = load_config()
    collector = SeasonalCollector(config)

    # Run collection
    result = collector.run_collection()

    # Show results
    print("\n" + "="*50)
    print("GLOBAL SEASONAL DATA COLLECTION RESULTS")
    print("="*50)
    
    metadata = result.get('collection_metadata', {})
    print(f"Global seasonal records: {metadata.get('global_seasonal_records', 0)}")
    print(f"Collection timestamp: {metadata.get('collection_timestamp', 'N/A')}")
    print(f"Data type: {metadata.get('data_type', 'N/A')}")
    print("="*50)
