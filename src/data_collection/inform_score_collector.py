"""Collects INFORM risk scores for disaster risk assessment"""

from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from .base_collector import ConfigurableCollector
except ImportError:
    from src.data_collection.base_collector import ConfigurableCollector


class InformScoreCollector(ConfigurableCollector):
    """Collects INFORM risk score data"""
    def get_data_directory(self) -> Path:
        inform_dir = self.config.paths.collection_raw_dir / "inform_score"
        inform_dir.mkdir(parents=True, exist_ok=True)
        return inform_dir

    def collect_data(self) -> Dict[str, Any]:
        self.logger.info("Starting INFORM score data collection...")
        inform_data = self._collect_inform_score_data()
        if inform_data:
            self.save_data(inform_data, "inform_score_data")
        record_count = self._count_inform_records(inform_data)
        return {
            'data': inform_data,
            'collection_metadata': {
                'inform_score_records': record_count,
                'collection_timestamp': datetime.now().isoformat(),
            }
        }

    def _collect_inform_score_data(self) -> Optional[Any]:
        """Collect INFORM score data from API or cache"""
        cached_data = self._load_from_cache('inform_score/global_inform_scores')
        if cached_data:
            record_count = self._count_inform_records(cached_data)
            self.logger.info(f"Loaded {record_count} INFORM score records from cache")
            return cached_data

        try:
            self.logger.info("Getting INFORM score data from API...")
            inform_data = self.make_api_request(
                'inform-score',
                params={'limit': 10000, 'format': 'json'},
                use_cache=True,
                cache_key='inform_score/global_inform_scores'
            )

            if inform_data:
                record_count = self._count_inform_records(inform_data)
                self.logger.info(f"Retrieved {record_count} INFORM score records from API")
                return inform_data
            else:
                self.logger.warning("No INFORM score data retrieved")
                return None

        except Exception as e:
            self.logger.error(f"Failed to collect INFORM score data: {e}")
            return None

    def _count_inform_records(self, data: Any) -> int:
        """Count total INFORM score records"""
        if not data:
            return 0

        record_count = 0
        if isinstance(data, dict):
            # Standard API response with results array
            record_count = len(data.get('results', []))
        elif isinstance(data, list):
            # Direct list of records
            record_count = len(data)

        return record_count


if __name__ == "__main__":
    from src.config import load_config

    config = load_config()
    collector = InformScoreCollector(config)

    result = collector.run_collection()

    print("\n" + "="*50)
    print("INFORM SCORE DATA COLLECTION RESULTS")
    print("="*50)

    metadata = result.get('collection_metadata', {})
    print(f"INFORM score records: {metadata.get('inform_score_records', 0)}")
    print(f"Collection timestamp: {metadata.get('collection_timestamp', 'N/A')}")
    print("="*50)