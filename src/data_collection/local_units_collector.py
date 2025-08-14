"""
Local Units Data Collector for IFRC API
Collects detailed information for all local units and saves to cache only
"""

from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import sys
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from .base_collector import ConfigurableCollector
except ImportError:
    from src.data_collection.base_collector import ConfigurableCollector


class LocalUnitsCollector(ConfigurableCollector):
    """Collects detailed local units data by ID and saves to cache only"""

    def collect_data(self) -> Dict[str, Any]:
        """Main data collection method - collects all local units details and caches them"""
        self.logger.info("Starting local units detail data collection...")

        # Get list of all local units IDs
        unit_ids = self._get_all_unit_ids()
        
        if not unit_ids:
            self.logger.warning("No local unit IDs found")
            return {
                'collection_metadata': {
                    'local_units_records': 0,
                    'collection_timestamp': datetime.now().isoformat(),
                    'data_type': 'local_units_cached_only'
                }
            }

        # Collect detailed data for each unit (saves to cache automatically)
        collected_count = self._collect_all_units_data(unit_ids)
        
        return {
            'collection_metadata': {
                'local_units_records': collected_count,
                'unit_ids_processed': len(unit_ids),
                'collection_timestamp': datetime.now().isoformat(),
                'data_type': 'local_units_cached_only'
            }
        }

    def _get_all_unit_ids(self) -> List[int]:
        """Get list of all local unit IDs from the public-local-units list API"""
        try:
            self.logger.info("Getting list of all local units...")
            units_list = self.make_api_request(
                'local_units_list',
                params={'limit': 10000, 'format': 'json'},
                use_cache=True,
                cache_key='local_units/units_list'
            )

            unit_ids = []
            if isinstance(units_list, dict) and 'results' in units_list:
                for unit in units_list['results']:
                    if isinstance(unit, dict) and 'id' in unit:
                        unit_ids.append(unit['id'])
            elif isinstance(units_list, list):
                for unit in units_list:
                    if isinstance(unit, dict) and 'id' in unit:
                        unit_ids.append(unit['id'])

            self.logger.info(f"Found {len(unit_ids)} local units to collect details for")
            return unit_ids

        except Exception as e:
            self.logger.error(f"Failed to get local unit IDs: {e}")
            return []
    
    def _collect_all_units_data(self, unit_ids: List[int]) -> int:
        """Collect detailed data for all local units and return count of successfully collected units"""
        collected_count = 0
        processed_count = 0
        
        for unit_id in unit_ids:
            try:
                # Check cache first for individual unit
                cached_unit = self._load_from_cache(f'local_units/unit_{unit_id}')
                if cached_unit:
                    collected_count += 1
                    processed_count += 1
                    continue

                # Get detailed data for this unit
                unit_detail = self._get_unit_detail_by_id(unit_id)
                
                if unit_detail:
                    collected_count += 1
                
                processed_count += 1
                
                # Progress logging
                if processed_count % 50 == 0:
                    self.logger.info(f"Processed {processed_count}/{len(unit_ids)} local units, collected {collected_count}")
                
                # Rate limiting
                time.sleep(self.config.api.rate_limit_delay)

            except Exception as e:
                self.logger.warning(f"Failed to collect data for unit ID {unit_id}: {e}")
                processed_count += 1
                continue

        self.logger.info(f"Completed collection: {collected_count} units successfully cached out of {len(unit_ids)} IDs")
        return collected_count

    def _get_unit_detail_by_id(self, unit_id: int) -> Optional[Dict]:
        """Get detailed data for a specific local unit by ID"""
        try:
            # Check cache first
            cached_data = self._load_from_cache(f'local_units/unit_{unit_id}')
            if cached_data:
                return cached_data

            # Build the full URL with ID
            base_url = self.config.api.base_url_v2
            detail_url = f"{base_url}/public-local-units/{unit_id}/"
            
            response = self.session.get(
                detail_url,
                params={'format': 'json'},
                timeout=self.config.api.timeout
            )
            response.raise_for_status()
            
            unit_data = response.json()
            
            # Cache the result
            self._save_to_cache(unit_data, f'local_units/unit_{unit_id}')
            
            return unit_data

        except Exception as e:
            self.logger.debug(f"Failed to get details for unit {unit_id}: {e}")
            return None


if __name__ == "__main__":
    from src.config import load_config

    config = load_config()
    collector = LocalUnitsCollector(config)

    # Run collection
    result = collector.run_collection()

    # Show results
    print("\n" + "="*50)
    print("LOCAL UNITS DATA COLLECTION RESULTS (CACHED ONLY)")
    print("="*50)
    
    metadata = result.get('collection_metadata', {})
    print(f"Local units records cached: {metadata.get('local_units_records', 0)}")
    print(f"Unit IDs processed: {metadata.get('unit_ids_processed', 0)}")
    print(f"Collection timestamp: {metadata.get('collection_timestamp', 'N/A')}")
    print(f"Data type: {metadata.get('data_type', 'N/A')}")
    print("="*50)
    print("Note: All data is cached in cache/local_units/unit_{id}.json files")
    print("Use query methods when needed to retrieve data from cache")
    print("="*50)