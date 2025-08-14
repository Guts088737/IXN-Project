"""
Region Data Collector for IFRC API
Collects regional information and detailed region data
"""

from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from .base_collector import ConfigurableCollector
except ImportError:
    from src.data_collection.base_collector import ConfigurableCollector


class RegionCollector(ConfigurableCollector):
    """Collects region data including list and detailed information"""

    def get_data_directory(self) -> Path:
        """Get region data save directory"""
        region_dir = self.config.paths.collection_raw_dir / "regions"
        region_dir.mkdir(parents=True, exist_ok=True)
        return region_dir

    def collect_data(self) -> Dict[str, Any]:
        """Main data collection method - collects all region data"""
        self.logger.info("Starting region data collection...")

        # First get all regions list
        regions_list = self._collect_regions_list()
        
        # Then get detailed data for each region
        regions_details = self._collect_regions_details(regions_list)
        
        # Combine all data
        complete_data = {
            'regions_list': regions_list,
            'regions_details': regions_details
        }
        
        # Save complete dataset
        if complete_data:
            self.save_data(complete_data, "complete_regions_data")
        
        # Also save regions list separately for other collectors to use
        if regions_list:
            self.save_data(regions_list, "regions_list")

        return {
            'data': complete_data,
            'collection_metadata': {
                'total_regions': len(regions_details) if regions_details else 0,
                'regions_with_details': len([r for r in regions_details.values() if r]) if regions_details else 0,
                'collection_timestamp': datetime.now().isoformat(),
            }
        }

    def _collect_regions_list(self) -> Optional[Any]:
        """Collect list of all regions"""
        # Check cache first
        cached_data = self._load_from_cache('regions/regions_list')
        if cached_data:
            record_count = self._count_regions_records(cached_data)
            self.logger.info(f"Loaded {record_count} regions from cache")
            return cached_data

        try:
            self.logger.info("Getting regions list from API...")
            regions_data = self.make_api_request(
                'regions',
                params={'limit': 1000, 'format': 'json'},
                use_cache=True,
                cache_key='regions/regions_list'
            )

            if regions_data:
                record_count = self._count_regions_records(regions_data)
                self.logger.info(f"Retrieved {record_count} regions from API")
                return regions_data
            else:
                self.logger.warning("No regions data retrieved")
                return None

        except Exception as e:
            self.logger.error(f"Failed to collect regions list: {e}")
            return None

    def _collect_regions_details(self, regions_list: Any) -> Dict[str, Any]:
        """Collect detailed data for each region"""
        regions_details = {}
        
        if not regions_list:
            return regions_details
            
        # Extract region IDs
        region_ids = self._extract_region_ids(regions_list)
        
        self.logger.info(f"Collecting detailed data for {len(region_ids)} regions")
        
        for region_id in region_ids:
            try:
                region_detail = self._get_region_detail_by_id(region_id)
                if region_detail:
                    # Use region name as key if available, otherwise use ID
                    region_name = region_detail.get('name', f'Region_{region_id}')
                    regions_details[region_name] = region_detail
                    self.logger.debug(f"Collected data for region: {region_name}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to collect details for region ID {region_id}: {e}")
                continue
        
        self.logger.info(f"Completed: collected details for {len(regions_details)} regions")
        return regions_details

    def _extract_region_ids(self, regions_list: Any) -> List[int]:
        """Extract region IDs from regions list"""
        region_ids = []
        
        if isinstance(regions_list, dict) and 'results' in regions_list:
            for region in regions_list['results']:
                if isinstance(region, dict) and 'id' in region:
                    region_ids.append(region['id'])
        elif isinstance(regions_list, list):
            for region in regions_list:
                if isinstance(region, dict) and 'id' in region:
                    region_ids.append(region['id'])
                    
        return region_ids

    def _get_region_detail_by_id(self, region_id: int) -> Optional[Dict]:
        """Get detailed data for a specific region by ID"""
        try:
            # Check cache first
            cached_data = self._load_from_cache(f'regions/region_{region_id}')
            if cached_data:
                return cached_data

            # Build the full URL with ID
            base_url = self.config.api.base_url_v2
            detail_url = f"{base_url}/region/{region_id}/"
            
            response = self.session.get(
                detail_url,
                params={'format': 'json'},
                timeout=self.config.api.timeout
            )
            response.raise_for_status()
            
            region_data = response.json()
            
            # Cache the result
            self._save_to_cache(region_data, f'regions/region_{region_id}')
            
            return region_data

        except Exception as e:
            self.logger.debug(f"Failed to get details for region {region_id}: {e}")
            return None

    def _count_regions_records(self, data: Any) -> int:
        """Count total region records"""
        if not data:
            return 0
            
        record_count = 0
        if isinstance(data, dict):
            record_count = len(data.get('results', []))
        elif isinstance(data, list):
            record_count = len(data)
        
        return record_count

    def get_all_regions_info(self, silent: bool = False) -> List[Dict]:
        """Get all regions information for use by other collectors"""
        try:
            # First try to load from cache
            cached_data = self._load_from_cache('regions/regions_list')
            if cached_data:
                if isinstance(cached_data, dict) and 'results' in cached_data:
                    regions_list = cached_data['results']
                elif isinstance(cached_data, list):
                    regions_list = cached_data
                else:
                    regions_list = []
                
                if not silent:
                    self.logger.info(f"Loaded {len(regions_list)} regions from cache")
                return regions_list
            
            # Try to load from saved data
            regions_data = self.load_data("regions_list")
            
            if regions_data:
                if isinstance(regions_data, dict) and 'results' in regions_data:
                    regions_list = regions_data['results']
                elif isinstance(regions_data, list):
                    regions_list = regions_data
                else:
                    regions_list = []
                
                if not silent:
                    self.logger.info(f"Loaded {len(regions_list)} regions from saved data")
                return regions_list
            
            # If no saved data, collect fresh data
            if not silent:
                self.logger.info("No cached regions data found, collecting fresh data...")
            fresh_data = self._collect_regions_list()
            
            if isinstance(fresh_data, dict) and 'results' in fresh_data:
                return fresh_data['results']
            elif isinstance(fresh_data, list):
                return fresh_data
            else:
                return []
                
        except Exception as e:
            if not silent:
                self.logger.error(f"Failed to get regions info: {e}")
            return []


if __name__ == "__main__":
    from src.config import load_config

    config = load_config()
    collector = RegionCollector(config)
    result = collector.run_collection()

    print("\n" + "="*50)
    print("REGION DATA COLLECTION RESULTS")
    print("="*50)
    
    metadata = result.get('collection_metadata', {})
    print(f"Total regions: {metadata.get('total_regions', 0)}")
    print(f"Regions with details: {metadata.get('regions_with_details', 0)}")
    print(f"Collection timestamp: {metadata.get('collection_timestamp', 'N/A')}")
    print("="*50)