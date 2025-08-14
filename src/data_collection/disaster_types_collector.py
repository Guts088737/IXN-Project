"""
Focus on collecting basic disaster type data
Only handles IFRC disaster type API
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


class DisasterTypesCollector(ConfigurableCollector):
    """
    Collects basic IFRC disaster type
    """

    def get_data_directory(self) -> Path:
        """Get disaster type data save directory"""
        return self.config.paths.disaster_types_dir

    def collect_data(self) -> Dict[str, Any]:
        """Collect basic disaster type data"""
        self.logger.info("Start collecting basic disaster type data...")

        # Get disaster types from API or cache
        disaster_types = self._get_disaster_types()

        # Save data
        self.save_data(disaster_types, "disaster_types")

        return {
            'data': disaster_types,
            'collection_metadata': {
                'types_collected': len(disaster_types),
                'collection_timestamp': datetime.now().isoformat(),
                'api_endpoint': 'disaster_types'
            }
        }

    def _get_disaster_types(self) -> List[Dict]:
        """Get disaster types from cache or API"""
        # Check cache first
        cached_data = self._load_from_cache('disaster_types/basic_disaster_types')
        if cached_data:
            # Extract results from cached API response
            if isinstance(cached_data, dict):
                types_list = cached_data.get('results', [])
            elif isinstance(cached_data, list):
                types_list = cached_data
            else:
                types_list = []
            
            self.logger.info(f"Load {len(types_list)} disaster types from cache")
            return types_list

        # Get from API
        self.logger.info("Get disaster type data from API...")
        disaster_types = self.make_api_request(
            'disaster_types',
            use_cache=True,
            cache_key='disaster_types/basic_disaster_types'
        )

        if isinstance(disaster_types, dict):
            types_list = disaster_types.get('results', [])
        elif isinstance(disaster_types, list):
            types_list = disaster_types
        else:
            self.logger.error(f"Unknown API response format: {type(disaster_types)}")
            return []

        self.logger.info(f"Retrieved {len(types_list)} disaster types")
        return types_list


    def get_disaster_type_by_id(self, type_id: int) -> Optional[Dict[str, Any]]:
        """Get specific disaster type by ID"""
        types = self._get_disaster_types()
        return next((t for t in types if t.get('id') == type_id), None)

    def get_disaster_type_by_name(self, type_name: str) -> Optional[Dict[str, Any]]:
        """Get specific disaster type by name"""
        types = self._get_disaster_types()
        return next((t for t in types if t.get('name', '').lower() == type_name.lower()), None)


    def list_all_disaster_types(self) -> List[Dict[str, Any]]:
        """List all disaster types"""
        return self._get_disaster_types()

    def _get_required_fields(self) -> List[str]:
        """Get required fields for disaster types"""
        return ['id', 'name']

if __name__ == "__main__":
    from src.config import load_config
    config = load_config()
    collector = DisasterTypesCollector(config)

    result = collector.run_collection()

    print("\n" + "="*50)
    print("COLLECTION RESULTS")
    print("="*50)
    metadata = result.get('collection_metadata', {})
    print(f"Number of disaster types collected: {metadata.get('types_collected', 0)}")

    

    
