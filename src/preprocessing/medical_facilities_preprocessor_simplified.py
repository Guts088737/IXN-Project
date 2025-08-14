"""
Data processing and feature engineering of local unit
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import sys
import re
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.config import ProjectConfig, load_config
except ImportError:
    from config import ProjectConfig, load_config

class MedicalFacilitiesPreprocessor:
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.facility_type_mapping = {}
        self.functionality_mapping = {}
        self._load_ifrc_mappings()
    
    def _load_ifrc_mappings(self):
        """Load the official IFRC data mapping"""
        try:
            options_file = self.config.paths.collection_raw_dir / "local_units_options" / "local_units_options_data.json"
            with open(options_file, 'r', encoding='utf-8') as f:
                options_data = json.load(f)
            
            data = options_data.get('data', {})
            
            # facility type mapping
            for facility_type in data.get('health_facility_type', []):
                self.facility_type_mapping[facility_type['id']] = facility_type['name']
            
            # functionality mapping
            for functionality in data.get('functionality', []):
                self.functionality_mapping[functionality['id']] = functionality['name']
                
        except Exception as e:
            self.logger.warning(f"Failed to load the IFRC mapping. Use the default mapping: {e}")
    
    def clean_data(self) -> Dict[str, Any]:
        """Main function of clearing data"""
        try:
            #Load the data
            raw_facilities = self._load_raw_data()
            if not raw_facilities:
                return self._create_empty_result()
            
            # clear the data
            processed_facilities = []
            for facility in raw_facilities:
                processed = self._process_single_facility(facility)
                if processed:
                    processed_facilities.append(processed)
            
            # set up feature engineering
            processed_facilities = self._add_features(processed_facilities)
            
            # save data
            output_file = self._save_results(processed_facilities)
            
            return {
                'processed_facilities': processed_facilities,
                'processing_summary': self._create_summary(raw_facilities, processed_facilities),
                'output_file': output_file
            }
            
        except Exception as e:
            self.logger.error(f"Failure to deal with data: {e}")
            return self._create_empty_result()
    
    def _load_raw_data(self) -> List[Dict]:
        cache_dir = self.config.paths.cache_dir / "local_units"
        unit_files = list(cache_dir.glob("unit_*.json"))
        
        facilities = []
        for unit_file in unit_files:
            try:
                with open(unit_file, 'r', encoding='utf-8') as f:
                    unit_data = json.load(f)
                
                if isinstance(unit_data, dict) and 'data' in unit_data:
                    facility_data = unit_data['data']
                    # Only deal with data of health facility type
                    if facility_data.get('type') == 2:
                        facilities.append(facility_data)
            except Exception as e:
                self.logger.error(f"Failure to load file {unit_file}: {e}")
        
        self.logger.info(f"The number of health facility type loaded: {len(facilities)} .")
        return facilities
    
    def _process_single_facility(self, facility: Dict) -> Optional[Dict]:
        try:
            if not facility.get('local_branch_name') and not facility.get('english_name'):
                return None
            
            coords = self._extract_coordinates(facility)
            if not coords:
                return None
            
            health_data = facility.get('health', {}) or {}

            processed = {
                # basic information
                'facility_id': self._generate_facility_id(facility),
                'facility_name': facility.get('local_branch_name', '') or facility.get('english_name', ''),
                'english_name': facility.get('english_name', ''),
                'country_name': facility.get('country_details', {}).get('name', ''),
                'coordinates': coords,
                
                # facility type and status
                'facility_type_id': health_data.get('health_facility_type'),
                'facility_type_name': self._get_facility_type_name(health_data.get('health_facility_type')),
                'functionality_status': self._get_functionality_status(health_data.get('functionality')),
                
                # data of health capacity
                'bed_capacity': self._safe_int(health_data.get('maximum_capacity')),
                'total_staff': self._safe_int(health_data.get('total_number_of_human_resource')),
                'doctors': self._safe_int(health_data.get('general_practitioner')) + self._safe_int(health_data.get('specialist')),
                'nurses': self._safe_int(health_data.get('nurse')),
                
                # service
                'general_services_count': len(health_data.get('general_medical_services_details', [])),
                'specialized_services_count': len(health_data.get('specialized_medical_beyond_primary_level_details', [])),
                'total_services_count': len(health_data.get('general_medical_services_details', [])) + 
                                     len(health_data.get('specialized_medical_beyond_primary_level_details', [])),
                
                # basic information of facilities
                'has_inpatient_capacity': health_data.get('is_in_patient_capacity', False),
                'has_isolation_facilities': health_data.get('is_isolation_rooms_wards', False),
                'has_cold_chain': health_data.get('is_cold_chain', False),
                'ambulance_count': (self._safe_int(health_data.get('ambulance_type_a')) + 
                                  self._safe_int(health_data.get('ambulance_type_b')) + 
                                  self._safe_int(health_data.get('ambulance_type_c'))),

                'created_at': facility.get('created_at', ''),
                'data_date': facility.get('date_of_data', '')
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Failure to clear facility data: {e}")
            return None
    
    def _add_features(self, facilities: List[Dict]) -> List[Dict]:
        if not facilities:
            return facilities
        
        # Collect global statistical data for standardization
        all_bed_capacities = [f['bed_capacity'] for f in facilities if f['bed_capacity'] > 0]
        all_staff_counts = [f['total_staff'] for f in facilities if f['total_staff'] > 0]
        all_service_counts = [f['total_services_count'] for f in facilities if f['total_services_count'] > 0]
        
        max_beds = max(all_bed_capacities) if all_bed_capacities else 1
        max_staff = max(all_staff_counts) if all_staff_counts else 1  
        max_services = max(all_service_counts) if all_service_counts else 1
        
        # Add standardized features to each facility
        for facility in facilities:
            facility.update({
                # Standardized scoring (0-10)
                'capacity_score': self._calculate_capacity_score(facility, max_beds, max_staff),
                'service_score': self._calculate_service_score(facility, max_services),
                'infrastructure_score': self._calculate_infrastructure_score(facility),
                'overall_score': 0,

                # Classification features
                'specialization_level': self._get_specialization_level(facility),
                'emergency_readiness': self._get_emergency_readiness(facility),
                
                # Geographical features
                'latitude': facility['coordinates'][1],
                'longitude': facility['coordinates'][0]
            })

            # Calculate the comprehensive score
            facility['overall_score'] = round(np.mean([
                facility['capacity_score'],
                facility['service_score'], 
                facility['infrastructure_score']
            ]), 2)
        
        return facilities
    
    def _calculate_capacity_score(self, facility: Dict, max_beds: int, max_staff: int) -> float:
        """Calculate capacity score"""
        bed_score = min(facility['bed_capacity'] / max_beds * 5, 5) if facility['bed_capacity'] > 0 else 0
        staff_score = min(facility['total_staff'] / max_staff * 5, 5) if facility['total_staff'] > 0 else 0
        return round(bed_score + staff_score, 2)
    
    def _calculate_service_score(self, facility: Dict, max_services: int) -> float:
        """Calculate service score"""
        service_score = min(facility['total_services_count'] / max_services * 10, 10) if facility['total_services_count'] > 0 else 0
        return round(service_score, 2)
    
    def _calculate_infrastructure_score(self, facility: Dict) -> float:
        """Calculate facility score"""
        infrastructure_features = [
            facility['has_inpatient_capacity'],
            facility['has_isolation_facilities'],
            facility['has_cold_chain'],
            facility['ambulance_count'] > 0
        ]
        score = sum(infrastructure_features) / len(infrastructure_features) * 10
        return round(score, 2)
    
    def _get_specialization_level(self, facility: Dict) -> str:
        """Obtain the degree of specialization"""
        if facility['specialized_services_count'] >= 3:
            return 'Highly Specialized'
        elif facility['specialized_services_count'] >= 1:
            return 'Moderately Specialized'
        else:
            return 'General Practice'
    
    def _get_emergency_readiness(self, facility: Dict) -> str:
        """Obtain the degree of emergency"""
        readiness_score = sum([
            facility['has_isolation_facilities'],
            facility['ambulance_count'] > 0,
            facility['has_cold_chain'],
            facility['bed_capacity'] > 0
        ])
        
        if readiness_score >= 3:
            return 'High'
        elif readiness_score >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def _extract_coordinates(self, facility: Dict) -> Optional[List[float]]:
        """Extract coordinates"""
        try:
            geojson = facility.get('location_geojson', {})
            if geojson and geojson.get('type') == 'Point':
                coords = geojson.get('coordinates', [])
                if len(coords) == 2:
                    return coords

            location_str = facility.get('location', '')
            if 'POINT' in location_str:
                match = re.search(r'POINT\s*\(\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*\)', location_str)
                if match:
                    return [float(match.group(1)), float(match.group(2))]
            
            return None
        except:
            return None
    
    def _generate_facility_id(self, facility: Dict) -> str:
        """Generate facility ID"""
        health_id = (facility.get('health') or {}).get('id')
        if health_id:
            return f"health_{health_id}"
        
        country_id = facility.get('country', 'unknown')
        name = facility.get('local_branch_name', 'unnamed')
        return f"facility_{country_id}_{hash(name) % 100000}"
    
    def _get_facility_type_name(self, type_id: int) -> str:
        """Obtain the name of facility type"""
        return self.facility_type_mapping.get(type_id, f'Unknown Type {type_id}')
    
    def _get_functionality_status(self, func_id: int) -> str:
        """Obtain the function status="""
        return self.functionality_mapping.get(func_id, 'Unknown Status')
    
    def _safe_int(self, value) -> int:
        """Safe integer conversion"""
        try:
            return int(value) if value is not None else 0
        except (ValueError, TypeError):
            return 0
    
    def _save_results(self, facilities: List[Dict]) -> str:
        output_dir = self.config.paths.processed_data_dir / "medical_facilities"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "processed_medical_facilities.json"
        
        output_data = {
            'metadata': {
                'processing_date': datetime.now().isoformat(),
                'total_facilities': len(facilities),
                'processor_version': 'simplified_v1.0'
            },
            'facilities': facilities
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"save the results to: {output_file}")
        return str(output_file)
    
    def _create_summary(self, raw_facilities: List[Dict], processed_facilities: List[Dict]) -> Dict[str, Any]:
        if not processed_facilities:
            return {
                'input_count': len(raw_facilities),
                'output_count': 0,
                'success_rate': 0
            }

        total_beds = sum(f['bed_capacity'] for f in processed_facilities)
        total_staff = sum(f['total_staff'] for f in processed_facilities)

        functionality_dist = defaultdict(int)
        for f in processed_facilities:
            functionality_dist[f['functionality_status']] += 1
        
        return {
            'input_count': len(raw_facilities),
            'output_count': len(processed_facilities),
            'success_rate': round(len(processed_facilities) / len(raw_facilities), 4),
            'total_bed_capacity': total_beds,
            'total_staff': total_staff,
            'functionality_distribution': dict(functionality_dist),
            'processing_date': datetime.now().isoformat()
        }
    
    def _create_empty_result(self) -> Dict[str, Any]:
        return {
            'processed_facilities': [],
            'processing_summary': {
                'input_count': 0,
                'output_count': 0,
                'success_rate': 0
            },
            'output_file': None
        }


if __name__ == "__main__":
    from src.config import load_config
    
    config = load_config()
    processor = MedicalFacilitiesPreprocessor(config)
    
    print("Start the data preprocessing of the local unit...")
    result = processor.clean_data()
    
    summary = result['processing_summary']
    print(f"Processing completed!")
    print(f"Number of facilities input: {summary['input_count']}")
    print(f"Number of facilities output: {summary['output_count']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    if summary.get('total_bed_capacity'):
        print(f"Total number of beds: {summary['total_bed_capacity']}")
        print(f"Total number of staff: {summary['total_staff']}")