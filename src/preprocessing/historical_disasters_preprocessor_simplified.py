"""
Historical disaster preprocessor
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.config import ProjectConfig, load_config
except ImportError:
    from config import ProjectConfig, load_config

class HistoricalDisastersPreprocessor:
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.disaster_type_mapping = {}
        self.country_mapping = {}
        self.field_reports_cache = {}
        self._load_ifrc_mappings()
        self._load_field_reports_cache()
    
    def _load_ifrc_mappings(self):
        """Loading mapping"""
        try:
            disaster_types_file = self.config.paths.collection_raw_dir / "disaster_types" / "disaster_types.json"
            if disaster_types_file.exists():
                with open(disaster_types_file, 'r', encoding='utf-8') as f:
                    disaster_types_data = json.load(f)

                if isinstance(disaster_types_data, list):
                    for dtype in disaster_types_data:
                        if isinstance(dtype, dict) and 'id' in dtype and 'name' in dtype:
                            self.disaster_type_mapping[dtype['id']] = dtype['name']
                elif isinstance(disaster_types_data, dict) and 'results' in disaster_types_data:
                    for dtype in disaster_types_data['results']:
                        if isinstance(dtype, dict) and 'id' in dtype and 'name' in dtype:
                            self.disaster_type_mapping[dtype['id']] = dtype['name']

            countries_file = self.config.paths.collection_raw_dir / "countries" / "all_countries.json"
            if countries_file.exists():
                with open(countries_file, 'r', encoding='utf-8') as f:
                    countries_data = json.load(f)
                
                if 'results' in countries_data:
                    for country in countries_data['results']:
                        self.country_mapping[country['id']] = country['name']
                        
        except Exception as e:
            self.logger.warning(f"Failed to load the IFRC mapping. Use the default mapping: {e}")
    
    def _load_field_reports_cache(self):
        try:
            cache_dir = self.config.paths.cache_dir / "field_reports"
            page_files = list(cache_dir.glob("page_*.json"))
            
            reports_loaded = 0
            for page_file in page_files:
                with open(page_file, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                
                if isinstance(page_data, dict) and 'data' in page_data:
                    results = page_data['data'].get('results', [])
                    for report in results:
                        if isinstance(report, dict) and 'id' in report:
                            self.field_reports_cache[report['id']] = report
                            reports_loaded += 1
            
            self.logger.info(f"Load the Field Reports cache: {reports_loaded} records")
            
        except Exception as e:
            self.logger.warning(f"Failed to load the Field Reports cache: {e}")
    
    def clean_data(self) -> Dict[str, Any]:
        try:
            raw_data = self._load_raw_data()
            if not raw_data:
                return self._create_empty_result()

            integrated_data = self._integrate_disasters_data(raw_data)
            self.logger.info(f"The number of integrated data records: {len(integrated_data)}")
            
            processed_data = []
            for i, disaster in enumerate(integrated_data):
                if i < 3:
                    self.logger.info(f"Handle the {i+1} th record: id={disaster.get('id')}, name={disaster.get('name')}, disaster_start_date={disaster.get('disaster_start_date')}")
                
                processed = self._process_single_disaster(disaster)
                if processed:
                    processed_data.append(processed)
                elif i < 3:
                    self.logger.warning(f"The processing of the {i+1} th record failed")

            output_file = self._save_results(processed_data)
            
            return {
                'processed_disasters': processed_data,
                'processing_summary': self._create_summary(raw_data, processed_data),
                'output_file': output_file
            }
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            return self._create_empty_result()
    
    def _load_raw_data(self) -> Optional[Dict]:
        raw_file = self.config.paths.collection_raw_dir / "historical_disasters" / "events_by_all_countries.json"
        
        if raw_file.exists():
            with open(raw_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Load data from the raw directory: {len(data) if isinstance(data, list) else 'N/A'} records")
            return data

        cache_dir = self.config.paths.cache_dir / "historical_disasters"
        cache_files = list(cache_dir.glob("country_*_events.json"))
        
        if not cache_files:
            self.logger.error("No historical disaster data files are found")
            return None

        all_disasters = []
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                    if isinstance(cache_data, dict) and 'data' in cache_data:
                        data_section = cache_data['data']
                        if isinstance(data_section, dict) and 'results' in data_section:
                            all_disasters.extend(data_section['results'])
                    elif isinstance(cache_data, dict) and 'results' in cache_data:
                        all_disasters.extend(cache_data['results'])
                    elif isinstance(cache_data, list):
                        all_disasters.extend(cache_data)
            except Exception as e:
                self.logger.warning(f"File loading failed {cache_file}: {e}")
        
        self.logger.info(f"Load data from the cache directory: {len(all_disasters)} records")
        return all_disasters
    
    def _integrate_disasters_data(self, raw_data) -> List[Dict]:
        if isinstance(raw_data, dict):
            disasters = []
            if 'data' in raw_data and isinstance(raw_data['data'], dict):
                data_section = raw_data['data']
                for country_name, country_data in data_section.items():
                    if isinstance(country_data, dict) and 'events' in country_data:
                        for event in country_data['events']:
                            event['source_country'] = country_name
                            disasters.append(event)
                    elif isinstance(country_data, dict) and 'results' in country_data:
                        for disaster in country_data['results']:
                            disaster['source_country'] = country_name
                            disasters.append(disaster)

            else:
                for country_name, country_data in raw_data.items():
                    if isinstance(country_data, dict) and 'results' in country_data:
                        for disaster in country_data['results']:
                            disaster['source_country'] = country_name
                            disasters.append(disaster)
                            
            return disasters
        
        elif isinstance(raw_data, list):
            return raw_data
        
        else:
            self.logger.error(f"不支持的数据格式: {type(raw_data)}")
            return []
    
    def _process_single_disaster(self, disaster: Dict) -> Optional[Dict]:
        try:
            if not disaster.get('name') and not disaster.get('id'):
                return None

            start_date = self._parse_date(disaster.get('disaster_start_date'))
            if not start_date or start_date.year < 2019:
                return None

            disaster_id = disaster.get('id') or disaster.get('event_id')
            disaster_type_obj = disaster.get('dtype') or disaster.get('disaster_type', {})
            
            processed = {
                'disaster_id': disaster_id or f"unknown_{hash(str(disaster)) % 100000}",
                'name': disaster.get('name', 'Unnamed Disaster'),
                'disaster_type': self._get_disaster_type_name(disaster_type_obj.get('id')),
                'disaster_type_id': disaster_type_obj.get('id'),

                'start_date': start_date.isoformat() if start_date else None,
                'year': start_date.year if start_date else None,
                'month': start_date.month if start_date else None,
                'season': self._get_season(start_date.month) if start_date else None,

                'country_name': self._extract_country_name(disaster),
                'country_id': self._extract_country_id(disaster),
                'region_name': disaster.get('region', {}).get('name', '') if isinstance(disaster.get('region'), dict) else '',
                'location_details': self._extract_location_details(disaster),

                'people_affected': self._safe_int(disaster.get('num_affected')),
                'people_injured': self._safe_int(disaster.get('num_injured')),
                'people_dead': self._safe_int(disaster.get('num_dead')),
                'people_displaced': self._safe_int(disaster.get('num_displaced')),

                'amount_requested': self._calculate_total_requested(disaster),
                'amount_funded': self._calculate_total_funded(disaster),
                'funding_coverage': 0,

                'appeals_count': len(disaster.get('appeals', [])),
                'appeals_ids': [appeal.get('id') for appeal in disaster.get('appeals', []) if isinstance(appeal, dict) and appeal.get('id')],
                'has_active_appeal': any(appeal.get('status') == 1 for appeal in disaster.get('appeals', []) if isinstance(appeal, dict)),

                'field_reports_count': len(disaster.get('field_reports', [])),
                'field_reports_ids': [report.get('id') for report in disaster.get('field_reports', []) if isinstance(report, dict) and report.get('id')],
                'has_field_assessment': len(disaster.get('field_reports', [])) > 0,
                'medical_needs': self._extract_medical_needs_from_reports(disaster.get('field_reports', [])),

                'emergency_response': disaster.get('is_featured', False),
                'status': disaster.get('status', 0),

                'created_at': disaster.get('created_at', ''),
                'updated_at': disaster.get('updated_at', ''),
            }

            if processed['amount_requested'] > 0:
                processed['funding_coverage'] = processed['amount_funded'] / processed['amount_requested']
            
            # 将field_reports中的伤亡数据同步到顶级字段
            medical_needs = processed.get('medical_needs', {})
            if medical_needs.get('total_injured_reported', 0) > 0:
                processed['people_injured'] = medical_needs['total_injured_reported']
            if medical_needs.get('total_dead_reported', 0) > 0:
                processed['people_dead'] = medical_needs['total_dead_reported']
            if medical_needs.get('total_displaced_reported', 0) > 0:  # 添加displaced同步
                processed['people_displaced'] = medical_needs['total_displaced_reported']
            if medical_needs.get('total_affected_reported', 0) > 0:
                # Field Reports中的数据通常更准确，优先使用
                # 如果field reports数据显著高于原数据（超过20%差异），或原数据为0，则使用field reports数据
                original_affected = processed.get('people_affected', 0)
                field_reports_affected = medical_needs['total_affected_reported']
                
                if original_affected == 0 or (field_reports_affected > original_affected * 1.2):
                    processed['people_affected'] = field_reports_affected
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Failed to process the disaster record: {e}")
            return None
    
    def _get_disaster_type_name(self, type_id: int) -> str:
        if not type_id:
            return 'Unknown'
        return self.disaster_type_mapping.get(type_id, f'Unknown Type {type_id}')
    
    def _get_country_name(self, country_id: int) -> str:
        if not country_id:
            return ''
        return self.country_mapping.get(country_id, f'Unknown Country {country_id}')
    
    def _extract_country_name(self, disaster: dict) -> str:
        if 'country' in disaster and isinstance(disaster['country'], dict):
            return disaster['country'].get('name', '')

        if 'countries' in disaster and isinstance(disaster['countries'], list) and disaster['countries']:
            return disaster['countries'][0].get('name', '')

        return disaster.get('source_country', '')
    
    def _extract_country_id(self, disaster: dict) -> int:
        if 'country' in disaster and isinstance(disaster['country'], dict):
            return disaster['country'].get('id', 0)

        if 'countries' in disaster and isinstance(disaster['countries'], list) and disaster['countries']:
            return disaster['countries'][0].get('id', 0)
        
        return 0
    
    def _calculate_total_requested(self, disaster: dict) -> float:
        total = 0.0
        appeals = disaster.get('appeals', [])
        if isinstance(appeals, list):
            for appeal in appeals:
                if isinstance(appeal, dict):
                    amount = appeal.get('amount_requested', 0)
                    total += self._safe_float(amount)
        if total == 0:
            total = self._safe_float(disaster.get('amount_requested', 0))
        
        return total
    
    def _calculate_total_funded(self, disaster: dict) -> float:
        total = 0.0
        appeals = disaster.get('appeals', [])
        if isinstance(appeals, list):
            for appeal in appeals:
                if isinstance(appeal, dict):
                    amount = appeal.get('amount_funded', 0)
                    total += self._safe_float(amount)

        if total == 0:
            total = self._safe_float(disaster.get('amount_funded', 0))
        
        return total
    
    def _extract_location_details(self, disaster: dict) -> dict:
        location_info = {
            'glide_code': disaster.get('glide', ''),
            'iso_code': '',
            'iso3_code': '',
            'region_id': None
        }

        country = disaster.get('country', {})
        if isinstance(country, dict):
            location_info['iso_code'] = country.get('iso', '')
            location_info['iso3_code'] = country.get('iso3', '')
            location_info['region_id'] = country.get('region')

        name = disaster.get('name', '')
        if name:
            if ':' in name:
                location_part = name.split(':')[0].strip()
                location_info['specific_location'] = location_part
            else:
                location_info['specific_location'] = ''
        
        return location_info
    
    def _extract_medical_needs_from_reports(self, field_reports: list) -> dict:
        medical_info = {
            'total_injured_reported': 0,
            'total_dead_reported': 0,
            'total_affected_reported': 0,
            'total_displaced_reported': 0,  # 添加displaced统计
            'total_assisted': 0,
            'has_medical_response': False,
            'medical_facilities_mentioned': False,
            'evacuation_mentioned': False,
            'field_reports_processed': 0,
            'field_reports_matched': 0,
            'field_reports_missing': 0
        }
        
        if not isinstance(field_reports, list):
            return medical_info
        
        medical_info['field_reports_processed'] = len(field_reports)
        
        for report in field_reports:
            if not isinstance(report, dict):
                continue
            
            report_id = report.get('id')
            detailed_report = None

            if report_id and report_id in self.field_reports_cache:
                detailed_report = self.field_reports_cache[report_id]
                medical_info['field_reports_matched'] += 1
            else:
                medical_info['field_reports_missing'] += 1
                detailed_report = report
            
            if detailed_report:
                medical_info['total_injured_reported'] += self._safe_int(detailed_report.get('num_injured'))
                medical_info['total_dead_reported'] += self._safe_int(detailed_report.get('num_dead')) 
                medical_info['total_affected_reported'] += self._safe_int(detailed_report.get('num_affected'))
                medical_info['total_displaced_reported'] += self._safe_int(detailed_report.get('num_displaced'))  # 添加displaced提取
                medical_info['total_assisted'] += self._safe_int(detailed_report.get('num_assisted'))

                medical_info['total_injured_reported'] += self._safe_int(detailed_report.get('gov_num_injured'))
                medical_info['total_injured_reported'] += self._safe_int(detailed_report.get('other_num_injured'))
                medical_info['total_dead_reported'] += self._safe_int(detailed_report.get('gov_num_dead'))
                medical_info['total_dead_reported'] += self._safe_int(detailed_report.get('other_num_dead'))
                medical_info['total_displaced_reported'] += self._safe_int(detailed_report.get('gov_num_displaced'))  # 添加政府displaced
                medical_info['total_displaced_reported'] += self._safe_int(detailed_report.get('other_num_displaced'))  # 添加其他displaced

                summary = str(detailed_report.get('summary', '')).lower()
                if any(keyword in summary for keyword in ['medical', 'hospital', 'health', 'doctor', 'nurse', 'clinic']):
                    medical_info['has_medical_response'] = True
                if any(keyword in summary for keyword in ['facility', 'hospital', 'clinic', 'health center']):
                    medical_info['medical_facilities_mentioned'] = True
                if any(keyword in summary for keyword in ['evacuation', 'evacuate', 'medevac']):
                    medical_info['evacuation_mentioned'] = True
        
        return medical_info
    
    def _get_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    def _parse_date(self, date_string) -> Optional[datetime]:
        if not date_string:
            return None
        
        try:
            date_str = str(date_string).replace('Z', '').replace('+00:00', '')
            if 'T' in date_str:
                return datetime.strptime(date_str[:19], '%Y-%m-%dT%H:%M:%S')
            else:
                return datetime.strptime(date_str[:10], '%Y-%m-%d')
        except:
            return None
    
    def _safe_int(self, value) -> int:
        try:
            return int(value) if value is not None and value != '' else 0
        except (ValueError, TypeError):
            return 0
    
    def _safe_float(self, value) -> float:
        try:
            return float(value) if value is not None and value != '' else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _save_results(self, disasters: List[Dict]) -> str:
        output_dir = self.config.paths.processed_data_dir / "historical_disasters"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "processed_historical_disasters.json"
        
        output_data = {
            'metadata': {
                'processing_date': datetime.now().isoformat(),
                'total_disasters': len(disasters),
                'processor_version': 'simplified_v1.0'
            },
            'disasters': disasters
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Save results to: {output_file}")
        return str(output_file)
    
    def _create_summary(self, raw_data, processed_disasters: List[Dict]) -> Dict[str, Any]:
        input_count = self._count_total_input_records(raw_data)
        
        if not processed_disasters:
            return {
                'input_count': input_count,
                'output_count': 0,
                'success_rate': 0
            }

        total_affected = sum(d['people_affected'] for d in processed_disasters)
        total_funding = sum(d['amount_requested'] for d in processed_disasters)

        disaster_types = {}
        countries = {}
        for d in processed_disasters:
            disaster_types[d['disaster_type']] = disaster_types.get(d['disaster_type'], 0) + 1
            countries[d['country_name']] = countries.get(d['country_name'], 0) + 1
        
        return {
            'input_count': input_count,
            'output_count': len(processed_disasters),
            'success_rate': round(len(processed_disasters) / max(1, input_count), 4),
            'total_people_affected': total_affected,
            'total_funding_requested': total_funding,
            'disaster_type_distribution': disaster_types,
            'country_distribution': countries,
            'processing_date': datetime.now().isoformat()
        }
    
    def _count_total_input_records(self, raw_data) -> int:
        if isinstance(raw_data, list):
            return len(raw_data)
        elif isinstance(raw_data, dict):
            if 'data' in raw_data:
                total = 0
                for country_data in raw_data['data'].values():
                    if isinstance(country_data, dict) and 'events' in country_data:
                        total += len(country_data['events'])
                    elif isinstance(country_data, dict) and 'results' in country_data:
                        total += len(country_data['results'])
                return total
            else:
                return sum(len(v.get('results', [])) for v in raw_data.values() if isinstance(v, dict))
        else:
            return 0
    
    def _create_empty_result(self) -> Dict[str, Any]:
        return {
            'processed_disasters': [],
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
    processor = HistoricalDisastersPreprocessor(config)
    
    print("Start the preprocessing of historical disaster data...")
    result = processor.clean_data()
    
    summary = result['processing_summary']
    print(f"Processing completed!")
    print(f"Input record count: {summary['input_count']}")
    print(f"Output record count: {summary['output_count']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    if summary.get('total_people_affected'):
        print(f"The total number of people affected: {summary['total_people_affected']:,}")
        print(f"Total capital requirement: ${summary['total_funding_requested']:,.2f}")