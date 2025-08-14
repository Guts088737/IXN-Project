"""
Field Report Data Collector for IFRC API
Collects all field report records from the API
"""

from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import sys
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from .base_collector import ConfigurableCollector
except ImportError:
    from src.data_collection.base_collector import ConfigurableCollector


class FieldReportCollector(ConfigurableCollector):
    """Collects all field report data"""

    def get_data_directory(self) -> Path:
        """Get field report data save directory"""
        report_dir = self.config.paths.collection_raw_dir / "field_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def collect_data(self) -> Dict[str, Any]:
        """Main data collection method - collects all field reports"""
        self.logger.info("Starting field report data collection...")

        # Collect all field reports
        all_reports = self._collect_all_field_reports()
        
        if not all_reports:
            self.logger.warning("No field reports data found")
            return {
                'data': [],
                'collection_metadata': {
                    'field_reports_records': 0,
                    'collection_timestamp': datetime.now().isoformat(),
                    'data_type': 'field_reports',
                    'time_range': {
                        'years_back': 5,
                        'start_date': (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d'),
                        'end_date': datetime.now().strftime('%Y-%m-%d')
                    }
                }
            }

        # Save all reports
        self.save_data(all_reports, "all_field_reports")
        
        return {
            'data': all_reports,
            'collection_metadata': {
                'field_reports_records': len(all_reports),
                'collection_timestamp': datetime.now().isoformat(),
                'data_type': 'field_reports',
                'time_range': {
                    'years_back': 5,
                    'start_date': (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d'),
                    'end_date': datetime.now().strftime('%Y-%m-%d')
                }
            }
        }

    def _collect_all_field_reports(self) -> List[Dict]:
        """Collect all field reports with pagination"""
        all_reports = []
        page = 1
        limit = 500  # Get 500 records per page
        
        try:
            while True:
                self.logger.info(f"Collecting field reports page {page}...")
                
                # Calculate offset
                offset = (page - 1) * limit
                
                # Make API request
                page_data = self._get_field_reports_page(limit, offset)
                
                if not page_data:
                    self.logger.warning(f"No data returned for page {page}")
                    break
                
                # Extract results
                if isinstance(page_data, dict):
                    if 'results' in page_data:
                        results = page_data['results']
                        if not results:
                            self.logger.info("No more results, collection complete")
                            break
                        
                        all_reports.extend(results)
                        self.logger.info(f"Page {page}: collected {len(results)} reports, total: {len(all_reports)}")
                        
                        # Check if we have more pages
                        if 'next' not in page_data or not page_data['next']:
                            self.logger.info("No more pages, collection complete")
                            break
                            
                    else:
                        # If no pagination structure, assume single response
                        all_reports.append(page_data)
                        break
                        
                elif isinstance(page_data, list):
                    all_reports.extend(page_data)
                    if len(page_data) < limit:
                        self.logger.info("Received fewer records than requested, collection complete")
                        break
                else:
                    self.logger.warning(f"Unexpected data format: {type(page_data)}")
                    break
                
                page += 1
                
                # Rate limiting
                time.sleep(self.config.api.rate_limit_delay)
                
        except Exception as e:
            self.logger.error(f"Error during field reports collection: {e}")
            
        self.logger.info(f"Completed field reports collection: {len(all_reports)} reports collected")
        return all_reports

    def _get_field_reports_page(self, limit: int, offset: int) -> Optional[Any]:
        """Get a single page of field reports"""
        try:
            # Check cache first
            cache_key = f'field_reports/page_{offset}_{limit}'
            cached_data = self._load_from_cache(cache_key)
            if cached_data:
                self.logger.debug(f"Loaded field reports page from cache (offset: {offset})")
                return cached_data

            # Build URL
            api_url = "https://goadmin-stage.ifrc.org/api/v2/field-report/"
            
            # Calculate time range for last 5 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5 * 365)
            
            # Parameters with date range filtering
            params = {
                'limit': limit,
                'offset': offset,
                'format': 'json',
                'created_at__gte': start_date.strftime('%Y-%m-%d'),
                'created_at__lte': end_date.strftime('%Y-%m-%d')
            }
            
            response = self.session.get(
                api_url,
                params=params,
                timeout=self.config.api.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the result
            self._save_to_cache(data, cache_key)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to get field reports page (offset: {offset}): {e}")
            return None


if __name__ == "__main__":
    from src.config import load_config

    config = load_config()
    collector = FieldReportCollector(config)

    # Run collection
    result = collector.run_collection()

    # Show results
    print("\n" + "="*50)
    print("FIELD REPORT DATA COLLECTION RESULTS")
    print("="*50)
    
    metadata = result.get('collection_metadata', {})
    print(f"Field reports records: {metadata.get('field_reports_records', 0)}")
    print(f"Collection timestamp: {metadata.get('collection_timestamp', 'N/A')}")
    print(f"Data type: {metadata.get('data_type', 'N/A')}")
    print("="*50)
    
    # Show sample of first few reports if available
    data = result.get('data', [])
    if data and len(data) > 0:
        print(f"\nSample of first 3 reports:")
        for i, report in enumerate(data[:3], 1):
            if isinstance(report, dict):
                report_id = report.get('id', 'N/A')
                summary = report.get('summary', 'N/A')
                created_at = report.get('created_at', 'N/A')
                print(f"  {i}. ID: {report_id}, Created: {created_at}")
                print(f"     Summary: {summary[:100]}..." if len(str(summary)) > 100 else f"     Summary: {summary}")
    print("="*50)