"""
02_data_collection/historical_disasters_collector.py

专注于历史灾害数据收集
使用IFRC Event API收集历史灾害事件数据，不包含数据预处理
"""

from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from .base_collector import ConfigurableCollector
    from .country_collector import CountryCollector
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from src.data_collection.base_collector import ConfigurableCollector
    from src.data_collection.country_collector import CountryCollector


class HistoricalDisastersCollector(ConfigurableCollector):
    """
    历史灾害数据收集器
    使用IFRC Event API收集历史灾害事件数据，不进行数据预处理
    """
    
    def __init__(self, config):
        """初始化收集器，创建国家收集器实例"""
        super().__init__(config)
        self.country_collector = CountryCollector(config)

    def get_data_directory(self) -> Path:
        """获取历史灾害数据保存目录"""
        return self.config.paths.historical_disasters_dir

    def collect_data(self) -> Dict[str, Any]:
        """
        收集历史灾害数据的主方法
        只负责数据收集，不进行预处理
        """
        self.logger.info("开始收集历史灾害数据...")

        # 从缓存获取所有国家ID和信息（静默模式）
        all_country_ids = self.country_collector.get_all_country_ids(silent=True)
        all_countries_info = self.country_collector.get_all_countries_info(silent=True)

        self.logger.info(f"获取到 {len(all_country_ids)} 个国家ID")

        # 按所有国家ID收集历史灾害原始数据
        country_disasters = self._collect_disasters_by_all_country_ids(all_country_ids, all_countries_info)

        # 保存原始数据
        self.save_data(country_disasters, "events_by_all_countries")

        result = {
            'raw_data': country_disasters,
            'all_countries_info': all_countries_info,
            'collection_metadata': {
                'total_countries': len(all_country_ids),
                'countries_with_disasters': len([c for c in country_disasters.values() if c.get('count', 0) > 0]),
                'total_disaster_records': sum(c.get('count', 0) for c in country_disasters.values()),
                'collection_timestamp': datetime.now().isoformat(),
                'time_range': self._get_collection_time_range()
            }
        }

        return result

    def _collect_disasters_by_all_country_ids(self, country_ids: List[int], countries_info: List[Dict]) -> Dict[str, Dict]:
        """使用所有国家ID收集历史灾害原始数据"""
        self.logger.info(f"开始为 {len(country_ids)} 个国家收集历史灾害数据...")

        # 创建国家ID到名称的映射
        id_to_name = {c['id']: c['name'] for c in countries_info}
        
        country_disasters = {}
        processed_count = 0
        total_disasters = 0

        for country_id in country_ids:
            country_name = id_to_name.get(country_id, f'Country_{country_id}')
            
            try:
                self.logger.debug(f"收集国家 {country_name} (ID: {country_id}) 的历史灾害...")

                # 收集原始灾害数据
                disasters = self._collect_disaster_data_for_country(country_id)

                country_disasters[country_name] = {
                    'country_id': country_id,
                    'country_name': country_name,
                    'events': disasters,  # 改为events表示这是事件数据
                    'count': len(disasters),
                    'collection_timestamp': datetime.now().isoformat()
                }

                total_disasters += len(disasters)
                processed_count += 1
                
                # 每处理50个国家打印一次进度
                if processed_count % 50 == 0:
                    self.logger.info(f"已处理 {processed_count}/{len(country_ids)} 个国家，累计收集 {total_disasters} 条灾害记录")

                if len(disasters) > 0:
                    self.logger.debug(f"{country_name}: 找到 {len(disasters)} 个灾害事件")

            except Exception as e:
                self.logger.error(f"收集 {country_name} (ID: {country_id}) 灾害数据失败: {e}")
                country_disasters[country_name] = {
                    'country_id': country_id,
                    'country_name': country_name,
                    'events': [],  # 改为events表示这是事件数据
                    'count': 0,
                    'error': str(e),
                    'collection_timestamp': datetime.now().isoformat()
                }

        self.logger.info(f"完成所有国家数据收集: 处理了 {processed_count} 个国家，总共收集 {total_disasters} 条灾害记录")
        return country_disasters

    def _collect_disaster_data_for_country(self, country_id: int) -> List[Dict]:
        """为指定国家ID使用event API收集历史灾害数据"""
        config = self.config.collection.historical_disasters

        # 计算时间范围 - 近5年
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.get('years_back', 5) * 365)

        # 基础参数 - 使用event API的参数
        params = {
            'limit': 1000,  # 每个国家最多1000条记录
            'countries__in': country_id,  # 按国家ID过滤
        }

        # 添加时间范围参数 - 基于灾害开始时间
        try:
            params.update({
                'disaster_start_date__gte': start_date.strftime('%Y-%m-%d'),
                'disaster_start_date__lte': end_date.strftime('%Y-%m-%d'),
            })
        except:
            pass

        # 从配置中添加可选参数（适配event API）
        optional_params = [
            'dtype', 'is_featured', 'regions__in', 'search'
        ]

        for param in optional_params:
            config_value = config.get(param)
            if config_value is not None and config_value != '':
                params[param] = config_value

        try:
            # 使用event API
            api_url = f"{self.config.api.base_url_v2}/event/"
            
            # 使用分类缓存
            cache_key = f'historical_disasters/country_{country_id}_events'
            events_response = self.make_api_request(
                api_url,
                params=params,
                use_cache=True,
                cache_key=cache_key
            )

            if events_response:
                # 处理API响应格式
                events = []
                if isinstance(events_response, list):
                    events = events_response
                elif isinstance(events_response, dict):
                    events = events_response.get('results', [])
                
                # 增强每个事件记录的信息
                enhanced_events = []
                for event in events:
                    enhanced_event = self._enhance_event_fields(event)
                    enhanced_events.append(enhanced_event)
                
                return enhanced_events
            else:
                return []

        except Exception as e:
            self.logger.error(f"收集国家 {country_id} 的event数据失败: {e}")
            return []

    def _enhance_event_fields(self, event: Dict) -> Dict:
        """精简event记录，只保留关键字段"""
        
        # 创建精简的事件记录
        simplified_event = {
            "event_id": event.get('id'),
            "name": event.get('name', 'Unknown Event'),
            "disaster_type": {
                "id": event.get('dtype', {}).get('id') if event.get('dtype') else None,
                "name": event.get('dtype', {}).get('name') if event.get('dtype') else None
            },
            "glide": event.get('glide'),
            "severity_level": event.get('severity_level_display'),
            "disaster_start_date": self._normalize_datetime(event.get('disaster_start_date')),
            "disaster_end_date": self._normalize_datetime(event.get('disaster_end_date')) if event.get('disaster_end_date') else None,
            "num_affected": event.get('num_affected')
        }
        
        # 提取主要国家信息
        countries = event.get('countries', [])
        if countries:
            primary_country = countries[0]
            simplified_event["country"] = {
                "id": primary_country.get('id'),
                "name": primary_country.get('name'),
                "iso": primary_country.get('iso', ''),
                "iso3": primary_country.get('iso3', ''),
                "region": primary_country.get('region'),
                "society_name": primary_country.get('society_name', '')
            }
        else:
            simplified_event["country"] = None
        
        # 精简appeals信息
        appeals = event.get('appeals', [])
        simplified_appeals = []
        for appeal in appeals:
            simplified_appeal = {
                "id": appeal.get('id'),
                "code": appeal.get('code'),
                "atype_display": appeal.get('atype_display'),
                "num_beneficiaries": appeal.get('num_beneficiaries'),
                "amount_requested": appeal.get('amount_requested'),
                "amount_funded": appeal.get('amount_funded'),
                "status_display": appeal.get('status_display'),
                "start_date": self._normalize_datetime(appeal.get('start_date')),
                "end_date": self._normalize_datetime(appeal.get('end_date'))
            }
            simplified_appeals.append(simplified_appeal)
        
        simplified_event["appeals"] = simplified_appeals
        
        # 精简field_reports信息
        field_reports = event.get('field_reports', [])
        simplified_reports = []
        for report in field_reports:
            simplified_report = {
                "id": report.get('id'),
                "status": report.get('status'),
                "created_at": self._normalize_datetime(report.get('created_at')),
                "report_date": self._normalize_datetime(report.get('report_date')),
                "summary": report.get('summary', ''),
                "num_affected": report.get('num_affected')
            }
            simplified_reports.append(simplified_report)
        
        simplified_event["field_reports"] = simplified_reports
        
        self.logger.debug(f"精简事件记录 - {simplified_event.get('name', 'Unknown')}: "
                         f"ID={simplified_event.get('event_id')}, "
                         f"国家={simplified_event.get('country', {}).get('name') if simplified_event.get('country') else 'Unknown'}, "
                         f"类型={simplified_event.get('disaster_type', {}).get('name')}")
        
        return simplified_event
    
    def _normalize_datetime(self, date_value) -> str:
        """标准化日期时间格式"""
        if not date_value:
            return None
            
        if isinstance(date_value, str):
            # 确保时间格式标准化
            if 'T' not in date_value and len(date_value) == 10:  # 只有日期
                return f"{date_value}T00:00:00Z"
            elif not date_value.endswith('Z') and 'T' in date_value:
                if '+' not in date_value and not date_value.endswith('Z'):
                    return f"{date_value}Z"
            return date_value
        else:
            return str(date_value)
    
    def _parse_datetime(self, date_value):
        """解析日期时间字符串为datetime对象"""
        if not date_value:
            return None
            
        try:
            from datetime import datetime
            date_str = str(date_value).replace('Z', '').replace('+00:00', '')
            
            if 'T' in date_str:
                return datetime.strptime(date_str[:19], '%Y-%m-%dT%H:%M:%S')
            else:
                return datetime.strptime(date_str[:10], '%Y-%m-%d')
        except Exception as e:
            self.logger.debug(f"解析日期失败 {date_value}: {e}")
            return None

    def _get_collection_time_range(self) -> Dict[str, str]:
        """获取收集的时间范围"""
        config = self.config.collection.historical_disasters
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.get('years_back', 5) * 365)

        return {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'years_covered': config.get('years_back', 5)
        }


    def _get_required_fields(self) -> List[str]:
        """获取历史灾害数据收集的必需字段"""
        return ['date', 'disaster_name']


if __name__ == "__main__":
    # 使用示例
    from src.config import load_config

    # 加载配置
    config = load_config()

    # 自定义历史灾害收集配置
    config.collection.historical_disasters.update({
        'years_back': 5,
        'max_appeals': 5000
    })

    # 创建收集器
    collector = HistoricalDisastersCollector(config)

    # 运行收集
    print("开始历史灾害原始数据收集...")
    result = collector.run_collection()

    print(f"收集完成!")
    
    # 安全访问collection_metadata
    metadata = result.get('collection_metadata', {})
    print(f"处理的国家总数: {metadata.get('total_countries', 0)} 个")
    print(f"有灾害数据的国家: {metadata.get('countries_with_disasters', 0)} 个")
    print(f"总原始灾害记录: {metadata.get('total_disaster_records', 0)} 条")

    # 显示收集到的国家信息
    all_countries = result.get('all_countries_info', [])
    print(f"使用的国家数: {len(all_countries)}")
