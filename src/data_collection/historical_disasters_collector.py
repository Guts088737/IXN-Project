"""
02_data_collection/historical_disasters_collector.py

配置驱动的历史灾害数据收集器
用于建立灾害-医疗资源需求的训练数据集
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from src.data_collection.base_collector import ConfigurableCollector
from src.config import get_config, ProjectConfig


class HistoricalDisastersCollector(ConfigurableCollector):
    """
    历史灾害数据收集器
    基于配置的IFRC历史灾害API数据收集
    """

    def get_data_directory(self) -> Path:
        """获取历史灾害数据保存目录"""
        return self.config.paths.historical_disasters_dir

    def collect_data(self) -> Dict[str, Any]:
        """
        收集历史灾害数据的主方法
        """
        self.logger.info("开始收集历史灾害数据...")

        # 1. 收集灾害类型定义
        disaster_types = self._collect_disaster_types()

        # 2. 收集历史灾害申请数据
        appeals_data = self._collect_disaster_appeals()

        # 3. 收集紧急行动数据
        operations_data = self._collect_emergency_operations()

        # 4. 按国家收集历史灾害
        country_disasters = self._collect_disasters_by_countries()

        # 5. 提取医疗相关灾害
        medical_relevant = self._extract_medical_relevant_disasters(
            appeals_data + operations_data
        )

        # 6. 数据整合和验证
        integrated_data = self._integrate_disaster_data({
            'appeals': appeals_data,
            'operations': operations_data,
            'country_disasters': country_disasters,
            'medical_relevant': medical_relevant
        })

        # 7. 保存分类数据
        self._save_categorized_data(integrated_data)

        result = {
            'data': integrated_data,
            'disaster_types': disaster_types,
            'collection_metadata': {
                'appeals_count': len(appeals_data),
                'operations_count': len(operations_data),
                'medical_relevant_count': len(medical_relevant),
                'countries_covered': len(country_disasters),
                'time_range': self._get_collection_time_range()
            }
        }

        return result

    def _collect_disaster_types(self) -> Dict[str, Any]:
        """收集灾害类型定义"""
        self.logger.info("收集灾害类型数据...")

        # 收集基础灾害类型
        disaster_types = self.make_api_request(
            'disaster_types',
            use_cache=True,
            cache_key='disaster_types'
        )


        # 收集聚合灾害类型
        aggregate_types = self.make_api_request(
            f"{self.config.api.base_url_v1}/aggregate_dtype/",
            params={'model_type': 'appeal'},  # 必需参数：指定数据模型类型
            use_cache=True,
            cache_key='aggregate_disaster_types'
        )

        types_data = {
            'basic_types': disaster_types.get('results', []) if disaster_types else [],
            'aggregate_types': aggregate_types if isinstance(aggregate_types, list) else (aggregate_types.get('results', []) if aggregate_types else []),
            'collection_timestamp': datetime.now().isoformat()
        }

        self.save_data(types_data, "disaster_types")
        self.logger.info(f"收集到 {len(types_data['basic_types'])} 种基础灾害类型")

        return types_data

    def _collect_disaster_appeals(self) -> List[Dict]:
        """收集灾害申请数据"""
        self.logger.info("收集灾害申请数据...")

        config = self.config.collection.historical_disasters

        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config['years_back'] * 365)

        params = {
            'start_date__gte': start_date.strftime('%Y-%m-%d'),
            'start_date__lte': end_date.strftime('%Y-%m-%d'),
            'ordering': '-start_date'
        }

        # 添加资金阈值过滤
        if config.get('minimum_funding_threshold', 0) > 0:
            params['amount_requested__gte'] = config['minimum_funding_threshold']

        appeals = self.make_paginated_request(
            'appeals',
            params=params,
            max_records=config.get('max_appeals', 500)
        )

        # 按国家过滤
        filtered_appeals = self.filter_by_countries(appeals)

        # 数据质量过滤
        quality_filtered = self._filter_by_data_quality(
            filtered_appeals,
            required_fields=['country', 'disaster_type', 'amount_requested']
        )

        self.save_data(quality_filtered, "disaster_appeals_raw")
        self.logger.info(f"收集到 {len(quality_filtered)} 条灾害申请记录")

        return quality_filtered

    def _collect_emergency_operations(self) -> List[Dict]:
        """收集紧急行动数据
        
        注意：emergency-operation API目前不可用
        此方法保留接口，待API可用时可快速集成
        """
        self.logger.info("紧急行动数据收集...")

        config = self.config.collection.historical_disasters

        if not config.get('include_operations', True):
            self.logger.info("配置中禁用了紧急行动数据收集")
            return []

        # TODO: 待emergency-operation API可用时，在此处添加API调用
        # 预期的API端点（待确认）:
        # - /api/v2/emergency-operation/
        # - /api/v2/operation/
        
        self.logger.info("emergency-operation API暂不可用，返回空数据")
        self.logger.info("预留位置：待API可用时，将在此处收集紧急行动数据")
        
        # 保存空的操作记录用于占位
        empty_operations = []
        self.save_data({
            'operations': empty_operations,
            'status': 'api_not_available',
            'message': 'emergency-operation API endpoints not available yet',
            'placeholder_timestamp': datetime.now().isoformat()
        }, "emergency_operations_placeholder")
        
        return empty_operations

    def _collect_disasters_by_countries(self) -> Dict[str, List[Dict]]:
        """按国家收集历史灾害"""
        self.logger.info("按国家收集历史灾害数据...")

        countries = self.config.collection.countries_of_interest
        country_disasters = {}

        for country in countries:
            try:
                self.logger.info(f"收集 {country} 的历史灾害...")

                # 搜索该国的灾害申请
                params = {
                    'country__name': country,
                    'limit': 100,
                    'ordering': '-start_date'
                }

                country_data = self.make_api_request(
                    'appeals',
                    params=params,
                    use_cache=True,
                    cache_key=f'country_disasters_{country.replace(" ", "_")}'
                )

                disasters = country_data.get('results', []) if country_data else []
                country_disasters[country] = disasters

                self.logger.info(f"{country}: 找到 {len(disasters)} 个灾害事件")

            except Exception as e:
                self.logger.error(f"收集 {country} 灾害数据失败: {e}")
                country_disasters[country] = []

        self.save_data(country_disasters, "disasters_by_country")
        return country_disasters

    def _extract_medical_relevant_disasters(self, all_disasters: List[Dict]) -> List[Dict]:
        """提取医疗相关的灾害数据"""
        self.logger.info("提取医疗相关灾害信息...")

        medical_keywords = [
            'health', 'medical', 'hospital', 'clinic', 'emergency',
            'trauma', 'injury', 'casualties', 'wounded', 'treatment',
            'healthcare', 'medicine', 'surgical', 'ambulance', 'nurse',
            'doctor', 'physician', 'patient', 'first aid'
        ]

        medical_relevant = []

        for disaster in all_disasters:
            relevance_score = self._calculate_medical_relevance(disaster, medical_keywords)

            if relevance_score > 0.1:  # 医疗相关性阈值
                enhanced_disaster = disaster.copy()
                enhanced_disaster.update({
                    '_medical_relevance_score': relevance_score,
                    '_medical_keywords_found': self._find_medical_keywords(disaster, medical_keywords),
                    '_estimated_medical_needs': self._estimate_medical_needs(disaster)
                })
                medical_relevant.append(enhanced_disaster)

        self.save_data(medical_relevant, "medical_relevant_disasters")
        self.logger.info(f"找到 {len(medical_relevant)} 个医疗相关灾害")

        return medical_relevant

    def _calculate_medical_relevance(self, disaster: Dict, keywords: List[str]) -> float:
        """计算灾害的医疗相关性得分"""
        import json

        disaster_text = json.dumps(disaster, default=str).lower()

        # 关键词权重
        keyword_weights = {
            'health': 0.3, 'medical': 0.3, 'hospital': 0.25, 'clinic': 0.2,
            'casualties': 0.4, 'injured': 0.4, 'wounded': 0.3, 'trauma': 0.3,
            'emergency': 0.15, 'treatment': 0.2, 'healthcare': 0.25,
            'surgical': 0.3, 'ambulance': 0.2, 'first aid': 0.2
        }

        relevance_score = 0.0
        for keyword in keywords:
            if keyword in disaster_text:
                weight = keyword_weights.get(keyword, 0.1)
                # 计算关键词频率
                frequency = disaster_text.count(keyword)
                relevance_score += weight * min(frequency, 3) / 3  # 最多计3次

        return min(relevance_score, 1.0)

    def _find_medical_keywords(self, disaster: Dict, keywords: List[str]) -> List[str]:
        """找出灾害中包含的医疗关键词"""
        import json

        disaster_text = json.dumps(disaster, default=str).lower()
        found_keywords = [kw for kw in keywords if kw in disaster_text]
        return found_keywords

    def _estimate_medical_needs(self, disaster: Dict) -> Dict[str, Any]:
        """基于灾害信息估算医疗需求"""
        # 获取受灾人口
        population_fields = [
            'num_affected', 'targeted_population', 'people_affected',
            'amount_requested', 'amount_funded'
        ]

        affected_population = 0
        funding_amount = 0

        for field in population_fields:
            value = disaster.get(field, 0)
            if isinstance(value, (int, float)) and value > 0:
                if 'population' in field or 'affected' in field:
                    affected_population = max(affected_population, int(value))
                elif 'amount' in field:
                    funding_amount = max(funding_amount, int(value))

        # 基础医疗需求估算（保守估计）
        estimated_needs = {
            'estimated_injured': int(affected_population * 0.05),  # 5%伤亡率
            'estimated_medical_budget': int(funding_amount * 0.2),  # 20%用于医疗
            'estimated_medical_staff_needed': max(1, int(affected_population / 5000)),
            'estimated_medical_facilities_needed': max(1, int(affected_population / 10000))
        }

        return estimated_needs

    def _filter_by_data_quality(self, data: List[Dict], required_fields: List[str]) -> List[Dict]:
        """按数据质量过滤"""
        quality_threshold = 0.5

        filtered_data = []
        for item in data:
            quality_score = self.calculate_data_quality_score(item, required_fields)
            if quality_score >= quality_threshold:
                filtered_data.append(item)

        return filtered_data

    def _filter_by_time_range(self, data: List[Dict]) -> List[Dict]:
        """按时间范围过滤"""
        config = self.config.collection.historical_disasters
        years_back = config.get('years_back', 5)

        cutoff_date = datetime.now() - timedelta(days=years_back * 365)

        filtered_data = []
        for item in data:
            # 尝试多个可能的日期字段
            date_fields = ['start_date', 'created_at', 'date', 'disaster_start_date']

            item_date = None
            for field in date_fields:
                if field in item and item[field]:
                    try:
                        # 使用基类的日期解析方法
                        date_str = str(item[field])
                        item_date = self._parse_iso_datetime(date_str)
                        # 如果解析结果不是1970年（解析失败标志），则使用
                        if item_date.year > 1970:
                            break
                    except:
                        continue

            if item_date and item_date >= cutoff_date:
                filtered_data.append(item)
            elif not item_date:
                # 如果没有日期信息，保留数据但标记
                item['_no_date_info'] = True
                filtered_data.append(item)

        return filtered_data

    def _integrate_disaster_data(self, data_sources: Dict[str, Any]) -> List[Dict]:
        """整合不同来源的灾害数据"""
        self.logger.info("整合灾害数据...")

        integrated = []
        processed_ids = set()

        # 处理申请数据
        for appeal in data_sources.get('appeals', []):
            disaster_id = appeal.get('id') or appeal.get('disaster_id')
            if disaster_id not in processed_ids:
                integrated_disaster = self._create_integrated_disaster_record(appeal, 'appeal')
                integrated.append(integrated_disaster)
                processed_ids.add(disaster_id)

        # 处理行动数据
        for operation in data_sources.get('operations', []):
            disaster_id = operation.get('id') or operation.get('disaster_id')
            if disaster_id not in processed_ids:
                integrated_disaster = self._create_integrated_disaster_record(operation, 'operation')
                integrated.append(integrated_disaster)
                processed_ids.add(disaster_id)

        # 添加医疗相关性信息
        medical_relevant_ids = {
            item.get('id') or item.get('disaster_id'): item
            for item in data_sources.get('medical_relevant', [])
        }

        for disaster in integrated:
            disaster_id = disaster.get('disaster_id')
            if disaster_id in medical_relevant_ids:
                medical_data = medical_relevant_ids[disaster_id]
                disaster.update({
                    'medical_relevance_score': medical_data.get('_medical_relevance_score', 0),
                    'medical_keywords_found': medical_data.get('_medical_keywords_found', []),
                    'estimated_medical_needs': medical_data.get('_estimated_medical_needs', {})
                })

        self.logger.info(f"整合了 {len(integrated)} 个灾害记录")
        return integrated

    def _create_integrated_disaster_record(self, raw_data: Dict, source_type: str) -> Dict:
        """创建整合的灾害记录"""
        integrated = {
            'disaster_id': raw_data.get('id') or raw_data.get('disaster_id'),
            'disaster_type': raw_data.get('disaster_type') or raw_data.get('dtype'),
            'country': raw_data.get('country'),
            'start_date': raw_data.get('start_date') or raw_data.get('created_at'),
            'amount_requested': self.safe_int_convert(raw_data.get('amount_requested', 0)),
            'amount_funded': self.safe_int_convert(raw_data.get('amount_funded', 0)),
            'people_affected': self.safe_int_convert(raw_data.get('num_affected', 0)),
            'source_type': source_type,
            'original_data': raw_data,
            '_integration_timestamp': datetime.now().isoformat()
        }

        return integrated

    def _save_categorized_data(self, integrated_data: List[Dict]) -> None:
        """保存分类的灾害数据"""
        # 按灾害类型保存
        by_type = {}
        for disaster in integrated_data:
            disaster_type = disaster.get('disaster_type', 'unknown')
            if disaster_type not in by_type:
                by_type[disaster_type] = []
            by_type[disaster_type].append(disaster)

        for disaster_type, disasters in by_type.items():
            safe_type = str(disaster_type).replace(' ', '_').replace('/', '_')
            self.save_data(disasters, f"disasters_by_type_{safe_type}")

        # 按国家保存
        by_country = {}
        for disaster in integrated_data:
            country = disaster.get('country', 'Unknown')
            if country not in by_country:
                by_country[country] = []
            by_country[country].append(disaster)

        for country, disasters in by_country.items():
            country_safe = str(country).replace(' ', '_').replace('/', '_')
            self.save_data(disasters, f"disasters_by_country_{country_safe}")

        # 按医疗相关性保存
        medical_relevant = [
            d for d in integrated_data
            if d.get('medical_relevance_score', 0) > 0.2
        ]
        if medical_relevant:
            self.save_data(medical_relevant, "high_medical_relevance_disasters")

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
        """获取历史灾害必需字段"""
        return ['country', 'disaster_type', 'start_date']

    def _apply_custom_filters(self, data: List[Dict]) -> List[Dict]:
        """应用历史灾害特定的过滤器"""
        config = self.config.collection.historical_disasters

        # 按灾害类型过滤
        disaster_types_filter = config.get('disaster_types_filter', [])
        if disaster_types_filter:
            data = [
                item for item in data
                if item.get('disaster_type') in disaster_types_filter
            ]

        # 按资金阈值过滤
        min_funding = config.get('minimum_funding_threshold', 0)
        if min_funding > 0:
            data = [
                item for item in data
                if self.safe_int_convert(item.get('amount_requested', 0)) >= min_funding
            ]

        return data


if __name__ == "__main__":
    # 使用示例
    from src.config import load_config

    # 加载配置
    config = load_config()

    # 自定义历史灾害收集配置
    config.collection.historical_disasters.update({
        'years_back': 3,
        'minimum_funding_threshold': 10000,
        'include_operations': True
    })

    # 创建收集器
    collector = HistoricalDisastersCollector(config)

    # 运行收集
    print("开始历史灾害数据收集...")
    result = collector.run_collection()

    print(f"收集完成!")
    
    # 安全访问collection_metadata
    metadata = result.get('collection_metadata', {})
    print(f"申请数据: {metadata.get('appeals_count', 0)} 条")
    print(f"行动数据: {metadata.get('operations_count', 0)} 条")
    print(f"医疗相关: {metadata.get('medical_relevant_count', 0)} 条")
    print(f"覆盖国家: {metadata.get('countries_covered', 0)} 个")

    # 数据质量报告
    validation = collector.validate_collected_data(result.get('data', []))
    print(f"数据质量评分: {validation['quality_score']:.2f}")