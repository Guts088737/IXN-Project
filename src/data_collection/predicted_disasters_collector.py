"""
02_data_collection/predicted_disasters_collector.py

配置驱动的预测灾害数据收集器
为未来的灾害预测API做好准备的框架
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))

from base_collector import ConfigurableCollector
from config import get_config, ProjectConfig


class PredictedDisastersCollector(ConfigurableCollector):
    """
    预测灾害数据收集器

    注意：目前partner表示没有专门的预测灾害API
    但这个框架为未来的预测数据做好了准备
    """

    def get_data_directory(self) -> Path:
        """获取预测灾害数据保存目录"""
        return self.config.paths.predicted_disasters_dir

    def collect_data(self) -> Dict[str, Any]:
        """
        收集预测灾害数据的主方法
        """
        self.logger.info("开始收集预测灾害数据...")

        # 定义所有预测数据源
        prediction_sources = self._get_prediction_sources_config()

        collected_predictions = {}
        collection_results = {}

        for source_name, source_config in prediction_sources.items():
            self.logger.info(f"收集 {source_name} 预测数据...")

            try:
                if source_config['enabled']:
                    data = self._collect_prediction_source(source_name, source_config)
                    collected_predictions[source_name] = data
                    collection_results[source_name] = {
                        'status': 'success',
                        'records_count': len(data) if isinstance(data, list) else 1,
                        'data_available': bool(data)
                    }
                else:
                    self.logger.info(f"{source_name} 被配置禁用")
                    collection_results[source_name] = {
                        'status': 'disabled',
                        'data_available': False
                    }

            except Exception as e:
                self.logger.error(f"收集 {source_name} 失败: {e}")
                collection_results[source_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'data_available': False
                }

        # 整合预测数据
        integrated_predictions = self._integrate_prediction_data(collected_predictions)

        # 生成预测摘要
        prediction_summary = self._generate_prediction_summary(integrated_predictions)

        # 如果没有真实预测数据，生成测试数据
        if not integrated_predictions and self._should_generate_mock_data():
            mock_data = self._generate_mock_predictions()
            integrated_predictions.extend(mock_data)
            collection_results['mock_data'] = {
                'status': 'generated',
                'records_count': len(mock_data),
                'note': '由于真实API不可用，生成了测试数据'
            }

        # 保存分类预测数据
        self._save_categorized_predictions(integrated_predictions)

        result = {
            'data': integrated_predictions,
            'collection_results': collection_results,
            'prediction_summary': prediction_summary,
            'collection_metadata': {
                'total_predictions': len(integrated_predictions),
                'successful_sources': len([r for r in collection_results.values() if r['status'] == 'success']),
                'failed_sources': len([r for r in collection_results.values() if r['status'] == 'failed']),
                'prediction_time_range': self._get_prediction_time_range()
            }
        }

        return result

    def _get_prediction_sources_config(self) -> Dict[str, Dict[str, Any]]:
        """获取预测数据源配置"""
        return {
            'earthquake_predictions': {
                'enabled': True,
                'api_endpoint': 'earthquake',
                'api_version': 'v1',
                'description': '地震预测数据',
                'cache_duration_hours': 6,
                'priority': 1
            },
            'global_exposure_data': {
                'enabled': True,
                'api_endpoint': 'global_exposure',
                'api_version': 'v1',
                'description': '全球风险暴露数据',
                'cache_duration_hours': 24,
                'priority': 2
            },
            'risk_scores': {
                'enabled': True,
                'api_endpoint': 'risk_score',
                'api_version': 'v1',
                'description': '灾害风险评分',
                'cache_duration_hours': 12,
                'priority': 3
            },
            'seasonal_predictions': {
                'enabled': True,
                'api_endpoint': 'seasonal',
                'api_version': 'v1',
                'description': '季节性预测数据',
                'cache_duration_hours': 48,
                'priority': 4
            },
            'country_seasonal': {
                'enabled': True,
                'api_endpoint': 'country_seasonal',
                'api_version': 'v1',
                'description': '国家季节性数据',
                'cache_duration_hours': 48,
                'priority': 4
            },
            'early_warning_data': {
                'enabled': True,
                'api_endpoint': 'early_actions',
                'api_version': 'v1',
                'description': '早期预警数据',
                'cache_duration_hours': 2,
                'priority': 1
            },
            'inform_risk_scores': {
                'enabled': True,
                'api_endpoint': 'inform_score',
                'api_version': 'v1',
                'description': 'INFORM风险指数',
                'cache_duration_hours': 24,
                'priority': 3
            },
            'adam_exposure': {
                'enabled': False,  # 默认禁用，可通过配置启用
                'api_endpoint': 'adam-exposure',
                'api_version': 'v1',
                'description': 'ADAM暴露数据',
                'cache_duration_hours': 24,
                'priority': 5
            }
        }

    def _collect_prediction_source(self, source_name: str, source_config: Dict) -> Any:
        """收集单个预测数据源"""
        endpoint = source_config['api_endpoint']
        cache_key = f"{source_name}_{datetime.now().strftime('%Y%m%d_%H')}"

        # 使用缓存，缓存时间基于配置
        data = self.make_api_request(
            endpoint,
            use_cache=True,
            cache_key=cache_key
        )

        if data:
            # 处理数据格式
            processed_data = self._process_prediction_data(data, source_name, source_config)

            # 保存原始数据
            self.save_data({
                'source': source_name,
                'raw_data': data,
                'processed_data': processed_data,
                'collection_timestamp': datetime.now().isoformat()
            }, f"{source_name}_raw")

            return processed_data

        return None

    def _process_prediction_data(self, raw_data: Any, source_name: str, source_config: Dict) -> Dict[str, Any]:
        """处理预测数据"""
        processed = {
            'source_name': source_name,
            'data_type': 'prediction',
            'description': source_config['description'],
            'collection_timestamp': datetime.now().isoformat(),
            'data_quality_score': 0.0,
            'prediction_confidence': 0.0,
            'geographic_coverage': [],
            'time_horizon': None,
            'raw_data': raw_data
        }

        # 根据数据源类型进行特定处理
        if source_name == 'earthquake_predictions':
            processed.update(self._process_earthquake_data(raw_data))
        elif source_name == 'global_exposure_data':
            processed.update(self._process_exposure_data(raw_data))
        elif source_name == 'risk_scores':
            processed.update(self._process_risk_score_data(raw_data))
        elif 'seasonal' in source_name:
            processed.update(self._process_seasonal_data(raw_data))
        elif source_name == 'early_warning_data':
            processed.update(self._process_early_warning_data(raw_data))
        elif source_name == 'inform_risk_scores':
            processed.update(self._process_inform_data(raw_data))

        # 计算数据质量评分
        processed['data_quality_score'] = self._calculate_prediction_quality(processed)

        return processed

    def _process_earthquake_data(self, data: Any) -> Dict[str, Any]:
        """处理地震预测数据"""
        if not data:
            return {}

        # 提取地震预测信息
        predictions = data.get('results', [data]) if isinstance(data, dict) else [data]

        processed_predictions = []
        for pred in predictions:
            if isinstance(pred, dict):
                processed_pred = {
                    'disaster_type': 'earthquake',
                    'magnitude': pred.get('magnitude'),
                    'location': {
                        'latitude': pred.get('lat') or pred.get('latitude'),
                        'longitude': pred.get('lon') or pred.get('longitude'),
                        'region': pred.get('region') or pred.get('location')
                    },
                    'prediction_date': pred.get('date') or pred.get('prediction_date'),
                    'confidence_level': pred.get('confidence', 0.5),
                    'depth': pred.get('depth'),
                    'affected_population_estimate': pred.get('population_at_risk', 0)
                }
                processed_predictions.append(processed_pred)

        return {
            'prediction_type': 'earthquake',
            'predictions': processed_predictions,
            'prediction_count': len(processed_predictions)
        }

    def _process_exposure_data(self, data: Any) -> Dict[str, Any]:
        """处理暴露数据"""
        if not data:
            return {}

        return {
            'prediction_type': 'exposure_analysis',
            'exposure_data': data,
            'geographic_coverage': self._extract_geographic_coverage(data),
            'population_exposure': self._extract_population_exposure(data)
        }

    def _process_risk_score_data(self, data: Any) -> Dict[str, Any]:
        """处理风险评分数据"""
        if not data:
            return {}

        risk_scores = data.get('results', [data]) if isinstance(data, dict) else [data]

        return {
            'prediction_type': 'risk_assessment',
            'risk_scores': risk_scores,
            'average_risk_score': self._calculate_average_risk(risk_scores),
            'high_risk_areas': self._identify_high_risk_areas(risk_scores)
        }

    def _process_seasonal_data(self, data: Any) -> Dict[str, Any]:
        """处理季节性预测数据"""
        if not data:
            return {}

        return {
            'prediction_type': 'seasonal_forecast',
            'seasonal_data': data,
            'forecast_period': self._extract_forecast_period(data),
            'seasonal_risks': self._extract_seasonal_risks(data)
        }

    def _process_early_warning_data(self, data: Any) -> Dict[str, Any]:
        """处理早期预警数据"""
        if not data:
            return {}

        warnings = data.get('results', [data]) if isinstance(data, dict) else [data]

        active_warnings = []
        for warning in warnings:
            if isinstance(warning, dict):
                warning_info = {
                    'warning_id': warning.get('id'),
                    'disaster_type': warning.get('disaster_type'),
                    'alert_level': warning.get('alert_level'),
                    'country': warning.get('country'),
                    'issued_date': warning.get('issued_date'),
                    'expiry_date': warning.get('expiry_date'),
                    'description': warning.get('description')
                }
                active_warnings.append(warning_info)

        return {
            'prediction_type': 'early_warning',
            'active_warnings': active_warnings,
            'warning_count': len(active_warnings),
            'countries_at_risk': list(set(w.get('country') for w in active_warnings if w.get('country')))
        }

    def _process_inform_data(self, data: Any) -> Dict[str, Any]:
        """处理INFORM风险指数数据"""
        if not data:
            return {}

        return {
            'prediction_type': 'inform_risk_index',
            'inform_data': data,
            'risk_categories': self._extract_inform_categories(data),
            'country_rankings': self._extract_country_rankings(data)
        }

    def _integrate_prediction_data(self, collected_data: Dict[str, Any]) -> List[Dict]:
        """整合不同来源的预测数据"""
        self.logger.info("整合预测数据...")

        integrated_predictions = []

        for source_name, prediction_data in collected_data.items():
            if not prediction_data:
                continue

            # 为每个预测数据源创建标准化记录
            if prediction_data.get('predictions'):
                # 处理具体预测（如地震）
                for pred in prediction_data['predictions']:
                    integrated_pred = self._create_standardized_prediction(pred, source_name)
                    integrated_predictions.append(integrated_pred)

            elif prediction_data.get('active_warnings'):
                # 处理早期预警
                for warning in prediction_data['active_warnings']:
                    integrated_pred = self._create_standardized_prediction(warning, source_name)
                    integrated_predictions.append(integrated_pred)

            else:
                # 处理其他类型的预测数据
                integrated_pred = self._create_standardized_prediction(prediction_data, source_name)
                integrated_predictions.append(integrated_pred)

        # 按国家过滤
        filtered_predictions = self.filter_by_countries(integrated_predictions)

        self.logger.info(f"整合了 {len(filtered_predictions)} 个预测记录")
        return filtered_predictions

    def _create_standardized_prediction(self, raw_prediction: Dict, source: str) -> Dict:
        """创建标准化的预测记录"""
        return {
            'prediction_id': f"{source}_{raw_prediction.get('id', int(time.time()))}",
            'data_source': source,
            'disaster_type': raw_prediction.get('disaster_type', 'unknown'),
            'country': raw_prediction.get('country'),
            'region': raw_prediction.get('region') or raw_prediction.get('location', {}).get('region'),
            'prediction_date': raw_prediction.get('prediction_date') or datetime.now().isoformat(),
            'confidence_level': raw_prediction.get('confidence_level', 0.5),
            'severity_estimate': raw_prediction.get('severity_estimate', 'medium'),
            'affected_population_estimate': self.safe_int_convert(
                raw_prediction.get('affected_population_estimate', 0)
            ),
            'geographic_coordinates': {
                'latitude': raw_prediction.get('location', {}).get('latitude'),
                'longitude': raw_prediction.get('location', {}).get('longitude')
            },
            'time_horizon': raw_prediction.get('time_horizon', '7_days'),
            'medical_impact_estimate': self._estimate_medical_impact(raw_prediction),
            'raw_data': raw_prediction,
            '_integration_timestamp': datetime.now().isoformat()
        }

    def _estimate_medical_impact(self, prediction: Dict) -> Dict[str, Any]:
        """估算预测灾害的医疗影响"""
        disaster_type = prediction.get('disaster_type', '').lower()
        affected_population = self.safe_int_convert(prediction.get('affected_population_estimate', 0))
        severity = prediction.get('severity_estimate', 'medium')

        # 基础医疗影响系数（保守估计）
        impact_multipliers = {
            'earthquake': {'low': 0.10, 'medium': 0.15, 'high': 0.25, 'extreme': 0.40},
            'flood': {'low': 0.03, 'medium': 0.05, 'high': 0.08, 'extreme': 0.15},
            'cyclone': {'low': 0.08, 'medium': 0.12, 'high': 0.20, 'extreme': 0.35},
            'drought': {'low': 0.02, 'medium': 0.05, 'high': 0.10, 'extreme': 0.20},
            'wildfire': {'low': 0.05, 'medium': 0.08, 'high': 0.12, 'extreme': 0.25}
        }

        multiplier = impact_multipliers.get(disaster_type, {}).get(severity, 0.10)

        return {
            'estimated_casualties': int(affected_population * multiplier),
            'estimated_injured': int(affected_population * multiplier * 3),
            'medical_facilities_at_risk': max(1, int(affected_population / 20000)),
            'medical_personnel_needed': max(5, int(affected_population / 2000)),
            'estimated_medical_supply_needs': {
                'emergency_kits': max(10, int(affected_population / 1000)),
                'surgical_supplies': max(5, int(affected_population / 5000)),
                'blood_units_needed': max(20, int(affected_population / 500))
            }
        }

    def _should_generate_mock_data(self) -> bool:
        """判断是否应该生成模拟数据"""
        # 如果配置允许且没有真实数据，则生成模拟数据
        return True  # 可以通过配置控制

    def _generate_mock_predictions(self) -> List[Dict]:
        """生成模拟预测数据用于测试框架"""
        self.logger.info("生成模拟预测数据用于测试...")

        mock_predictions = []
        countries = self.config.collection.countries_of_interest[:3]  # 限制为3个国家

        disaster_types = ['earthquake', 'flood', 'cyclone']
        severities = ['medium', 'high']

        for i, country in enumerate(countries):
            disaster_type = disaster_types[i % len(disaster_types)]
            severity = severities[i % len(severities)]

            mock_prediction = {
                'prediction_id': f'mock_prediction_{i + 1}_{int(time.time())}',
                'data_source': 'mock_data_generator',
                'disaster_type': disaster_type,
                'country': country,
                'prediction_date': (datetime.now() + timedelta(days=7)).isoformat(),
                'confidence_level': 0.75,
                'severity_estimate': severity,
                'affected_population_estimate': 50000 + (i * 20000),
                'time_horizon': '7_days',
                'medical_impact_estimate': {
                    'estimated_casualties': 2500 + (i * 1000),
                    'estimated_injured': 7500 + (i * 3000),
                    'medical_facilities_at_risk': 2 + i,
                    'medical_personnel_needed': 25 + (i * 10)
                },
                '_is_mock_data': True,
                '_generation_timestamp': datetime.now().isoformat()
            }

            mock_predictions.append(mock_prediction)

        # 保存模拟数据
        self.save_data(mock_predictions, "mock_predictions")

        return mock_predictions

    def _generate_prediction_summary(self, predictions: List[Dict]) -> Dict[str, Any]:
        """生成预测数据摘要"""
        if not predictions:
            return {'summary': '没有预测数据'}

        # 按灾害类型统计
        by_disaster_type = {}
        by_country = {}
        confidence_levels = []

        for pred in predictions:
            # 灾害类型统计
            disaster_type = pred.get('disaster_type', 'unknown')
            by_disaster_type[disaster_type] = by_disaster_type.get(disaster_type, 0) + 1

            # 国家统计
            country = pred.get('country', 'unknown')
            by_country[country] = by_country.get(country, 0) + 1

            # 置信度统计
            confidence = pred.get('confidence_level', 0)
            confidence_levels.append(confidence)

        # 计算平均置信度
        avg_confidence = sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0

        # 识别高风险预测
        high_risk_predictions = [
            p for p in predictions
            if p.get('severity_estimate') in ['high', 'extreme'] or p.get('confidence_level', 0) > 0.8
        ]

        return {
            'total_predictions': len(predictions),
            'by_disaster_type': by_disaster_type,
            'by_country': by_country,
            'average_confidence': round(avg_confidence, 2),
            'high_risk_predictions_count': len(high_risk_predictions),
            'high_risk_predictions': high_risk_predictions[:5],  # 显示前5个
            'time_range_covered': self._get_prediction_time_range(),
            'data_sources_used': list(set(p.get('data_source') for p in predictions))
        }

    def _save_categorized_predictions(self, predictions: List[Dict]) -> None:
        """保存分类的预测数据"""
        if not predictions:
            return

        # 按灾害类型保存
        by_type = {}
        for pred in predictions:
            disaster_type = pred.get('disaster_type', 'unknown')
            if disaster_type not in by_type:
                by_type[disaster_type] = []
            by_type[disaster_type].append(pred)

        for disaster_type, preds in by_type.items():
            self.save_data(preds, f"predictions_by_type_{disaster_type}")

        # 按国家保存
        by_country = {}
        for pred in predictions:
            country = pred.get('country', 'Unknown')
            if country not in by_country:
                by_country[country] = []
            by_country[country].append(pred)

        for country, preds in by_country.items():
            country_safe = str(country).replace(' ', '_').replace('/', '_')
            self.save_data(preds, f"predictions_by_country_{country_safe}")

        # 按紧急程度保存
        urgent_predictions = [
            p for p in predictions
            if p.get('severity_estimate') in ['high', 'extreme'] or p.get('confidence_level', 0) > 0.8
        ]
        if urgent_predictions:
            self.save_data(urgent_predictions, "urgent_predictions")

    def _get_prediction_time_range(self) -> Dict[str, str]:
        """获取预测的时间范围"""
        now = datetime.now()
        return {
            'prediction_start_date': now.isoformat(),
            'prediction_end_date': (now + timedelta(days=30)).isoformat(),
            'forecast_horizon': '30_days'
        }

    def _calculate_prediction_quality(self, prediction_data: Dict) -> float:
        """计算预测数据质量评分"""
        quality_score = 0.0

        # 检查必要字段
        required_fields = ['source_name', 'data_type', 'collection_timestamp']
        available_fields = sum(1 for field in required_fields if prediction_data.get(field))
        quality_score += (available_fields / len(required_fields)) * 0.4

        # 检查数据内容
        if prediction_data.get('raw_data'):
            quality_score += 0.3

        if prediction_data.get('prediction_confidence', 0) > 0:
            quality_score += 0.2

        if prediction_data.get('geographic_coverage'):
            quality_score += 0.1

        return min(quality_score, 1.0)

    # 辅助方法
    def _extract_geographic_coverage(self, data: Any) -> List[str]:
        """提取地理覆盖范围"""
        # 实现根据具体API返回格式
        return []

    def _extract_population_exposure(self, data: Any) -> Dict[str, Any]:
        """提取人口暴露信息"""
        return {}

    def _calculate_average_risk(self, risk_scores: List[Dict]) -> float:
        """计算平均风险得分"""
        if not risk_scores:
            return 0.0

        scores = []
        for score_data in risk_scores:
            if isinstance(score_data, dict):
                score = score_data.get('risk_score') or score_data.get('score', 0)
                scores.append(float(score))

        return sum(scores) / len(scores) if scores else 0.0

    def _identify_high_risk_areas(self, risk_scores: List[Dict]) -> List[Dict]:
        """识别高风险区域"""
        high_risk = []
        for score_data in risk_scores:
            if isinstance(score_data, dict):
                risk_level = score_data.get('risk_level') or score_data.get('level')
                if risk_level in ['high', 'very_high', 'extreme']:
                    high_risk.append(score_data)
        return high_risk

    def _extract_forecast_period(self, data: Any) -> str:
        """提取预报周期"""
        return "seasonal"  # 默认返回

    def _extract_seasonal_risks(self, data: Any) -> List[str]:
        """提取季节性风险"""
        return []

    def _extract_inform_categories(self, data: Any) -> List[str]:
        """提取INFORM风险类别"""
        return []

    def _extract_country_rankings(self, data: Any) -> List[Dict]:
        """提取国家风险排名"""
        return []

    def _get_required_fields(self) -> List[str]:
        """获取预测数据必需字段"""
        return ['prediction_id', 'data_source', 'disaster_type']


if __name__ == "__main__":
    # 使用示例
    from config import load_config

    # 加载配置
    config = load_config()

    # 创建收集器
    collector = PredictedDisastersCollector(config)

    # 运行收集
    print("开始预测灾害数据收集...")
    result = collector.run_collection()

    print(f"收集完成!")
    print(f"预测记录数: {result['collection_metadata']['total_predictions']}")
    print(f"成功的数据源: {result['collection_metadata']['successful_sources']}")
    print(f"失败的数据源: {result['collection_metadata']['failed_sources']}")

    # 预测摘要
    summary = result.get('prediction_summary', {})
    if summary.get('high_risk_predictions_count', 0) > 0:
        print(f"高风险预测: {summary['high_risk_predictions_count']} 个")

    # 数据质量报告
    validation = collector.validate_collected_data(result.get('data', []))
    print(f"数据质量评分: {validation['quality_score']:.2f}")