"""
灾害预测流水线 - 整合所有组件的主控制器
完全数据驱动的端到端灾害预测系统
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.preprocessing.feature_reversal_engineer import FeatureReversalEngineer
from src.models.spatial_analysis import DisasterSpatialAnalyzer
from src.models.mixed_expert_predictor import MixedExpertPredictor
from src.models.random_forest_expert_ensemble import RandomForestExpertEnsemble
from src.training.expert_trainer import ExpertTrainer
from src.training.feature_validation import FeatureValidator
from src.output.prediction_formatter import PredictionFormatter


class DisasterPredictionPipeline:
    """完整的数据驱动灾害预测流水线"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 初始化所有组件
        self.feature_engineer = FeatureReversalEngineer()
        self.spatial_analyzer = DisasterSpatialAnalyzer()
        self.expert_predictor = MixedExpertPredictor()
        self.rf_ensemble = RandomForestExpertEnsemble()
        self.expert_trainer = ExpertTrainer()
        self.feature_validator = FeatureValidator()
        self.formatter = PredictionFormatter()
        
        # 系统状态
        self.is_initialized = False
        self.training_completed = False
        
        # 缓存训练结果
        self.expert_specializations = {}
        self.country_risk_factors = {}
        self.spatial_features = {}
        self.validation_results = {}
    
    def initialize_system(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """初始化预测系统"""
        self.logger.info("初始化灾害预测系统...")
        
        initialization_results = {
            'data_analysis': {},
            'feature_engineering': {},
            'spatial_analysis': {},
            'system_status': 'initializing'
        }
        
        try:
            # 1. 数据质量分析
            data_quality = self._analyze_data_quality(historical_data)
            initialization_results['data_analysis'] = data_quality
            
            if data_quality['overall_quality'] < 0.3:
                raise ValueError(f"数据质量过低 ({data_quality['overall_quality']:.2f})，无法进行可靠预测")
            
            # 2. 特征工程
            self.logger.info("执行特征反推工程...")
            feature_results = self.feature_engineer.generate_data_driven_expert_features(historical_data)
            
            self.expert_specializations = feature_results['expert_specializations']
            self.country_risk_factors = feature_results['country_risk_factors']
            initialization_results['feature_engineering'] = {
                'experts_discovered': len(self.expert_specializations),
                'countries_analyzed': len(self.country_risk_factors),
                'feature_correlations_computed': len(feature_results['feature_correlations'])
            }
            
            # 3. 空间分析
            self.logger.info("执行空间模式分析...")
            self.spatial_features = self.spatial_analyzer.create_spatial_risk_features(historical_data)
            
            initialization_results['spatial_analysis'] = {
                'spatial_features_generated': len(self.spatial_features),
                'hotspots_identified': self._count_identified_hotspots(historical_data),
                'corridors_analyzed': len(self.spatial_analyzer.analyze_disaster_corridors(historical_data))
            }
            
            # 4. 特征验证
            self.logger.info("验证特征质量...")
            self.validation_results = self.feature_validator.generate_feature_quality_report(
                historical_data, feature_results
            )
            
            overall_quality = self.validation_results.get('overall_quality_score', 0.0)
            if overall_quality < 0.4:
                self.logger.warning(f"特征质量较低 ({overall_quality:.2f})，预测可能不够准确")
            
            self.is_initialized = True
            initialization_results['system_status'] = 'ready_for_training'
            
            self.logger.info("系统初始化完成")
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            initialization_results['system_status'] = 'initialization_failed'
            initialization_results['error'] = str(e)
        
        return initialization_results
    
    def train_prediction_system(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """训练预测系统"""
        if not self.is_initialized:
            raise ValueError("系统未初始化，请先调用initialize_system")
        
        self.logger.info("开始训练预测系统...")
        
        # 使用专家训练器训练所有专家
        training_results = self.expert_trainer.train_all_experts(
            historical_data, self.expert_specializations, 
            self.country_risk_factors, self.spatial_features
        )
        
        # 将训练好的模型加载到预测器
        self.expert_predictor.expert_models = training_results['trained_experts']
        
        # 训练随机森林集成模型
        self.logger.info("训练随机森林集成模型...")
        rf_training_results = self.rf_ensemble.train_random_forest_models(
            historical_data, self.country_risk_factors, self.spatial_features
        )
        training_results['random_forest_performance'] = rf_training_results
        
        # 计算融合权重
        self._calculate_dynamic_fusion_weights(training_results['validation_scores'])
        
        self.training_completed = True
        
        # 优化建议
        optimization_suggestions = self.expert_trainer.optimize_expert_configuration(training_results)
        training_results['optimization_suggestions'] = optimization_suggestions
        
        self.logger.info("预测系统训练完成")
        return training_results
    
    def predict_disasters(self, current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """执行灾害预测"""
        if not self.training_completed:
            raise ValueError("系统未训练完成，请先调用train_prediction_system")
        
        self.logger.info(f"预测国家{current_conditions.get('country_id')}的灾害...")
        
        # 1. 使用混合专家模型进行预测
        expert_prediction = self.expert_predictor.predict_disasters(current_conditions, self.spatial_features)
        
        # 2. 使用随机森林+专家集成预测（解决多样性问题）
        ensemble_prediction = self.rf_ensemble.predict_with_ensemble(
            current_conditions, expert_prediction, self.country_risk_factors, self.spatial_features
        )
        
        # 格式化预测结果
        formatted_results = self.formatter.format_prediction_results(
            ensemble_prediction, current_conditions
        )
        
        # 添加系统诊断信息
        formatted_results['system_diagnostics'] = {
            'experts_used': len(self.expert_predictor.expert_models),
            'random_forest_enabled': self.rf_ensemble.is_trained,
            'spatial_features_applied': current_conditions.get('country_id') in self.spatial_features,
            'fusion_weights_count': len(self.expert_predictor.fusion_weights),
            'diversity_enhanced': ensemble_prediction.get('diversity_enhanced', False),
            'prediction_method': 'expert_rf_ensemble',
            'prediction_pipeline_health': 'operational'
        }
        
        return formatted_results
    
    def batch_predict(self, prediction_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量预测"""
        self.logger.info(f"执行批量预测 ({len(prediction_requests)}个请求)...")
        
        batch_results = []
        
        for i, request in enumerate(prediction_requests):
            try:
                self.logger.info(f"处理预测请求 {i+1}/{len(prediction_requests)}")
                result = self.predict_disasters(request)
                result['batch_index'] = i
                result['request_id'] = request.get('request_id', f'batch_{i}')
                batch_results.append(result)
                
            except Exception as e:
                self.logger.error(f"预测请求{i}处理失败: {e}")
                error_result = {
                    'batch_index': i,
                    'request_id': request.get('request_id', f'batch_{i}'),
                    'error': str(e),
                    'status': 'failed'
                }
                batch_results.append(error_result)
        
        return batch_results
    
    def _analyze_data_quality(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """分析历史数据质量"""
        required_columns = ['country_id', 'disaster_type_id', 'year', 'month', 'people_affected']
        missing_columns = [col for col in required_columns if col not in historical_data.columns]
        
        # 计算各项质量指标
        completeness = 1.0 - historical_data.isnull().sum().sum() / (len(historical_data) * len(historical_data.columns))
        
        time_span = len(historical_data['year'].unique()) if 'year' in historical_data.columns else 0
        temporal_coverage = min(time_span / 10.0, 1.0)
        
        geographic_coverage = len(historical_data['country_id'].unique()) if 'country_id' in historical_data.columns else 0
        geographic_score = min(geographic_coverage / 50.0, 1.0)
        
        disaster_diversity = len(historical_data['disaster_type_id'].unique()) if 'disaster_type_id' in historical_data.columns else 0
        diversity_score = min(disaster_diversity / 20.0, 1.0)
        
        sample_size_score = min(len(historical_data) / 1000.0, 1.0)
        
        overall_quality = np.mean([completeness, temporal_coverage, geographic_score, diversity_score, sample_size_score])
        
        return {
            'overall_quality': overall_quality,
            'completeness': completeness,
            'temporal_coverage': temporal_coverage,
            'geographic_coverage': geographic_score,
            'disaster_diversity': diversity_score,
            'sample_adequacy': sample_size_score,
            'missing_columns': missing_columns,
            'total_records': len(historical_data),
            'quality_assessment': 'excellent' if overall_quality > 0.8 else 
                                'good' if overall_quality > 0.6 else
                                'fair' if overall_quality > 0.4 else 'poor'
        }
    
    def _count_identified_hotspots(self, historical_data: pd.DataFrame) -> int:
        """计算识别的热点数量"""
        hotspots = self.spatial_analyzer.identify_disaster_hotspots(historical_data)
        return sum(len(type_hotspots) for type_hotspots in hotspots.values())
    
    def _calculate_dynamic_fusion_weights(self, validation_scores: Dict[str, float]) -> None:
        """计算动态融合权重"""
        if not validation_scores:
            return
        
        # 基于验证性能计算权重
        total_score = sum(validation_scores.values())
        if total_score > 0:
            normalized_weights = {name: score / total_score for name, score in validation_scores.items()}
        else:
            # 均匀权重作为后备
            equal_weight = 1.0 / len(validation_scores)
            normalized_weights = {name: equal_weight for name in validation_scores.keys()}
        
        self.expert_predictor.fusion_weights = normalized_weights
        self.expert_predictor.is_trained = True
        
        self.logger.info(f"设置动态融合权重: {normalized_weights}")
    
    def run_full_pipeline(self, historical_data: pd.DataFrame, 
                         prediction_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行完整的预测流水线"""
        self.logger.info("开始运行完整预测流水线...")
        
        pipeline_results = {
            'initialization': {},
            'training': {},
            'predictions': [],
            'system_performance': {},
            'pipeline_status': 'running'
        }
        
        try:
            # 1. 系统初始化
            self.logger.info("步骤1: 系统初始化")
            pipeline_results['initialization'] = self.initialize_system(historical_data)
            
            if not self.is_initialized:
                raise ValueError("系统初始化失败")
            
            # 2. 模型训练
            self.logger.info("步骤2: 模型训练")
            pipeline_results['training'] = self.train_prediction_system(historical_data)
            
            # 3. 执行预测
            self.logger.info("步骤3: 执行预测")
            pipeline_results['predictions'] = self.batch_predict(prediction_requests)
            
            # 4. 系统性能评估
            pipeline_results['system_performance'] = self._evaluate_pipeline_performance(
                pipeline_results['training'], pipeline_results['predictions']
            )
            
            pipeline_results['pipeline_status'] = 'completed_successfully'
            self.logger.info("预测流水线执行完成")
            
        except Exception as e:
            self.logger.error(f"流水线执行失败: {e}")
            pipeline_results['pipeline_status'] = 'failed'
            pipeline_results['error'] = str(e)
        
        return pipeline_results
    
    def _evaluate_pipeline_performance(self, training_results: Dict[str, Any], 
                                     prediction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估整个流水线性能"""
        performance_metrics = {
            'training_performance': {},
            'prediction_performance': {},
            'system_reliability': {}
        }
        
        # 训练性能
        training_metadata = training_results.get('training_metadata', {})
        performance_metrics['training_performance'] = {
            'experts_trained': training_metadata.get('total_experts_trained', 0),
            'average_validation_score': training_metadata.get('average_validation_score', 0.0),
            'best_expert_performance': training_metadata.get('average_validation_score', 0.0),
            'training_data_utilization': training_metadata.get('data_split_ratio', 0.0)
        }
        
        # 预测性能
        successful_predictions = [r for r in prediction_results if 'error' not in r]
        failed_predictions = [r for r in prediction_results if 'error' in r]
        
        performance_metrics['prediction_performance'] = {
            'success_rate': len(successful_predictions) / len(prediction_results),
            'total_predictions': len(prediction_results),
            'successful_predictions': len(successful_predictions),
            'failed_predictions': len(failed_predictions),
            'average_confidence': self._calculate_average_prediction_confidence(successful_predictions)
        }
        
        # 系统可靠性
        performance_metrics['system_reliability'] = {
            'initialization_success': self.is_initialized,
            'training_success': self.training_completed,
            'feature_validation_passed': self.validation_results.get('overall_quality_score', 0) > 0.4,
            'expert_models_operational': len(self.expert_predictor.expert_models) > 0,
            'spatial_features_available': len(self.spatial_features) > 0
        }
        
        return performance_metrics
    
    def _calculate_average_prediction_confidence(self, successful_predictions: List[Dict[str, Any]]) -> float:
        """计算预测的平均置信度"""
        if not successful_predictions:
            return 0.0
        
        confidences = []
        for prediction in successful_predictions:
            confidence_analysis = prediction.get('confidence_analysis', {})
            confidence = confidence_analysis.get('overall_confidence_score', 0.0)
            confidences.append(confidence)
        
        return np.mean(confidences) if confidences else 0.0
    
    def generate_system_report(self) -> str:
        """生成系统状态报告"""
        report_lines = []
        
        report_lines.append("=== 灾害预测系统状态报告 ===")
        report_lines.append("")
        
        # 系统状态
        report_lines.append("【系统状态】")
        report_lines.append(f"  初始化状态: {'完成' if self.is_initialized else '未完成'}")
        report_lines.append(f"  训练状态: {'完成' if self.training_completed else '未完成'}")
        report_lines.append(f"  专家模型数量: {len(self.expert_predictor.expert_models)}")
        report_lines.append(f"  空间特征覆盖: {len(self.spatial_features)}个国家")
        report_lines.append("")
        
        # 特征质量
        if self.validation_results:
            overall_quality = self.validation_results.get('overall_quality_score', 0.0)
            report_lines.append("【特征质量】")
            report_lines.append(f"  整体质量评分: {overall_quality:.3f}")
            
            recommendations = self.validation_results.get('recommendations', [])
            if recommendations:
                report_lines.append("  系统建议:")
                for rec in recommendations[:3]:  # 最多显示3条建议
                    report_lines.append(f"    - {rec}")
            report_lines.append("")
        
        # 专家配置
        if self.expert_specializations:
            report_lines.append("【专家配置】")
            for expert_name, spec in self.expert_specializations.items():
                disaster_count = len(spec.get('disaster_types', []))
                report_lines.append(f"  {expert_name}: 负责{disaster_count}种灾害类型")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def export_predictions(self, prediction_results: Dict[str, Any], 
                         output_dir: str, format_type: str = 'json') -> str:
        """导出预测结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type.lower() == 'json':
            output_path = os.path.join(output_dir, f'disaster_predictions_{timestamp}.json')
            self.formatter.export_to_json(prediction_results, output_path)
        elif format_type.lower() == 'csv':
            output_path = os.path.join(output_dir, f'disaster_predictions_{timestamp}.csv')
            self.formatter.export_to_csv(prediction_results, output_path)
        else:
            raise ValueError(f"不支持的输出格式: {format_type}")
        
        return output_path
    
    def create_prediction_alerts(self, prediction_results: Dict[str, Any], 
                               alert_threshold: float = 0.7) -> Dict[str, Any]:
        """创建预测告警"""
        return self.formatter.create_alert_system_output(prediction_results, alert_threshold)
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """获取系统诊断信息"""
        diagnostics = {
            'system_health': {
                'initialization_status': self.is_initialized,
                'training_status': self.training_completed,
                'components_loaded': {
                    'feature_engineer': bool(self.feature_engineer),
                    'spatial_analyzer': bool(self.spatial_analyzer),
                    'expert_predictor': bool(self.expert_predictor),
                    'expert_trainer': bool(self.expert_trainer),
                    'feature_validator': bool(self.feature_validator),
                    'formatter': bool(self.formatter)
                }
            },
            'data_status': {
                'expert_specializations_count': len(self.expert_specializations),
                'country_risk_factors_count': len(self.country_risk_factors),
                'spatial_features_count': len(self.spatial_features),
                'trained_experts_count': len(self.expert_predictor.expert_models)
            },
            'model_status': {
                'expert_models_trained': len(self.expert_predictor.expert_models),
                'fusion_weights_configured': len(self.expert_predictor.fusion_weights),
                'model_training_completed': self.expert_predictor.is_trained
            }
        }
        
        # 整体健康评分
        health_checks = [
            self.is_initialized,
            self.training_completed,
            len(self.expert_predictor.expert_models) > 0,
            len(self.spatial_features) > 0,
            len(self.country_risk_factors) > 0
        ]
        
        health_score = sum(health_checks) / len(health_checks)
        
        if health_score >= 1.0:
            system_health = 'excellent'
        elif health_score >= 0.8:
            system_health = 'good'
        elif health_score >= 0.6:
            system_health = 'fair'
        else:
            system_health = 'poor'
        
        diagnostics['overall_health'] = {
            'health_score': health_score,
            'health_level': system_health,
            'ready_for_prediction': health_score >= 0.8
        }
        
        return diagnostics