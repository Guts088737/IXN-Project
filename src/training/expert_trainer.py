"""
专家训练器 - 数据驱动的专家模型训练系统
完全基于历史数据自动训练和优化专家模型
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict


class ExpertTrainer:
    """混合专家模型训练器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.training_history = {}
        self.validation_metrics = {}
        
    def train_all_experts(self, historical_data: pd.DataFrame,
                         expert_specializations: Dict[str, Dict[str, Any]],
                         country_risk_factors: Dict[int, Dict[str, float]],
                         spatial_features: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """训练所有专家并返回训练结果"""
        self.logger.info("开始训练所有专家模型...")
        
        training_results = {
            'trained_experts': {},
            'validation_scores': {},
            'training_metadata': {},
            'performance_comparison': {}
        }
        
        # 数据分割
        train_data, validation_data = self._temporal_split(historical_data, train_ratio=0.8)
        
        for expert_name, specialization in expert_specializations.items():
            self.logger.info(f"训练专家: {expert_name}")
            
            # 过滤该专家的数据
            expert_disaster_types = specialization['disaster_types']
            expert_train_data = train_data[train_data['disaster_type_id'].isin(expert_disaster_types)]
            expert_val_data = validation_data[validation_data['disaster_type_id'].isin(expert_disaster_types)]
            
            if expert_train_data.empty:
                continue
            
            # 训练专家模型
            expert_model = self._train_expert_with_validation(
                expert_name, expert_train_data, expert_val_data, 
                specialization, country_risk_factors, spatial_features
            )
            
            training_results['trained_experts'][expert_name] = expert_model
            
            # 验证专家性能
            validation_score = self._validate_expert_performance(
                expert_name, expert_model, expert_val_data, spatial_features
            )
            training_results['validation_scores'][expert_name] = validation_score
        
        # 计算整体训练统计
        training_results['training_metadata'] = {
            'total_experts_trained': len(training_results['trained_experts']),
            'training_data_size': len(train_data),
            'validation_data_size': len(validation_data),
            'average_validation_score': np.mean(list(training_results['validation_scores'].values())),
            'best_performing_expert': max(training_results['validation_scores'].items(), key=lambda x: x[1])[0],
            'data_split_ratio': len(train_data) / len(historical_data)
        }
        
        # 专家性能对比
        training_results['performance_comparison'] = self._compare_expert_performance(
            training_results['validation_scores']
        )
        
        self.logger.info(f"训练完成，平均验证分数: {training_results['training_metadata']['average_validation_score']:.3f}")
        return training_results
    
    def _temporal_split(self, historical_data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """基于时间的数据分割"""
        sorted_data = historical_data.sort_values('year')
        split_index = int(len(sorted_data) * train_ratio)
        
        train_data = sorted_data.iloc[:split_index]
        validation_data = sorted_data.iloc[split_index:]
        
        self.logger.info(f"数据分割: 训练集{len(train_data)}条, 验证集{len(validation_data)}条")
        return train_data, validation_data
    
    def _train_expert_with_validation(self, expert_name: str, train_data: pd.DataFrame, 
                                    val_data: pd.DataFrame, specialization: Dict[str, Any],
                                    country_risk_factors: Dict[int, Dict[str, float]],
                                    spatial_features: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """带验证的专家模型训练"""
        expert_model = {
            'expert_name': expert_name,
            'specialization': specialization,
            'country_predictors': {},
            'disaster_type_models': {},
            'feature_importance': {},
            'training_stats': {}
        }
        
        # 1. 为每个国家学习预测器
        for country_id in train_data['country_id'].unique():
            if pd.isna(country_id):
                continue
                
            country_train = train_data[train_data['country_id'] == country_id]
            if len(country_train) < 3:
                continue
            
            country_predictor = self._build_country_predictor(
                int(country_id), country_train, country_risk_factors, spatial_features
            )
            expert_model['country_predictors'][int(country_id)] = country_predictor
        
        # 2. 为每种灾害类型学习模型
        for disaster_type in specialization['disaster_types']:
            type_train_data = train_data[train_data['disaster_type_id'] == disaster_type]
            if len(type_train_data) < 5:
                continue
            
            disaster_model = self._build_disaster_type_model(disaster_type, type_train_data)
            expert_model['disaster_type_models'][disaster_type] = disaster_model
        
        # 3. 计算特征重要性
        expert_model['feature_importance'] = self._calculate_feature_importance(train_data, specialization)
        
        # 4. 训练统计
        expert_model['training_stats'] = {
            'training_samples': len(train_data),
            'countries_covered': len(expert_model['country_predictors']),
            'disaster_types_covered': len(expert_model['disaster_type_models']),
            'data_quality_score': self._assess_training_data_quality(train_data)
        }
        
        return expert_model
    
    def _build_country_predictor(self, country_id: int, country_data: pd.DataFrame,
                               country_risk_factors: Dict[int, Dict[str, float]],
                               spatial_features: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """为单个国家构建预测器"""
        predictor = {
            'country_id': country_id,
            'historical_patterns': {},
            'risk_profile': {},
            'spatial_context': {}
        }
        
        # 历史模式学习
        predictor['historical_patterns'] = {
            'annual_disaster_count': len(country_data) / len(country_data['year'].unique()),
            'seasonal_distribution': country_data['month'].value_counts(normalize=True).to_dict(),
            'disaster_type_distribution': country_data['disaster_type_id'].value_counts(normalize=True).to_dict(),
            'impact_distribution': {
                'median': country_data['people_affected'].median(),
                'q75': country_data['people_affected'].quantile(0.75),
                'q25': country_data['people_affected'].quantile(0.25),
                'variance': country_data['people_affected'].var()
            }
        }
        
        # 风险档案
        if country_id in country_risk_factors:
            predictor['risk_profile'] = country_risk_factors[country_id]
        
        # 空间上下文
        if country_id in spatial_features:
            predictor['spatial_context'] = spatial_features[country_id]
        
        return predictor
    
    def _build_disaster_type_model(self, disaster_type: int, type_data: pd.DataFrame) -> Dict[str, Any]:
        """为单种灾害类型构建模型"""
        model = {
            'disaster_type_id': disaster_type,
            'occurrence_patterns': {},
            'impact_characteristics': {},
            'geographic_preferences': {},
            'temporal_preferences': {}
        }
        
        # 发生模式
        model['occurrence_patterns'] = {
            'global_frequency': len(type_data) / len(type_data['year'].unique()),
            'country_frequency_distribution': type_data['country_id'].value_counts(normalize=True).to_dict(),
            'regional_preference': type_data['region_id'].value_counts(normalize=True).to_dict() if 'region_id' in type_data.columns else {}
        }
        
        # 影响特征
        if 'people_affected' in type_data.columns:
            impacts = type_data['people_affected'].dropna()
            model['impact_characteristics'] = {
                'typical_scale': impacts.median(),
                'impact_variability': impacts.std() / max(impacts.mean(), 1),
                'extreme_threshold': impacts.quantile(0.9),
                'minimum_threshold': impacts.quantile(0.1)
            }
        
        # 地理偏好
        if 'latitude' in type_data.columns and 'longitude' in type_data.columns:
            model['geographic_preferences'] = {
                'preferred_latitude_range': (type_data['latitude'].quantile(0.25), type_data['latitude'].quantile(0.75)),
                'latitude_variance': type_data['latitude'].var(),
                'longitude_variance': type_data['longitude'].var(),
                'geographic_concentration': self._calculate_geographic_concentration(type_data)
            }
        
        # 时间偏好
        if 'month' in type_data.columns:
            monthly_dist = type_data['month'].value_counts(normalize=True)
            model['temporal_preferences'] = {
                'seasonal_pattern': monthly_dist.to_dict(),
                'peak_season': monthly_dist.idxmax(),
                'seasonality_strength': monthly_dist.var() * 12,
                'active_months': monthly_dist[monthly_dist > monthly_dist.mean()].index.tolist()
            }
        
        return model
    
    def _calculate_geographic_concentration(self, data: pd.DataFrame) -> float:
        """计算地理集中度"""
        if 'latitude' not in data.columns or 'longitude' not in data.columns:
            return 0.5
        
        lat_range = data['latitude'].max() - data['latitude'].min()
        lng_range = data['longitude'].max() - data['longitude'].min()
        
        # 地理跨度越小，集中度越高
        geographic_span = np.sqrt(lat_range**2 + lng_range**2)
        concentration = 1.0 / (1.0 + geographic_span / 100.0)  # 100度作为参考
        
        return concentration
    
    def _calculate_feature_importance(self, train_data: pd.DataFrame, 
                                    specialization: Dict[str, Any]) -> Dict[str, float]:
        """计算特征重要性"""
        feature_importance = {}
        
        # 基于专家特化模式计算特征重要性
        dominant_features = specialization.get('dominant_features', {})
        
        for feature, significance in dominant_features.items():
            # 标准化重要性分数
            normalized_importance = min(significance / 5.0, 1.0)
            feature_importance[feature] = normalized_importance
        
        # 添加自动发现的重要特征
        if 'month' in train_data.columns:
            seasonal_variance = train_data['month'].value_counts(normalize=True).var()
            feature_importance['seasonal_pattern'] = seasonal_variance * 12
        
        if 'people_affected' in train_data.columns:
            impact_predictability = 1.0 / (1.0 + train_data['people_affected'].std() / max(train_data['people_affected'].mean(), 1))
            feature_importance['impact_scale'] = impact_predictability
        
        return feature_importance
    
    def _assess_training_data_quality(self, train_data: pd.DataFrame) -> float:
        """评估训练数据质量"""
        quality_factors = []
        
        # 数据完整性
        completeness = 1.0 - train_data.isnull().sum().sum() / (len(train_data) * len(train_data.columns))
        quality_factors.append(completeness)
        
        # 时间跨度
        time_span = len(train_data['year'].unique())
        temporal_quality = min(time_span / 10.0, 1.0)
        quality_factors.append(temporal_quality)
        
        # 地理多样性
        geographic_diversity = len(train_data['country_id'].unique())
        geographic_quality = min(geographic_diversity / 20.0, 1.0)
        quality_factors.append(geographic_quality)
        
        # 样本数量
        sample_adequacy = min(len(train_data) / 100.0, 1.0)
        quality_factors.append(sample_adequacy)
        
        return np.mean(quality_factors)
    
    def _validate_expert_performance(self, expert_name: str, expert_model: Dict[str, Any],
                                   validation_data: pd.DataFrame,
                                   spatial_features: Dict[int, Dict[str, float]]) -> float:
        """验证专家模型性能"""
        if validation_data.empty:
            return 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        for _, event in validation_data.iterrows():
            country_id = event['country_id']
            actual_disaster_type = event['disaster_type_id']
            actual_impact = event['people_affected']
            event_month = event['month']
            
            if pd.isna(country_id) or pd.isna(actual_disaster_type):
                continue
            
            # 使用专家模型进行预测
            prediction = self._expert_predict_single(
                expert_model, int(country_id), event_month, spatial_features
            )
            
            # 检查预测准确性
            if self._is_prediction_accurate(prediction, actual_disaster_type, actual_impact):
                correct_predictions += 1
            
            total_predictions += 1
        
        accuracy = correct_predictions / max(total_predictions, 1)
        self.validation_metrics[expert_name] = {
            'accuracy': accuracy,
            'validation_samples': total_predictions,
            'correct_predictions': correct_predictions
        }
        
        return accuracy
    
    def _expert_predict_single(self, expert_model: Dict[str, Any], country_id: int, 
                             month: int, spatial_features: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """使用专家模型进行单次预测"""
        prediction = {
            'predicted_disaster_types': [],
            'predicted_impacts': {},
            'confidence': 0.0
        }
        
        # 国家预测器
        if country_id in expert_model['country_predictors']:
            country_predictor = expert_model['country_predictors'][country_id]
            
            # 基于历史模式预测
            seasonal_dist = country_predictor['historical_patterns'].get('seasonal_distribution', {})
            disaster_type_dist = country_predictor['historical_patterns'].get('disaster_type_distribution', {})
            
            # 月份权重
            month_weight = seasonal_dist.get(month, 1.0/12)
            
            # 预测最可能的灾害类型
            for disaster_type, base_prob in disaster_type_dist.items():
                adjusted_prob = base_prob * month_weight * 12  # 标准化
                
                if adjusted_prob > 0.1:  # 阈值过滤
                    prediction['predicted_disaster_types'].append(disaster_type)
                    
                    # 影响预测
                    impact_dist = country_predictor['historical_patterns']['impact_distribution']
                    predicted_impact = impact_dist.get('median', 1000)
                    prediction['predicted_impacts'][disaster_type] = predicted_impact
        
        # 空间调制
        if country_id in spatial_features:
            spatial_risk = spatial_features[country_id].get('avg_hotspot_risk', 0.5)
            prediction['confidence'] = spatial_risk
        else:
            prediction['confidence'] = 0.3
        
        return prediction
    
    def _is_prediction_accurate(self, prediction: Dict[str, Any], 
                              actual_disaster_type: int, actual_impact: float) -> bool:
        """判断预测是否准确"""
        # 灾害类型预测准确性
        predicted_types = prediction.get('predicted_disaster_types', [])
        type_correct = actual_disaster_type in predicted_types
        
        # 影响规模预测准确性
        predicted_impacts = prediction.get('predicted_impacts', {})
        impact_correct = True
        
        if actual_disaster_type in predicted_impacts:
            predicted_impact = predicted_impacts[actual_disaster_type]
            if predicted_impact > 0 and actual_impact > 0:
                relative_error = abs(predicted_impact - actual_impact) / actual_impact
                impact_correct = relative_error < 1.0  # 100%误差内认为正确
        
        return type_correct and impact_correct
    
    def _compare_expert_performance(self, validation_scores: Dict[str, float]) -> Dict[str, Any]:
        """比较专家性能"""
        if not validation_scores:
            return {}
        
        scores = list(validation_scores.values())
        performance_comparison = {
            'best_expert': max(validation_scores.items(), key=lambda x: x[1]),
            'worst_expert': min(validation_scores.items(), key=lambda x: x[1]),
            'performance_variance': np.var(scores),
            'performance_range': max(scores) - min(scores),
            'experts_above_threshold': sum(1 for score in scores if score > 0.6),
            'overall_system_performance': np.mean(scores)
        }
        
        # 性能分级
        excellent_experts = [name for name, score in validation_scores.items() if score > 0.8]
        good_experts = [name for name, score in validation_scores.items() if 0.6 <= score <= 0.8]
        poor_experts = [name for name, score in validation_scores.items() if score < 0.6]
        
        performance_comparison['performance_tiers'] = {
            'excellent': excellent_experts,
            'good': good_experts,
            'needs_improvement': poor_experts
        }
        
        return performance_comparison
    
    def optimize_expert_configuration(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """基于训练结果优化专家配置"""
        self.logger.info("优化专家配置...")
        
        optimization_suggestions = {
            'weight_adjustments': {},
            'group_reorganization': [],
            'feature_recommendations': {},
            'training_improvements': []
        }
        
        validation_scores = training_results['validation_scores']
        performance_comparison = training_results['performance_comparison']
        
        # 1. 权重调整建议
        for expert_name, score in validation_scores.items():
            if score > 0.8:
                optimization_suggestions['weight_adjustments'][expert_name] = 'increase'
            elif score < 0.4:
                optimization_suggestions['weight_adjustments'][expert_name] = 'decrease'
            else:
                optimization_suggestions['weight_adjustments'][expert_name] = 'maintain'
        
        # 2. 分组重组建议
        poor_experts = performance_comparison['performance_tiers']['needs_improvement']
        if len(poor_experts) > 1:
            optimization_suggestions['group_reorganization'].append(
                f"考虑合并表现较差的专家: {', '.join(poor_experts)}"
            )
        
        # 3. 特征改进建议
        for expert_name, expert_model in training_results['trained_experts'].items():
            feature_importance = expert_model.get('feature_importance', {})
            low_importance_features = [f for f, imp in feature_importance.items() if imp < 0.3]
            
            if low_importance_features:
                optimization_suggestions['feature_recommendations'][expert_name] = {
                    'remove_features': low_importance_features,
                    'suggested_action': '寻找替代特征或增加更多训练数据'
                }
        
        # 4. 训练改进建议
        avg_score = performance_comparison['overall_system_performance']
        if avg_score < 0.6:
            optimization_suggestions['training_improvements'].extend([
                '增加更多历史训练数据',
                '检查数据质量和完整性',
                '考虑调整专家分组策略'
            ])
        elif avg_score > 0.8:
            optimization_suggestions['training_improvements'].append(
                '模型性能良好，可以部署到生产环境'
            )
        
        return optimization_suggestions
    
    def save_training_checkpoint(self, training_results: Dict[str, Any], checkpoint_path: str) -> None:
        """保存训练检查点"""
        import json
        
        # 准备可序列化的数据
        checkpoint_data = {
            'training_metadata': training_results['training_metadata'],
            'validation_scores': training_results['validation_scores'],
            'performance_comparison': training_results['performance_comparison'],
            'expert_count': len(training_results['trained_experts']),
            'training_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # 保存检查点
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"训练检查点已保存: {checkpoint_path}")
    
    def load_training_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """加载训练检查点"""
        import json
        
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            self.logger.info(f"成功加载训练检查点: {checkpoint_path}")
            return checkpoint_data
        
        except FileNotFoundError:
            self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return None
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return None