"""
特征验证器 - 验证反推特征的有效性
数据驱动的特征质量评估和验证
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict


class FeatureValidator:
    """反推特征有效性验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_reversed_features(self, historical_data: pd.DataFrame, 
                                 reversed_features: Dict[str, Dict[int, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """验证反推特征的预测有效性"""
        self.logger.info("验证反推特征的有效性...")
        
        validation_results = {}
        
        for feature_type, country_features in reversed_features.items():
            if not country_features:
                continue
                
            # 为该类特征计算验证指标
            validation_metrics = {
                'coverage': self._calculate_feature_coverage(country_features, historical_data),
                'discriminative_power': self._calculate_discriminative_power(feature_type, country_features, historical_data),
                'consistency': self._calculate_feature_consistency(country_features),
                'predictive_value': self._calculate_predictive_value(feature_type, country_features, historical_data)
            }
            
            # 综合有效性评分
            overall_score = np.mean(list(validation_metrics.values()))
            validation_metrics['overall_effectiveness'] = overall_score
            
            validation_results[feature_type] = validation_metrics
            
            self.logger.info(f"{feature_type}特征有效性: {overall_score:.3f}")
        
        return validation_results
    
    def _calculate_feature_coverage(self, country_features: Dict[int, Dict[str, float]], 
                                  historical_data: pd.DataFrame) -> float:
        """计算特征覆盖率"""
        total_countries = len(historical_data['country_id'].unique())
        covered_countries = len(country_features)
        return covered_countries / max(total_countries, 1)
    
    def _calculate_discriminative_power(self, feature_type: str, 
                                      country_features: Dict[int, Dict[str, float]], 
                                      historical_data: pd.DataFrame) -> float:
        """计算特征的区分能力"""
        if not country_features:
            return 0.0
        
        # 获取第一个特征名作为代表
        sample_features = list(list(country_features.values())[0].keys())
        if not sample_features:
            return 0.0
        
        discriminative_scores = []
        
        for feature_name in sample_features:
            # 计算该特征在不同国家间的变异程度
            feature_values = []
            for country_id, features in country_features.items():
                if feature_name in features:
                    feature_values.append(features[feature_name])
            
            if len(feature_values) > 1:
                # 变异系数作为区分能力指标
                feature_mean = np.mean(feature_values)
                feature_std = np.std(feature_values)
                cv = feature_std / max(feature_mean, 1e-8)
                discriminative_scores.append(min(cv, 2.0))  # 限制最大值
        
        return np.mean(discriminative_scores) if discriminative_scores else 0.0
    
    def _calculate_feature_consistency(self, country_features: Dict[int, Dict[str, float]]) -> float:
        """计算特征的一致性（避免异常值）"""
        if not country_features:
            return 0.0
        
        sample_features = list(list(country_features.values())[0].keys())
        consistency_scores = []
        
        for feature_name in sample_features:
            feature_values = []
            for features in country_features.values():
                if feature_name in features:
                    feature_values.append(features[feature_name])
            
            if len(feature_values) > 2:
                # 检查异常值比例
                q25 = np.percentile(feature_values, 25)
                q75 = np.percentile(feature_values, 75)
                iqr = q75 - q25
                
                # 异常值定义：超出 Q75 + 1.5*IQR 或低于 Q25 - 1.5*IQR
                outlier_threshold_high = q75 + 1.5 * iqr
                outlier_threshold_low = q25 - 1.5 * iqr
                
                outliers = [v for v in feature_values 
                          if v > outlier_threshold_high or v < outlier_threshold_low]
                outlier_ratio = len(outliers) / len(feature_values)
                
                # 一致性 = 1 - 异常值比例
                consistency = 1.0 - outlier_ratio
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_predictive_value(self, feature_type: str, 
                                  country_features: Dict[int, Dict[str, float]], 
                                  historical_data: pd.DataFrame) -> float:
        """计算特征的预测价值"""
        # 简单的预测价值评估：特征与实际灾害发生的关联性
        
        if not country_features:
            return 0.0
        
        predictive_scores = []
        sample_features = list(list(country_features.values())[0].keys())
        
        for feature_name in sample_features:
            # 获取所有国家的该特征值
            feature_country_mapping = {}
            for country_id, features in country_features.items():
                if feature_name in features:
                    feature_country_mapping[country_id] = features[feature_name]
            
            if len(feature_country_mapping) < 3:
                continue
            
            # 将国家按特征值分为高、中、低三组
            feature_values = list(feature_country_mapping.values())
            q33 = np.percentile(feature_values, 33)
            q67 = np.percentile(feature_values, 67)
            
            high_risk_countries = [cid for cid, val in feature_country_mapping.items() if val > q67]
            medium_risk_countries = [cid for cid, val in feature_country_mapping.items() if q33 <= val <= q67]
            low_risk_countries = [cid for cid, val in feature_country_mapping.items() if val < q33]
            
            # 计算各组的实际灾害发生率
            high_group_rate = self._calculate_disaster_rate(high_risk_countries, historical_data)
            medium_group_rate = self._calculate_disaster_rate(medium_risk_countries, historical_data)
            low_group_rate = self._calculate_disaster_rate(low_risk_countries, historical_data)
            
            # 预测价值 = 高风险组与低风险组的灾害率差异
            if low_group_rate > 0:
                predictive_power = (high_group_rate - low_group_rate) / low_group_rate
                predictive_scores.append(max(predictive_power, 0.0))
        
        return np.mean(predictive_scores) if predictive_scores else 0.0
    
    def _calculate_disaster_rate(self, country_list: List[int], historical_data: pd.DataFrame) -> float:
        """计算国家组的灾害发生率"""
        if not country_list:
            return 0.0
        
        total_events = 0
        total_years = 0
        
        for country_id in country_list:
            country_data = historical_data[historical_data['country_id'] == country_id]
            if not country_data.empty:
                country_years = len(country_data['year'].unique())
                total_events += len(country_data)
                total_years += country_years
        
        return total_events / max(total_years, 1)
    
    def cross_validate_expert_features(self, historical_data: pd.DataFrame, 
                                     expert_groups: Dict[str, List[int]], 
                                     reversed_features: Dict[str, Dict[int, Dict[str, float]]]) -> Dict[str, float]:
        """交叉验证专家特征的有效性"""
        self.logger.info("交叉验证专家特征...")
        
        # 数据分割（80%训练，20%测试）
        train_data, test_data = self._temporal_split(historical_data, train_ratio=0.8)
        
        expert_effectiveness = {}
        
        for expert_name, disaster_types in expert_groups.items():
            # 该专家负责的灾害类型数据
            expert_train_data = train_data[train_data['disaster_type_id'].isin(disaster_types)]
            expert_test_data = test_data[test_data['disaster_type_id'].isin(disaster_types)]
            
            if expert_train_data.empty or expert_test_data.empty:
                expert_effectiveness[expert_name] = 0.0
                continue
            
            # 基于训练数据构建简单预测模型
            country_risk_profiles = self._build_country_risk_profiles(expert_train_data)
            
            # 在测试数据上评估预测效果
            prediction_accuracy = self._evaluate_predictions(expert_test_data, country_risk_profiles)
            
            expert_effectiveness[expert_name] = prediction_accuracy
            self.logger.info(f"{expert_name}专家预测准确率: {prediction_accuracy:.3f}")
        
        return expert_effectiveness
    
    def _temporal_split(self, historical_data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """基于时间的数据分割"""
        # 按年份排序
        sorted_data = historical_data.sort_values('year')
        
        # 找到分割点
        split_index = int(len(sorted_data) * train_ratio)
        
        train_data = sorted_data.iloc[:split_index]
        test_data = sorted_data.iloc[split_index:]
        
        return train_data, test_data
    
    def _build_country_risk_profiles(self, train_data: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """基于训练数据构建国家风险档案"""
        country_profiles = {}
        
        for country_id in train_data['country_id'].unique():
            if pd.isna(country_id):
                continue
                
            country_data = train_data[train_data['country_id'] == country_id]
            
            # 计算该国家的灾害风险指标
            disaster_rate = len(country_data) / max(len(country_data['year'].unique()), 1)
            avg_impact = country_data['people_affected'].mean()
            disaster_diversity = len(country_data['disaster_type_id'].unique())
            
            country_profiles[int(country_id)] = {
                'disaster_rate': disaster_rate,
                'avg_impact': avg_impact,
                'disaster_diversity': disaster_diversity
            }
        
        return country_profiles
    
    def _evaluate_predictions(self, test_data: pd.DataFrame, 
                            country_profiles: Dict[int, Dict[str, float]]) -> float:
        """评估预测准确率"""
        correct_predictions = 0
        total_predictions = 0
        
        for _, event in test_data.iterrows():
            country_id = event['country_id']
            actual_impact = event['people_affected']
            
            if pd.isna(country_id) or country_id not in country_profiles:
                continue
            
            # 使用国家风险档案预测
            profile = country_profiles[country_id]
            predicted_impact = profile['avg_impact']
            
            # 简单的准确性评估（相对误差<50%认为正确）
            if predicted_impact > 0 and actual_impact > 0:
                relative_error = abs(predicted_impact - actual_impact) / actual_impact
                if relative_error < 0.5:
                    correct_predictions += 1
            
            total_predictions += 1
        
        return correct_predictions / max(total_predictions, 1)
    
    def generate_feature_quality_report(self, historical_data: pd.DataFrame, 
                                      reversed_features: Dict[str, Dict[int, Dict[str, float]]]) -> Dict[str, Any]:
        """生成特征质量报告"""
        self.logger.info("生成特征质量报告...")
        
        # 验证所有反推特征
        validation_results = self.validate_reversed_features(historical_data, reversed_features)
        
        # 自动发现专家分组
        expert_groups = {}
        if 'expert_specializations' in reversed_features:
            for expert_name, spec in reversed_features['expert_specializations'].items():
                expert_groups[expert_name] = spec.get('disaster_types', [])
        
        # 交叉验证
        cv_results = {}
        if expert_groups:
            cv_results = self.cross_validate_expert_features(historical_data, expert_groups, reversed_features)
        
        # 综合质量评分
        quality_scores = {}
        for feature_type, metrics in validation_results.items():
            quality_scores[feature_type] = metrics.get('overall_effectiveness', 0.0)
        
        overall_quality = np.mean(list(quality_scores.values())) if quality_scores else 0.0
        
        return {
            'validation_results': validation_results,
            'cross_validation_results': cv_results,
            'feature_quality_scores': quality_scores,
            'overall_quality_score': overall_quality,
            'recommendations': self._generate_recommendations(validation_results, cv_results),
            'data_stats': {
                'total_events': len(historical_data),
                'total_countries': len(historical_data['country_id'].unique()),
                'date_range': f"{historical_data['year'].min()}-{historical_data['year'].max()}",
                'disaster_types_count': len(historical_data['disaster_type_id'].unique())
            }
        }
    
    def _generate_recommendations(self, validation_results: Dict, cv_results: Dict) -> List[str]:
        """基于验证结果生成改进建议"""
        recommendations = []
        
        # 基于验证结果的建议
        for feature_type, metrics in validation_results.items():
            effectiveness = metrics.get('overall_effectiveness', 0.0)
            coverage = metrics.get('coverage', 0.0)
            
            if effectiveness < 0.5:
                recommendations.append(f"{feature_type}特征效果较低({effectiveness:.2f})，建议增加更多相关历史数据")
            
            if coverage < 0.7:
                recommendations.append(f"{feature_type}特征覆盖率不足({coverage:.2f})，建议扩大数据收集范围")
        
        # 基于交叉验证的建议
        if cv_results:
            low_performing_experts = [name for name, score in cv_results.items() if score < 0.6]
            if low_performing_experts:
                recommendations.append(f"以下专家模型效果不佳，建议合并或重新分组: {', '.join(low_performing_experts)}")
        
        # 通用建议
        if not recommendations:
            recommendations.append("所有特征质量良好，可以开始专家模型训练")
        
        return recommendations
    
    def validate_expert_group_balance(self, expert_groups: Dict[str, List[int]], 
                                    historical_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """验证专家分组的数据平衡性"""
        self.logger.info("验证专家分组平衡性...")
        
        balance_analysis = {}
        
        total_events = len(historical_data)
        
        for expert_name, disaster_types in expert_groups.items():
            expert_data = historical_data[historical_data['disaster_type_id'].isin(disaster_types)]
            
            # 数据量分析
            data_share = len(expert_data) / total_events
            
            # 地理覆盖分析
            expert_countries = len(expert_data['country_id'].unique())
            total_countries = len(historical_data['country_id'].unique())
            geographic_coverage = expert_countries / total_countries
            
            # 时间覆盖分析
            expert_years = len(expert_data['year'].unique()) 
            total_years = len(historical_data['year'].unique())
            temporal_coverage = expert_years / total_years
            
            # 平衡性评分
            balance_score = min(data_share * 5, 1.0)  # 理想情况下每个专家负责20%数据
            
            balance_analysis[expert_name] = {
                'data_share': data_share,
                'geographic_coverage': geographic_coverage, 
                'temporal_coverage': temporal_coverage,
                'balance_score': balance_score,
                'sample_size': len(expert_data),
                'disaster_types_count': len(disaster_types)
            }
        
        # 检查整体平衡性
        data_shares = [metrics['data_share'] for metrics in balance_analysis.values()]
        overall_balance = 1.0 - np.std(data_shares)  # 标准差越小越平衡
        
        balance_analysis['overall_balance'] = {
            'balance_coefficient': overall_balance,
            'data_distribution_fairness': min(overall_balance * 2, 1.0),
            'recommendation': "平衡" if overall_balance > 0.8 else "需要重新平衡分组"
        }
        
        return balance_analysis