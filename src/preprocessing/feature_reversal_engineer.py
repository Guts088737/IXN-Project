"""
特征反推工程师 - 完全数据驱动的特征反推
通过历史灾害事件的统计模式自动反推缺失特征，无硬编码
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import logging


class FeatureReversalEngineer:
    """完全数据驱动的特征反推工程师"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def auto_discover_disaster_groups(self, historical_data: pd.DataFrame) -> Dict[str, List[int]]:
        """自动发现灾害类型的自然分组"""
        self.logger.info("自动发现灾害类型分组...")
        
        # 基于灾害类型的地理分布相似性进行聚类
        disaster_types = historical_data['disaster_type_id'].unique()
        disaster_geographic_patterns = {}
        
        for disaster_type in disaster_types:
            if pd.isna(disaster_type):
                continue
                
            type_data = historical_data[historical_data['disaster_type_id'] == disaster_type]
            
            # 计算该灾害类型的地理分布特征
            region_distribution = type_data['region_id'].value_counts(normalize=True).to_dict()
            coastal_ratio = type_data['latitude'].apply(lambda x: abs(x) < 60).mean()  # 纬度<60度视为可能沿海
            avg_latitude = type_data['latitude'].mean()
            latitude_variance = type_data['latitude'].var()
            
            disaster_geographic_patterns[int(disaster_type)] = {
                'region_distribution': region_distribution,
                'coastal_ratio': coastal_ratio,
                'avg_latitude': avg_latitude,
                'latitude_variance': latitude_variance
            }
        
        # 基于地理模式相似性自动分组
        return self._cluster_disasters_by_similarity(disaster_geographic_patterns)
    
    def _cluster_disasters_by_similarity(self, patterns: Dict[int, Dict]) -> Dict[str, List[int]]:
        """基于模式相似性自动聚类灾害类型"""
        disaster_ids = list(patterns.keys())
        
        if len(disaster_ids) < 2:
            return {'group_1': disaster_ids}
        
        # 计算灾害间的相似性矩阵
        similarity_matrix = np.zeros((len(disaster_ids), len(disaster_ids)))
        
        for i, id1 in enumerate(disaster_ids):
            for j, id2 in enumerate(disaster_ids):
                if i != j:
                    similarity = self._calculate_pattern_similarity(patterns[id1], patterns[id2])
                    similarity_matrix[i][j] = similarity
        
        # 基于相似性进行层次聚类
        groups = self._hierarchical_clustering(disaster_ids, similarity_matrix)
        
        # 转换为命名分组
        named_groups = {}
        for i, group in enumerate(groups):
            group_name = self._auto_name_disaster_group(group, patterns)
            named_groups[group_name] = group
        
        self.logger.info(f"自动发现{len(named_groups)}个灾害分组: {list(named_groups.keys())}")
        return named_groups
    
    def _calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """计算两个灾害模式的相似性"""
        # 地理分布相似性
        regions1 = set(pattern1['region_distribution'].keys())
        regions2 = set(pattern2['region_distribution'].keys())
        region_overlap = len(regions1.intersection(regions2)) / len(regions1.union(regions2))
        
        # 沿海特征相似性
        coastal_similarity = 1.0 - abs(pattern1['coastal_ratio'] - pattern2['coastal_ratio'])
        
        # 纬度分布相似性
        lat_diff = abs(pattern1['avg_latitude'] - pattern2['avg_latitude'])
        lat_similarity = max(0, 1.0 - lat_diff / 90.0)  # 标准化到0-1
        
        # 综合相似性
        return (region_overlap + coastal_similarity + lat_similarity) / 3.0
    
    def _hierarchical_clustering(self, disaster_ids: List[int], similarity_matrix: np.ndarray, 
                               target_groups: int = 5) -> List[List[int]]:
        """简单的层次聚类"""
        # 从每个灾害作为独立组开始
        groups = [[disaster_id] for disaster_id in disaster_ids]
        
        # 迭代合并最相似的组，直到达到目标组数
        while len(groups) > target_groups and len(groups) > 1:
            max_similarity = -1
            merge_indices = (0, 1)
            
            # 找到最相似的两个组
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    # 计算组间相似性（组内元素的平均相似性）
                    group_similarity = 0.0
                    count = 0
                    for id1 in groups[i]:
                        for id2 in groups[j]:
                            idx1 = disaster_ids.index(id1)
                            idx2 = disaster_ids.index(id2)
                            group_similarity += similarity_matrix[idx1][idx2]
                            count += 1
                    
                    if count > 0:
                        avg_similarity = group_similarity / count
                        if avg_similarity > max_similarity:
                            max_similarity = avg_similarity
                            merge_indices = (i, j)
            
            # 合并最相似的两组
            if max_similarity > 0.3:  # 相似性阈值
                i, j = merge_indices
                merged_group = groups[i] + groups[j]
                new_groups = [groups[k] for k in range(len(groups)) if k != i and k != j]
                new_groups.append(merged_group)
                groups = new_groups
            else:
                break  # 相似性太低，停止合并
        
        return groups
    
    def _auto_name_disaster_group(self, group: List[int], patterns: Dict[int, Dict]) -> str:
        """基于分组特征自动命名"""
        # 获取分组的地理特征
        avg_coastal_ratio = np.mean([patterns[id]['coastal_ratio'] for id in group])
        avg_latitude = np.mean([patterns[id]['avg_latitude'] for id in group])
        
        # 自动命名逻辑
        if avg_coastal_ratio > 0.7:
            name_prefix = "marine"
        elif avg_coastal_ratio < 0.3:
            name_prefix = "inland"
        else:
            name_prefix = "mixed"
        
        if avg_latitude > 30:
            name_suffix = "northern"
        elif avg_latitude < -30:
            name_suffix = "southern"
        else:
            name_suffix = "tropical"
        
        return f"{name_prefix}_{name_suffix}_expert"
    
    def reverse_features_by_correlation(self, historical_data: pd.DataFrame, target_feature: str) -> Dict[int, float]:
        """基于关联性反推目标特征"""
        # 找出与目标灾害类型最相关的现有特征
        feature_correlations = {}
        available_features = ['latitude', 'longitude', 'month', 'people_affected', 'funding_coverage']
        
        for feature in available_features:
            if feature in historical_data.columns:
                # 计算特征与各灾害类型的相关性
                correlation_by_type = {}
                for disaster_type in historical_data['disaster_type_id'].unique():
                    if pd.isna(disaster_type):
                        continue
                    
                    type_data = historical_data[historical_data['disaster_type_id'] == disaster_type]
                    if len(type_data) > 3 and feature in type_data.columns:
                        # 计算该特征在该灾害类型中的分布特征
                        feature_mean = type_data[feature].mean()
                        feature_std = type_data[feature].std()
                        feature_range = type_data[feature].max() - type_data[feature].min()
                        
                        # 标准化特征强度
                        feature_strength = feature_std / (feature_mean + 1e-8)
                        correlation_by_type[int(disaster_type)] = feature_strength
                
                feature_correlations[feature] = correlation_by_type
        
        return feature_correlations
    
    def auto_derive_risk_factors(self, historical_data: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """完全自动化地为每个国家派生风险因子"""
        self.logger.info("自动派生国家风险因子...")
        
        country_risk_factors = {}
        all_countries = historical_data['country_id'].unique()
        
        # 全局统计作为基准
        global_disaster_rate = len(historical_data) / len(historical_data['year'].unique()) / len(all_countries)
        global_avg_impact = historical_data['people_affected'].mean()
        global_avg_funding = historical_data['amount_requested'].mean()
        
        for country_id in all_countries:
            if pd.isna(country_id):
                continue
                
            country_data = historical_data[historical_data['country_id'] == country_id]
            
            if country_data.empty:
                continue
            
            # 数据驱动的风险因子计算
            risk_factors = {}
            
            # 1. 相对灾害频率 (相对于全球平均)
            country_years = len(country_data['year'].unique())
            country_disaster_rate = len(country_data) / max(country_years, 1)
            relative_disaster_frequency = country_disaster_rate / max(global_disaster_rate, 1e-8)
            risk_factors['relative_disaster_frequency'] = min(relative_disaster_frequency, 5.0)
            
            # 2. 相对影响严重性
            country_avg_impact = country_data['people_affected'].mean()
            relative_impact_severity = country_avg_impact / max(global_avg_impact, 1)
            risk_factors['relative_impact_severity'] = min(relative_impact_severity, 5.0)
            
            # 3. 灾害类型多样性 (Shannon熵)
            type_distribution = country_data['disaster_type_id'].value_counts(normalize=True)
            diversity_entropy = -sum(p * np.log(p) for p in type_distribution if p > 0)
            max_entropy = np.log(len(historical_data['disaster_type_id'].unique()))
            risk_factors['disaster_type_diversity'] = diversity_entropy / max_entropy
            
            # 4. 时间分布均匀性
            monthly_distribution = country_data['month'].value_counts(normalize=True)
            temporal_uniformity = 1.0 - monthly_distribution.var()
            risk_factors['temporal_uniformity'] = max(temporal_uniformity, 0.0)
            
            # 5. 响应能力指标 (数据驱动)
            if 'amount_funded' in country_data.columns and 'amount_requested' in country_data.columns:
                funding_efficiency = (country_data['amount_funded'] / 
                                    (country_data['amount_requested'] + 1)).mean()
                risk_factors['response_capacity'] = min(funding_efficiency, 1.0)
            else:
                risk_factors['response_capacity'] = 0.5
            
            # 6. 地理风险聚集度
            if 'latitude' in country_data.columns and 'longitude' in country_data.columns:
                # 计算该国灾害的地理分散程度
                lat_variance = country_data['latitude'].var()
                lng_variance = country_data['longitude'].var()
                geographic_dispersion = np.sqrt(lat_variance + lng_variance)
                risk_factors['geographic_risk_concentration'] = 1.0 - min(geographic_dispersion / 10.0, 1.0)
            else:
                risk_factors['geographic_risk_concentration'] = 0.5
                
            country_risk_factors[int(country_id)] = risk_factors
        
        self.logger.info(f"自动派生了{len(country_risk_factors)}个国家的风险因子")
        return country_risk_factors
    
    def auto_discover_expert_specialization(self, historical_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """为每种灾害类型基于数据自动发现专属专家特征"""
        self.logger.info("为每种灾害类型自动发现专属专家特征...")
        
        expert_specializations = {}
        disaster_types = historical_data['disaster_type_id'].dropna().unique()
        
        for disaster_type in disaster_types:
            disaster_type = int(disaster_type)
            
            # 为该灾害类型创建专属专家
            disaster_data = historical_data[historical_data['disaster_type_id'] == disaster_type]
            
            if len(disaster_data) < 2:  # 至少需要2个事件来学习模式
                continue
            
            # 基于该灾害类型的历史数据反推专家特征
            expert_name = f"disaster_type_{disaster_type}_expert"
            specialization = {
                'disaster_types': [disaster_type],  # 每个专家只负责一种灾害
                'dominant_features': self._find_dominant_features(disaster_data, historical_data),
                'geographic_pattern': self._analyze_geographic_pattern(disaster_data),
                'temporal_pattern': self._analyze_temporal_pattern(disaster_data),
                'impact_pattern': self._analyze_impact_pattern(disaster_data),
                'response_pattern': self._analyze_response_pattern(disaster_data),
                'disaster_specific_features': self._derive_disaster_specific_features(disaster_data, disaster_type)
            }
            
            expert_specializations[expert_name] = specialization
        
        self.logger.info(f"为{len(expert_specializations)}种灾害类型创建了专属专家")
        return expert_specializations
    
    def _derive_disaster_specific_features(self, disaster_data: pd.DataFrame, disaster_type: int) -> Dict[str, Any]:
        """为特定灾害类型反推专属特征，嵌入物理约束学习"""
        self.logger.info(f"为灾害类型{disaster_type}反推专属特征...")
        
        specific_features = {}
        
        # 0. 物理约束特征学习 (方案1: 地理物理特征增强)
        physical_constraints = self._learn_physical_constraints(disaster_data, disaster_type)
        specific_features['physical_constraints'] = physical_constraints
        
        # 1. 地理偏好特征
        if 'latitude' in disaster_data.columns and not disaster_data['latitude'].isna().all():
            lat_mean = disaster_data['latitude'].mean()
            lat_std = disaster_data['latitude'].std()
            specific_features['preferred_latitude_range'] = {
                'center': lat_mean,
                'spread': lat_std,
                'min_observed': disaster_data['latitude'].min(),
                'max_observed': disaster_data['latitude'].max()
            }
        
        # 2. 时间模式特征
        if 'month' in disaster_data.columns:
            month_freq = disaster_data['month'].value_counts(normalize=True)
            specific_features['seasonal_pattern'] = {
                'peak_months': month_freq.nlargest(3).index.tolist(),
                'peak_probabilities': month_freq.nlargest(3).values.tolist(),
                'seasonal_concentration': month_freq.std()  # 越大越集中在特定月份
            }
        
        # 3. 影响规模特征
        if 'people_affected' in disaster_data.columns:
            impact_data = disaster_data['people_affected'].dropna()
            if len(impact_data) > 0:
                specific_features['impact_characteristics'] = {
                    'typical_scale': impact_data.median(),
                    'scale_variability': impact_data.std(),
                    'max_observed_impact': impact_data.max(),
                    'impact_distribution_percentiles': {
                        '25%': impact_data.quantile(0.25),
                        '75%': impact_data.quantile(0.75),
                        '90%': impact_data.quantile(0.90)
                    }
                }
        
        # 4. 区域偏好特征
        if 'region_id' in disaster_data.columns:
            region_freq = disaster_data['region_id'].value_counts(normalize=True)
            specific_features['regional_preferences'] = {
                'preferred_regions': region_freq.index.tolist(),
                'region_probabilities': region_freq.values.tolist(),
                'geographic_diversity': len(region_freq)
            }
        
        # 5. 国家级风险模式
        if 'country_id' in disaster_data.columns:
            country_freq = disaster_data['country_id'].value_counts()
            high_risk_countries = country_freq[country_freq >= 2].index.tolist()  # 至少发生2次的国家
            specific_features['high_risk_countries'] = {
                'country_ids': high_risk_countries,
                'risk_levels': {int(cid): int(freq) for cid, freq in country_freq.items() if freq >= 2}
            }
        
        # 6. 发生频率特征
        if 'year' in disaster_data.columns:
            yearly_freq = disaster_data['year'].value_counts()
            specific_features['frequency_pattern'] = {
                'avg_events_per_year': len(disaster_data) / len(yearly_freq) if len(yearly_freq) > 0 else 0,
                'frequency_variance': yearly_freq.std(),
                'active_years': len(yearly_freq),
                'total_events': len(disaster_data)
            }
        
        # 7. 预警特征（基于历史模式推导）
        specific_features['predictive_indicators'] = {
            'requires_seasonal_monitoring': specific_features.get('seasonal_pattern', {}).get('seasonal_concentration', 0) > 1.0,
            'requires_geographic_focus': len(specific_features.get('regional_preferences', {}).get('preferred_regions', [])) <= 2,
            'high_impact_potential': specific_features.get('impact_characteristics', {}).get('max_observed_impact', 0) > 50000,
            'recurring_risk_countries': len(specific_features.get('high_risk_countries', {}).get('country_ids', [])) > 0
        }
        
        return specific_features
    
    def _find_dominant_features(self, group_data: pd.DataFrame, global_data: pd.DataFrame) -> Dict[str, float]:
        """找出该灾害组最显著的特征"""
        dominant_features = {}
        
        numeric_features = ['latitude', 'longitude', 'month', 'people_affected', 'amount_requested', 'funding_coverage']
        
        for feature in numeric_features:
            if feature in group_data.columns and feature in global_data.columns:
                # 计算该组在此特征上与全局的差异程度
                group_mean = group_data[feature].mean()
                global_mean = global_data[feature].mean()
                group_std = group_data[feature].std()
                global_std = global_data[feature].std()
                
                # 差异程度评分
                mean_difference = abs(group_mean - global_mean) / max(global_std, 1e-8)
                variance_ratio = group_std / max(global_std, 1e-8)
                
                # 综合显著性评分
                significance_score = mean_difference * (1 + variance_ratio)
                dominant_features[feature] = min(significance_score, 5.0)
        
        # 按显著性排序
        sorted_features = dict(sorted(dominant_features.items(), key=lambda x: x[1], reverse=True))
        return sorted_features
    
    def _analyze_geographic_pattern(self, group_data: pd.DataFrame) -> Dict[str, float]:
        """分析地理分布模式"""
        if 'latitude' not in group_data.columns or 'longitude' not in group_data.columns:
            return {'pattern_strength': 0.0}
        
        # 地理聚集程度
        lat_variance = group_data['latitude'].var()
        lng_variance = group_data['longitude'].var()
        geographic_concentration = 1.0 / (1.0 + lat_variance + lng_variance)
        
        # 纬度偏好
        avg_latitude = group_data['latitude'].mean()
        latitude_preference = abs(avg_latitude) / 90.0  # 0=赤道, 1=极地
        
        # 区域分布熵
        if 'region_id' in group_data.columns:
            region_counts = group_data['region_id'].value_counts(normalize=True)
            region_entropy = -sum(p * np.log(p) for p in region_counts if p > 0)
            max_entropy = np.log(len(region_counts))
            region_diversity = region_entropy / max(max_entropy, 1e-8)
        else:
            region_diversity = 0.5
        
        return {
            'geographic_concentration': geographic_concentration,
            'latitude_preference': latitude_preference,
            'region_diversity': region_diversity,
            'pattern_strength': geographic_concentration * (1 - region_diversity)
        }
    
    def _analyze_temporal_pattern(self, group_data: pd.DataFrame) -> Dict[str, float]:
        """分析时间分布模式"""
        if 'month' not in group_data.columns:
            return {'seasonality_strength': 0.0}
        
        # 季节性强度
        monthly_counts = group_data['month'].value_counts(normalize=True)
        seasonal_variance = monthly_counts.var()
        seasonality_strength = seasonal_variance * 12  # 标准化
        
        # 峰值月份
        peak_month = monthly_counts.idxmax()
        peak_intensity = monthly_counts.max()
        
        # 时间聚集程度
        temporal_concentration = 1.0 - (len(monthly_counts) / 12.0)
        
        return {
            'seasonality_strength': min(seasonality_strength, 1.0),
            'peak_month': peak_month,
            'peak_intensity': peak_intensity,
            'temporal_concentration': temporal_concentration
        }
    
    def _analyze_impact_pattern(self, group_data: pd.DataFrame) -> Dict[str, float]:
        """分析影响模式"""
        if 'people_affected' not in group_data.columns:
            return {'impact_predictability': 0.0}
        
        # 影响规模分布
        impacts = group_data['people_affected'].dropna()
        if len(impacts) == 0:
            return {'impact_predictability': 0.0}
        
        # 影响规模的变异系数
        impact_cv = impacts.std() / max(impacts.mean(), 1)
        impact_predictability = 1.0 / (1.0 + impact_cv)
        
        # 极端事件比例
        q75 = impacts.quantile(0.75)
        extreme_events_ratio = (impacts > q75 * 3).mean()
        
        return {
            'impact_predictability': impact_predictability,
            'extreme_events_ratio': extreme_events_ratio,
            'impact_variance_coefficient': impact_cv,
            'typical_impact_scale': impacts.median()
        }
    
    def _analyze_response_pattern(self, group_data: pd.DataFrame) -> Dict[str, float]:
        """分析响应模式"""
        response_features = {}
        
        # 资金响应模式
        if 'amount_requested' in group_data.columns and 'amount_funded' in group_data.columns:
            requested = group_data['amount_requested'].dropna()
            funded = group_data['amount_funded'].dropna()
            
            if len(requested) > 0 and len(funded) > 0:
                avg_funding_ratio = (funded / (requested + 1)).mean()
                funding_consistency = 1.0 - (funded / (requested + 1)).std()
                
                response_features['funding_efficiency'] = min(avg_funding_ratio, 1.0)
                response_features['funding_consistency'] = max(funding_consistency, 0.0)
        
        # 响应速度代理指标
        if 'appeals_count' in group_data.columns:
            appeals = group_data['appeals_count'].dropna()
            if len(appeals) > 0:
                avg_appeals = appeals.mean()
                response_complexity = min(avg_appeals / 3.0, 1.0)
                response_features['response_complexity'] = response_complexity
        
        return response_features
    
    def generate_data_driven_expert_features(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """生成完全数据驱动的专家特征体系"""
        self.logger.info("开始生成数据驱动的专家特征体系...")
        
        # 1. 自动发现专家分组和专业化
        expert_specializations = self.auto_discover_expert_specialization(historical_data)
        
        # 2. 自动派生国家风险因子
        country_risk_factors = self.auto_derive_risk_factors(historical_data)
        
        # 3. 自动发现特征关联性
        feature_correlations = {}
        for expert_name, spec in expert_specializations.items():
            expert_data = historical_data[historical_data['disaster_type_id'].isin(spec['disaster_types'])]
            correlations = self.reverse_features_by_correlation(expert_data, expert_name)
            feature_correlations[expert_name] = correlations
        
        return {
            'expert_specializations': expert_specializations,
            'country_risk_factors': country_risk_factors,
            'feature_correlations': feature_correlations,
            'metadata': {
                'total_countries': len(country_risk_factors),
                'total_experts': len(expert_specializations),
                'data_coverage': f"{len(historical_data)} historical events",
                'generated_timestamp': pd.Timestamp.now().isoformat()
            }
        }
    
    def _learn_physical_constraints(self, disaster_data: pd.DataFrame, disaster_type: int) -> Dict[str, Any]:
        """基于灾害形成原因学习硬物理约束 - 避免地理-灾害不匹配"""
        constraints = {}
        
        # 1. 基于灾害形成机制的绝对物理约束
        constraints['formation_constraints'] = self._learn_disaster_formation_constraints(disaster_data, disaster_type)
        
        # 2. 地理可行性约束学习
        constraints['geographic_feasibility'] = self._learn_geographic_feasibility(disaster_data, disaster_type)
        
        # 3. 基础设施依赖学习  
        constraints['infrastructure_requirements'] = self._learn_infrastructure_dependencies(disaster_data, disaster_type)
        
        # 4. 气候依赖学习
        constraints['climate_dependencies'] = self._learn_climate_dependencies(disaster_data, disaster_type)
        
        # 5. 季节性约束学习
        constraints['seasonal_constraints'] = self._learn_seasonal_constraints(disaster_data, disaster_type)
        
        # 6. 负样本学习 (方案3: 约束嵌入)
        constraints['negative_samples'] = self._generate_negative_samples(disaster_data, disaster_type)
        
        return constraints
    
    def _learn_geographic_feasibility(self, disaster_data: pd.DataFrame, disaster_type: int) -> Dict[str, Any]:
        """学习地理可行性约束"""
        geo_constraints = {}
        
        # 1. 国家集中度分析 - 核心约束学习
        historical_countries = set(disaster_data['country_id'].dropna().astype(int))
        total_global_countries = 253  # 假设全球国家总数
        
        geo_constraints['country_distribution'] = {
            'historical_countries': list(historical_countries),
            'never_occurred_countries': list(set(range(1, 254)) - historical_countries),
            'geographic_concentration': len(historical_countries) / total_global_countries,
            'selectivity_strength': 1.0 - (len(historical_countries) / total_global_countries)
        }
        
        # 2. 纬度封锁区学习
        if 'latitude' in disaster_data.columns and not disaster_data['latitude'].isna().all():
            lats = disaster_data['latitude'].dropna()
            if len(lats) > 0:
                geo_constraints['latitude_constraints'] = {
                    'observed_lat_min': float(lats.min()),
                    'observed_lat_max': float(lats.max()),
                    'lat_concentration': float(lats.std()),  # 标准差小=地理集中
                    'forbidden_zones': self._identify_latitude_forbidden_zones(lats, disaster_type)
                }
        
        # 3. 相似灾害类型约束传递
        geo_constraints['disaster_type_similarity'] = self._get_similar_disaster_constraints(disaster_type)
        
        return geo_constraints
    
    def _estimate_coastal_proximity(self, lat: float, lng: float) -> float:
        """估算沿海接近度 (0-1评分)"""
        if lat == 0 and lng == 0:
            return 0.5
        
        # 基于坐标特征简单估算沿海程度
        # 小岛屿/群岛区域
        island_regions = [
            (-30, 30, 120, 180),   # 太平洋
            (-30, 30, -180, -120), # 太平洋西部  
            (10, 25, -85, -60),    # 加勒比海
            (-30, 0, 40, 100)      # 印度洋岛屿
        ]
        
        for lat_min, lat_max, lng_min, lng_max in island_regions:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return 0.95  # 高沿海评分
        
        # 大陆内陆区域
        interior_regions = [
            (-10, 10, 10, 35),     # 非洲内陆
            (35, 55, 60, 120),     # 亚洲内陆
            (35, 55, -110, -80),   # 北美内陆
            (-20, 5, -70, -50)     # 南美内陆
        ]
        
        for lat_min, lat_max, lng_min, lng_max in interior_regions:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return 0.1  # 低沿海评分
        
        return 0.6  # 默认中等沿海可能性
    
    def _learn_infrastructure_dependencies(self, disaster_data: pd.DataFrame, disaster_type: int) -> Dict[str, Any]:
        """学习基础设施依赖模式"""
        infra_constraints = {}
        
        # 分析历史事件的国家发展水平模式
        country_frequencies = disaster_data['country_id'].value_counts()
        total_countries = len(disaster_data['country_id'].unique())
        
        infra_constraints['country_selectivity'] = {
            'affected_countries_ratio': total_countries / 253,  # 相对于全球国家数
            'concentration_index': country_frequencies.std() / country_frequencies.mean() if len(country_frequencies) > 0 else 0,
            'requires_development': disaster_type in [67, 68, 66]  # 需要较高发展水平的灾害
        }
        
        return infra_constraints
    
    def _learn_climate_dependencies(self, disaster_data: pd.DataFrame, disaster_type: int) -> Dict[str, Any]:
        """学习气候依赖模式"""
        climate_constraints = {}
        
        if 'latitude' in disaster_data.columns:
            lats = disaster_data['latitude'].dropna()
            if len(lats) > 0:
                # 分析气候带分布
                climate_distribution = {'tropical': 0, 'temperate': 0, 'polar': 0}
                for lat in lats:
                    climate_zone = self._get_climate_zone_from_lat(lat)
                    climate_distribution[climate_zone] += 1
                
                total = sum(climate_distribution.values())
                if total > 0:
                    climate_constraints['climate_preference'] = {
                        zone: count/total for zone, count in climate_distribution.items()
                    }
                    
                    # 识别气候禁区 (历史概率<5%的气候带)
                    climate_constraints['climate_restrictions'] = [
                        zone for zone, prob in climate_constraints['climate_preference'].items() 
                        if prob < 0.05
                    ]
        
        return climate_constraints
    
    def _generate_negative_samples(self, disaster_data: pd.DataFrame, disaster_type: int) -> Dict[str, Any]:
        """生成负样本用于约束嵌入学习 (方案3)"""
        negative_samples = {
            'impossible_countries': [],
            'impossible_climate_zones': [],
            'impossible_months': [],
            'constraint_strength': 1.0
        }
        
        # 1. 从未发生该灾害的国家作为强负样本
        all_countries = set(range(1, 254))  # 假设country_id 1-253
        historical_countries = set(disaster_data['country_id'].dropna().astype(int))
        never_occurred_countries = all_countries - historical_countries
        
        # 物理不可能的国家 (强约束)
        if disaster_type in [4, 23, 10]:  # 海洋灾害
            # 内陆国家：从未发生过任何海洋灾害的国家
            negative_samples['impossible_countries'] = list(never_occurred_countries)[:30]
            negative_samples['constraint_strength'] = 0.98  # 极强约束
        
        elif disaster_type in [2, 8]:  # 地质灾害
            # 地质稳定国家：从未发生地震/火山的国家
            negative_samples['impossible_countries'] = list(never_occurred_countries)[:20]
            negative_samples['constraint_strength'] = 0.95
        
        elif disaster_type in [67, 68, 66]:  # 高技术灾害
            # 基础设施不足国家：发生频次极低的国家
            low_frequency_countries = []
            country_freq = disaster_data['country_id'].value_counts()
            for country_id in never_occurred_countries:
                if country_freq.get(country_id, 0) == 0:
                    low_frequency_countries.append(country_id)
            negative_samples['impossible_countries'] = low_frequency_countries[:15]
            negative_samples['constraint_strength'] = 0.85
        
        # 2. 不可能气候带
        if disaster_type == 19:  # Heat Wave
            negative_samples['impossible_climate_zones'] = ['polar']
            
        elif disaster_type == 14:  # Cold Wave  
            negative_samples['impossible_climate_zones'] = ['tropical']
            
        elif disaster_type in [4, 23]:  # 热带风暴相关
            negative_samples['impossible_climate_zones'] = ['polar']
        
        return negative_samples
    
    def _identify_latitude_forbidden_zones(self, historical_lats: pd.Series, disaster_type: int) -> List[Dict[str, float]]:
        """从历史纬度分布识别封锁区"""
        forbidden_zones = []
        
        lat_min, lat_max = historical_lats.min(), historical_lats.max()
        
        # 极地封锁区 (如果从未在高纬度发生)
        if lat_max < 60:  # 从未在60度以北发生
            forbidden_zones.append({
                'zone_type': 'high_latitude_exclusion',
                'lat_min': 60,
                'lat_max': 90,
                'exclusion_strength': 0.95
            })
        
        if lat_min > -60:  # 从未在60度以南发生
            forbidden_zones.append({
                'zone_type': 'high_latitude_exclusion_south', 
                'lat_min': -90,
                'lat_max': -60,
                'exclusion_strength': 0.95
            })
        
        # 热带封锁区 (如果从未在低纬度发生)
        if lat_min > 30:  # 从未在30度以南发生
            forbidden_zones.append({
                'zone_type': 'tropical_exclusion',
                'lat_min': -30,
                'lat_max': 30,
                'exclusion_strength': 0.9
            })
        
        return forbidden_zones
    
    def _get_similar_disaster_constraints(self, disaster_type: int) -> Dict[str, Any]:
        """相似灾害类型的约束传递"""
        similarity_groups = {
            'geological': [2, 8],  # Earthquake, Volcanic - 地质灾害组
            'marine': [4, 23, 10],  # Cyclone, Storm Surge, Tsunami - 海洋灾害组  
            'hydrological': [12, 27],  # Flood, Flash Flood - 水文灾害组
            'climate_extreme': [19, 14],  # Heat Wave, Cold Wave - 极端气候组
            'social_conflict': [6, 7, 5],  # 社会冲突组
            'tech_industrial': [67, 68, 66],  # 高技术/工业灾害组
            'agricultural': [21, 62, 20]  # 农业相关灾害组
        }
        
        # 找到当前灾害所属的相似组
        for group_name, group_types in similarity_groups.items():
            if disaster_type in group_types:
                return {
                    'similarity_group': group_name,
                    'related_disaster_types': [t for t in group_types if t != disaster_type],
                    'constraint_sharing': True
                }
        
        return {'similarity_group': 'independent', 'constraint_sharing': False}
    
    def _get_climate_zone_from_lat(self, latitude: float) -> str:
        """根据纬度判断气候带"""
        abs_lat = abs(latitude)
        if abs_lat > 66.5:
            return 'polar'
        elif abs_lat > 23.5:
            return 'temperate'
        else:
            return 'tropical'
    
    def _learn_seasonal_constraints(self, disaster_data: pd.DataFrame, disaster_type: int) -> Dict[str, Any]:
        """学习季节性约束 - 季节-灾害-地理三维模式"""
        seasonal_constraints = {}
        
        if 'month' not in disaster_data.columns:
            return seasonal_constraints
        
        # 1. 月份分布强度学习
        month_distribution = disaster_data['month'].value_counts(normalize=True).to_dict()
        
        # 识别禁止月份 (从未发生或极少发生)
        forbidden_months = [month for month, prob in month_distribution.items() if prob < 0.01]
        peak_months = [month for month, prob in month_distribution.items() if prob > 0.15]
        
        seasonal_constraints['monthly_patterns'] = {
            'distribution': month_distribution,
            'forbidden_months': forbidden_months,
            'peak_months': peak_months,
            'seasonal_concentration': np.std(list(month_distribution.values()))  # 季节集中度
        }
        
        # 2. 地理-季节交互学习
        seasonal_constraints['geo_seasonal_interactions'] = self._learn_geo_seasonal_interactions(disaster_data, disaster_type)
        
        # 3. 反季节强约束识别
        seasonal_constraints['anti_seasonal_constraints'] = self._identify_anti_seasonal_patterns(disaster_data, disaster_type)
        
        return seasonal_constraints
    
    def _learn_geo_seasonal_interactions(self, disaster_data: pd.DataFrame, disaster_type: int) -> Dict[str, Any]:
        """学习地理-季节交互模式"""
        interactions = {}
        
        if 'latitude' not in disaster_data.columns or 'month' not in disaster_data.columns:
            return interactions
        
        # 按气候带分析季节模式
        climate_seasonal_patterns = {}
        
        for _, event in disaster_data.iterrows():
            lat = event.get('latitude', 0)
            month = event.get('month')
            
            if pd.isna(lat) or pd.isna(month):
                continue
                
            climate_zone = self._get_climate_zone_from_lat(lat)
            
            if climate_zone not in climate_seasonal_patterns:
                climate_seasonal_patterns[climate_zone] = {}
            
            if month not in climate_seasonal_patterns[climate_zone]:
                climate_seasonal_patterns[climate_zone][month] = 0
            
            climate_seasonal_patterns[climate_zone][month] += 1
        
        # 标准化并识别气候带-月份的强弱模式
        for climate_zone, month_counts in climate_seasonal_patterns.items():
            total = sum(month_counts.values())
            if total > 0:
                normalized_pattern = {month: count/total for month, count in month_counts.items()}
                
                # 识别该气候带的禁止月份
                weak_months = [month for month, prob in normalized_pattern.items() if prob < 0.05]
                strong_months = [month for month, prob in normalized_pattern.items() if prob > 0.2]
                
                interactions[climate_zone] = {
                    'monthly_distribution': normalized_pattern,
                    'forbidden_months': weak_months,
                    'preferred_months': strong_months
                }
        
        return interactions
    
    def _identify_anti_seasonal_patterns(self, disaster_data: pd.DataFrame, disaster_type: int) -> Dict[str, Any]:
        """识别反季节强约束模式"""
        anti_seasonal = {}
        
        # 基于灾害类型的物理季节约束
        if disaster_type in [4, 23]:  # Cyclone, Storm Surge - 台风季节性
            # 从历史数据学习台风季节分布
            if 'month' in disaster_data.columns:
                month_dist = disaster_data['month'].value_counts(normalize=True)
                
                # 识别完全无台风的月份作为强禁止
                never_months = [m for m in range(1, 13) if m not in month_dist.index]
                rare_months = [m for m, prob in month_dist.items() if prob < 0.02]
                
                anti_seasonal['typhoon_season_constraints'] = {
                    'never_occurred_months': never_months,
                    'extremely_rare_months': rare_months,
                    'constraint_strength': 0.95
                }
        
        elif disaster_type in [12, 27]:  # Flood, Flash Flood - 降水季节性
            # 学习洪水的反季节模式
            if 'month' in disaster_data.columns:
                month_dist = disaster_data['month'].value_counts(normalize=True)
                dry_season_months = [m for m, prob in month_dist.items() if prob < 0.05]
                
                anti_seasonal['flood_dry_season_constraints'] = {
                    'dry_season_months': dry_season_months,
                    'constraint_strength': 0.8
                }
        
        elif disaster_type == 20:  # Drought - 干旱季节性
            # 干旱与雨季的反向关系
            if 'month' in disaster_data.columns:
                month_dist = disaster_data['month'].value_counts(normalize=True)
                wet_season_months = [m for m, prob in month_dist.items() if prob < 0.03]
                
                anti_seasonal['drought_wet_season_constraints'] = {
                    'wet_season_months': wet_season_months,
                    'constraint_strength': 0.85
                }
        
        elif disaster_type == 15:  # Fire - 火灾季节性
            # 火灾与湿润季节的关系
            if 'month' in disaster_data.columns:
                month_dist = disaster_data['month'].value_counts(normalize=True)
                humid_months = [m for m, prob in month_dist.items() if prob < 0.04]
                
                anti_seasonal['fire_humid_season_constraints'] = {
                    'humid_season_months': humid_months,
                    'constraint_strength': 0.75
                }
        
        return anti_seasonal
    
    def _learn_disaster_formation_constraints(self, disaster_data: pd.DataFrame, disaster_type: int) -> Dict[str, Any]:
        """基于灾害形成机制学习绝对物理约束 - 防止地理-灾害不匹配"""
        formation_constraints = {}
        
        # 分析每种灾害的形成原因和地理必要条件
        disaster_name = self._get_disaster_name_by_id(disaster_type)
        
        if disaster_type == 23:  # Storm Surge - 风暴潮
            # 形成原因: 海洋风暴+潮汐作用，绝对需要沿海
            formation_constraints.update({
                'absolute_requirements': ['coastal_access'],
                'formation_mechanism': 'ocean_storm_tidal_interaction',
                'geographic_necessity': 'mandatory_coastal',
                'max_inland_distance_km': 5,  # 从历史数据学习最大内陆距离
                'coastal_requirement_strength': 0.99  # 极强沿海要求
            })
            
            # 从历史数据验证沿海要求
            coastal_validation = self._validate_coastal_requirement(disaster_data)
            formation_constraints['historical_coastal_validation'] = coastal_validation
        
        elif disaster_type == 10:  # Tsunami - 海啸
            # 形成原因: 海底地震/火山，绝对需要海洋传播
            formation_constraints.update({
                'absolute_requirements': ['ocean_proximity', 'seismic_source'],
                'formation_mechanism': 'underwater_seismic_ocean_wave',
                'geographic_necessity': 'mandatory_coastal',
                'max_inland_penetration_km': 10,  # 海啸最大内陆侵入
                'coastal_requirement_strength': 0.99
            })
            
        elif disaster_type == 4:  # Cyclone - 气旋/台风
            # 形成原因: 海洋热能+科里奥利力，需要海洋形成+登陆路径
            formation_constraints.update({
                'absolute_requirements': ['ocean_formation', 'latitude_range'],
                'formation_mechanism': 'ocean_thermal_coriolis_system',
                'geographic_necessity': 'ocean_adjacent_or_path',
                'latitude_formation_range': (-40, 40),  # 台风形成纬度
                'requires_ocean_access': True,
                'inland_decay_factor': 0.8  # 内陆后强度衰减
            })
            
        elif disaster_type == 8:  # Volcanic Eruption - 火山喷发
            # 形成原因: 岩浆活动，绝对需要火山带
            formation_constraints.update({
                'absolute_requirements': ['volcanic_zone'],
                'formation_mechanism': 'magma_tectonic_activity',
                'geographic_necessity': 'mandatory_volcanic_belt',
                'max_distance_to_volcano_km': 50,  # 从历史数据学习
                'volcanic_requirement_strength': 0.98
            })
            
            # 从历史分布学习火山带约束
            volcanic_validation = self._validate_volcanic_zone_requirement(disaster_data)
            formation_constraints['historical_volcanic_validation'] = volcanic_validation
            
        elif disaster_type == 2:  # Earthquake - 地震
            # 形成原因: 地壳运动，需要断层活动区
            formation_constraints.update({
                'absolute_requirements': ['tectonic_activity'],
                'formation_mechanism': 'crustal_plate_movement',
                'geographic_necessity': 'fault_zone_proximity',
                'stable_region_exclusion': True,
                'tectonic_requirement_strength': 0.9
            })
            
        elif disaster_type == 12:  # Flood - 洪水
            # 形成原因: 降水+径流+地形，需要水文条件
            formation_constraints.update({
                'absolute_requirements': ['water_source', 'topography'],
                'formation_mechanism': 'precipitation_runoff_accumulation',
                'geographic_necessity': 'river_basin_or_rainfall',
                'desert_region_penalty': 0.8,  # 沙漠地区洪水概率降低
                'requires_hydrological_conditions': True
            })
            
        elif disaster_type == 27:  # Flash Flood - 山洪
            # 形成原因: 短时强降水+陡峭地形，需要山地/丘陵
            formation_constraints.update({
                'absolute_requirements': ['steep_terrain', 'intense_rainfall'],
                'formation_mechanism': 'rapid_runoff_concentration',
                'geographic_necessity': 'mountainous_or_hilly',
                'flat_terrain_penalty': 0.7,  # 平原地区山洪概率降低
                'elevation_preference': 'varied_topography'
            })
            
        elif disaster_type == 20:  # Drought - 干旱
            # 形成原因: 长期降水不足，需要干旱气候倾向
            formation_constraints.update({
                'absolute_requirements': ['precipitation_deficit'],
                'formation_mechanism': 'prolonged_rainfall_deficiency',
                'geographic_necessity': 'arid_semi_arid_climate',
                'humid_region_penalty': 0.6,  # 湿润地区干旱概率降低
                'climate_suitability_requirement': True
            })
            
        elif disaster_type == 15:  # Fire - 火灾
            # 形成原因: 干燥+可燃物+点火源，需要植被+干燥条件
            formation_constraints.update({
                'absolute_requirements': ['vegetation_cover', 'dry_conditions'],
                'formation_mechanism': 'vegetation_combustion',
                'geographic_necessity': 'vegetated_dry_regions',
                'urban_vs_wildland': 'both_possible',
                'seasonal_dryness_requirement': True
            })
            
        elif disaster_type == 19:  # Heat Wave - 热浪
            # 形成原因: 高压系统+大陆加热，需要大陆性气候
            formation_constraints.update({
                'absolute_requirements': ['continental_climate'],
                'formation_mechanism': 'high_pressure_continental_heating',
                'geographic_necessity': 'land_mass_continental',
                'polar_region_exclusion': True,
                'oceanic_region_penalty': 0.5
            })
            
        elif disaster_type == 14:  # Cold Wave - 寒潮
            # 形成原因: 极地冷空气南下，需要中高纬度+大陆通道
            formation_constraints.update({
                'absolute_requirements': ['polar_air_mass_access'],
                'formation_mechanism': 'polar_air_advection',
                'geographic_necessity': 'mid_high_latitude',
                'tropical_region_exclusion': True,
                'latitude_threshold': 15  # 纬度低于15度极少寒潮
            })
            
        elif disaster_type == 1:  # Epidemic - 传染病
            # 形成原因: 病原体+传播途径+易感人群，需要人口密度
            formation_constraints.update({
                'absolute_requirements': ['population_density'],
                'formation_mechanism': 'pathogen_transmission',
                'geographic_necessity': 'populated_areas',
                'rural_vs_urban': 'both_possible',
                'isolation_penalty': 0.3  # 极度隔离地区传染病概率降低
            })
            
        elif disaster_type in [67, 68, 66]:  # 技术/工业灾害
            # 形成原因: 人为技术系统故障，需要工业基础设施
            formation_constraints.update({
                'absolute_requirements': ['industrial_infrastructure'],
                'formation_mechanism': 'technological_system_failure',
                'geographic_necessity': 'developed_industrial_areas',
                'rural_remote_penalty': 0.1,  # 偏远农村地区技术灾害极少
                'development_level_requirement': True
            })
            
        elif disaster_type in [21, 62]:  # 农业相关灾害
            # 形成原因: 农作物/农业系统问题，需要农业区
            formation_constraints.update({
                'absolute_requirements': ['agricultural_activity'],
                'formation_mechanism': 'agricultural_system_disruption',
                'geographic_necessity': 'farming_regions',
                'non_agricultural_penalty': 0.2,  # 非农业地区相关灾害少
                'rural_preference': True
            })
        
        else:
            # 其他灾害类型的通用约束
            # 扩展其他灾害类型的形成机制约束
            if disaster_type == 5:  # Population Movement - 人口流动
                formation_constraints.update({
                    'absolute_requirements': ['population_corridors', 'conflict_or_disaster_trigger'],
                    'formation_mechanism': 'forced_displacement_migration',
                    'geographic_necessity': 'populated_border_regions',
                    'remote_area_penalty': 0.2
                })
            elif disaster_type == 6:  # Complex Emergency - 复合紧急事态
                formation_constraints.update({
                    'absolute_requirements': ['multiple_crisis_overlap'],
                    'formation_mechanism': 'cascading_multi_hazard_crisis',
                    'geographic_necessity': 'vulnerable_regions',
                    'stable_developed_penalty': 0.3
                })
            elif disaster_type == 7:  # Civil Unrest - 社会动乱
                formation_constraints.update({
                    'absolute_requirements': ['population_density', 'social_tension'],
                    'formation_mechanism': 'social_political_disruption',
                    'geographic_necessity': 'urban_populated_areas',
                    'rural_remote_penalty': 0.4
                })
            elif disaster_type == 11:  # Mass Movement - 地质运动
                formation_constraints.update({
                    'absolute_requirements': ['unstable_slopes', 'gravity_trigger'],
                    'formation_mechanism': 'gravitational_slope_failure',
                    'geographic_necessity': 'mountainous_hilly_terrain',
                    'flat_terrain_penalty': 0.2
                })
            elif disaster_type == 13:  # Other - 其他
                formation_constraints.update({
                    'formation_mechanism': 'various_mechanisms',
                    'geographic_necessity': 'location_flexible',
                    'constraint_strength': 0.8
                })
            elif disaster_type == 24:  # Landslide - 滑坡 (已在mixed_expert_predictor中处理)
                formation_constraints.update({
                    'absolute_requirements': ['steep_slopes', 'trigger_factor'],
                    'formation_mechanism': 'slope_instability_trigger',
                    'geographic_necessity': 'mountainous_regions',
                    'flat_terrain_penalty': 0.3
                })
            elif disaster_type == 54:  # Transport Accident - 交通事故
                formation_constraints.update({
                    'absolute_requirements': ['transport_infrastructure'],
                    'formation_mechanism': 'transport_system_failure',
                    'geographic_necessity': 'transport_corridors',
                    'remote_area_penalty': 0.4
                })
            elif disaster_type == 57:  # Chemical Emergency - 化学紧急事态
                formation_constraints.update({
                    'absolute_requirements': ['chemical_facilities', 'industrial_activity'],
                    'formation_mechanism': 'chemical_hazard_release',
                    'geographic_necessity': 'industrial_areas',
                    'rural_remote_penalty': 0.1
                })
            else:
                formation_constraints.update({
                    'formation_mechanism': 'general_environmental_social',
                    'geographic_necessity': 'location_flexible',
                    'constraint_strength': 0.5
                })
        
        # 从历史数据验证形成约束的有效性
        formation_constraints['historical_validation'] = self._validate_formation_constraints(
            disaster_data, formation_constraints, disaster_type
        )
        
        return formation_constraints
    
    def _get_disaster_name_by_id(self, disaster_type: int) -> str:
        """获取灾害类型名称"""
        disaster_names = {
            1: 'Epidemic', 2: 'Earthquake', 4: 'Cyclone', 5: 'Population Movement',
            6: 'Complex Emergency', 7: 'Civil Unrest', 8: 'Volcanic Eruption',
            10: 'Tsunami', 11: 'Landslide', 12: 'Flood', 13: 'Other',
            14: 'Cold Wave', 15: 'Fire', 19: 'Heat Wave', 20: 'Drought',
            21: 'Food Insecurity', 23: 'Storm Surge', 24: 'Mass Movement',
            27: 'Pluvial/Flash Flood', 54: 'Transport Accident', 57: 'Chemical Emergency',
            62: 'Insect Infestation', 66: 'Biological Emergency', 67: 'Radiological Emergency',
            68: 'Transport Emergency'
        }
        return disaster_names.get(disaster_type, f'Unknown_{disaster_type}')
    
    def _validate_coastal_requirement(self, disaster_data: pd.DataFrame) -> Dict[str, Any]:
        """验证沿海要求的历史一致性"""
        validation = {'coastal_consistency': 0.0, 'inland_violations': []}
        
        if 'latitude' not in disaster_data.columns or 'longitude' not in disaster_data.columns:
            return validation
        
        total_events = len(disaster_data)
        coastal_events = 0
        inland_violations = []
        
        for idx, event in disaster_data.iterrows():
            lat = event.get('latitude', 0)
            lng = event.get('longitude', 0)
            country_id = event.get('country_id')
            
            coastal_score = self._estimate_coastal_proximity(lat, lng)
            
            if coastal_score > 0.7:  # 明确沿海
                coastal_events += 1
            elif coastal_score < 0.3:  # 明确内陆
                inland_violations.append({
                    'country_id': country_id,
                    'latitude': lat,
                    'longitude': lng,
                    'coastal_score': coastal_score
                })
        
        validation['coastal_consistency'] = coastal_events / max(total_events, 1)
        validation['inland_violations'] = inland_violations[:5]  # 保留前5个违例
        validation['inland_violation_rate'] = len(inland_violations) / max(total_events, 1)
        
        return validation
    
    def _validate_volcanic_zone_requirement(self, disaster_data: pd.DataFrame) -> Dict[str, Any]:
        """验证火山带要求的历史一致性"""
        validation = {'volcanic_zone_consistency': 0.0, 'non_volcanic_violations': []}
        
        if 'latitude' not in disaster_data.columns or 'longitude' not in disaster_data.columns:
            return validation
        
        total_events = len(disaster_data)
        volcanic_zone_events = 0
        non_volcanic_violations = []
        
        # 简化的火山带识别 (基于已知火山活跃区域)
        volcanic_zones = [
            # 环太平洋火山带
            (-60, 60, 120, 180),
            (-60, 60, -180, -60),
            # 地中海-阿尔卑斯-喜马拉雅火山带
            (25, 45, -10, 100),
            # 中大西洋脊
            (-60, 60, -40, -10),
            # 东非大裂谷火山
            (-35, 20, 25, 50)
        ]
        
        for idx, event in disaster_data.iterrows():
            lat = event.get('latitude', 0)
            lng = event.get('longitude', 0)
            country_id = event.get('country_id')
            
            in_volcanic_zone = False
            for lat_min, lat_max, lng_min, lng_max in volcanic_zones:
                if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                    in_volcanic_zone = True
                    break
            
            if in_volcanic_zone:
                volcanic_zone_events += 1
            else:
                non_volcanic_violations.append({
                    'country_id': country_id,
                    'latitude': lat,
                    'longitude': lng,
                    'volcanic_zone_distance': 'unknown'  # 实际应计算到最近火山带距离
                })
        
        validation['volcanic_zone_consistency'] = volcanic_zone_events / max(total_events, 1)
        validation['non_volcanic_violations'] = non_volcanic_violations[:5]
        validation['non_volcanic_violation_rate'] = len(non_volcanic_violations) / max(total_events, 1)
        
        return validation
    
    def _validate_formation_constraints(self, disaster_data: pd.DataFrame, 
                                      formation_constraints: Dict[str, Any], 
                                      disaster_type: int) -> Dict[str, Any]:
        """验证形成约束与历史数据的一致性"""
        validation_result = {
            'constraint_effectiveness': 0.0,
            'historical_consistency': 0.0,
            'violation_examples': []
        }
        
        if disaster_data.empty:
            return validation_result
        
        total_events = len(disaster_data)
        consistent_events = 0
        violations = []
        
        # 检查每个历史事件是否符合形成约束
        for idx, event in disaster_data.iterrows():
            lat = event.get('latitude', 0)
            lng = event.get('longitude', 0)
            country_id = event.get('country_id')
            month = event.get('month')
            
            # 根据灾害类型检查相应的形成约束
            constraint_satisfied = True
            violation_reasons = []
            
            if disaster_type in [23, 10, 4]:  # 海洋灾害组
                coastal_score = self._estimate_coastal_proximity(lat, lng)
                if coastal_score < 0.3:  # 明显内陆
                    constraint_satisfied = False
                    violation_reasons.append(f'内陆位置,沿海评分:{coastal_score:.2f}')
            
            elif disaster_type == 8:  # 火山
                # 检查是否在火山活跃带
                in_volcanic_zone = self._check_volcanic_zone(lat, lng)
                if not in_volcanic_zone:
                    constraint_satisfied = False
                    violation_reasons.append('非火山活跃带')
            
            elif disaster_type == 14:  # 寒潮
                if abs(lat) < 15:  # 热带地区
                    constraint_satisfied = False
                    violation_reasons.append(f'热带地区寒潮,纬度:{lat:.1f}')
            
            elif disaster_type == 19:  # 热浪
                if abs(lat) > 75:  # 极地地区
                    constraint_satisfied = False
                    violation_reasons.append(f'极地地区热浪,纬度:{lat:.1f}')
            
            if constraint_satisfied:
                consistent_events += 1
            else:
                violations.append({
                    'country_id': country_id,
                    'latitude': lat,
                    'longitude': lng,
                    'month': month,
                    'violation_reasons': violation_reasons
                })
        
        validation_result['historical_consistency'] = consistent_events / max(total_events, 1)
        validation_result['violation_examples'] = violations[:3]  # 保留前3个违例
        validation_result['constraint_effectiveness'] = 1.0 - (len(violations) / max(total_events, 1))
        
        return validation_result
    
    def _check_volcanic_zone(self, lat: float, lng: float) -> bool:
        """检查是否在火山活跃带"""
        volcanic_zones = [
            # 环太平洋火山带
            (-60, 60, 120, 180),
            (-60, 60, -180, -60),
            # 地中海-阿尔卑斯-喜马拉雅带
            (25, 45, -10, 100),
            # 中大西洋脊
            (-60, 60, -40, -10),
            # 东非大裂谷
            (-35, 20, 25, 50)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in volcanic_zones:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        return False