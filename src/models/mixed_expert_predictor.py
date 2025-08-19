"""
混合专家预测器 - 数据驱动的多专家灾害预测
完全基于历史数据自动学习的专家模型融合
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict


class MixedExpertPredictor:
    """数据驱动的混合专家预测系统"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.expert_models = {}
        self.fusion_weights = {}
        self.is_trained = False
        
    def train_expert_models(self, historical_data: pd.DataFrame, 
                          expert_specializations: Dict[str, Dict[str, Any]],
                          country_risk_factors: Dict[int, Dict[str, float]]) -> None:
        """训练所有专家模型"""
        self.logger.info("开始训练混合专家模型...")
        
        # 存储历史数据引用用于地理约束分析
        self.historical_data = historical_data
        
        # 基于历史数据建立气候-灾害排斥约束
        self._build_climate_disaster_exclusions()
        
        # 建立技术设施相关约束
        self._build_technical_infrastructure_constraints()
        
        for expert_name, specialization in expert_specializations.items():
            self.logger.info(f"训练{expert_name}...")
            
            # 过滤该专家负责的灾害类型数据
            expert_disaster_types = specialization['disaster_types']
            expert_data = historical_data[historical_data['disaster_type_id'].isin(expert_disaster_types)]
            
            if expert_data.empty:
                continue
            
            # 为该专家训练模型
            expert_model = self._train_single_expert(expert_data, specialization, country_risk_factors)
            self.expert_models[expert_name] = expert_model
            
            # 计算该专家的融合权重
            self.fusion_weights[expert_name] = self._calculate_expert_weight(expert_data, expert_model)
        
        # 标准化融合权重
        self._normalize_fusion_weights()
        self.is_trained = True
        
        self.logger.info(f"成功训练了{len(self.expert_models)}个专家模型")
    
    def _get_historical_event_count(self, disaster_type: int, country_id: int) -> int:
        """获取特定国家特定灾害类型的历史事件数量"""
        # 这里需要存储历史数据的引用，简化实现
        # 在实际使用中应该在初始化时存储historical_data
        if hasattr(self, '_historical_data'):
            country_disaster_data = self._historical_data[
                (self._historical_data['country_id'] == country_id) &
                (self._historical_data['disaster_type_id'] == disaster_type)
            ]
            return len(country_disaster_data)
        return 3  # 默认假设有3次历史事件
    
    def _apply_bayesian_shrinkage(self, base_prob: float, historical_events: int, 
                                global_mean: float = 0.05, smoothing_factor: int = 5) -> float:
        """应用贝叶斯收缩 - 历史事件少时向全局均值收缩"""
        if historical_events == 0:
            return global_mean * 0.1  # 无历史记录时大幅降低
        
        # 贝叶斯收缩公式
        shrinkage_weight = historical_events / (historical_events + smoothing_factor)
        shrunk_prob = shrinkage_weight * base_prob + (1 - shrinkage_weight) * global_mean
        
        # 额外的稀疏性惩罚
        if historical_events < 3:
            sparsity_penalty = 0.5 + 0.5 * (historical_events / 3.0)
            shrunk_prob *= sparsity_penalty
        
        return shrunk_prob
    
    def set_historical_data_reference(self, historical_data: pd.DataFrame) -> None:
        """设置历史数据引用用于贝叶斯收缩"""
        self._historical_data = historical_data
    
    def _train_single_expert(self, expert_data: pd.DataFrame, 
                           specialization: Dict[str, Any],
                           country_risk_factors: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """训练单个专家模型"""
        expert_model = {
            'disaster_types': specialization['disaster_types'],
            'country_patterns': {},
            'temporal_patterns': {},
            'geographic_patterns': {},
            'impact_patterns': {}
        }
        
        # 1. 学习国家级模式
        for country_id in expert_data['country_id'].unique():
            if pd.isna(country_id):
                continue
                
            country_data = expert_data[expert_data['country_id'] == country_id]
            if len(country_data) < 2:
                continue
            
            # 基于历史数据学习该国家的模式
            country_pattern = {
                'disaster_frequency': len(country_data) / len(country_data['year'].unique()),
                'avg_impact': country_data['people_affected'].mean(),
                'impact_variability': country_data['people_affected'].std(),
                'seasonal_distribution': country_data['month'].value_counts(normalize=True).to_dict(),
                'disaster_type_probabilities': country_data['disaster_type_id'].value_counts(normalize=True).to_dict()
            }
            
            # 结合风险因子
            if country_id in country_risk_factors:
                country_pattern.update(country_risk_factors[country_id])
            
            expert_model['country_patterns'][int(country_id)] = country_pattern
        
        # 2. 学习时间模式
        temporal_pattern = specialization['temporal_pattern']
        expert_model['temporal_patterns'] = {
            'seasonal_weights': self._derive_seasonal_weights(expert_data),
            'peak_months': self._identify_peak_months(expert_data),
            'temporal_concentration': temporal_pattern.get('temporal_concentration', 0.5)
        }
        
        # 3. 学习地理模式
        geographic_pattern = specialization['geographic_pattern']
        expert_model['geographic_patterns'] = {
            'latitude_preference': geographic_pattern.get('latitude_preference', 0.0),
            'geographic_concentration': geographic_pattern.get('geographic_concentration', 0.5),
            'regional_weights': self._derive_regional_weights(expert_data)
        }
        
        # 4. 学习影响模式
        expert_model['impact_patterns'] = self._learn_impact_patterns(expert_data)
        
        return expert_model
    
    def _derive_seasonal_weights(self, expert_data: pd.DataFrame) -> Dict[int, float]:
        """基于数据派生季节权重"""
        if 'month' not in expert_data.columns:
            return {i: 1.0/12 for i in range(1, 13)}
        
        monthly_freq = expert_data['month'].value_counts(normalize=True)
        seasonal_weights = {}
        
        for month in range(1, 13):
            weight = monthly_freq.get(month, 0.01)  # 最小权重0.01
            seasonal_weights[month] = weight
        
        return seasonal_weights
    
    def _identify_peak_months(self, expert_data: pd.DataFrame) -> List[int]:
        """识别高发月份"""
        if 'month' not in expert_data.columns:
            return list(range(1, 13))
        
        monthly_freq = expert_data['month'].value_counts(normalize=True)
        threshold = monthly_freq.mean() + monthly_freq.std()
        
        peak_months = [month for month, freq in monthly_freq.items() if freq > threshold]
        return peak_months if peak_months else [monthly_freq.idxmax()]
    
    def _derive_regional_weights(self, expert_data: pd.DataFrame) -> Dict[int, float]:
        """基于数据派生区域权重"""
        if 'region_id' not in expert_data.columns:
            return {}
        
        regional_freq = expert_data['region_id'].value_counts(normalize=True)
        return regional_freq.to_dict()
    
    def _learn_impact_patterns(self, expert_data: pd.DataFrame) -> Dict[str, float]:
        """学习影响模式"""
        if 'people_affected' not in expert_data.columns:
            return {'default_impact_scale': 1000.0}
        
        impacts = expert_data['people_affected'].dropna()
        if len(impacts) == 0:
            return {'default_impact_scale': 1000.0}
        
        return {
            'typical_impact': impacts.median(),
            'impact_volatility': impacts.std() / max(impacts.mean(), 1),
            'min_observed_impact': impacts.min(),
            'max_observed_impact': impacts.max(),
            'q75_impact_threshold': impacts.quantile(0.75)
        }
    
    def _calculate_expert_weight(self, expert_data: pd.DataFrame, expert_model: Dict[str, Any]) -> float:
        """基于专家数据质量计算融合权重"""
        # 数据量权重
        data_volume_score = min(len(expert_data) / 100.0, 1.0)
        
        # 时间跨度权重
        time_span = len(expert_data['year'].unique())
        temporal_coverage_score = min(time_span / 10.0, 1.0)
        
        # 地理覆盖权重
        geographic_coverage = len(expert_data['country_id'].unique())
        geographic_score = min(geographic_coverage / 20.0, 1.0)
        
        # 灾害类型覆盖权重
        disaster_diversity = len(expert_data['disaster_type_id'].unique())
        diversity_score = min(disaster_diversity / 5.0, 1.0)
        
        # 综合权重
        overall_weight = (data_volume_score * 0.3 + temporal_coverage_score * 0.25 + 
                         geographic_score * 0.25 + diversity_score * 0.2)
        
        return overall_weight
    
    def _normalize_fusion_weights(self) -> None:
        """标准化融合权重"""
        total_weight = sum(self.fusion_weights.values())
        if total_weight > 0:
            for expert_name in self.fusion_weights:
                self.fusion_weights[expert_name] /= total_weight
    
    def predict_disasters(self, current_conditions: Dict[str, Any], 
                         spatial_features: Dict[int, Dict[str, float]],
                         districts_map: Dict[int, Dict] = None) -> Dict[str, Any]:
        """使用混合专家模型进行预测"""
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train_expert_models")
        
        self.logger.info("使用混合专家模型进行预测...")
        
        # 存储districts_map用于地理约束检查
        if districts_map:
            self.districts_map = districts_map
        
        expert_predictions = {}
        
        # 获取每个专家的预测
        for expert_name, expert_model in self.expert_models.items():
            prediction = self._expert_predict(expert_name, expert_model, current_conditions, spatial_features)
            expert_predictions[expert_name] = prediction
        
        # 融合所有专家预测
        final_prediction = self._fuse_expert_predictions(expert_predictions)
        
        return final_prediction
    
    def _expert_predict(self, expert_name: str, expert_model: Dict[str, Any], 
                       current_conditions: Dict[str, Any], 
                       spatial_features: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """单个专家模型预测 - 集成物理约束和分层架构"""
        country_id = current_conditions.get('country_id')
        target_month = current_conditions.get('month', pd.Timestamp.now().month)
        latitude = current_conditions.get('latitude', 0)
        longitude = current_conditions.get('longitude', 0)
        
        # 初始化预测结果
        prediction = {
            'disaster_probabilities': {},
            'impact_estimates': {},
            'confidence_scores': {},
            'expert_confidence': 0.0
        }
        
        # 从国家模式中获取基础概率
        base_probabilities = {}
        if country_id in expert_model['country_patterns']:
            country_pattern = expert_model['country_patterns'][country_id]
            base_probabilities = country_pattern.get('disaster_type_probabilities', {})
        
        # 应用时间调制
        temporal_weights = expert_model['temporal_patterns']['seasonal_weights']
        monthly_modifier = temporal_weights.get(target_month, 1.0)
        
        # 应用空间调制
        spatial_modifier = 1.0
        if country_id in spatial_features:
            country_spatial = spatial_features[country_id]
            spatial_risk = country_spatial.get('avg_hotspot_risk', 0.5)
            spatial_modifier = 1.0 + spatial_risk  # 空间风险增强
        
        # 计算每种灾害类型的预测概率 - 分层架构 (方案2)
        for disaster_type in expert_model['disaster_types']:
            # 第1层: 物理可行性专家评估
            feasibility_score = self._assess_physical_feasibility(
                disaster_type, country_id, latitude, longitude, target_month, expert_model
            )
            
            # 第2层: 概率预测专家 (仅在可行时激活)
            if feasibility_score > 0.1:  # 物理可行阈值
                base_prob = base_probabilities.get(disaster_type, 0.01)
                
                # 应用贝叶斯收缩 - 历史数据稀疏性惩罚
                historical_events = self._get_historical_event_count(disaster_type, country_id)
                bayesian_shrinkage = self._apply_bayesian_shrinkage(base_prob, historical_events)
                
                # 综合调制
                probability_layer_output = bayesian_shrinkage * monthly_modifier * spatial_modifier
                
                # 最终概率 = 可行性评分 × 概率预测
                final_prob = feasibility_score * probability_layer_output
                prediction['disaster_probabilities'][disaster_type] = min(final_prob, 0.95)
            else:
                # 物理不可行，直接设为极低概率
                prediction['disaster_probabilities'][disaster_type] = 0.001
            
            # 影响估计
            impact_patterns = expert_model['impact_patterns']
            estimated_impact = impact_patterns.get('typical_impact', 1000) * spatial_modifier
            prediction['impact_estimates'][disaster_type] = estimated_impact
            
            # 置信度评分
            confidence = self._calculate_prediction_confidence(
                country_id, disaster_type, expert_model, current_conditions
            )
            prediction['confidence_scores'][disaster_type] = confidence
        
        # 专家整体置信度
        prediction['expert_confidence'] = np.mean(list(prediction['confidence_scores'].values()))
        
        return prediction
    
    def _assess_physical_feasibility(self, disaster_type: int, country_id: int, 
                                   latitude: float, longitude: float, target_month: int,
                                   expert_model: Dict[str, Any]) -> float:
        """第1层: 基于灾害形成原因的物理可行性评估 - 防止地理-灾害不匹配"""
        feasibility_score = 1.0
        
        # 获取该专家学习的物理约束
        disaster_features = expert_model.get('disaster_specific_features', {})
        physical_constraints = disaster_features.get('physical_constraints', {})
        
        # 0. 绝对物理法则检查 - 基于灾害形成机制
        # 传入target_month用于季节性约束
        enhanced_constraints = physical_constraints.copy()
        enhanced_constraints.setdefault('formation_constraints', {})['target_month'] = target_month
        
        formation_feasibility = self._assess_formation_mechanism_feasibility(
            disaster_type, latitude, longitude, enhanced_constraints
        )
        feasibility_score *= formation_feasibility
        
        # 如果形成机制不可行，直接返回极低分数 - 加强海洋灾害内陆约束
        if formation_feasibility < 0.001:  # 提高阈值，更严格的物理约束
            return formation_feasibility
        
        # 地形约束检查 - 新增地形物理约束
        terrain_feasibility = self._assess_terrain_constraints(disaster_type, latitude, longitude)
        feasibility_score *= terrain_feasibility
        
        # 如果地形不可行，直接返回极低分数
        if terrain_feasibility < 0.001:
            return terrain_feasibility
        
        # 1. 国家历史分布约束 - 核心约束
        geo_feasibility = physical_constraints.get('geographic_feasibility', {})
        country_dist = geo_feasibility.get('country_distribution', {})
        
        if country_dist:
            # 检查当前国家是否在历史发生国家列表中
            historical_countries = country_dist.get('historical_countries', [])
            never_occurred_countries = country_dist.get('never_occurred_countries', [])
            selectivity_strength = country_dist.get('selectivity_strength', 0)
            
            if country_id in never_occurred_countries:
                # 根据该灾害的地理选择性强度调整惩罚
                exclusion_penalty = 0.98 * selectivity_strength  # 选择性越强，惩罚越重
                feasibility_score *= (1 - exclusion_penalty)
        
        # 2. 纬度封锁区约束
        lat_constraints = geo_feasibility.get('latitude_constraints', {})
        if lat_constraints:
            forbidden_zones = lat_constraints.get('forbidden_zones', [])
            for zone in forbidden_zones:
                zone_lat_min = zone.get('lat_min', -90)
                zone_lat_max = zone.get('lat_max', 90) 
                exclusion_strength = zone.get('exclusion_strength', 0.9)
                
                if zone_lat_min <= latitude <= zone_lat_max:
                    feasibility_score *= (1 - exclusion_strength)
        
        # 3. 相似灾害类型约束传递
        similarity_info = geo_feasibility.get('disaster_type_similarity', {})
        if similarity_info.get('constraint_sharing'):
            # 如果是地质灾害组，地震和火山共享约束
            if disaster_type in [2, 8]:  # Earthquake, Volcanic
                # 从相似灾害的历史分布推断约束
                if country_id in never_occurred_countries:
                    feasibility_score *= 0.05  # 地质稳定区域强约束
            
            # 如果是海洋灾害组，共享沿海约束
            elif disaster_type in [4, 23, 10]:  # Marine disasters
                coastal_score = self._estimate_coastal_proximity_score(latitude, longitude)
                if coastal_score < 0.3:  # 明显内陆
                    feasibility_score *= 0.02  # 海洋灾害内陆强约束
        
        # 4. 季节性物理约束 (三维约束) 
        seasonal_feasibility = self._assess_seasonal_feasibility(
            disaster_type, latitude, longitude, target_month, physical_constraints
        )
        feasibility_score *= seasonal_feasibility
        
        # 5. 负样本学习约束 (方案3)
        negative_samples = physical_constraints.get('negative_samples', {})
        
        # 不可能国家约束
        impossible_countries = negative_samples.get('impossible_countries', [])
        if country_id in impossible_countries:
            constraint_strength = negative_samples.get('constraint_strength', 0.9)
            feasibility_score *= (1 - constraint_strength)
        
        # 不可能气候带约束
        impossible_climates = negative_samples.get('impossible_climate_zones', [])
        current_climate = self._get_climate_zone_from_lat(latitude)
        if current_climate in impossible_climates:
            feasibility_score *= 0.02
        
        return min(feasibility_score, 1.0)
    
    def _assess_formation_mechanism_feasibility(self, disaster_type: int, latitude: float, 
                                             longitude: float, physical_constraints: Dict[str, Any]) -> float:
        """基于灾害形成机制的绝对物理可行性检查"""
        formation_score = 1.0
        
        # 0. 绝对物理不可能组合 - 优先级最高的硬约束
        absolute_impossibility = self._check_absolute_physical_impossibilities(disaster_type, latitude, longitude)
        if absolute_impossibility:
            return 0.0001  # 绝对禁止
        
        formation_constraints = physical_constraints.get('formation_constraints', {})
        if not formation_constraints:
            return formation_score
        
        formation_mechanism = formation_constraints.get('formation_mechanism', '')
        
        # 1. Storm Surge (风暴潮) - 绝对需要沿海
        if disaster_type == 23:  # Storm Surge
            # 首先检查是否为绝对内陆国家
            if self._is_absolutely_landlocked_country(latitude, longitude):
                return 0.0001  # 绝对内陆国家禁止Storm Surge
            
            # 检查区域级沿海接近性
            regional_coastal_score = self._assess_regional_coastal_access(latitude, longitude)
            if regional_coastal_score < 0.3:  # 该区域完全内陆
                return 0.0001  # 区域级内陆禁止
            
            # 精确距离海岸线检查
            coastal_distance_tier = self._get_coastal_distance_tier(latitude, longitude)
            if coastal_distance_tier == 'impossible':  # >30km
                return 0.0001
            elif coastal_distance_tier == 'low_risk':  # 10-30km
                formation_score *= 0.05
            elif coastal_distance_tier == 'medium_risk':  # 0-10km
                formation_score *= 0.8
        
        # 2. Tsunami (海啸) - 绝对需要海洋传播
        elif disaster_type == 10:  # Tsunami
            # 检查绝对内陆国家
            if self._is_absolutely_landlocked_country(latitude, longitude):
                return 0.0001  # 绝对内陆国家禁止Tsunami
            
            # 检查海洋连通性
            ocean_connectivity = self._check_ocean_connectivity(latitude, longitude)
            if not ocean_connectivity:
                return 0.0001  # 无海洋连通禁止Tsunami
            
            # 精确距离约束
            coastal_distance_tier = self._get_coastal_distance_tier(latitude, longitude)
            if coastal_distance_tier == 'impossible':  # >30km
                return 0.0001
            elif coastal_distance_tier == 'low_risk':  # 10-30km  
                formation_score *= 0.01
        
        # 3. Cyclone (气旋/台风) - 需要海洋形成或登陆路径
        elif disaster_type == 4:  # Cyclone
            # 检查绝对内陆国家
            if self._is_absolutely_landlocked_country(latitude, longitude):
                return 0.0001  # 绝对内陆国家禁止Cyclone
            
            latitude_range = formation_constraints.get('latitude_formation_range', (-40, 40))
            
            # 纬度约束
            if not (latitude_range[0] <= latitude <= latitude_range[1]):
                return 0.0001  # 台风形成纬度绝对约束
            
            # 海洋接近性约束 - 更严格
            coastal_distance_tier = self._get_coastal_distance_tier(latitude, longitude)
            if coastal_distance_tier == 'impossible':  # >30km内陆
                return 0.0001  # 绝对禁止深度内陆Cyclone
            elif coastal_distance_tier == 'low_risk':  # 10-30km
                formation_score *= 0.02
        
        # 4. Volcanic Eruption (火山喷发) - 绝对需要火山带
        elif disaster_type == 8:  # Volcanic
            # 检查是否为从未发生过火山灾害的国家
            possible_country_id = self._estimate_country_from_coordinates(latitude, longitude)
            if hasattr(self, '_non_volcanic_countries') and possible_country_id in self._non_volcanic_countries:
                return 0.0001  # 从未发生过火山灾害的国家绝对禁止
            
            # 使用精确的火山带检测
            in_volcanic_zone = self._refine_volcanic_zone_detection(latitude, longitude)
            if not in_volcanic_zone:
                return 0.0001  # 绝对禁止非火山带火山喷发
        
        # 5. Earthquake (地震) - 需要构造活动区
        elif disaster_type == 2:  # Earthquake
            tectonic_requirement = formation_constraints.get('tectonic_requirement_strength', 0.9)
            
            in_active_zone = self._check_tectonic_activity_zone(latitude, longitude)
            if not in_active_zone:
                formation_score *= (1 - tectonic_requirement)
        
        # 6. Heat Wave (热浪) - 需要大陆性气候
        elif disaster_type == 19:  # Heat Wave
            if abs(latitude) > 75:  # 极地地区
                formation_score *= 0.01  # 极地热浪不可能
            
            oceanic_penalty = formation_constraints.get('oceanic_region_penalty', 0.5)
            coastal_score = self._estimate_coastal_proximity_score(latitude, longitude)
            if coastal_score > 0.8:  # 海洋性气候区域
                formation_score *= oceanic_penalty
        
        # 7. Cold Wave (寒潮) - 需要极地冷空气通道
        elif disaster_type == 14:  # Cold Wave
            latitude_threshold = formation_constraints.get('latitude_threshold', 15)
            if abs(latitude) < latitude_threshold:  # 热带地区
                formation_score *= 0.01  # 热带寒潮不可能
        
        # 8. Drought (干旱) - 需要干旱气候倾向，强化气候微区识别
        elif disaster_type == 20:  # Drought
            # 使用气候微区进行精确约束
            climate_microzone = self._get_climate_microzone(latitude, longitude)
            
            if climate_microzone == 'oceanic':
                formation_score *= 0.01  # 海洋性气候极少干旱（如冰岛）
            elif climate_microzone == 'coastal_transition':
                formation_score *= 0.1   # 海陆过渡区域干旱概率大幅降低
            elif climate_microzone in ['continental_arid', 'continental_semi']:
                formation_score *= 1.8   # 大陆性干旱区域干旱概率增加
            
            # 特别约束：小岛国绝对不可能干旱
            coastal_score = self._estimate_coastal_proximity_score(latitude, longitude)
            if coastal_score > 0.95:  # 小岛屿（如汤加）
                return 0.0001  # 接近绝对禁止
        
        # 9. 技术/工业灾害 (67, 68, 66) - 需要工业基础设施
        elif disaster_type in [67, 68, 66]:  # Radiological, Transport, Biological Emergency
            # 检查国家工业化水平
            possible_country_id = self._estimate_country_from_coordinates(latitude, longitude)
            
            if hasattr(self, '_country_industrialization_scores'):
                country_industrialization = self._country_industrialization_scores.get(possible_country_id, 0.0)
                
                # 特殊约束：核辐射紧急事态需要特定核工业基础
                if disaster_type == 67:  # Radiological Emergency
                    nuclear_capable_countries = self._identify_nuclear_capable_countries()
                    if possible_country_id not in nuclear_capable_countries:
                        return 0.0001  # 非核技术国家绝对禁止核辐射灾害
                    
                    # 即使是核技术国家，小国核设施有限
                    if country_industrialization < 0.2:  # 提高核设施阈值
                        return 0.0001  # 核工业基础不足
                
                # 一般技术灾害约束
                elif country_industrialization < 0.01:  # 从未有技术灾害记录
                    return 0.0001  # 绝对禁止无工业基础的技术灾害
                elif country_industrialization < 0.05:  # 极低工业化
                    formation_score *= 0.01
                elif country_industrialization < 0.15:  # 低工业化
                    formation_score *= 0.1
            
            # 检查区域技术灾害密度
            if hasattr(self, '_technical_disaster_density_map'):
                grid_key = f"{int(latitude)}_{int(longitude)}"
                regional_density = self._technical_disaster_density_map.get(grid_key, 0)
                
                if regional_density == 0:  # 该区域从未有技术灾害
                    formation_score *= 0.05  # 严重惩罚
                elif regional_density < 2:  # 技术灾害稀少区域
                    formation_score *= 0.2
            
            # 检查灾害共现模式
            if hasattr(self, '_disaster_cooccurrence_patterns'):
                country_pattern = self._disaster_cooccurrence_patterns.get(possible_country_id, {})
                rural_indicator = country_pattern.get('rural_indicator', 0.0)
                
                if rural_indicator > 0.8:  # 纯农业国家特征
                    formation_score *= 0.02  # 农业国家技术灾害概率极低
        
        # 10. 水文灾害 (洪水、山洪、滑坡) - 需要降雨条件
        elif disaster_type in [5, 12, 13]:  # Landslide, Flood, Flash Flood
            # 检查是否为极干旱国家
            possible_country_id = self._estimate_country_from_coordinates(latitude, longitude)
            if hasattr(self, '_arid_countries') and possible_country_id in self._arid_countries:
                return 0.0001  # 极干旱国家禁止水文灾害
            
            # 季节性干旱检查 - 沙漠气候在旱季
            seasonal_aridity = self._assess_seasonal_aridity(latitude, longitude, 
                                                           formation_constraints.get('target_month', 9))
            if seasonal_aridity > 0.9:  # 极度干旱季节
                formation_score *= 0.01
        
        # 11. 农业相关灾害 (21, 62) - 需要农业区域
        elif disaster_type in [21, 62]:  # Food Insecurity, Insect Infestation
            non_agricultural_penalty = formation_constraints.get('non_agricultural_penalty', 0.2)
            
            # 城市化极高或沙漠地区农业灾害少
            if abs(latitude) > 50:  # 高纬度非农业区
                formation_score *= non_agricultural_penalty
        
        # 11. Landslide (24) - 需要坡度和降水/地震触发
        elif disaster_type == 24:  # Landslide
            # 形成原因: 坡度+触发因子(降水/地震)，需要山地丘陵
            if abs(latitude) > 70:  # 极地平原区域
                formation_score *= 0.3
            # 极平坦地区(某些平原)滑坡概率低
            flat_regions = [  # 大平原区域
                (35, 55, -110, -80),   # 北美大平原
                (45, 60, 20, 60),      # 东欧平原
                (-30, -10, -60, -40)   # 阿根廷潘帕斯草原
            ]
            for lat_min, lat_max, lng_min, lng_max in flat_regions:
                if lat_min <= latitude <= lat_max and lng_min <= longitude <= lng_max:
                    formation_score *= 0.4
                    break
        
        # 12. Fire (15) - 需要植被+干燥条件
        elif disaster_type == 15:  # Fire
            # 形成原因: 可燃植被+干燥+点火源
            if abs(latitude) > 70:  # 极地苔原区植被稀少
                formation_score *= 0.3
            # 沙漠地区植被覆盖不足
            arid_regions = [
                (15, 35, -20, 50),     # 撒哈拉-阿拉伯沙漠带
                (-30, -15, 110, 140),  # 澳洲内陆
                (25, 40, -120, -100)   # 美国西南沙漠
            ]
            for lat_min, lat_max, lng_min, lng_max in arid_regions:
                if lat_min <= latitude <= lat_max and lng_min <= longitude <= lng_max:
                    formation_score *= 0.4  # 极度干旱区植被不足
                    break
        
        # 13. Population Movement (5) - 需要人口流动通道
        elif disaster_type == 5:  # Population Movement
            # 形成原因: 冲突/灾害驱动+边界/交通，需要人口集中区
            if abs(latitude) > 65:  # 极地稀人区
                formation_score *= 0.1
            # 人口极稀少区域
            if self._estimate_coastal_proximity_score(latitude, longitude) < 0.2 and abs(latitude) > 45:
                formation_score *= 0.3  # 内陆高纬度人口稀少
        
        # 14. Complex Emergency (6) - 需要多重危机条件
        elif disaster_type == 6:  # Complex Emergency
            # 形成原因: 多重危机叠加，通常需要脆弱国家/冲突区
            # 极度稳定发达地区复合紧急事态概率低
            if abs(latitude) > 55:  # 高纬度发达稳定区
                formation_score *= 0.4
        
        # 15. Civil Unrest (7) - 需要人口密度和社会张力
        elif disaster_type == 7:  # Civil Unrest
            # 形成原因: 社会矛盾+人口集中，需要城市化区域
            if abs(latitude) > 60:  # 极地稀人区
                formation_score *= 0.2
            # 极度偏远农村地区社会动乱相对少
            rural_score = self._estimate_coastal_proximity_score(latitude, longitude)
            if rural_score < 0.2 and abs(latitude) > 50:
                formation_score *= 0.5
        
        # 16. Mass Movement (11) - 需要不稳定斜坡
        elif disaster_type == 11:  # Mass Movement (类似Landslide)
            # 形成原因: 重力+不稳定斜坡，需要地形起伏
            if abs(latitude) > 70:  # 极地平原
                formation_score *= 0.3
            # 检查是否在平坦区域
            if self._is_in_flat_plains(latitude, longitude):
                formation_score *= 0.2
        
        # 17. Other (13) - 通用灾害，地理约束相对宽松
        elif disaster_type == 13:  # Other
            # 其他类型灾害，保持较高的地理灵活性
            formation_score *= 0.8  # 轻微降低但不严格约束
        
        return min(formation_score, 1.0)
    
    def _is_in_flat_plains(self, lat: float, lng: float) -> bool:
        """检查是否在大平原区域"""
        flat_plains = [
            (35, 55, -110, -80),   # 北美大平原
            (45, 60, 20, 60),      # 东欧平原
            (-30, -10, -60, -40),  # 潘帕斯草原
            (20, 40, 100, 130),    # 华北平原
            (-10, 10, 10, 30)      # 非洲萨赫勒平原
        ]
        
        for lat_min, lat_max, lng_min, lng_max in flat_plains:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        return False
    
    def _check_volcanic_zone_proximity(self, lat: float, lng: float) -> bool:
        """检查是否在火山活跃带附近"""
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
    
    def _check_tectonic_activity_zone(self, lat: float, lng: float) -> bool:
        """检查是否在构造活动带"""
        tectonic_zones = [
            # 环太平洋地震带
            (-60, 60, 120, 180),
            (-60, 60, -180, -60),
            # 欧亚地震带
            (25, 45, -10, 100),
            # 中大西洋脊
            (-60, 60, -40, -10),
            # 东非大裂谷
            (-35, 20, 25, 50),
            # 地中海地震带
            (30, 45, 0, 40)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in tectonic_zones:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        return False
    
    def _is_absolutely_landlocked_country(self, lat: float, lng: float) -> bool:
        """基于历史数据识别绝对内陆国家 - 数据驱动方法"""
        if not hasattr(self, '_landlocked_countries_cache'):
            self._landlocked_countries_cache = self._identify_landlocked_countries_from_history()
        
        # 根据坐标判断可能的国家，然后检查是否在内陆国家列表中
        possible_country_id = self._estimate_country_from_coordinates(lat, lng)
        return possible_country_id in self._landlocked_countries_cache
    
    def _identify_landlocked_countries_from_history(self) -> set:
        """从历史数据识别从未发生过海洋灾害的国家"""
        if not hasattr(self, 'historical_data') or self.historical_data is None:
            return set()
        
        # 海洋相关灾害类型
        marine_disaster_types = {4, 10, 23}  # Cyclone, Tsunami, Storm Surge
        
        # 所有国家
        all_countries = set(self.historical_data['country_id'].dropna().astype(int))
        
        # 有海洋灾害记录的国家
        marine_countries = set()
        for disaster_type in marine_disaster_types:
            countries_with_marine = set(
                self.historical_data[
                    self.historical_data['disaster_type_id'] == disaster_type
                ]['country_id'].dropna().astype(int)
            )
            marine_countries.update(countries_with_marine)
        
        # 从未有海洋灾害的国家 = 可能的内陆国家
        potential_landlocked = all_countries - marine_countries
        
        self.logger.info(f"识别出{len(potential_landlocked)}个可能的内陆国家，{len(marine_countries)}个沿海国家")
        return potential_landlocked
    
    def _estimate_country_from_coordinates(self, lat: float, lng: float) -> int:
        """根据坐标估算国家ID - 简化版本"""
        # 基于已知的地理区域粗略估算
        # 这里使用districts_map进行反向查找
        if hasattr(self, 'districts_map'):
            min_distance = float('inf')
            closest_country = None
            
            for district_id, district_info in self.districts_map.items():
                d_lat = district_info.get('latitude', 0)
                d_lng = district_info.get('longitude', 0)
                
                if d_lat == 0 and d_lng == 0:
                    continue
                
                # 计算简单欧几里得距离
                distance = ((lat - d_lat)**2 + (lng - d_lng)**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_country = district_info.get('country_id')
            
            return closest_country if closest_country else 1
        
        return 1  # 默认值
    
    def _assess_regional_coastal_access(self, lat: float, lng: float) -> float:
        """评估区域级沿海接近性 - 检查该区域是否有任何沿海地带"""
        # 定义区域范围 (以当前点为中心的3度x3度区域)
        region_size = 1.5  # 度数
        region_bounds = {
            'lat_min': lat - region_size,
            'lat_max': lat + region_size,
            'lng_min': lng - region_size,
            'lng_max': lng + region_size
        }
        
        # 检查该区域内是否有任何已知沿海districts
        if hasattr(self, 'districts_map'):
            coastal_districts_in_region = 0
            total_districts_in_region = 0
            
            for district_id, district_info in self.districts_map.items():
                d_lat = district_info.get('latitude', 0)
                d_lng = district_info.get('longitude', 0)
                
                if (region_bounds['lat_min'] <= d_lat <= region_bounds['lat_max'] and
                    region_bounds['lng_min'] <= d_lng <= region_bounds['lng_max']):
                    total_districts_in_region += 1
                    
                    # 检查该district是否沿海
                    if self._estimate_coastal_proximity_score(d_lat, d_lng) > 0.7:
                        coastal_districts_in_region += 1
            
            if total_districts_in_region > 0:
                return coastal_districts_in_region / total_districts_in_region
        
        # 后备：使用现有的点级沿海检查
        return self._estimate_coastal_proximity_score(lat, lng)
    
    def _get_coastal_distance_tier(self, lat: float, lng: float) -> str:
        """基于坐标获取海岸距离等级"""
        coastal_score = self._estimate_coastal_proximity_score(lat, lng)
        
        # 数据驱动的距离分级
        if coastal_score > 0.9:
            return 'high_risk'      # 0-10km 高风险
        elif coastal_score > 0.6:
            return 'medium_risk'    # 10-30km 中风险  
        elif coastal_score > 0.3:
            return 'low_risk'       # 30-100km 低风险
        else:
            return 'impossible'     # >100km 不可能
    
    def _check_ocean_connectivity(self, lat: float, lng: float) -> bool:
        """检查海洋连通性 - 区分真正海洋vs内陆湖泊"""
        # 主要海洋区域定义
        major_oceans = [
            # 太平洋
            (-60, 60, 100, 180),
            (-60, 60, -180, -100),
            # 大西洋  
            (-60, 70, -80, 20),
            # 印度洋
            (-60, 30, 20, 100),
            # 北冰洋
            (60, 90, -180, 180)
        ]
        
        # 内陆湖泊/海域 (需要排除)
        inland_water_bodies = [
            # 里海
            (35, 47, 46, 55),
            # 咸海
            (43, 46, 58, 62),
            # 贝加尔湖  
            (51, 56, 103, 110),
            # 维多利亚湖
            (-3, 0, 31, 35)
        ]
        
        # 检查是否在内陆水体附近 (应排除)
        for lat_min, lat_max, lng_min, lng_max in inland_water_bodies:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return False  # 内陆水体不支持海洋灾害
        
        # 检查是否接近主要海洋
        extended_lat_lng = 2.0  # 扩展2度范围检查海洋接近性
        for lat_min, lat_max, lng_min, lng_max in major_oceans:
            if (lat_min - extended_lat_lng <= lat <= lat_max + extended_lat_lng and 
                lng_min - extended_lat_lng <= lng <= lng_max + extended_lat_lng):
                return True
        
        return False
    
    def _build_climate_disaster_exclusions(self) -> None:
        """基于历史数据建立气候-灾害排斥约束"""
        if not hasattr(self, 'historical_data') or self.historical_data is None:
            return
        
        # 识别极干旱国家 - 从未发生过水文灾害的国家
        hydrological_disasters = {5, 12, 13}  # Landslide, Flood, Flash Flood
        self._arid_countries = self._identify_countries_without_disaster_types(hydrological_disasters)
        
        # 识别非火山国家 - 从未发生过火山灾害的国家  
        volcanic_disasters = {8}  # Volcanic Eruption
        self._non_volcanic_countries = self._identify_countries_without_disaster_types(volcanic_disasters)
        
        # 识别极地/高纬度国家 - 从未发生过热带灾害的国家
        tropical_disasters = {4, 19, 20}  # Cyclone, Heat Wave, Drought
        self._polar_countries = self._identify_countries_without_disaster_types(tropical_disasters)
        
        self.logger.info(f"识别出极干旱国家{len(self._arid_countries)}个，非火山国家{len(self._non_volcanic_countries)}个，极地国家{len(self._polar_countries)}个")
    
    def _build_technical_infrastructure_constraints(self) -> None:
        """基于历史数据建立技术设施约束"""
        if not hasattr(self, 'historical_data') or self.historical_data is None:
            return
        
        # 技术设施相关灾害类型
        technical_disasters = {66, 67, 68}  # Biological Emergency, Radiological Emergency, Transport Accident
        
        # 1. 计算国家级工业化评分
        self._country_industrialization_scores = self._calculate_country_industrialization_scores(technical_disasters)
        
        # 2. 识别技术灾害密集区域
        self._technical_disaster_density_map = self._calculate_regional_technical_density()
        
        # 3. 识别灾害类型共现模式
        self._disaster_cooccurrence_patterns = self._analyze_disaster_cooccurrence_patterns()
        
        # 4. 建立地形约束
        self._build_terrain_constraints()
        
        self.logger.info(f"建立技术设施约束：{len(self._country_industrialization_scores)}个国家工业化评分，{len(self._technical_disaster_density_map)}个区域密度")
    
    def _calculate_country_industrialization_scores(self, technical_disasters: set) -> Dict[int, float]:
        """计算国家工业化水平评分"""
        country_scores = {}
        
        # 获取所有国家
        all_countries = set(self.historical_data['country_id'].dropna().astype(int))
        
        for country_id in all_countries:
            country_data = self.historical_data[self.historical_data['country_id'] == country_id]
            
            # 技术灾害事件统计
            tech_disaster_count = 0
            tech_disaster_types = set()
            
            for disaster_type in technical_disasters:
                country_type_events = country_data[country_data['disaster_type_id'] == disaster_type]
                tech_disaster_count += len(country_type_events)
                if len(country_type_events) > 0:
                    tech_disaster_types.add(disaster_type)
            
            # 总灾害事件数
            total_disasters = len(country_data)
            
            # 计算工业化评分
            if total_disasters > 0:
                # 技术灾害占比
                tech_ratio = tech_disaster_count / total_disasters
                
                # 技术灾害种类多样性 (0-1)
                tech_diversity = len(tech_disaster_types) / len(technical_disasters)
                
                # 综合工业化评分 (0-1)
                industrialization_score = (tech_ratio * 0.7 + tech_diversity * 0.3)
                
                country_scores[country_id] = industrialization_score
            else:
                country_scores[country_id] = 0.0
        
        return country_scores
    
    def _calculate_regional_technical_density(self) -> Dict[str, float]:
        """计算区域技术灾害密度"""
        if not hasattr(self, 'districts_map'):
            return {}
        
        regional_density = {}
        technical_disasters = {66, 67, 68}
        
        # 按1度x1度网格计算技术灾害密度
        for _, disaster_row in self.historical_data.iterrows():
            if disaster_row.get('disaster_type_id') in technical_disasters:
                lat = disaster_row.get('latitude', 0)
                lng = disaster_row.get('longitude', 0)
                
                if lat != 0 or lng != 0:
                    # 网格化坐标 (1度精度)
                    grid_key = f"{int(lat)}_{int(lng)}"
                    regional_density[grid_key] = regional_density.get(grid_key, 0) + 1
        
        return regional_density
    
    def _analyze_disaster_cooccurrence_patterns(self) -> Dict[str, Dict[str, float]]:
        """分析灾害类型共现模式"""
        cooccurrence = {}
        technical_disasters = {66, 67, 68}
        agricultural_disasters = {21, 62}  # Food Insecurity, Insect Infestation
        
        # 按国家分析灾害共现
        for country_id in self.historical_data['country_id'].dropna().unique():
            country_data = self.historical_data[self.historical_data['country_id'] == country_id]
            
            disaster_types_in_country = set(country_data['disaster_type_id'].dropna().astype(int))
            
            # 技术灾害共现分析
            tech_disasters_present = disaster_types_in_country.intersection(technical_disasters)
            agri_disasters_present = disaster_types_in_country.intersection(agricultural_disasters)
            
            country_pattern = {
                'technical_diversity': len(tech_disasters_present) / len(technical_disasters),
                'agricultural_ratio': len(agri_disasters_present) / len(agricultural_disasters),
                'industrial_indicator': 1.0 if len(tech_disasters_present) >= 2 else 0.0,
                'rural_indicator': 1.0 if len(agri_disasters_present) >= 1 and len(tech_disasters_present) == 0 else 0.0
            }
            
            cooccurrence[int(country_id)] = country_pattern
        
        return cooccurrence
    
    def _build_terrain_constraints(self) -> None:
        """基于历史数据建立地形约束"""
        if not hasattr(self, 'historical_data') or self.historical_data is None:
            return
        
        # 1. 基于滑坡事件分布推断地形复杂度
        self._terrain_complexity_map = self._analyze_terrain_from_landslides()
        
        # 2. 基于洪水模式识别地形类型
        self._flood_terrain_patterns = self._analyze_flood_terrain_patterns()
        
        # 3. 识别平坦地区（无滑坡/雪崩历史）
        self._flat_terrain_countries = self._identify_flat_terrain_countries()
        
        # 4. 建立海拔代理指标
        self._elevation_proxy_map = self._calculate_elevation_proxy_indicators()
        
        # 5. 建立气候微区识别
        self._climate_microzone_map = self._build_climate_microzone_identification()
        
        self.logger.info(f"建立地形约束：{len(self._terrain_complexity_map)}个地区地形复杂度，{len(self._flat_terrain_countries)}个平坦地形国家")
    
    def _analyze_terrain_from_landslides(self) -> Dict[str, float]:
        """从滑坡事件分布推断地形复杂度"""
        terrain_complexity = {}
        landslide_disasters = {5}  # Landslide
        
        # 统计各地区滑坡事件密度
        for _, disaster_row in self.historical_data.iterrows():
            if disaster_row.get('disaster_type_id') in landslide_disasters:
                lat = disaster_row.get('latitude', 0)
                lng = disaster_row.get('longitude', 0)
                
                if lat != 0 or lng != 0:
                    # 网格化坐标 (2度精度用于地形分析)
                    grid_key = f"{int(lat//2)*2}_{int(lng//2)*2}"
                    terrain_complexity[grid_key] = terrain_complexity.get(grid_key, 0) + 1
        
        # 归一化为地形复杂度评分
        if terrain_complexity:
            max_complexity = max(terrain_complexity.values())
            if max_complexity > 0:
                for grid_key in terrain_complexity:
                    terrain_complexity[grid_key] = terrain_complexity[grid_key] / max_complexity
        
        return terrain_complexity
    
    def _analyze_flood_terrain_patterns(self) -> Dict[str, Dict[str, float]]:
        """分析洪水模式识别地形类型"""
        flood_patterns = {}
        flood_disasters = {12, 27}  # Flood, Flash Flood
        
        # 按国家统计洪水类型分布
        for country_id in self.historical_data['country_id'].dropna().unique():
            country_data = self.historical_data[
                (self.historical_data['country_id'] == country_id) & 
                (self.historical_data['disaster_type_id'].isin(flood_disasters))
            ]
            
            if len(country_data) > 0:
                flood_counts = country_data['disaster_type_id'].value_counts()
                total_floods = len(country_data)
                
                pattern = {
                    'regular_flood_ratio': flood_counts.get(12, 0) / total_floods,  # 常规洪水 = 平原地形
                    'flash_flood_ratio': flood_counts.get(27, 0) / total_floods,    # 山洪 = 山地地形
                    'terrain_type': 'mountainous' if flood_counts.get(27, 0) > flood_counts.get(12, 0) else 'plains'
                }
                
                flood_patterns[int(country_id)] = pattern
        
        return flood_patterns
    
    def _identify_flat_terrain_countries(self) -> set:
        """识别平坦地形国家 - 从未发生过地形相关灾害的国家"""
        terrain_disasters = {5, 54}  # Landslide, Avalanche
        
        # 所有国家
        all_countries = set(self.historical_data['country_id'].dropna().astype(int))
        
        # 有地形灾害记录的国家
        countries_with_terrain_disasters = set()
        for disaster_type in terrain_disasters:
            countries_with_type = set(
                self.historical_data[
                    self.historical_data['disaster_type_id'] == disaster_type
                ]['country_id'].dropna().astype(int)
            )
            countries_with_terrain_disasters.update(countries_with_type)
        
        # 从未有地形灾害的国家 = 平坦地形国家
        flat_countries = all_countries - countries_with_terrain_disasters
        return flat_countries
    
    def _calculate_elevation_proxy_indicators(self) -> Dict[int, float]:
        """计算海拔代理指标 - 基于气候和灾害模式"""
        elevation_proxy = {}
        
        # 雪崩和高海拔相关灾害用于推断海拔
        high_elevation_disasters = {54, 14}  # Avalanche, Cold Wave
        
        for country_id in self.historical_data['country_id'].dropna().unique():
            country_data = self.historical_data[self.historical_data['country_id'] == country_id]
            
            # 计算高海拔指标
            high_alt_events = len(country_data[country_data['disaster_type_id'].isin(high_elevation_disasters)])
            total_events = len(country_data)
            
            if total_events > 0:
                elevation_indicator = high_alt_events / total_events
                
                # 纬度修正 - 高纬度地区可能低海拔也有寒潮
                avg_lat = country_data['latitude'].mean() if 'latitude' in country_data.columns else 0
                if abs(avg_lat) > 50:  # 高纬度地区
                    elevation_indicator *= 0.7
                
                elevation_proxy[int(country_id)] = min(elevation_indicator, 1.0)
        
        return elevation_proxy
    
    def _identify_nuclear_capable_countries(self) -> set:
        """识别具备核技术能力的国家 - 基于核辐射灾害历史和工业化水平"""
        nuclear_countries = set()
        
        # 1. 有核辐射紧急事态历史记录的国家
        radiological_disasters = {67}  # Radiological Emergency
        countries_with_nuclear_history = set(
            self.historical_data[
                self.historical_data['disaster_type_id'].isin(radiological_disasters)
            ]['country_id'].dropna().astype(int)
        )
        nuclear_countries.update(countries_with_nuclear_history)
        
        # 2. 高工业化且有多种技术灾害的国家
        if hasattr(self, '_country_industrialization_scores'):
            for country_id, industrial_score in self._country_industrialization_scores.items():
                if industrial_score >= 0.3:  # 高工业化国家
                    # 检查技术灾害多样性
                    country_data = self.historical_data[self.historical_data['country_id'] == country_id]
                    tech_disaster_types = set(country_data[
                        country_data['disaster_type_id'].isin({66, 67, 68})
                    ]['disaster_type_id'].dropna().astype(int))
                    
                    # 有2种以上技术灾害的高工业化国家推定为核技术国家
                    if len(tech_disaster_types) >= 2:
                        nuclear_countries.add(country_id)
        
        # 3. 基于已知核技术国家的地理特征推断（数据驱动）
        known_nuclear_regions = []
        for country_id in nuclear_countries:
            country_data = self.historical_data[self.historical_data['country_id'] == country_id]
            if len(country_data) > 0:
                avg_lat = country_data['latitude'].mean()
                avg_lng = country_data['longitude'].mean()
                if not (pd.isna(avg_lat) or pd.isna(avg_lng)):
                    known_nuclear_regions.append((avg_lat, avg_lng))
        
        # 基于核技术国家的地理聚集推断其他可能的核技术国家
        for country_id in self.historical_data['country_id'].dropna().unique():
            if country_id not in nuclear_countries:
                country_data = self.historical_data[self.historical_data['country_id'] == country_id]
                if len(country_data) > 0:
                    avg_lat = country_data['latitude'].mean()
                    avg_lng = country_data['longitude'].mean()
                    
                    # 检查是否接近已知核技术区域且有足够工业化基础
                    industrial_score = self._country_industrialization_scores.get(country_id, 0.0)
                    if industrial_score >= 0.25:  # 高工业化
                        for nuclear_lat, nuclear_lng in known_nuclear_regions:
                            distance = ((avg_lat - nuclear_lat)**2 + (avg_lng - nuclear_lng)**2)**0.5
                            if distance < 15:  # 地理接近核技术区域
                                nuclear_countries.add(country_id)
                                break
        
        return nuclear_countries
    
    def _assess_terrain_constraints(self, disaster_type: int, latitude: float, longitude: float) -> float:
        """评估地形约束 - 基于历史数据推断的地形特征"""
        terrain_score = 1.0
        
        # 1. 滑坡地形约束 - 需要复杂地形
        if disaster_type == 5:  # Landslide
            # 检查是否为平坦地形国家
            possible_country_id = self._estimate_country_from_coordinates(latitude, longitude)
            if hasattr(self, '_flat_terrain_countries') and possible_country_id in self._flat_terrain_countries:
                return 0.0001  # 从未有滑坡记录的平坦国家绝对禁止
            
            # 检查地区地形复杂度
            grid_key = f"{int(latitude//2)*2}_{int(longitude//2)*2}"
            if hasattr(self, '_terrain_complexity_map'):
                complexity = self._terrain_complexity_map.get(grid_key, 0)
                if complexity < 0.1:  # 地形复杂度极低
                    terrain_score *= 0.01
                else:
                    terrain_score *= (0.5 + 0.5 * complexity)  # 复杂度越高滑坡概率越大
        
        # 2. 雪崩地形约束 - 需要山地+高海拔
        elif disaster_type == 54:  # Avalanche  
            possible_country_id = self._estimate_country_from_coordinates(latitude, longitude)
            
            # 检查平坦地形国家
            if hasattr(self, '_flat_terrain_countries') and possible_country_id in self._flat_terrain_countries:
                return 0.0001  # 平坦国家绝对禁止雪崩
            
            # 检查海拔代理指标
            if hasattr(self, '_elevation_proxy_map'):
                elevation_indicator = self._elevation_proxy_map.get(possible_country_id, 0)
                if elevation_indicator < 0.05:  # 海拔极低
                    terrain_score *= 0.001
                else:
                    terrain_score *= (0.3 + 0.7 * elevation_indicator)
            
            # 纬度-海拔交互约束
            if abs(latitude) < 30 and elevation_indicator < 0.3:  # 低纬度+低海拔
                terrain_score *= 0.001  # 热带低地不可能雪崩
        
        # 3. 洪水地形约束 - 基于地形类型
        elif disaster_type in [12, 27]:  # Flood, Flash Flood
            possible_country_id = self._estimate_country_from_coordinates(latitude, longitude)
            
            if hasattr(self, '_flood_terrain_patterns') and possible_country_id in self._flood_terrain_patterns:
                pattern = self._flood_terrain_patterns[possible_country_id]
                
                if disaster_type == 12:  # 常规洪水 - 偏好平原
                    if pattern['terrain_type'] == 'mountainous':
                        terrain_score *= 0.6  # 山地地区常规洪水概率降低
                elif disaster_type == 27:  # 山洪 - 偏好山地
                    if pattern['terrain_type'] == 'plains':
                        terrain_score *= 0.7  # 平原地区山洪概率降低
        
        # 4. 干旱地形约束 - 内陆平原/高原更易干旱
        elif disaster_type == 20:  # Drought
            # 沿海地区海洋调节作用
            coastal_score = self._estimate_coastal_proximity_score(latitude, longitude)
            if coastal_score > 0.8:  # 强海洋性气候
                terrain_score *= 0.4  # 海洋性气候干旱概率降低
        
        return terrain_score
    
    def _check_absolute_physical_impossibilities(self, disaster_type: int, latitude: float, longitude: float) -> bool:
        """检查绝对物理不可能的灾害-地理组合"""
        
        # 1. 极地地区热浪 (>70度纬度)
        if disaster_type == 19 and abs(latitude) > 70:
            return True  # 绝对不可能
        
        # 2. 赤道地区寒潮/雪崩 (<15度纬度)
        if disaster_type in [14, 54] and abs(latitude) < 15:
            return True  # 热带地区绝对不可能寒潮/雪崩
        
        # 3. 沙漠中心洪水 - 极干旱区域不可能洪水
        if disaster_type in [12, 27]:  # Flood types
            # 撒哈拉沙漠中心
            if 15 <= latitude <= 30 and -5 <= longitude <= 35:
                # 撒哈拉核心区域
                if 18 <= latitude <= 27 and 0 <= longitude <= 25:
                    return True
            
            # 阿拉伯沙漠核心
            elif 20 <= latitude <= 30 and 40 <= longitude <= 55:
                return True
            
            # 澳洲内陆沙漠
            elif -30 <= latitude <= -20 and 125 <= longitude <= 140:
                return True
        
        # 4. 海洋性小岛干旱 - 小岛屿不可能干旱
        if disaster_type == 20:  # Drought
            coastal_score = self._estimate_coastal_proximity_score(latitude, longitude)
            if coastal_score > 0.95 and abs(latitude) < 60:  # 热带/温带小岛
                return True
        
        # 5. 地中海气旋 - 地中海不形成热带气旋
        if disaster_type == 4:  # Cyclone
            # 地中海区域
            if 30 <= latitude <= 45 and 0 <= longitude <= 40:
                return True
            
            # 北大西洋高纬度
            if latitude > 45 and -60 <= longitude <= 0:
                return True
        
        # 6. 平原滑坡 - 极平坦地区不可能滑坡
        if disaster_type == 5:  # Landslide
            possible_country_id = self._estimate_country_from_coordinates(latitude, longitude)
            if hasattr(self, '_flat_terrain_countries') and possible_country_id in self._flat_terrain_countries:
                # 额外检查：荷兰、孟加拉等极平坦国家
                flat_regions = [
                    (51, 54, 3, 8),    # 荷兰
                    (20, 27, 88, 93),  # 孟加拉
                    (44, 54, -104, -95), # 美国大平原
                ]
                
                for lat_min, lat_max, lng_min, lng_max in flat_regions:
                    if lat_min <= latitude <= lat_max and lng_min <= longitude <= lng_max:
                        return True
        
        return False  # 不是绝对不可能的组合
    
    def _build_climate_microzone_identification(self) -> Dict[str, str]:
        """建立气候微区识别 - 区分海洋性vs大陆性气候"""
        climate_microzone = {}
        
        # 基于历史干旱事件识别大陆性气候区域
        drought_disasters = {20}  # Drought
        continental_climate_indicators = {}
        
        # 统计各网格区域的干旱频率
        for _, disaster_row in self.historical_data.iterrows():
            if disaster_row.get('disaster_type_id') in drought_disasters:
                lat = disaster_row.get('latitude', 0)
                lng = disaster_row.get('longitude', 0)
                
                if lat != 0 or lng != 0:
                    grid_key = f"{int(lat//3)*3}_{int(lng//3)*3}"  # 3度网格
                    continental_climate_indicators[grid_key] = continental_climate_indicators.get(grid_key, 0) + 1
        
        # 基于沿海接近度和干旱频率分类气候类型
        for grid_key, drought_count in continental_climate_indicators.items():
            parts = grid_key.split('_')
            grid_lat, grid_lng = int(parts[0]), int(parts[1])
            
            coastal_score = self._estimate_coastal_proximity_score(grid_lat, grid_lng)
            
            # 气候类型判断
            if coastal_score < 0.3 and drought_count >= 2:
                climate_microzone[grid_key] = 'continental_arid'      # 大陆性干旱
            elif coastal_score < 0.5 and drought_count >= 1:
                climate_microzone[grid_key] = 'continental_semi'     # 大陆性半干旱
            elif coastal_score > 0.8:
                climate_microzone[grid_key] = 'oceanic'              # 海洋性
            elif coastal_score > 0.5:
                climate_microzone[grid_key] = 'coastal_transition'   # 海陆过渡
            else:
                climate_microzone[grid_key] = 'continental_humid'    # 大陆性湿润
        
        return climate_microzone
    
    def _get_climate_microzone(self, latitude: float, longitude: float) -> str:
        """获取气候微区类型"""
        grid_key = f"{int(latitude//3)*3}_{int(longitude//3)*3}"
        
        if hasattr(self, '_climate_microzone_map'):
            return self._climate_microzone_map.get(grid_key, 'temperate_default')
        
        # 后备分类
        coastal_score = self._estimate_coastal_proximity_score(latitude, longitude)
        if coastal_score > 0.8:
            return 'oceanic'
        elif coastal_score < 0.3:
            return 'continental_arid'
        else:
            return 'coastal_transition'
    
    def _identify_countries_without_disaster_types(self, disaster_types: set) -> set:
        """识别从未发生过特定类型灾害的国家"""
        if not hasattr(self, 'historical_data'):
            return set()
        
        # 所有国家
        all_countries = set(self.historical_data['country_id'].dropna().astype(int))
        
        # 有这些灾害记录的国家
        countries_with_disasters = set()
        for disaster_type in disaster_types:
            countries_with_type = set(
                self.historical_data[
                    self.historical_data['disaster_type_id'] == disaster_type
                ]['country_id'].dropna().astype(int)
            )
            countries_with_disasters.update(countries_with_type)
        
        # 从未有这些灾害的国家
        countries_without = all_countries - countries_with_disasters
        return countries_without
    
    def _refine_volcanic_zone_detection(self, lat: float, lng: float) -> bool:
        """精确火山带检测 - 区分地震带vs火山带"""
        # 真正的活火山带(更精确的范围)
        active_volcanic_zones = [
            # 印度尼西亚火山弧
            (-10, 6, 95, 141),
            # 日本火山弧  
            (24, 46, 129, 146),
            # 菲律宾火山弧
            (4, 19, 119, 127),
            # 安第斯山脉火山带
            (-56, 11, -78, -66),
            # 中美洲火山弧
            (8, 22, -92, -83),
            # 意大利火山区
            (36, 46, 12, 16),
            # 冰岛火山带
            (63, 67, -25, -13),
            # 新西兰火山区
            (-47, -34, 166, 179),
            # 夏威夷火山带
            (18, 22, -160, -154),
            # 阿拉斯加阿留申火山弧
            (51, 65, -180, -130),
            # 喀斯喀特火山带
            (40, 49, -125, -121),
            # 东非裂谷火山带
            (-15, 15, 35, 42)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in active_volcanic_zones:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        return False
    
    def _assess_seasonal_aridity(self, lat: float, lng: float, target_month: int) -> float:
        """评估季节性干旱程度 - 基于地理气候特征"""
        aridity_score = 0.0
        
        # 1. 基于纬度和经度的气候分类
        abs_lat = abs(lat)
        
        # 沙漠气候区域检测
        major_desert_regions = [
            # 撒哈拉沙漠
            (10, 35, -20, 50),
            # 阿拉伯沙漠  
            (15, 35, 35, 60),
            # 中亚沙漠
            (35, 50, 50, 80),
            # 澳洲内陆
            (-35, -10, 115, 150),
            # 美国西南沙漠
            (25, 40, -120, -100),
            # 智利阿塔卡马
            (-30, -15, -75, -65)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in major_desert_regions:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                aridity_score += 0.8  # 沙漠区域基础干旱
                break
        
        # 2. 季节性干旱调整
        # 北半球9月：夏末秋初，多数地区相对干燥
        if lat > 0:  # 北半球
            if target_month in [7, 8, 9]:  # 夏末秋初
                aridity_score += 0.3
            elif abs_lat > 30 and target_month in [6, 7, 8]:  # 温带夏季
                aridity_score += 0.2
        else:  # 南半球
            if target_month in [12, 1, 2]:  # 南半球夏季
                aridity_score += 0.2
        
        # 3. 极干旱气候带额外调整
        if abs_lat > 20 and abs_lat < 35:  # 副热带高压带
            aridity_score += 0.4
        
        return min(aridity_score, 1.0)
    
    def _assess_seasonal_feasibility(self, disaster_type: int, latitude: float, longitude: float,
                                   target_month: int, physical_constraints: Dict[str, Any]) -> float:
        """评估季节性物理可行性 - 季节-灾害-地理三维约束"""
        seasonal_score = 1.0
        
        seasonal_constraints = physical_constraints.get('seasonal_constraints', {})
        if not seasonal_constraints:
            return seasonal_score
        
        # 1. 月份禁止约束
        monthly_patterns = seasonal_constraints.get('monthly_patterns', {})
        forbidden_months = monthly_patterns.get('forbidden_months', [])
        peak_months = monthly_patterns.get('peak_months', [])
        
        if target_month in forbidden_months:
            seasonal_score *= 0.05  # 历史从未发生的月份强约束
        elif target_month in peak_months:
            seasonal_score *= 1.2  # 高发月份增强
        
        # 2. 地理-季节交互约束
        geo_seasonal = seasonal_constraints.get('geo_seasonal_interactions', {})
        current_climate = self._get_climate_zone_from_lat(latitude)
        
        if current_climate in geo_seasonal:
            climate_pattern = geo_seasonal[current_climate]
            climate_forbidden = climate_pattern.get('forbidden_months', [])
            climate_preferred = climate_pattern.get('preferred_months', [])
            
            if target_month in climate_forbidden:
                seasonal_score *= 0.1  # 该气候带该月份历史极少
            elif target_month in climate_preferred:
                seasonal_score *= 1.1  # 该气候带该月份历史偏好
        
        # 3. 反季节强约束检查
        anti_seasonal = seasonal_constraints.get('anti_seasonal_constraints', {})
        
        # 台风季节约束
        if disaster_type in [4, 23] and 'typhoon_season_constraints' in anti_seasonal:
            typhoon_constraints = anti_seasonal['typhoon_season_constraints']
            never_months = typhoon_constraints.get('never_occurred_months', [])
            rare_months = typhoon_constraints.get('extremely_rare_months', [])
            
            if target_month in never_months:
                seasonal_score *= 0.01  # 从未发生过的月份
            elif target_month in rare_months:
                seasonal_score *= 0.2   # 极少发生的月份
        
        # 洪水干季约束
        elif disaster_type in [12, 27] and 'flood_dry_season_constraints' in anti_seasonal:
            flood_constraints = anti_seasonal['flood_dry_season_constraints']
            dry_months = flood_constraints.get('dry_season_months', [])
            
            if target_month in dry_months:
                seasonal_score *= 0.3  # 干季洪水概率降低
        
        # 干旱雨季约束
        elif disaster_type == 20 and 'drought_wet_season_constraints' in anti_seasonal:
            drought_constraints = anti_seasonal['drought_wet_season_constraints']
            wet_months = drought_constraints.get('wet_season_months', [])
            
            if target_month in wet_months:
                seasonal_score *= 0.2  # 雨季干旱概率大幅降低
        
        # 火灾湿润季约束
        elif disaster_type == 15 and 'fire_humid_season_constraints' in anti_seasonal:
            fire_constraints = anti_seasonal['fire_humid_season_constraints']
            humid_months = fire_constraints.get('humid_season_months', [])
            
            if target_month in humid_months:
                seasonal_score *= 0.4  # 湿润季火灾概率降低
        
        return min(seasonal_score, 1.0)
    
    def _estimate_coastal_proximity_score(self, lat: float, lng: float) -> float:
        """估算沿海接近度评分"""
        if lat == 0 and lng == 0:
            return 0.5
        
        # 小岛屿/群岛区域 - 高沿海评分
        island_regions = [
            (-30, 30, 120, 180),   # 太平洋
            (-30, 30, -180, -120), # 太平洋西部  
            (10, 25, -85, -60),    # 加勒比海
            (-30, 0, 40, 100)      # 印度洋岛屿
        ]
        
        for lat_min, lat_max, lng_min, lng_max in island_regions:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return 0.95
        
        # 大陆内陆区域 - 低沿海评分
        interior_regions = [
            (-10, 10, 10, 35),     # 非洲内陆
            (35, 55, 60, 120),     # 亚洲内陆
            (35, 55, -110, -80),   # 北美内陆
            (-20, 5, -70, -50)     # 南美内陆
        ]
        
        for lat_min, lat_max, lng_min, lng_max in interior_regions:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return 0.1
        
        return 0.6  # 默认中等沿海可能性
    
    def _get_climate_zone_from_lat(self, latitude: float) -> str:
        """根据纬度判断气候带"""
        abs_lat = abs(latitude)
        if abs_lat > 66.5:
            return 'polar'
        elif abs_lat > 23.5:
            return 'temperate'
        else:
            return 'tropical'
    
    def _calculate_prediction_confidence(self, country_id: int, disaster_type: int, 
                                       expert_model: Dict[str, Any], 
                                       current_conditions: Dict[str, Any]) -> float:
        """计算预测置信度"""
        confidence_factors = []
        
        # 1. 数据可用性置信度
        if country_id in expert_model['country_patterns']:
            data_confidence = 0.8
        else:
            data_confidence = 0.3
        confidence_factors.append(data_confidence)
        
        # 2. 专家专业性置信度
        if disaster_type in expert_model['disaster_types']:
            specialization_confidence = 0.9
        else:
            specialization_confidence = 0.1
        confidence_factors.append(specialization_confidence)
        
        # 3. 时间模式置信度
        target_month = current_conditions.get('month', pd.Timestamp.now().month)
        peak_months = expert_model['temporal_patterns'].get('peak_months', [])
        if target_month in peak_months:
            temporal_confidence = 0.8
        else:
            temporal_confidence = 0.4
        confidence_factors.append(temporal_confidence)
        
        return np.mean(confidence_factors)
    
    def _fuse_expert_predictions(self, expert_predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """融合所有专家预测"""
        # 收集所有灾害类型
        all_disaster_types = set()
        for prediction in expert_predictions.values():
            all_disaster_types.update(prediction['disaster_probabilities'].keys())
        
        # 初始化融合结果
        fused_prediction = {
            'disaster_probabilities': {},
            'impact_estimates': {},
            'confidence_scores': {},
            'expert_contributions': {},
            'final_confidence': 0.0
        }
        
        # 对每种灾害类型进行加权融合
        for disaster_type in all_disaster_types:
            weighted_prob = 0.0
            weighted_impact = 0.0
            weighted_confidence = 0.0
            contributing_experts = []
            
            total_weight = 0.0
            
            for expert_name, prediction in expert_predictions.items():
                if disaster_type in prediction['disaster_probabilities']:
                    expert_weight = self.fusion_weights.get(expert_name, 0.0)
                    expert_confidence = prediction['confidence_scores'].get(disaster_type, 0.0)
                    
                    # 动态权重调整：专家置信度影响权重
                    dynamic_weight = expert_weight * (0.5 + 0.5 * expert_confidence)
                    
                    weighted_prob += prediction['disaster_probabilities'][disaster_type] * dynamic_weight
                    weighted_impact += prediction['impact_estimates'][disaster_type] * dynamic_weight
                    weighted_confidence += expert_confidence * dynamic_weight
                    
                    total_weight += dynamic_weight
                    contributing_experts.append({
                        'expert': expert_name,
                        'weight': dynamic_weight,
                        'probability': prediction['disaster_probabilities'][disaster_type],
                        'confidence': expert_confidence
                    })
            
            # 标准化
            if total_weight > 0:
                fused_prediction['disaster_probabilities'][disaster_type] = weighted_prob / total_weight
                fused_prediction['impact_estimates'][disaster_type] = weighted_impact / total_weight
                fused_prediction['confidence_scores'][disaster_type] = weighted_confidence / total_weight
            else:
                fused_prediction['disaster_probabilities'][disaster_type] = 0.0
                fused_prediction['impact_estimates'][disaster_type] = 1000.0
                fused_prediction['confidence_scores'][disaster_type] = 0.1
            
            fused_prediction['expert_contributions'][disaster_type] = contributing_experts
        
        # 计算整体置信度
        fused_prediction['final_confidence'] = np.mean(list(fused_prediction['confidence_scores'].values()))
        
        return fused_prediction
    
    def predict_top_disasters(self, current_conditions: Dict[str, Any],
                            spatial_features: Dict[int, Dict[str, float]],
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """预测最可能的K种灾害"""
        full_prediction = self.predict_disasters(current_conditions, spatial_features, getattr(self, 'districts_map', None))
        
        # 按概率排序
        disaster_probs = full_prediction['disaster_probabilities']
        sorted_disasters = sorted(disaster_probs.items(), key=lambda x: x[1], reverse=True)
        
        top_predictions = []
        for i, (disaster_type, probability) in enumerate(sorted_disasters[:top_k]):
            prediction_entry = {
                'rank': i + 1,
                'disaster_type_id': disaster_type,
                'probability': probability,
                'estimated_impact': full_prediction['impact_estimates'].get(disaster_type, 0),
                'confidence': full_prediction['confidence_scores'].get(disaster_type, 0),
                'contributing_experts': full_prediction['expert_contributions'].get(disaster_type, [])
            }
            top_predictions.append(prediction_entry)
        
        return top_predictions
    
    def analyze_expert_performance(self, historical_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """分析各专家模型的性能"""
        if not self.is_trained:
            return {}
        
        self.logger.info("分析专家模型性能...")
        
        performance_analysis = {}
        
        for expert_name, expert_model in self.expert_models.items():
            expert_disaster_types = expert_model['disaster_types']
            expert_data = historical_data[historical_data['disaster_type_id'].isin(expert_disaster_types)]
            
            if expert_data.empty:
                continue
            
            # 性能指标计算
            performance_metrics = {
                'data_coverage': len(expert_data) / len(historical_data),
                'geographic_coverage': len(expert_data['country_id'].unique()) / len(historical_data['country_id'].unique()),
                'temporal_coverage': len(expert_data['year'].unique()) / len(historical_data['year'].unique()),
                'disaster_type_specialization': len(expert_disaster_types) / len(historical_data['disaster_type_id'].unique()),
                'average_impact_accuracy': self._estimate_impact_accuracy(expert_data, expert_model),
                'fusion_weight': self.fusion_weights.get(expert_name, 0.0)
            }
            
            # 专业化程度评分
            specialization_score = 1.0 - performance_metrics['disaster_type_specialization']
            performance_metrics['specialization_strength'] = specialization_score
            
            performance_analysis[expert_name] = performance_metrics
        
        return performance_analysis
    
    def _estimate_impact_accuracy(self, expert_data: pd.DataFrame, expert_model: Dict[str, Any]) -> float:
        """估算影响预测准确性"""
        if expert_data.empty:
            return 0.0
        
        typical_impact = expert_model['impact_patterns'].get('typical_impact', 1000)
        actual_impacts = expert_data['people_affected'].dropna()
        
        if len(actual_impacts) == 0:
            return 0.0
        
        # 计算相对误差
        relative_errors = []
        for actual in actual_impacts:
            if actual > 0:
                relative_error = abs(typical_impact - actual) / actual
                relative_errors.append(min(relative_error, 2.0))  # 限制最大误差
        
        if relative_errors:
            avg_relative_error = np.mean(relative_errors)
            accuracy = max(0.0, 1.0 - avg_relative_error)
            return accuracy
        
        return 0.0
    
    def generate_prediction_explanation(self, prediction_result: Dict[str, Any], 
                                      current_conditions: Dict[str, Any]) -> Dict[str, str]:
        """生成预测解释"""
        explanations = {}
        
        top_disaster = max(prediction_result['disaster_probabilities'].items(), key=lambda x: x[1])
        disaster_type, probability = top_disaster
        
        # 主要预测解释
        explanations['primary_prediction'] = (
            f"预测灾害类型{disaster_type}的概率为{probability:.3f}，"
            f"预计影响{prediction_result['impact_estimates'].get(disaster_type, 0):.0f}人"
        )
        
        # 专家贡献解释
        contributing_experts = prediction_result['expert_contributions'].get(disaster_type, [])
        if contributing_experts:
            top_expert = max(contributing_experts, key=lambda x: x['weight'])
            explanations['expert_analysis'] = (
                f"主要由{top_expert['expert']}专家模型贡献(权重{top_expert['weight']:.3f})，"
                f"该专家对此预测的置信度为{top_expert['confidence']:.3f}"
            )
        
        # 置信度解释
        final_confidence = prediction_result['final_confidence']
        if final_confidence > 0.7:
            confidence_level = "高"
        elif final_confidence > 0.4:
            confidence_level = "中等"
        else:
            confidence_level = "低"
        
        explanations['confidence_assessment'] = f"整体预测置信度为{confidence_level}({final_confidence:.3f})"
        
        return explanations
    
    def update_expert_weights_online(self, new_feedback: Dict[str, float]) -> None:
        """基于在线反馈动态更新专家权重"""
        for expert_name, feedback_score in new_feedback.items():
            if expert_name in self.fusion_weights:
                # 学习率为0.1的在线更新
                current_weight = self.fusion_weights[expert_name]
                updated_weight = current_weight * 0.9 + feedback_score * 0.1
                self.fusion_weights[expert_name] = updated_weight
        
        # 重新标准化
        self._normalize_fusion_weights()
        self.logger.info("已更新专家权重")