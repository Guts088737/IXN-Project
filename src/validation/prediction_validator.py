"""
预测验证器 - 整合所有地理合理性验证规则
数据驱动的预测后验证和概率重新分配
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict


class PredictionValidator:
    """预测验证器 - 整合所有验证规则"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = {}
        self.historical_constraints = {}
        
    def initialize_validation_rules(self, historical_data: pd.DataFrame,
                                  districts_data: Dict[int, Dict],
                                  countries_data: Dict[int, Dict]) -> None:
        """初始化所有验证规则"""
        self.logger.info("初始化预测验证规则...")
        
        # 1. 基于历史数据建立硬约束
        self._build_historical_constraints(historical_data)
        
        # 2. 建立地理物理约束
        self._build_geographic_physical_constraints(districts_data)
        
        # 3. 建立气候约束
        self._build_climate_constraints(historical_data, districts_data)
        
        # 4. 建立灾害类型特定约束
        self._build_disaster_specific_constraints(historical_data)
        
        # 5. 构建国家-灾害统计数据
        self._build_country_disaster_statistics(historical_data)
        
        self.logger.info("验证规则初始化完成")
    
    def _build_country_disaster_statistics(self, historical_data: pd.DataFrame) -> None:
        """构建详细的国家-灾害统计数据"""
        self._country_disaster_stats = {}
        
        # 统计每个国家每种灾害的历史事件数
        for _, event in historical_data.iterrows():
            country_id = event.get('country_id')
            disaster_type = event.get('disaster_type_id')
            
            if pd.isna(country_id) or pd.isna(disaster_type):
                continue
            
            key = (int(disaster_type), int(country_id))
            self._country_disaster_stats[key] = self._country_disaster_stats.get(key, 0) + 1
    
    def _build_historical_constraints(self, historical_data: pd.DataFrame) -> None:
        """基于历史数据建立硬约束"""
        self.historical_constraints = {}
        
        # 每种灾害类型的历史发生国家
        for disaster_type in historical_data['disaster_type_id'].unique():
            if pd.isna(disaster_type):
                continue
                
            disaster_type = int(disaster_type)
            type_data = historical_data[historical_data['disaster_type_id'] == disaster_type]
            
            # 历史发生过该灾害的国家
            historical_countries = set(type_data['country_id'].dropna().astype(int))
            
            # 历史发生的月份分布
            historical_months = type_data['month'].value_counts(normalize=True).to_dict()
            
            # 历史影响规模范围
            impacts = type_data['people_affected'].dropna()
            impact_range = {
                'min': impacts.min() if len(impacts) > 0 else 100,
                'max': impacts.max() if len(impacts) > 0 else 100000,
                'median': impacts.median() if len(impacts) > 0 else 5000
            }
            
            self.historical_constraints[disaster_type] = {
                'allowed_countries': historical_countries,
                'historical_months': historical_months,
                'impact_range': impact_range,
                'total_historical_events': len(type_data),
                'min_events_threshold': 1  # 至少1次历史记录才允许预测
            }
    
    def _build_geographic_physical_constraints(self, districts_data: Dict[int, Dict]) -> None:
        """建立地理物理约束"""
        self.validation_rules['geographic'] = {}
        
        # 海洋灾害约束
        self.validation_rules['geographic']['marine_disasters'] = {
            'disaster_types': [4, 23, 10],  # Cyclone, Storm Surge, Tsunami
            'requires_coastal': True,
            'max_inland_distance_km': 50
        }
        
        # 地质灾害约束
        self.validation_rules['geographic']['geological_disasters'] = {
            'disaster_types': [2, 8],  # Earthquake, Volcanic
            'requires_active_geology': True,
            'known_stable_regions': [
                # 某些地质稳定区域的坐标范围
                (55, 70, 5, 30),   # 斯堪的纳维亚
                (45, 55, -10, 10), # 西欧平原
            ]
        }
        
        # 气候相关灾害约束
        self.validation_rules['geographic']['climate_disasters'] = {
            19: {'min_latitude': -60, 'max_latitude': 60, 'exclude_polar': True},  # Heat Wave
            14: {'preferred_latitudes': [40, 70], 'exclude_tropical': True},      # Cold Wave
            20: {'arid_zones_preferred': True, 'exclude_high_latitude': 60}       # Drought
        }
    
    def _build_climate_constraints(self, historical_data: pd.DataFrame,
                                 districts_data: Dict[int, Dict]) -> None:
        """建立气候约束"""
        self.validation_rules['climate'] = {}
        
        # 分析每种灾害的历史气候带分布
        for disaster_type in historical_data['disaster_type_id'].unique():
            if pd.isna(disaster_type):
                continue
                
            disaster_type = int(disaster_type)
            type_data = historical_data[historical_data['disaster_type_id'] == disaster_type]
            
            # 计算该灾害类型的气候带分布
            climate_distribution = self._analyze_disaster_climate_distribution(type_data)
            
            self.validation_rules['climate'][disaster_type] = climate_distribution
    
    def _analyze_disaster_climate_distribution(self, disaster_data: pd.DataFrame) -> Dict[str, float]:
        """分析灾害的气候带分布"""
        if disaster_data.empty or 'latitude' not in disaster_data.columns:
            return {'tropical': 0.33, 'temperate': 0.33, 'polar': 0.33}
        
        climate_counts = {'tropical': 0, 'temperate': 0, 'polar': 0}
        
        for lat in disaster_data['latitude'].dropna():
            climate_zone = self._get_climate_zone(lat)
            climate_counts[climate_zone] += 1
        
        total = sum(climate_counts.values())
        if total > 0:
            return {zone: count/total for zone, count in climate_counts.items()}
        else:
            return {'tropical': 0.33, 'temperate': 0.33, 'polar': 0.33}
    
    def _get_climate_zone(self, latitude: float) -> str:
        """根据纬度判断气候带"""
        abs_lat = abs(latitude)
        if abs_lat > 66.5:
            return 'polar'
        elif abs_lat > 23.5:
            return 'temperate'
        else:
            return 'tropical'
    
    def _build_disaster_specific_constraints(self, historical_data: pd.DataFrame) -> None:
        """建立灾害类型特定约束"""
        self.validation_rules['disaster_specific'] = {
            # 传染病 - 人口密度相关
            1: {'requires_population_density': True, 'min_urban_areas': True},
            
            # 地震 - 板块边界
            2: {'requires_tectonic_activity': True, 'fault_line_proximity': True},
            
            # 台风 - 海洋形成
            4: {'requires_ocean_access': True, 'latitude_range': (-40, 40)},
            
            # 火山 - 火山带
            8: {'requires_volcanic_activity': True, 'known_volcanic_regions': True},
            
            # 洪水 - 河流/降水
            12: {'river_systems': True, 'seasonal_rainfall': True},
            
            # 火灾 - 植被/干燥
            15: {'vegetation_cover': True, 'dry_seasons': True},
            
            # 热浪 - 大陆性气候
            19: {'continental_climate': True, 'summer_months': [6,7,8]},
            
            # 干旱 - 降水不足
            20: {'arid_regions': True, 'rainfall_deficiency': True},
            
            # 粮食不安全 - 农业依赖
            21: {'agricultural_regions': True, 'rural_areas': True},
            
            # 风暴潮 - 沿海低地
            23: {'coastal_lowlands': True, 'storm_seasons': True},
            
            # 虫害 - 农业区
            62: {'agricultural_zones': True, 'suitable_climate': True},
            
            # 生物紧急事态 - 生态敏感区
            66: {'biodiversity_hotspots': True, 'ecosystem_sensitivity': True},
            
            # 辐射紧急事态 - 核设施
            67: {'nuclear_facilities': True, 'industrial_areas': True},
            
            # 交通紧急事态 - 交通枢纽
            68: {'transport_hubs': True, 'infrastructure_density': True}
        }
    
    def validate_prediction(self, prediction: Dict[str, Any],
                          districts_data: Dict[int, Dict],
                          countries_data: Dict[int, Dict]) -> Tuple[bool, List[str], float]:
        """强化验证 - 物理约束优先级高于统计学习"""
        issues = []
        confidence_penalty = 0.0
        
        disaster_type = prediction.get('disaster_type_id')
        country_id = prediction.get('country_id')
        probability = prediction.get('probability', 0)
        latitude = prediction.get('latitude', 0)
        longitude = prediction.get('longitude', 0)
        
        # 0. 最高优先级：绝对物理不可能检查
        absolute_issues = self._check_absolute_physical_impossibilities_enhanced(
            disaster_type, latitude, longitude, prediction, districts_data
        )
        if absolute_issues:
            return False, absolute_issues, 0.0  # 绝对拒绝
        
        # 1. 极端概率异常检查 (兜底保险)
        if probability > 0.98 or probability < 0.0001:
            issues.append(f"概率异常: {probability}")
            confidence_penalty += 0.3
        
        # 2. 数值安全检查
        if np.isnan(probability) or np.isinf(probability):
            issues.append(f"概率数值错误: {probability}")
            confidence_penalty += 0.5
        
        # 3. 气候-灾害兼容性检查
        climate_issues = self._check_climate_disaster_compatibility(
            disaster_type, latitude, longitude
        )
        issues.extend(climate_issues)
        if climate_issues:
            confidence_penalty += 0.4
        
        # 简化其他验证，主要依赖物理约束
        is_reasonable = len(issues) == 0
        adjusted_confidence = max(0.0, 1.0 - confidence_penalty)
        
        return is_reasonable, issues, adjusted_confidence
    
    def _check_extreme_physical_impossibilities(self, disaster_type: int, country_id: int,
                                              prediction: Dict[str, Any], 
                                              districts_data: Dict[int, Dict]) -> List[str]:
        """检查极端物理不可能情况 - 专家可能遗漏的边缘案例"""
        issues = []
        
        # 只检查最明显的物理矛盾
        district_id = prediction.get('district_id')
        if district_id and district_id in districts_data:
            lat = districts_data[district_id].get('latitude', 0)
            
            # 极地地区热浪 (>75度纬度)
            if disaster_type == 19 and abs(lat) > 75:
                issues.append("极地地区热浪概率异常")
            
            # 赤道地区寒潮 (<10度纬度)  
            elif disaster_type == 14 and abs(lat) < 10:
                issues.append("赤道地区寒潮概率异常")
        
        return issues
    
    def _check_absolute_physical_impossibilities_enhanced(self, disaster_type: int, latitude: float, 
                                                        longitude: float, prediction: Dict[str, Any],
                                                        districts_data: Dict[int, Dict]) -> List[str]:
        """增强的绝对物理不可能检查"""
        issues = []
        
        # 1. 冰岛干旱 - 海洋性气候不可能严重干旱
        if disaster_type == 20:  # Drought
            # 检查是否为强海洋性气候（如冰岛 63-67°N, -25--13°W）
            if (63 <= latitude <= 67 and -25 <= longitude <= -13):
                issues.append("冰岛海洋性气候不可能发生干旱")
            
            # 小岛屿干旱检查
            elif self._is_small_island(latitude, longitude):
                issues.append("小岛屿海洋性气候不可能干旱")
        
        # 2. 瓦吉尔洪水 - 极干旱地区不可能洪水  
        elif disaster_type in [12, 27]:  # Flood types
            if self._is_extreme_arid_region(latitude, longitude):
                issues.append("极干旱地区不可能发生洪水")
        
        # 3. 地中海气旋 - 地中海不形成热带气旋
        elif disaster_type == 4:  # Cyclone
            if self._is_mediterranean_region(latitude, longitude):
                issues.append("地中海区域不形成热带气旋")
        
        # 4. 汤加干旱 - 太平洋岛国不可能干旱
        elif disaster_type == 20 and self._is_pacific_island(latitude, longitude):
            issues.append("太平洋岛国海洋性气候不可能干旱")
        
        # 5. 博茨瓦纳滑坡 - 极平坦地形不可能滑坡
        elif disaster_type == 5:  # Landslide
            if self._is_extremely_flat_terrain(latitude, longitude):
                issues.append("极平坦地形不可能发生滑坡")
        
        return issues
    
    def _check_climate_disaster_compatibility(self, disaster_type: int, latitude: float, longitude: float) -> List[str]:
        """检查气候-灾害兼容性"""
        issues = []
        
        # 获取气候微区
        climate_zone = self._get_detailed_climate_zone(latitude, longitude)
        
        # 气候-灾害兼容性矩阵
        incompatible_combinations = {
            # 海洋性气候不兼容的灾害
            'oceanic': [20],  # 干旱
            # 极干旱气候不兼容的灾害  
            'extreme_arid': [12, 27],  # 洪水类型
            # 热带气候不兼容的灾害
            'tropical': [14, 54],  # 寒潮、雪崩
            # 极地气候不兼容的灾害
            'polar': [19, 4],  # 热浪、气旋
        }
        
        for climate_type, incompatible_disasters in incompatible_combinations.items():
            if climate_zone == climate_type and disaster_type in incompatible_disasters:
                issues.append(f"{climate_type}气候与灾害类型{disaster_type}不兼容")
        
        return issues
    
    def _get_detailed_climate_zone(self, latitude: float, longitude: float) -> str:
        """获取详细气候分区"""
        # 海洋性气候识别
        if self._is_oceanic_climate(latitude, longitude):
            return 'oceanic'
        
        # 极干旱气候识别
        elif self._is_extreme_arid_region(latitude, longitude):
            return 'extreme_arid'
        
        # 基础气候带
        abs_lat = abs(latitude)
        if abs_lat > 66.5:
            return 'polar'
        elif abs_lat < 23.5:
            return 'tropical'
        else:
            return 'temperate'
    
    def _is_oceanic_climate(self, latitude: float, longitude: float) -> bool:
        """判断是否为海洋性气候"""
        # 冰岛
        if 63 <= latitude <= 67 and -25 <= longitude <= -13:
            return True
        
        # 英国群岛
        elif 50 <= latitude <= 60 and -10 <= longitude <= 2:
            return True
        
        # 太平洋岛屿
        elif self._is_pacific_island(latitude, longitude):
            return True
        
        # 其他小岛屿
        elif self._is_small_island(latitude, longitude):
            return True
        
        return False
    
    def _is_extreme_arid_region(self, latitude: float, longitude: float) -> bool:
        """判断是否为极干旱地区"""
        # 瓦吉尔等半沙漠地区 (肯尼亚北部)
        if 0 <= latitude <= 4 and 39 <= longitude <= 42:
            return True
        
        # 撒哈拉沙漠核心
        elif 18 <= latitude <= 27 and 0 <= longitude <= 25:
            return True
        
        # 阿拉伯沙漠
        elif 20 <= latitude <= 30 and 40 <= longitude <= 55:
            return True
        
        # 澳洲内陆沙漠
        elif -30 <= latitude <= -20 and 125 <= longitude <= 140:
            return True
        
        return False
    
    def _is_mediterranean_region(self, latitude: float, longitude: float) -> bool:
        """判断是否为地中海区域"""
        return 30 <= latitude <= 45 and 0 <= longitude <= 40
    
    def _is_pacific_island(self, latitude: float, longitude: float) -> bool:
        """判断是否为太平洋岛国"""
        # 汤加区域
        if -25 <= latitude <= -15 and -180 <= longitude <= -170:
            return True
        
        # 其他太平洋岛屿
        elif -30 <= latitude <= 30 and ((120 <= longitude <= 180) or (-180 <= longitude <= -120)):
            return True
        
        return False
    
    def _is_small_island(self, latitude: float, longitude: float) -> bool:
        """判断是否为小岛屿"""
        # 加勒比海岛屿
        if 10 <= latitude <= 25 and -85 <= longitude <= -60:
            return True
        
        # 印度洋岛屿  
        elif -30 <= latitude <= 0 and 40 <= longitude <= 100:
            return True
        
        return False
    
    def _is_extremely_flat_terrain(self, latitude: float, longitude: float) -> bool:
        """判断是否为极平坦地形"""
        # 博茨瓦纳平原
        if -26 <= latitude <= -17 and 20 <= longitude <= 29:
            return True
        
        # 荷兰低地
        elif 51 <= latitude <= 54 and 3 <= longitude <= 8:
            return True
        
        # 孟加拉平原
        elif 20 <= latitude <= 27 and 88 <= longitude <= 93:
            return True
        
        return False
    
    def _validate_historical_constraints(self, disaster_type: int, country_id: int,
                                       month: int) -> Tuple[bool, List[str]]:
        """验证历史约束"""
        issues = []
        
        if disaster_type not in self.historical_constraints:
            issues.append(f"灾害类型{disaster_type}缺少历史约束数据")
            return False, issues
        
        constraints = self.historical_constraints[disaster_type]
        
        # 检查国家历史兼容性 (放宽约束)
        allowed_countries = constraints['allowed_countries']
        if country_id not in allowed_countries:
            # 只有当该灾害类型历史事件数量充足且国家完全无记录时才严格拒绝
            if constraints['total_historical_events'] >= 10:
                issues.append(f"国家{country_id}历史上从未发生过灾害类型{disaster_type}")
            # 否则仅作为轻微警告，不拒绝预测
        
        # 检查季节性合理性 (放宽阈值)
        historical_months = constraints['historical_months']
        if month in historical_months:
            month_prob = historical_months[month]
            if month_prob < 0.005:  # 放宽到0.5%
                issues.append(f"灾害类型{disaster_type}在{month}月的历史发生概率极低({month_prob:.1%})")
        
        # 检查历史事件数量阈值
        if constraints['total_historical_events'] < constraints['min_events_threshold']:
            issues.append(f"灾害类型{disaster_type}历史事件过少({constraints['total_historical_events']}次)")
        
        return len(issues) == 0, issues
    
    def _validate_geographic_constraints(self, disaster_type: int, district_id: int,
                                       district_info: Dict) -> Tuple[bool, List[str]]:
        """验证地理约束"""
        issues = []
        lat = district_info.get('latitude', 0)
        lng = district_info.get('longitude', 0)
        
        # 海洋灾害约束
        if disaster_type in [4, 23, 10]:  # Cyclone, Storm Surge, Tsunami
            if not self._is_coastal_location(lat, lng):
                issues.append(f"海洋灾害{disaster_type}不应在内陆district{district_id}发生")
        
        # 地质灾害约束
        elif disaster_type in [2, 8]:  # Earthquake, Volcanic
            if not self._is_geological_active_zone(lat, lng):
                stable_regions = self.validation_rules['geographic']['geological_disasters']['known_stable_regions']
                for lat_min, lat_max, lng_min, lng_max in stable_regions:
                    if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                        issues.append(f"地质灾害{disaster_type}不应在稳定地质区域district{district_id}发生")
                        break
        
        return len(issues) == 0, issues
    
    def _validate_climate_constraints(self, disaster_type: int, district_id: int,
                                    district_info: Dict) -> Tuple[bool, List[str]]:
        """验证气候约束"""
        issues = []
        lat = district_info.get('latitude', 0)
        climate_zone = self._get_climate_zone(lat)
        
        # 获取该灾害类型的历史气候分布 (放宽气候约束)
        if disaster_type in self.validation_rules['climate']:
            climate_dist = self.validation_rules['climate'][disaster_type]
            current_climate_prob = climate_dist.get(climate_zone, 0)
            
            # 如果当前气候带的历史概率极低 (放宽到1%)
            if current_climate_prob < 0.01:
                issues.append(f"灾害类型{disaster_type}在{climate_zone}气候带的历史概率极低({current_climate_prob:.1%})")
        
        # 特定气候约束检查
        climate_rules = self.validation_rules['geographic'].get('climate_disasters', {})
        if disaster_type in climate_rules:
            rules = climate_rules[disaster_type]
            
            # 纬度范围约束
            if 'min_latitude' in rules and lat < rules['min_latitude']:
                issues.append(f"灾害类型{disaster_type}纬度过低({lat})")
            if 'max_latitude' in rules and lat > rules['max_latitude']:
                issues.append(f"灾害类型{disaster_type}纬度过高({lat})")
            
            # 气候带排除
            if rules.get('exclude_polar') and climate_zone == 'polar':
                issues.append(f"灾害类型{disaster_type}不适宜极地气候")
            if rules.get('exclude_tropical') and climate_zone == 'tropical':
                issues.append(f"灾害类型{disaster_type}不适宜热带气候")
        
        return len(issues) == 0, issues
    
    def _validate_disaster_specific_constraints(self, disaster_type: int, prediction: Dict[str, Any],
                                              districts_data: Dict[int, Dict]) -> Tuple[bool, List[str]]:
        """验证灾害类型特定约束"""
        issues = []
        
        if disaster_type not in self.validation_rules['disaster_specific']:
            return True, issues  # 无特定约束
        
        constraints = self.validation_rules['disaster_specific'][disaster_type]
        district_id = prediction.get('district_id')
        
        if not district_id or district_id not in districts_data:
            return True, issues
        
        district_info = districts_data[district_id]
        lat = district_info.get('latitude', 0)
        lng = district_info.get('longitude', 0)
        
        # 台风纬度约束
        if disaster_type == 4 and 'latitude_range' in constraints:
            lat_min, lat_max = constraints['latitude_range']
            if not (lat_min <= lat <= lat_max):
                issues.append(f"台风形成纬度范围约束违反: {lat}不在{lat_min}-{lat_max}范围内")
        
        # 热浪季节约束
        if disaster_type == 19 and 'summer_months' in constraints:
            month = prediction.get('predicted_month')
            summer_months = constraints['summer_months']
            # 考虑南北半球季节差异
            if lat < 0:  # 南半球
                summer_months = [12, 1, 2]  # 南半球夏季
            if month not in summer_months:
                issues.append(f"热浪季节约束违反: {month}月不是夏季月份")
        
        return len(issues) == 0, issues
    
    def _validate_probability_reasonableness(self, disaster_type: int, probability: float,
                                           country_id: int) -> Tuple[bool, List[str]]:
        """验证概率合理性 - 动态自适应阈值"""
        issues = []
        
        # 概率范围检查
        if probability < 0 or probability > 1:
            issues.append(f"概率值超出合理范围: {probability}")
        
        # 极低概率检查
        if probability < 0.001:
            issues.append(f"概率过低: {probability}")
        
        # 动态概率阈值检查
        if disaster_type in self.historical_constraints:
            constraints = self.historical_constraints[disaster_type]
            
            # 计算该灾害类型的全球平均发生率
            global_event_rate = self._calculate_global_disaster_rate(disaster_type)
            
            # 计算该国家的相对风险水平
            country_relative_risk = self._calculate_country_relative_risk(disaster_type, country_id)
            
            # 动态阈值 = 全球平均率 × 国家相对风险 × 调整因子 (放宽阈值)
            dynamic_threshold = global_event_rate * country_relative_risk * 5.0
            
            # 如果概率超过动态阈值太多 (放宽倍数)
            if probability > dynamic_threshold * 8:
                if country_id in constraints['allowed_countries']:
                    historical_events = self._get_country_historical_events(disaster_type, country_id)
                    if historical_events < 3:
                        issues.append(f"概率{probability:.3f}超过基于历史数据的合理阈值{dynamic_threshold*8:.3f}")
                else:
                    # 放宽无历史记录的概率阈值
                    if probability > 0.3:
                        issues.append(f"无历史记录但概率过高: {probability}")
        
        return len(issues) == 0, issues
    
    def _calculate_global_disaster_rate(self, disaster_type: int) -> float:
        """计算全球该灾害类型的平均发生率"""
        if disaster_type not in self.historical_constraints:
            return 0.01
        
        constraints = self.historical_constraints[disaster_type]
        total_events = constraints['total_historical_events']
        total_countries = len(constraints['allowed_countries'])
        
        # 全球平均发生率 (每个国家每月的期望概率)
        if total_countries > 0:
            annual_rate = total_events / total_countries / 10  # 假设10年数据
            monthly_rate = annual_rate / 12
            return min(monthly_rate, 0.1)  # 限制最大月度概率
        
        return 0.01
    
    def _calculate_country_relative_risk(self, disaster_type: int, country_id: int) -> float:
        """计算国家相对风险水平"""
        if disaster_type not in self.historical_constraints:
            return 1.0
        
        constraints = self.historical_constraints[disaster_type]
        
        if country_id not in constraints['allowed_countries']:
            return 0.1  # 无历史记录的国家低风险
        
        # 计算该国家在该灾害类型中的相对活跃度
        country_events = self._get_country_historical_events(disaster_type, country_id)
        total_events = constraints['total_historical_events']
        total_countries = len(constraints['allowed_countries'])
        
        if total_countries > 0 and total_events > 0:
            expected_events_per_country = total_events / total_countries
            relative_risk = country_events / max(expected_events_per_country, 1)
            return min(relative_risk, 5.0)  # 限制最大相对风险
        
        return 1.0
    
    def _get_country_historical_events(self, disaster_type: int, country_id: int) -> int:
        """获取国家历史事件数量"""
        # 简化实现 - 在实际使用中应存储详细的国家-灾害统计
        if hasattr(self, '_country_disaster_stats'):
            return self._country_disaster_stats.get((disaster_type, country_id), 0)
        return 2  # 默认假设
    
    def _is_coastal_location(self, lat: float, lng: float) -> bool:
        """判断是否为沿海位置"""
        # 岛屿/沿海区域特征
        coastal_regions = [
            # 太平洋岛屿
            (-30, 30, 120, 180),
            (-30, 30, -180, -120),
            # 加勒比海
            (10, 25, -85, -60),
            # 地中海
            (30, 45, 0, 40),
            # 印度洋岛屿
            (-30, 0, 40, 100)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in coastal_regions:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        
        # 大陆边缘判断 (简化)
        continent_interiors = [
            # 非洲内陆
            (-10, 10, 10, 35),
            # 亚洲内陆
            (35, 55, 60, 120),
            # 北美内陆
            (35, 55, -110, -80),
            # 南美内陆
            (-20, 5, -70, -50)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in continent_interiors:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return False  # 明确的内陆区域
        
        return True  # 默认可能沿海
    
    def _is_geological_active_zone(self, lat: float, lng: float) -> bool:
        """判断是否为地质活跃区域"""
        active_zones = [
            # 环太平洋火山地震带
            (-60, 60, 120, 180),
            (-60, 60, -180, -60),
            # 地中海-阿尔卑斯-喜马拉雅带
            (25, 45, -10, 100),
            # 中大西洋脊
            (-60, 60, -40, -10),
            # 东非大裂谷
            (-35, 20, 25, 50)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in active_zones:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        return False
    
    def _get_climate_zone(self, latitude: float) -> str:
        """根据纬度判断气候带"""
        abs_lat = abs(latitude)
        if abs_lat > 66.5:
            return 'polar'
        elif abs_lat > 23.5:
            return 'temperate'
        else:
            return 'tropical'
    
    def filter_and_redistribute_predictions(self, predictions: List[Dict[str, Any]],
                                          districts_data: Dict[int, Dict],
                                          countries_data: Dict[int, Dict]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """过滤不合理预测并重新分配概率"""
        reasonable_predictions = []
        filtered_predictions = []
        total_filtered_probability = 0.0
        
        # 验证每个预测
        for prediction in predictions:
            is_reasonable, issues, confidence = self.validate_prediction(
                prediction, districts_data, countries_data
            )
            
            if is_reasonable:
                prediction['validation_confidence'] = confidence
                reasonable_predictions.append(prediction)
            else:
                prediction['rejection_reasons'] = issues
                prediction['validation_confidence'] = confidence
                filtered_predictions.append(prediction)
                total_filtered_probability += prediction.get('probability', 0)
        
        # 重新分配被过滤的概率
        if total_filtered_probability > 0 and reasonable_predictions:
            self._redistribute_probabilities(reasonable_predictions, total_filtered_probability)
        
        # 生成验证报告
        validation_report = self._generate_validation_report(
            len(predictions), len(reasonable_predictions), filtered_predictions
        )
        
        self.logger.info(f"验证完成: {len(reasonable_predictions)}/{len(predictions)}通过")
        
        return reasonable_predictions, validation_report
    
    def _redistribute_probabilities(self, reasonable_predictions: List[Dict[str, Any]],
                                  filtered_probability: float) -> None:
        """重新分配概率并确保不超过1.0"""
        if not reasonable_predictions:
            return
        
        # 按原概率权重分配
        total_reasonable_prob = sum(pred.get('probability', 0) for pred in reasonable_predictions)
        
        if total_reasonable_prob > 0:
            for prediction in reasonable_predictions:
                original_prob = prediction.get('probability', 0)
                weight = original_prob / total_reasonable_prob
                additional_prob = filtered_probability * weight
                
                prediction['original_probability'] = original_prob
                prediction['probability_boost'] = additional_prob
                
                # 数值保护和NaN处理
                try:
                    # 使用clip确保概率在有效范围内
                    current_prob = np.clip(original_prob, 1e-6, 1-1e-6)
                    additional_prob = np.clip(additional_prob, 0, 1-current_prob)
                    
                    # 检查NaN值
                    if np.isnan(current_prob) or np.isnan(additional_prob):
                        self.logger.warning(f"发现NaN概率值: current={current_prob}, additional={additional_prob}")
                        prediction['probability'] = np.clip(original_prob, 0.001, 0.999)
                        continue
                    
                    # 安全的logit变换
                    current_logit = np.log(current_prob / (1 - current_prob))
                    
                    # 对additional_prob进行保护
                    safe_additional = np.clip(additional_prob, 1e-6, 1-1e-6)
                    boost_logit = np.log(safe_additional / (1 - safe_additional))
                    
                    # 在logit空间中累加 (降低boost强度)
                    new_logit = current_logit + boost_logit * 0.05
                    
                    # 检查logit值的有效性
                    if np.isnan(new_logit) or np.isinf(new_logit):
                        self.logger.warning(f"Logit计算异常: {new_logit}, 使用原始概率")
                        new_prob = current_prob
                    else:
                        # 安全的sigmoid变换
                        new_logit = np.clip(new_logit, -10, 10)  # 限制logit范围
                        new_prob = 1 / (1 + np.exp(-new_logit))
                    
                    # 最终概率保护
                    final_prob = np.clip(new_prob, 0.001, 0.95)
                    
                    # NaN最终检查
                    if np.isnan(final_prob):
                        final_prob = 0.001
                        self.logger.warning("最终概率为NaN，设置为0.001")
                    
                    prediction['probability'] = float(final_prob)
                    
                except Exception as e:
                    self.logger.error(f"概率重新分配计算错误: {e}")
                    # 错误时保持原始概率
                    prediction['probability'] = np.clip(original_prob, 0.001, 0.95)
    
    def _generate_validation_report(self, total: int, passed: int,
                                  filtered: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成验证报告"""
        # 统计拒绝原因
        rejection_reasons = defaultdict(int)
        disaster_type_rejections = defaultdict(int)
        
        for pred in filtered:
            disaster_type = pred.get('disaster_type_id', 'unknown')
            disaster_type_rejections[disaster_type] += 1
            
            for reason in pred.get('rejection_reasons', []):
                rejection_reasons[reason] += 1
        
        return {
            'total_predictions': total,
            'passed_predictions': passed,
            'filtered_predictions': len(filtered),
            'pass_rate': passed / total if total > 0 else 0,
            'common_rejection_reasons': dict(rejection_reasons),
            'rejected_by_disaster_type': dict(disaster_type_rejections),
            'constraints_summary': {
                'historical_constraints': len(self.historical_constraints),
                'geographic_rules': len(self.validation_rules.get('geographic', {})),
                'climate_rules': len(self.validation_rules.get('climate', {})),
                'disaster_specific_rules': len(self.validation_rules.get('disaster_specific', {}))
            }
        }