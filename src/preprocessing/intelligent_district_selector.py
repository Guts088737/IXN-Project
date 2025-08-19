"""
智能district选择器 - 基于历史灾害密度的最优位置选择
替代随机选择，提供数据驱动的district选择
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict


class IntelligentDistrictSelector:
    """智能district选择器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.district_disaster_profiles = {}
        self.country_disaster_patterns = {}
        
    def build_district_disaster_profiles(self, historical_data: pd.DataFrame,
                                       districts_data: Dict[int, Dict],
                                       countries_data: Dict[int, Dict]) -> None:
        """构建每个district的灾害档案"""
        self.logger.info("构建district灾害档案...")
        
        # 1. 通过country_id关联历史事件与districts
        self._map_historical_events_to_districts(historical_data, districts_data, countries_data)
        
        # 2. 计算每个district的灾害密度和偏好
        self._calculate_district_disaster_densities(historical_data, districts_data)
        
        # 3. 建立country级别的灾害模式
        self._build_country_disaster_patterns(historical_data)
        
        self.logger.info(f"构建了{len(self.district_disaster_profiles)}个district的灾害档案")
    
    def _map_historical_events_to_districts(self, historical_data: pd.DataFrame,
                                          districts_data: Dict[int, Dict],
                                          countries_data: Dict[int, Dict]) -> None:
        """将历史事件映射到districts"""
        # 为每个country的districts建立灾害事件计数
        country_district_events = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # 遍历历史事件
        for _, event in historical_data.iterrows():
            country_id = event.get('country_id')
            disaster_type = event.get('disaster_type_id')
            
            if pd.isna(country_id) or pd.isna(disaster_type):
                continue
            
            country_id = int(country_id)
            disaster_type = int(disaster_type)
            
            # 找到该国家的所有districts
            country_name = countries_data.get(country_id, {}).get('name', '')
            if country_name:
                # 为该国家的每个district增加该类灾害的权重
                # (假设国家内的灾害可能发生在任何district，但有不同概率)
                for district_id, district_info in districts_data.items():
                    if district_info.get('country_name') == country_name:
                        # 基于地理特征计算该district发生此类灾害的可能性
                        likelihood = self._calculate_event_likelihood_for_district(
                            district_id, district_info, disaster_type, event
                        )
                        country_district_events[country_id][district_id][disaster_type] += likelihood
        
        # 转换为district档案格式
        for country_id, district_events in country_district_events.items():
            for district_id, disaster_counts in district_events.items():
                self.district_disaster_profiles[district_id] = {
                    'country_id': country_id,
                    'disaster_densities': dict(disaster_counts),
                    'total_disaster_weight': sum(disaster_counts.values()),
                    'disaster_type_diversity': len(disaster_counts)
                }
    
    def _calculate_event_likelihood_for_district(self, district_id: int, district_info: Dict,
                                               disaster_type: int, event: Dict) -> float:
        """计算历史事件在特定district发生的可能性权重"""
        likelihood = 0.5  # 基础权重
        
        district_lat = district_info.get('latitude', 0)
        district_lng = district_info.get('longitude', 0)
        
        # 1. 地理位置匹配度
        event_lat = event.get('latitude', 0)
        event_lng = event.get('longitude', 0)
        
        if event_lat != 0 and event_lng != 0 and district_lat != 0 and district_lng != 0:
            # 如果历史事件有具体坐标，距离越近权重越高
            distance = self._calculate_distance(district_lat, district_lng, event_lat, event_lng)
            if distance < 100:  # 100km内
                likelihood += 0.4
            elif distance < 500:  # 500km内
                likelihood += 0.2
        
        # 2. 灾害类型地理特征匹配
        type_specific_weight = self._get_disaster_type_geographic_weight(
            disaster_type, district_lat, district_lng
        )
        likelihood += type_specific_weight * 0.3
        
        # 3. 时间模式匹配 (季节性)
        event_month = event.get('month', 6)
        seasonal_weight = self._get_seasonal_weight(disaster_type, event_month, district_lat)
        likelihood += seasonal_weight * 0.2
        
        return min(likelihood, 1.0)
    
    def _get_disaster_type_geographic_weight(self, disaster_type: int, lat: float, lng: float) -> float:
        """获取灾害类型的地理特征权重"""
        # 沿海灾害
        if disaster_type in [4, 23, 10]:  # Cyclone, Storm Surge, Tsunami
            if self._is_coastal_location(lat, lng):
                return 0.8
            else:
                return 0.1
        
        # 地质灾害 - 板块边界和火山带
        elif disaster_type in [2, 8]:  # Earthquake, Volcanic
            if self._is_geological_active_zone(lat, lng):
                return 0.9
            else:
                return 0.2
        
        # 干旱 - 干旱带偏好
        elif disaster_type == 20:  # Drought
            if self._is_arid_zone(lat, lng):
                return 0.8
            else:
                return 0.4
        
        # 洪水 - 河流/低地偏好
        elif disaster_type in [12, 27]:  # Flood, Flash Flood
            if self._is_flood_prone_area(lat, lng):
                return 0.8
            else:
                return 0.5
        
        return 0.5  # 默认权重
    
    def _is_coastal_location(self, lat: float, lng: float) -> bool:
        """判断是否为沿海位置"""
        # 岛屿特征坐标范围
        island_regions = [
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
        
        for lat_min, lat_max, lng_min, lng_max in island_regions:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        
        return False
    
    def _is_geological_active_zone(self, lat: float, lng: float) -> bool:
        """判断是否为地质活跃区域"""
        # 主要地质活跃带
        active_zones = [
            # 环太平洋火山地震带
            (-60, 60, 120, 180),
            (-60, 60, -180, -120),
            # 地中海-喜马拉雅地震带
            (25, 45, 0, 100),
            # 大西洋中脊
            (-60, 60, -30, -10)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in active_zones:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        
        return False
    
    def _is_arid_zone(self, lat: float, lng: float) -> bool:
        """判断是否为干旱区域"""
        # 主要干旱带
        arid_zones = [
            # 撒哈拉及中东
            (10, 35, -20, 60),
            # 澳洲内陆
            (-35, -10, 110, 160),
            # 美国西南/墨西哥北部
            (25, 40, -120, -100),
            # 南美洲西海岸
            (-30, 0, -80, -65),
            # 南非卡拉哈里
            (-35, -20, 15, 30)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in arid_zones:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        
        return False
    
    def _is_flood_prone_area(self, lat: float, lng: float) -> bool:
        """判断是否为洪水易发区域"""
        # 主要河流流域和季风区
        flood_zones = [
            # 南亚季风区
            (5, 30, 65, 95),
            # 东南亚
            (-10, 25, 90, 140),
            # 中美洲
            (5, 20, -90, -75),
            # 西非萨赫勒
            (10, 18, -20, 20),
            # 南美洲北部
            (-10, 15, -75, -45)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in flood_zones:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        
        return False
    
    def _get_seasonal_weight(self, disaster_type: int, month: int, lat: float) -> float:
        """获取季节性权重 - 考虑南北半球差异和区域特定模式"""
        
        # 1. 南北半球季节调整
        if lat < 0:  # 南半球
            adjusted_month = ((month + 6 - 1) % 12) + 1
        else:  # 北半球
            adjusted_month = month
        
        # 2. 区域特定季节模式
        regional_seasonal_adjustments = self._get_regional_seasonal_adjustments(lat, month)
        
        # 3. 基础季节性权重（北半球基准）
        base_seasonal_weights = {
            4: {6: 0.9, 7: 0.9, 8: 0.9, 9: 0.8, 10: 0.7},  # Cyclone - 夏秋
            19: {5: 0.7, 6: 0.9, 7: 0.9, 8: 0.9, 9: 0.7},  # Heat Wave - 夏季
            14: {11: 0.7, 12: 0.9, 1: 0.9, 2: 0.9, 3: 0.7},  # Cold Wave - 冬季
            15: {3: 0.6, 4: 0.8, 5: 0.8, 6: 0.7, 7: 0.9, 8: 0.8, 9: 0.7, 10: 0.6},  # Fire - 春夏秋
            20: {3: 0.7, 4: 0.8, 5: 0.9, 6: 0.9, 7: 0.9, 8: 0.8, 9: 0.6},  # Drought - 春夏(9月开始缓解)
            12: {4: 0.6, 5: 0.7, 6: 0.8, 7: 0.9, 8: 0.9, 9: 0.8, 10: 0.7},  # Flood - 雨季
            27: {4: 0.6, 5: 0.7, 6: 0.8, 7: 0.9, 8: 0.9, 9: 0.8, 10: 0.7},  # Flash Flood - 雨季
            1: {1: 0.7, 2: 0.8, 3: 0.7, 9: 0.6, 10: 0.7, 11: 0.8, 12: 0.8},  # Epidemic - 季节转换期
            21: {3: 0.8, 4: 0.9, 5: 0.9, 6: 0.8, 9: 0.7, 10: 0.8, 11: 0.9}  # Food Insecurity - 干旱+收获期
        }
        
        # 4. 获取基础权重
        base_weight = 0.5  # 默认权重
        if disaster_type in base_seasonal_weights:
            base_weight = base_seasonal_weights[disaster_type].get(adjusted_month, 0.3)
        
        # 5. 应用区域调整
        adjusted_weight = base_weight * regional_seasonal_adjustments.get(disaster_type, 1.0)
        
        return min(max(adjusted_weight, 0.1), 1.0)  # 限制在0.1-1.0范围内
    
    def _get_regional_seasonal_adjustments(self, lat: float, month: int) -> Dict[int, float]:
        """基于全局气候模式的季节调整因子 - 数据驱动无硬编码"""
        adjustments = {}
        
        # 1. 基于纬度的通用季节模式
        climate_zone = self._get_climate_zone(lat)
        hemisphere = 'north' if lat >= 0 else 'south'
        
        # 2. 南北半球季节调整
        effective_month = month
        if hemisphere == 'south':
            # 南半球季节相位差6个月
            effective_month = ((month + 6 - 1) % 12) + 1
        
        # 3. 气候带通用调整规律
        seasonal_climate_factors = self._calculate_climate_seasonal_factors(climate_zone, effective_month)
        
        # 4. 应用气候因子到各灾害类型
        for disaster_type, climate_sensitivity in seasonal_climate_factors.items():
            adjustments[disaster_type] = climate_sensitivity
        
        return adjustments
    
    def _calculate_climate_seasonal_factors(self, climate_zone: str, month: int) -> Dict[int, float]:
        """基于气候带计算季节因子"""
        factors = {}
        
        # 热带气候带 (全年温暖)
        if climate_zone == 'tropical':
            if month in [6, 7, 8, 9]:  # 北半球夏季/雨季
                factors[12] = 1.2  # 洪水增强
                factors[27] = 1.2  # 快速洪水增强
                factors[4] = 1.1   # 台风增强
                factors[20] = 0.6  # 干旱降低
            elif month in [12, 1, 2, 3]:  # 北半球冬季/旱季
                factors[20] = 1.1  # 干旱增强
                factors[21] = 1.1  # 粮食不安全增强
                factors[12] = 0.7  # 洪水降低
        
        # 温带气候带 (四季分明)
        elif climate_zone == 'temperate':
            if month in [6, 7, 8]:  # 夏季
                factors[19] = 1.2  # 热浪增强
                factors[15] = 1.3  # 火灾增强
                factors[20] = 1.1  # 干旱增强
                factors[14] = 0.3  # 寒潮大幅降低
            elif month in [12, 1, 2]:  # 冬季
                factors[14] = 1.2  # 寒潮增强
                factors[19] = 0.2  # 热浪大幅降低
                factors[15] = 0.4  # 火灾降低
            elif month in [4, 5, 9, 10]:  # 春秋季
                factors[12] = 1.1  # 洪水适中增强
                factors[15] = 0.8  # 火灾适中
        
        # 极地气候带 (寒冷为主)
        elif climate_zone == 'polar':
            if month in [6, 7, 8]:  # 极地夏季
                factors[19] = 0.5  # 热浪仍然较低
                factors[14] = 0.6  # 寒潮降低
            else:  # 极地冬季
                factors[14] = 1.1  # 寒潮增强
                factors[19] = 0.1  # 热浪极低
                factors[15] = 0.3  # 火灾极低
        
        return factors
    
    def _get_climate_zone(self, latitude: float) -> str:
        """根据纬度判断气候带"""
        abs_lat = abs(latitude)
        if abs_lat > 66.5:
            return 'polar'
        elif abs_lat > 23.5:
            return 'temperate'
        else:
            return 'tropical'
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """计算地理距离(km)"""
        R = 6371
        lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))
    
    def _calculate_district_disaster_densities(self, historical_data: pd.DataFrame,
                                             districts_data: Dict[int, Dict]) -> None:
        """计算district灾害密度"""
        # 为每个district计算各类灾害的相对密度评分
        for district_id in self.district_disaster_profiles:
            profile = self.district_disaster_profiles[district_id]
            densities = profile['disaster_densities']
            
            # 标准化密度评分 (相对于该district的总灾害权重)
            total_weight = profile['total_disaster_weight']
            if total_weight > 0:
                for disaster_type in densities:
                    densities[disaster_type] = densities[disaster_type] / total_weight
    
    def _build_country_disaster_patterns(self, historical_data: pd.DataFrame) -> None:
        """建立国家级灾害模式"""
        for country_id in historical_data['country_id'].unique():
            if pd.isna(country_id):
                continue
                
            country_id = int(country_id)
            country_data = historical_data[historical_data['country_id'] == country_id]
            
            # 该国家的灾害类型分布
            disaster_distribution = country_data['disaster_type_id'].value_counts(normalize=True)
            
            # 季节性模式
            seasonal_pattern = country_data['month'].value_counts(normalize=True)
            
            # 影响规模分布
            impact_distribution = country_data['people_affected'].describe()
            
            self.country_disaster_patterns[country_id] = {
                'disaster_type_probabilities': disaster_distribution.to_dict(),
                'seasonal_probabilities': seasonal_pattern.to_dict(),
                'typical_impact_scale': impact_distribution['50%'],
                'impact_variability': impact_distribution['std'],
                'total_historical_events': len(country_data),
                'active_years': len(country_data['year'].unique())
            }
    
    def select_optimal_district_for_disaster(self, country_id: int, disaster_type: int,
                                           districts_by_country: Dict[str, List[int]],
                                           districts_data: Dict[int, Dict],
                                           countries_data: Dict[int, Dict],
                                           target_month: int = None) -> Tuple[Optional[int], float, Dict[str, Any]]:
        """为特定灾害类型选择最优district"""
        
        # 获取国家名称和districts
        country_name = countries_data.get(country_id, {}).get('name', '')
        if not country_name or country_name not in districts_by_country:
            return None, 0.0, {'reason': 'country_not_found'}
        
        country_districts = districts_by_country[country_name]
        if not country_districts:
            return None, 0.0, {'reason': 'no_districts'}
        
        # 计算每个district的适宜性评分
        district_scores = []
        
        for district_id in country_districts:
            if district_id not in districts_data:
                continue
            
            district_info = districts_data[district_id]
            
            # 综合评分
            score_components = self._calculate_comprehensive_district_score(
                district_id, district_info, disaster_type, country_id, target_month
            )
            
            total_score = sum(score_components.values())
            district_scores.append((district_id, total_score, score_components))
        
        if not district_scores:
            return None, 0.0, {'reason': 'no_valid_districts'}
        
        # 排序并选择最优district
        district_scores.sort(key=lambda x: x[1], reverse=True)
        best_district_id, best_score, score_breakdown = district_scores[0]
        
        return best_district_id, best_score, {
            'selection_method': 'historical_density_based',
            'score_breakdown': score_breakdown,
            'total_candidates': len(district_scores)
        }
    
    def _calculate_comprehensive_district_score(self, district_id: int, district_info: Dict,
                                              disaster_type: int, country_id: int,
                                              target_month: Optional[int]) -> Dict[str, float]:
        """计算district的综合适宜性评分"""
        scores = {}
        
        # 1. 历史灾害密度评分 (40%)
        historical_score = self._get_historical_disaster_density_score(district_id, disaster_type)
        scores['historical_density'] = historical_score * 0.4
        
        # 2. 地理特征匹配评分 (30%)
        geographic_score = self._get_geographic_feature_score(district_info, disaster_type)
        scores['geographic_match'] = geographic_score * 0.3
        
        # 3. 气候适宜性评分 (20%)
        climate_score = self._get_climate_suitability_score(district_info, disaster_type)
        scores['climate_suitability'] = climate_score * 0.2
        
        # 4. 季节性匹配评分 (10%)
        if target_month:
            seasonal_score = self._get_seasonal_match_score(district_info, disaster_type, target_month)
            scores['seasonal_match'] = seasonal_score * 0.1
        else:
            scores['seasonal_match'] = 0.05  # 默认季节评分
        
        return scores
    
    def _get_historical_disaster_density_score(self, district_id: int, disaster_type: int) -> float:
        """获取历史灾害密度评分"""
        if district_id not in self.district_disaster_profiles:
            return 0.3  # 无历史数据的默认评分
        
        profile = self.district_disaster_profiles[district_id]
        disaster_densities = profile['disaster_densities']
        
        # 该district该类灾害的密度
        disaster_density = disaster_densities.get(disaster_type, 0)
        
        # 相对密度评分 (相对于该district的所有灾害类型)
        if profile['total_disaster_weight'] > 0:
            relative_density = disaster_density / profile['total_disaster_weight']
            return min(relative_density * 2, 1.0)  # 放大评分
        
        return 0.3
    
    def _get_geographic_feature_score(self, district_info: Dict, disaster_type: int) -> float:
        """获取地理特征匹配评分 - 精细化地质特征匹配"""
        lat = district_info.get('latitude', 0)
        lng = district_info.get('longitude', 0)
        
        # 沿海灾害 - 精确距海判断
        if disaster_type in [4, 23, 10]:
            return 0.9 if self._is_coastal_location(lat, lng) else 0.1
        
        # 地质灾害 - 精细化地质特征匹配
        elif disaster_type in [2, 8]:
            return self._get_geological_precision_score(lat, lng, disaster_type)
        
        # 干旱 - 气候带精确匹配
        elif disaster_type == 20:
            return self._get_drought_precision_score(lat, lng)
        
        # 洪水 - 水文地理精确匹配
        elif disaster_type in [12, 27]:
            return self._get_flood_precision_score(lat, lng)
        
        # 山火 - 植被和气候精确匹配
        elif disaster_type == 15:
            return self._get_fire_precision_score(lat, lng)
        
        return 0.5
    
    def _get_geological_precision_score(self, lat: float, lng: float, disaster_type: int) -> float:
        """精细化地质特征评分"""
        base_score = 0.2
        
        # 主要地质活跃带精确定位
        high_activity_zones = [
            # 环太平洋火山地震带
            {'lat_range': (35, 45), 'lng_range': (135, 145), 'weight': 0.95},  # 日本
            {'lat_range': (-15, 5), 'lng_range': (-80, -70), 'weight': 0.9},   # 安第斯山脉
            {'lat_range': (10, 25), 'lng_range': (120, 130), 'weight': 0.85},  # 菲律宾
            {'lat_range': (-45, -35), 'lng_range': (165, 180), 'weight': 0.9}, # 新西兰
            {'lat_range': (55, 65), 'lng_range': (-25, -15), 'weight': 0.8},   # 冰岛
            
            # 地中海-阿尔卑斯-喜马拉雅带
            {'lat_range': (35, 42), 'lng_range': (12, 20), 'weight': 0.8},     # 意大利
            {'lat_range': (35, 45), 'lng_range': (25, 35), 'weight': 0.75},    # 土耳其
            {'lat_range': (25, 35), 'lng_range': (70, 85), 'weight': 0.8},     # 喜马拉雅
        ]
        
        for zone in high_activity_zones:
            lat_min, lat_max = zone['lat_range']
            lng_min, lng_max = zone['lng_range']
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                # 火山特别加权
                if disaster_type == 8:  # Volcanic Eruption
                    return min(zone['weight'] + 0.05, 1.0)
                else:  # Earthquake
                    return zone['weight']
        
        # 中等活跃区域
        moderate_activity_zones = [
            # 大西洋中脊
            {'lat_range': (-60, 60), 'lng_range': (-40, -10), 'weight': 0.6},
            # 东非大裂谷
            {'lat_range': (-35, 20), 'lng_range': (25, 50), 'weight': 0.7},
        ]
        
        for zone in moderate_activity_zones:
            lat_min, lat_max = zone['lat_range']
            lng_min, lng_max = zone['lng_range']
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return zone['weight']
        
        return base_score
    
    def _get_drought_precision_score(self, lat: float, lng: float) -> float:
        """精细化干旱风险评分"""
        # 主要干旱带精确定位
        drought_zones = [
            # 撒哈拉-萨赫勒
            {'lat_range': (10, 30), 'lng_range': (-20, 40), 'weight': 0.9},
            # 阿拉伯半岛
            {'lat_range': (15, 30), 'lng_range': (35, 60), 'weight': 0.85},
            # 澳洲内陆
            {'lat_range': (-30, -15), 'lng_range': (115, 150), 'weight': 0.8},
            # 美国西南
            {'lat_range': (30, 40), 'lng_range': (-115, -100), 'weight': 0.75},
            # 南非卡拉哈里
            {'lat_range': (-30, -20), 'lng_range': (15, 30), 'weight': 0.7},
            # 南美洲巴塔哥尼亚
            {'lat_range': (-50, -35), 'lng_range': (-75, -65), 'weight': 0.6}
        ]
        
        for zone in drought_zones:
            lat_min, lat_max = zone['lat_range']
            lng_min, lng_max = zone['lng_range']
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return zone['weight']
        
        # 温带湿润区域降低权重
        if 40 <= abs(lat) <= 60:  # 温带海洋性气候
            return 0.3
        
        return 0.5
    
    def _get_flood_precision_score(self, lat: float, lng: float) -> float:
        """精细化洪水风险评分"""
        # 主要洪水易发区域
        flood_zones = [
            # 南亚季风区
            {'lat_range': (8, 30), 'lng_range': (65, 95), 'weight': 0.9},
            # 东南亚
            {'lat_range': (-10, 25), 'lng_range': (90, 140), 'weight': 0.85},
            # 西非几内亚湾沿岸
            {'lat_range': (4, 15), 'lng_range': (-20, 15), 'weight': 0.8},
            # 南美洲亚马逊
            {'lat_range': (-15, 10), 'lng_range': (-75, -45), 'weight': 0.8},
            # 中国南方
            {'lat_range': (20, 35), 'lng_range': (100, 125), 'weight': 0.75},
            # 中美洲
            {'lat_range': (5, 20), 'lng_range': (-95, -75), 'weight': 0.7}
        ]
        
        for zone in flood_zones:
            lat_min, lat_max = zone['lat_range']
            lng_min, lng_max = zone['lng_range']
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return zone['weight']
        
        # 沙漠地区降低洪水权重
        if self._is_arid_zone(lat, lng):
            return 0.2
        
        return 0.5
    
    def _get_fire_precision_score(self, lat: float, lng: float) -> float:
        """精细化火灾风险评分"""
        # 山火高发区域
        fire_zones = [
            # 地中海气候区
            {'lat_range': (30, 45), 'lng_range': (-10, 40), 'weight': 0.9},
            # 加州
            {'lat_range': (32, 42), 'lng_range': (-125, -115), 'weight': 0.95},
            # 澳洲东南
            {'lat_range': (-40, -25), 'lng_range': (140, 155), 'weight': 0.9},
            # 南非fynbos
            {'lat_range': (-35, -30), 'lng_range': (15, 25), 'weight': 0.8},
            # 智利中部
            {'lat_range': (-40, -30), 'lng_range': (-75, -70), 'weight': 0.8},
            # 北美西部内陆
            {'lat_range': (40, 50), 'lng_range': (-120, -110), 'weight': 0.75}
        ]
        
        for zone in fire_zones:
            lat_min, lat_max = zone['lat_range']
            lng_min, lng_max = zone['lng_range']
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return zone['weight']
        
        # 湿润热带地区降低火灾权重
        if abs(lat) < 10:  # 赤道附近湿润
            return 0.3
        
        return 0.5
    
    def _is_geological_active_zone(self, lat: float, lng: float) -> bool:
        """判断是否为地质活跃区域"""
        # 环太平洋火山地震带
        pacific_ring = [
            (-60, 60, 120, 180),  # 西太平洋
            (-60, 60, -180, -60), # 东太平洋
            # 地中海-阿尔卑斯-喜马拉雅带
            (25, 45, -10, 100),
            # 中大西洋脊
            (-60, 60, -40, -10)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in pacific_ring:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        return False
    
    def _is_arid_zone(self, lat: float, lng: float) -> bool:
        """判断是否为干旱区域"""
        arid_zones = [
            # 非洲撒哈拉
            (10, 30, -20, 40),
            # 中东
            (20, 40, 35, 65),
            # 澳洲内陆
            (-35, -10, 110, 160),
            # 美国西南
            (30, 40, -120, -100),
            # 南美智利北部
            (-30, -15, -75, -65)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in arid_zones:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        return False
    
    def _is_flood_prone_area(self, lat: float, lng: float) -> bool:
        """判断是否为洪水易发区域"""
        flood_zones = [
            # 南亚季风区
            (8, 30, 65, 95),
            # 东南亚
            (-10, 25, 90, 140),
            # 西非
            (5, 20, -20, 20),
            # 南美洲亚马逊流域
            (-15, 10, -75, -45),
            # 中国南方
            (20, 35, 100, 125)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in flood_zones:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        return False
    
    def _is_fire_prone_area(self, lat: float, lng: float) -> bool:
        """判断是否为山火易发区域"""
        fire_zones = [
            # 地中海气候区
            (30, 45, -10, 40),
            # 加州
            (32, 42, -125, -115),
            # 澳洲东南
            (-40, -25, 140, 155),
            # 南非
            (-35, -25, 15, 30)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in fire_zones:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        return False
    
    def _get_climate_suitability_score(self, district_info: Dict, disaster_type: int) -> float:
        """获取气候适宜性评分"""
        lat = district_info.get('latitude', 0)
        climate_zone = self._get_climate_zone(lat)
        
        # 各灾害类型的气候偏好
        climate_preferences = {
            4: ['tropical', 'temperate'],      # Cyclone
            19: ['tropical', 'temperate'],     # Heat Wave  
            14: ['temperate', 'polar'],        # Cold Wave
            20: ['tropical', 'temperate'],     # Drought
            15: ['temperate'],                 # Fire
            12: ['tropical', 'temperate'],     # Flood
            1: ['tropical'],                   # Epidemic
            21: ['tropical'],                  # Food Insecurity
            62: ['tropical', 'temperate']      # Insect Infestation
        }
        
        preferred_climates = climate_preferences.get(disaster_type, ['all'])
        if 'all' in preferred_climates or climate_zone in preferred_climates:
            return 0.8
        else:
            return 0.3
    
    def _get_climate_zone(self, latitude: float) -> str:
        """根据纬度判断气候带"""
        abs_lat = abs(latitude)
        if abs_lat > 66.5:
            return 'polar'
        elif abs_lat > 23.5:
            return 'temperate'
        else:
            return 'tropical'
    
    def _get_seasonal_match_score(self, district_info: Dict, disaster_type: int, target_month: int) -> float:
        """获取季节性匹配评分"""
        lat = district_info.get('latitude', 0)
        return self._get_seasonal_weight(disaster_type, target_month, lat)
    
    def select_multiple_districts_with_probabilities(self, country_id: int, disaster_type: int,
                                                   districts_by_country: Dict[str, List[int]],
                                                   districts_data: Dict[int, Dict],
                                                   countries_data: Dict[int, Dict],
                                                   target_month: int = None,
                                                   max_districts: int = 3) -> List[Tuple[int, float, Dict]]:
        """选择多个districts并返回概率分布"""
        
        country_name = countries_data.get(country_id, {}).get('name', '')
        if not country_name or country_name not in districts_by_country:
            return []
        
        country_districts = districts_by_country[country_name]
        district_scores = []
        
        # 计算所有districts的评分
        for district_id in country_districts:
            if district_id not in districts_data:
                continue
            
            district_info = districts_data[district_id]
            score_components = self._calculate_comprehensive_district_score(
                district_id, district_info, disaster_type, country_id, target_month
            )
            
            total_score = sum(score_components.values())
            district_scores.append((district_id, total_score, score_components))
        
        # 排序并取前N个
        district_scores.sort(key=lambda x: x[1], reverse=True)
        top_districts = district_scores[:max_districts]
        
        # 转换评分为概率分布
        total_score = sum(score for _, score, _ in top_districts)
        if total_score > 0:
            probabilistic_districts = [
                (district_id, score/total_score, components) 
                for district_id, score, components in top_districts
            ]
        else:
            # 如果所有评分都是0，等概率分布
            equal_prob = 1.0 / len(top_districts) if top_districts else 0
            probabilistic_districts = [
                (district_id, equal_prob, components)
                for district_id, score, components in top_districts
            ]
        
        return probabilistic_districts
    
    def get_district_disaster_profile(self, district_id: int) -> Optional[Dict[str, Any]]:
        """获取district的灾害档案"""
        return self.district_disaster_profiles.get(district_id, None)
    
    def analyze_country_district_distribution(self, country_id: int,
                                            districts_by_country: Dict[str, List[int]],
                                            districts_data: Dict[int, Dict],
                                            countries_data: Dict[int, Dict]) -> Dict[str, Any]:
        """分析国家内districts的灾害分布特征"""
        country_name = countries_data.get(country_id, {}).get('name', '')
        if not country_name or country_name not in districts_by_country:
            return {}
        
        country_districts = districts_by_country[country_name]
        
        analysis = {
            'total_districts': len(country_districts),
            'coastal_districts': 0,
            'inland_districts': 0,
            'geological_active_districts': 0,
            'district_disaster_diversity': {},
            'high_risk_districts': [],
            'country_disaster_pattern': self.country_disaster_patterns.get(country_id, {})
        }
        
        # 统计各类district特征
        for district_id in country_districts:
            if district_id in self.coastal_districts:
                analysis['coastal_districts'] += 1
            if district_id in self.inland_districts:
                analysis['inland_districts'] += 1
            if district_id in self.geological_active_districts:
                analysis['geological_active_districts'] += 1
            
            # district的灾害多样性
            if district_id in self.district_disaster_profiles:
                profile = self.district_disaster_profiles[district_id]
                diversity = profile['disaster_type_diversity']
                analysis['district_disaster_diversity'][district_id] = diversity
                
                # 高风险district (总权重高)
                if profile['total_disaster_weight'] > 2.0:
                    analysis['high_risk_districts'].append({
                        'district_id': district_id,
                        'risk_weight': profile['total_disaster_weight'],
                        'disaster_types': list(profile['disaster_densities'].keys())
                    })
        
        return analysis