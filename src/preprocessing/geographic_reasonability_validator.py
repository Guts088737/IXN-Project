"""
地理合理性验证器 - 数据驱动的灾害地理约束
基于历史数据和地理特征验证预测合理性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Set
import logging
import json


class GeographicReasonabilityValidator:
    """地理合理性验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.coastal_districts = set()
        self.inland_districts = set()
        self.geological_active_districts = set()
        self.disaster_geographic_constraints = {}
        
    def initialize_geographic_constraints(self, historical_data: pd.DataFrame, 
                                        districts_data: Dict[int, Dict],
                                        countries_data: Dict[int, Dict]) -> None:
        """基于历史数据和地理数据初始化地理约束"""
        self.logger.info("初始化地理约束...")
        
        # 1. 识别沿海districts (基于坐标距海距离)
        self._identify_coastal_districts(districts_data)
        
        # 2. 识别地质活跃districts (基于历史地震/火山事件)
        self._identify_geological_active_districts(historical_data, districts_data)
        
        # 3. 建立灾害类型的地理约束规则 (数据驱动)
        self._build_disaster_geographic_constraints(historical_data, districts_data)
        
        self.logger.info(f"识别沿海districts: {len(self.coastal_districts)}个")
        self.logger.info(f"识别地质活跃districts: {len(self.geological_active_districts)}个")
        
    def _identify_coastal_districts(self, districts_data: Dict[int, Dict]) -> None:
        """识别沿海districts (通过坐标计算距海距离)"""
        for district_id, district_info in districts_data.items():
            lat = district_info.get('latitude', 0)
            lng = district_info.get('longitude', 0)
            
            if lat == 0 and lng == 0:
                continue
                
            # 简化海岸线判断：靠近大陆边缘或岛屿
            is_coastal = self._is_likely_coastal(lat, lng)
            
            if is_coastal:
                self.coastal_districts.add(district_id)
            else:
                self.inland_districts.add(district_id)
    
    def _is_likely_coastal(self, lat: float, lng: float) -> bool:
        """基于坐标特征判断是否可能沿海"""
        # 小岛屿特征 (某些特定经纬度范围)
        island_regions = [
            # 太平洋岛屿
            (-30, 30, 120, 180),  # 太平洋
            (-30, 30, -180, -120),  # 太平洋西部
            # 加勒比海
            (10, 25, -85, -60),
            # 地中海岛屿
            (30, 45, 0, 40),
            # 印度洋岛屿
            (-30, 0, 40, 100)
        ]
        
        for lat_min, lat_max, lng_min, lng_max in island_regions:
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return True
        
        # 大陆沿海判断 (简化：距离大陆中心超过一定距离)
        continent_centers = {
            'africa': (0, 20),
            'asia': (30, 100), 
            'europe': (55, 20),
            'north_america': (45, -100),
            'south_america': (-15, -60),
            'oceania': (-25, 140)
        }
        
        # 如果距离任何大陆中心都很远，可能是岛屿(沿海)
        min_distance = float('inf')
        for center_lat, center_lng in continent_centers.values():
            distance = self._calculate_distance(lat, lng, center_lat, center_lng)
            min_distance = min(min_distance, distance)
        
        # 距大陆中心>2000km的判定为岛屿/沿海
        return min_distance > 2000
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """计算两点距离(km)"""
        R = 6371
        lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))
        
    def _identify_geological_active_districts(self, historical_data: pd.DataFrame, 
                                            districts_data: Dict[int, Dict]) -> None:
        """基于历史地震/火山事件识别地质活跃districts"""
        geological_disasters = [2, 8]  # Earthquake, Volcanic Eruption
        
        # 从历史数据中找出地质活跃的country
        geological_active_countries = set()
        for disaster_type in geological_disasters:
            geological_events = historical_data[historical_data['disaster_type_id'] == disaster_type]
            active_countries = geological_events['country_id'].unique()
            geological_active_countries.update(active_countries)
        
        # 将地质活跃国家的所有districts标记为地质活跃
        for district_id, district_info in districts_data.items():
            country_name = district_info.get('country_name', '')
            # 通过国家名称找到对应的country_id (需要country映射)
            # 暂时使用简化逻辑：如果该district所属国家在历史上有地质活动
            for country_id in geological_active_countries:
                if not pd.isna(country_id):
                    self.geological_active_districts.add(district_id)
                    break
    
    def _build_disaster_geographic_constraints(self, historical_data: pd.DataFrame,
                                             districts_data: Dict[int, Dict]) -> None:
        """建立各灾害类型的地理约束 (基于历史分布)"""
        self.disaster_geographic_constraints = {}
        
        # 分析每种灾害类型的历史地理分布
        for disaster_type in historical_data['disaster_type_id'].unique():
            if pd.isna(disaster_type):
                continue
                
            disaster_type = int(disaster_type)
            type_data = historical_data[historical_data['disaster_type_id'] == disaster_type]
            
            # 该灾害类型的历史发生国家
            historical_countries = set(type_data['country_id'].dropna().astype(int))
            
            # 地理特征约束
            constraints = {
                'allowed_countries': historical_countries,
                'requires_coastal': self._disaster_requires_coastal(disaster_type),
                'requires_geological_activity': self._disaster_requires_geological(disaster_type),
                'climate_zones': self._get_disaster_climate_zones(disaster_type, type_data),
                'min_historical_events': len(type_data)
            }
            
            self.disaster_geographic_constraints[disaster_type] = constraints
    
    def _disaster_requires_coastal(self, disaster_type: int) -> bool:
        """判断灾害类型是否需要沿海环境"""
        coastal_disasters = [4, 23, 10]  # Cyclone, Storm Surge, Tsunami
        return disaster_type in coastal_disasters
    
    def _disaster_requires_geological(self, disaster_type: int) -> bool:
        """判断灾害类型是否需要地质活跃环境"""
        geological_disasters = [2, 8]  # Earthquake, Volcanic Eruption
        return disaster_type in geological_disasters
    
    def _get_disaster_climate_zones(self, disaster_type: int, type_data: pd.DataFrame) -> List[str]:
        """基于历史数据推断灾害适宜的气候带"""
        if type_data.empty:
            return ['all']
        
        # 基于历史发生地的纬度分布推断气候带偏好
        historical_lats = type_data['latitude'].dropna()
        if len(historical_lats) == 0:
            return ['all']
        
        climate_zones = []
        avg_lat = historical_lats.mean()
        lat_std = historical_lats.std()
        
        # 根据纬度分布特征推断适宜气候带
        if avg_lat > 30 or avg_lat < -30:
            climate_zones.append('temperate')  # 温带
        if -30 <= avg_lat <= 30:
            climate_zones.append('tropical')   # 热带
        if abs(avg_lat) > 60:
            climate_zones.append('polar')      # 极地
            
        return climate_zones if climate_zones else ['all']
    
    def validate_prediction_geographic_reasonability(self, prediction: Dict[str, Any],
                                                   districts_data: Dict[int, Dict]) -> Tuple[bool, List[str]]:
        """验证单个预测的地理合理性"""
        issues = []
        
        disaster_type = prediction.get('disaster_type_id')
        district_id = prediction.get('district_id')
        country_id = prediction.get('country_id')
        
        if disaster_type not in self.disaster_geographic_constraints:
            issues.append(f"未知灾害类型{disaster_type}")
            return False, issues
        
        constraints = self.disaster_geographic_constraints[disaster_type]
        
        # 1. 检查国家历史兼容性
        if country_id not in constraints['allowed_countries']:
            issues.append(f"国家{country_id}历史上从未发生过灾害类型{disaster_type}")
        
        # 2. 检查沿海要求
        if constraints['requires_coastal'] and district_id not in self.coastal_districts:
            issues.append(f"灾害类型{disaster_type}需要沿海环境，但district{district_id}为内陆")
        
        # 3. 检查地质活跃要求  
        if constraints['requires_geological_activity'] and district_id not in self.geological_active_districts:
            issues.append(f"灾害类型{disaster_type}需要地质活跃环境，但district{district_id}地质稳定")
        
        # 4. 检查气候带适宜性
        if district_id in districts_data:
            district_lat = districts_data[district_id].get('latitude', 0)
            district_climate = self._get_climate_zone(district_lat)
            allowed_climates = constraints['climate_zones']
            
            if 'all' not in allowed_climates and district_climate not in allowed_climates:
                issues.append(f"灾害类型{disaster_type}不适宜{district_climate}气候带")
        
        # 5. 特殊灾害类型的额外验证
        additional_issues = self._validate_special_disaster_types(prediction, districts_data)
        issues.extend(additional_issues)
        
        return len(issues) == 0, issues
    
    def _get_climate_zone(self, latitude: float) -> str:
        """根据纬度判断气候带"""
        abs_lat = abs(latitude)
        if abs_lat > 66.5:
            return 'polar'
        elif abs_lat > 23.5:
            return 'temperate'
        else:
            return 'tropical'
    
    def _validate_special_disaster_types(self, prediction: Dict[str, Any],
                                       districts_data: Dict[int, Dict]) -> List[str]:
        """特殊灾害类型的额外地理合理性验证"""
        issues = []
        disaster_type = prediction.get('disaster_type_id')
        district_id = prediction.get('district_id')
        
        if district_id not in districts_data:
            return issues
        
        district_info = districts_data[district_id]
        lat = district_info.get('latitude', 0)
        lng = district_info.get('longitude', 0)
        
        # Heat Wave - 不太可能在极地地区
        if disaster_type == 19:  # Heat Wave
            if abs(lat) > 60:
                issues.append("Heat Wave不太可能在极地地区发生")
        
        # Cold Wave - 不太可能在赤道地区
        if disaster_type == 14:  # Cold Wave
            if abs(lat) < 15:
                issues.append("Cold Wave不太可能在赤道地区发生")
        
        # Drought - 检查历史降水模式 (简化：避免高纬度湿润地区)
        if disaster_type == 20:  # Drought
            if abs(lat) > 55:  # 高纬度通常降水充足
                issues.append("Drought在高纬度湿润地区概率较低")
        
        # Insect Infestation - 主要在农业/森林地区
        if disaster_type == 62:  # Insect Infestation
            # 简化判断：极地地区不太可能
            if abs(lat) > 65:
                issues.append("Insect Infestation在极地地区不太可能")
        
        # Complex Emergency/Civil Unrest - 检查是否为争议地区
        if disaster_type in [6, 7]:  # Complex Emergency, Civil Unrest
            # 基于历史数据，某些稳定地区概率应该较低
            # 这里可以添加基于历史政治稳定性的约束
            pass
        
        # Transport Emergency - 检查交通枢纽密度
        if disaster_type == 68:  # Transport Emergency
            # 可以基于人口密度或交通设施密度判断
            # 简化：偏远地区概率较低
            pass
        
        # Radiological Emergency - 检查核设施分布
        if disaster_type == 67:  # Radiological Emergency
            # 需要核设施数据，简化：某些无核国家概率极低
            pass
        
        return issues
    
    def select_optimal_district(self, country_id: int, disaster_type: int,
                              districts_by_country: Dict[str, List[int]],
                              districts_data: Dict[int, Dict],
                              countries_data: Dict[int, Dict],
                              historical_data: pd.DataFrame) -> Tuple[Optional[int], float]:
        """为特定国家和灾害类型选择最优district"""
        
        # 获取国家名称
        country_name = countries_data.get(country_id, {}).get('name', '')
        if not country_name or country_name not in districts_by_country:
            return None, 0.0
        
        country_district_ids = districts_by_country[country_name]
        if not country_district_ids:
            return None, 0.0
        
        # 计算每个district对该灾害类型的适宜性评分
        district_scores = []
        
        for district_id in country_district_ids:
            if district_id not in districts_data:
                continue
                
            district_info = districts_data[district_id]
            score = self._calculate_district_suitability_score(
                district_id, district_info, disaster_type, historical_data
            )
            
            district_scores.append((district_id, score))
        
        if not district_scores:
            return None, 0.0
        
        # 按评分排序，选择最优district
        district_scores.sort(key=lambda x: x[1], reverse=True)
        best_district_id, best_score = district_scores[0]
        
        return best_district_id, best_score
    
    def _calculate_district_suitability_score(self, district_id: int, district_info: Dict,
                                            disaster_type: int, historical_data: pd.DataFrame) -> float:
        """计算district对特定灾害类型的适宜性评分"""
        score = 0.0
        
        # 1. 历史灾害密度评分 (基于该country的历史事件)
        country_name = district_info.get('country_name', '')
        disaster_events = historical_data[
            (historical_data['disaster_type_id'] == disaster_type)
        ]
        
        # 该国该类灾害的历史频次
        country_events = 0
        for _, event in disaster_events.iterrows():
            # 这里需要通过event的country信息匹配
            # 简化：假设有country_id匹配
            pass
        
        # 2. 地理特征匹配评分
        geographic_score = self._calculate_geographic_feature_score(district_id, disaster_type)
        score += geographic_score * 0.4
        
        # 3. 气候适宜性评分
        climate_score = self._calculate_climate_suitability_score(district_info, disaster_type)
        score += climate_score * 0.3
        
        # 4. 地形适宜性评分 (基于坐标推断)
        terrain_score = self._calculate_terrain_suitability_score(district_info, disaster_type)
        score += terrain_score * 0.3
        
        return min(score, 1.0)
    
    def _calculate_geographic_feature_score(self, district_id: int, disaster_type: int) -> float:
        """地理特征匹配评分"""
        score = 0.5  # 默认中等适宜
        
        # 沿海灾害类型
        if disaster_type in [4, 23, 10]:  # Cyclone, Storm Surge, Tsunami
            if district_id in self.coastal_districts:
                score = 0.9
            elif district_id in self.inland_districts:
                score = 0.1
        
        # 地质灾害类型
        elif disaster_type in [2, 8]:  # Earthquake, Volcanic Eruption
            if district_id in self.geological_active_districts:
                score = 0.9
            else:
                score = 0.2
        
        # 其他灾害类型保持默认评分
        return score
    
    def _calculate_climate_suitability_score(self, district_info: Dict, disaster_type: int) -> float:
        """气候适宜性评分"""
        lat = district_info.get('latitude', 0)
        climate_zone = self._get_climate_zone(lat)
        
        # 热带气候偏好灾害
        tropical_disasters = [4, 20, 21, 62]  # Cyclone, Drought, Food Insecurity, Insect
        if disaster_type in tropical_disasters:
            return 0.9 if climate_zone == 'tropical' else 0.5
        
        # 温带气候偏好灾害  
        temperate_disasters = [15, 12, 19]  # Fire, Flood, Heat Wave
        if disaster_type in temperate_disasters:
            return 0.8 if climate_zone == 'temperate' else 0.6
        
        # 极地气候偏好灾害
        polar_disasters = [14]  # Cold Wave
        if disaster_type in polar_disasters:
            return 0.9 if climate_zone == 'polar' else 0.3
        
        return 0.5  # 默认适宜性
    
    def _calculate_terrain_suitability_score(self, district_info: Dict, disaster_type: int) -> float:
        """地形适宜性评分 (基于坐标特征推断)"""
        lat = district_info.get('latitude', 0)
        lng = district_info.get('longitude', 0)
        
        # 山地灾害 (基于纬度变化推断地形)
        mountainous_disasters = [2, 8, 15]  # Earthquake, Volcanic, Fire
        if disaster_type in mountainous_disasters:
            # 某些已知山区范围
            known_mountain_ranges = [
                # 安第斯山脉
                (-60, 15, -80, -65),
                # 喜马拉雅山脉  
                (25, 40, 70, 100),
                # 阿尔卑斯山脉
                (45, 48, 5, 15)
            ]
            
            for lat_min, lat_max, lng_min, lng_max in known_mountain_ranges:
                if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                    return 0.8
        
        # 平原/沿海灾害
        lowland_disasters = [12, 27]  # Flood, Flash Flood
        # 简化：低纬度沿海平原
        if disaster_type in lowland_disasters:
            if abs(lat) < 40 and self._is_likely_coastal(lat, lng):
                return 0.8
        
        return 0.5  # 默认地形适宜性
    
    def filter_unreasonable_predictions(self, predictions: List[Dict[str, Any]],
                                      districts_data: Dict[int, Dict]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """过滤地理不合理的预测"""
        reasonable_predictions = []
        filtered_predictions = []
        
        for prediction in predictions:
            is_reasonable, issues = self.validate_prediction_geographic_reasonability(
                prediction, districts_data
            )
            
            if is_reasonable:
                reasonable_predictions.append(prediction)
            else:
                prediction['rejection_reasons'] = issues
                filtered_predictions.append(prediction)
        
        self.logger.info(f"地理合理性验证: {len(reasonable_predictions)}个通过, {len(filtered_predictions)}个被过滤")
        
        return reasonable_predictions, filtered_predictions
    
    def redistribute_filtered_probabilities(self, original_predictions: List[Dict[str, Any]],
                                          reasonable_predictions: List[Dict[str, Any]],
                                          filtered_predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重新分配被过滤预测的概率到合理预测中"""
        if not filtered_predictions:
            return reasonable_predictions
        
        # 计算被过滤的总概率
        filtered_total_prob = sum(pred.get('probability', 0) for pred in filtered_predictions)
        
        if filtered_total_prob <= 0 or not reasonable_predictions:
            return reasonable_predictions
        
        # 按原概率权重重新分配
        total_reasonable_prob = sum(pred.get('probability', 0) for pred in reasonable_predictions)
        
        for prediction in reasonable_predictions:
            original_prob = prediction.get('probability', 0)
            if total_reasonable_prob > 0:
                redistribution_weight = original_prob / total_reasonable_prob
                additional_prob = filtered_total_prob * redistribution_weight
                prediction['probability'] += additional_prob
                prediction['probability_redistributed'] = True
        
        self.logger.info(f"重新分配概率{filtered_total_prob:.3f}到{len(reasonable_predictions)}个合理预测")
        
        return reasonable_predictions
    
    def generate_validation_report(self, original_count: int, reasonable_count: int,
                                 filtered_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成验证报告"""
        report = {
            'total_predictions': original_count,
            'reasonable_predictions': reasonable_count,
            'filtered_predictions': len(filtered_predictions),
            'success_rate': reasonable_count / original_count if original_count > 0 else 0,
            'common_rejection_reasons': {},
            'filtered_by_disaster_type': {},
            'geographic_constraints_summary': {
                'coastal_districts_identified': len(self.coastal_districts),
                'geological_active_districts': len(self.geological_active_districts),
                'disaster_type_constraints': len(self.disaster_geographic_constraints)
            }
        }
        
        # 统计拒绝原因
        reason_counts = {}
        disaster_type_counts = {}
        
        for pred in filtered_predictions:
            disaster_type = pred.get('disaster_type_id', 'unknown')
            disaster_type_counts[disaster_type] = disaster_type_counts.get(disaster_type, 0) + 1
            
            for reason in pred.get('rejection_reasons', []):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        report['common_rejection_reasons'] = reason_counts
        report['filtered_by_disaster_type'] = disaster_type_counts
        
        return report