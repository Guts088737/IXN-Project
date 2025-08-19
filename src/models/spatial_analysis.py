"""
空间分析模型 - 用于灾害空间模式识别
完全数据驱动的空间聚类和模式分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict


class DisasterSpatialAnalyzer:
    """灾害空间分析器 - 识别灾害的空间分布模式"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def identify_disaster_hotspots_by_country(self, historical_data: pd.DataFrame, min_events: int = 2) -> Dict[int, List[Dict[str, float]]]:
        """识别各类灾害的热点区域"""
        self.logger.info("识别灾害热点区域...")
        
        disaster_hotspots = {}
        disaster_types = historical_data['disaster_type_id'].dropna().unique()
        
        for disaster_type in disaster_types:
            disaster_type = int(disaster_type)
            type_data = historical_data[historical_data['disaster_type_id'] == disaster_type]
            
            if len(type_data) < min_events:
                continue
            
            # 统计每个国家的灾害情况
            country_stats = {}
            
            for _, event in type_data.iterrows():
                country_id = event.get('country_id')
                if pd.isna(country_id):
                    continue
                
                country_id = int(country_id)
                impact = event.get('people_affected', 1000)
                
                if country_id not in country_stats:
                    country_stats[country_id] = {
                        'event_count': 0,
                        'total_impact': 0,
                        'impacts': [],
                        'region_id': event.get('region_id', 1)
                    }
                
                country_stats[country_id]['event_count'] += 1
                country_stats[country_id]['total_impact'] += impact
                country_stats[country_id]['impacts'].append(impact)
            
            # 识别热点国家
            hotspot_countries = self._identify_hotspot_countries(country_stats, min_events)
            
            if hotspot_countries:
                disaster_hotspots[disaster_type] = hotspot_countries
                self.logger.info(f"灾害类型{disaster_type}发现{len(hotspot_countries)}个热点国家")
        
        total_hotspots = sum(len(hotspots) for hotspots in disaster_hotspots.values())
        self.logger.info(f"共识别{len(disaster_hotspots)}种灾害的{total_hotspots}个热点国家")
        return disaster_hotspots
    
    def _identify_hotspot_countries(self, country_stats: Dict[int, Dict[str, Any]], min_events: int) -> List[Dict[str, float]]:
        """识别热点国家"""
        if not country_stats:
            return []
        
        # 计算所有国家的最大值用于标准化
        all_events = [stats['event_count'] for stats in country_stats.values()]
        all_impacts = [stats['total_impact'] for stats in country_stats.values()]
        
        max_events = max(all_events) if all_events else 1
        max_impact = max(all_impacts) if all_impacts and max(all_impacts) > 0 else 1
        
        hotspot_countries = []
        
        for country_id, stats in country_stats.items():
            if stats['event_count'] >= min_events:
                # 计算热点强度
                frequency_score = stats['event_count'] / max(max_events, 1)
                impact_score = stats['total_impact'] / max(max_impact, 1)
                
                # 计算影响一致性（方差越小越一致）
                impact_std = np.std(stats['impacts']) if len(stats['impacts']) > 1 else 0.0
                impact_consistency = 1.0 / (1.0 + impact_std) if impact_std >= 0 else 1.0
                
                # 综合热点评分
                hotspot_intensity = (frequency_score * 0.4 + impact_score * 0.4 + impact_consistency * 0.2)
                
                # 动态阈值：取Top 30%的国家作为热点
                if hotspot_intensity > 0.2:  # 降低阈值以识别更多热点
                    hotspot_countries.append({
                        'country_id': country_id,
                        'hotspot_intensity': hotspot_intensity,
                        'event_count': stats['event_count'],
                        'total_impact': stats['total_impact'],
                        'avg_impact': stats['total_impact'] / max(stats['event_count'], 1),
                        'impact_consistency': impact_consistency,
                        'region_id': stats['region_id'],
                        'risk_level': 'high' if hotspot_intensity > 0.6 else ('medium' if hotspot_intensity > 0.4 else 'moderate')
                    })
        
        # 按热点强度排序
        return sorted(hotspot_countries, key=lambda x: x['hotspot_intensity'], reverse=True)
    
    def _grid_based_hotspot_detection(self, coordinates: np.ndarray, impacts: np.ndarray, 
                                    grid_size: float = 5.0) -> List[Dict[str, float]]:
        """基于网格的热点检测"""
        if len(coordinates) == 0:
            return []
        
        # 创建网格
        lat_min, lat_max = coordinates[:, 0].min(), coordinates[:, 0].max()
        lng_min, lng_max = coordinates[:, 1].min(), coordinates[:, 1].max()
        
        # 计算网格数量
        lat_bins = max(int((lat_max - lat_min) / grid_size), 1)
        lng_bins = max(int((lng_max - lng_min) / grid_size), 1)
        
        # 网格统计
        grid_stats = defaultdict(lambda: {'count': 0, 'total_impact': 0, 'coords': []})
        
        for i, (lat, lng) in enumerate(coordinates):
            lat_idx = int((lat - lat_min) / max(lat_max - lat_min, 1e-8) * lat_bins)
            lng_idx = int((lng - lng_min) / max(lng_max - lng_min, 1e-8) * lng_bins)
            
            grid_key = (lat_idx, lng_idx)
            grid_stats[grid_key]['count'] += 1
            grid_stats[grid_key]['total_impact'] += impacts[i] if i < len(impacts) else 0
            grid_stats[grid_key]['coords'].append((lat, lng))
        
        # 识别热点（事件密度和影响都高的网格）
        hotspots = []
        max_count = max((stats['count'] for stats in grid_stats.values()), default=1)
        max_impact = max((stats['total_impact'] for stats in grid_stats.values()), default=1)
        
        for grid_key, stats in grid_stats.items():
            if stats['count'] >= 2:  # 至少2个事件
                # 计算热点强度
                density_score = stats['count'] / max_count
                impact_score = stats['total_impact'] / max_impact
                hotspot_intensity = (density_score + impact_score) / 2
                
                if hotspot_intensity > 0.3:  # 动态阈值
                    # 计算网格中心坐标
                    center_lat = np.mean([coord[0] for coord in stats['coords']])
                    center_lng = np.mean([coord[1] for coord in stats['coords']])
                    
                    hotspots.append({
                        'center_latitude': center_lat,
                        'center_longitude': center_lng,
                        'hotspot_intensity': hotspot_intensity,
                        'event_count': stats['count'],
                        'total_impact': stats['total_impact'],
                        'grid_area': grid_size * grid_size
                    })
        
        # 按热点强度排序
        return sorted(hotspots, key=lambda x: x['hotspot_intensity'], reverse=True)
    
    def calculate_country_risk_proximity(self, historical_data: pd.DataFrame, 
                                       hotspots: Dict[int, List[Dict[str, float]]]) -> Dict[int, Dict[str, float]]:
        """计算每个国家到各类灾害热点的风险距离"""
        self.logger.info("计算国家-热点风险距离...")
        
        country_risk_proximity = {}
        all_countries = historical_data['country_id'].unique()
        
        for country_id in all_countries:
            if pd.isna(country_id):
                continue
                
            country_data = historical_data[historical_data['country_id'] == country_id]
            if country_data.empty:
                continue
            
            # 计算国家中心坐标
            country_center_lat = country_data['latitude'].mean()
            country_center_lng = country_data['longitude'].mean()
            
            if pd.isna(country_center_lat) or pd.isna(country_center_lng):
                continue
            
            proximity_scores = {}
            
            # 计算到各类灾害热点的最近距离
            for disaster_type, type_hotspots in hotspots.items():
                if not type_hotspots:
                    proximity_scores[f'distance_to_disaster_{disaster_type}_hotspot'] = 1.0
                    continue
                
                # 找到最近的热点
                min_distance = float('inf')
                max_intensity = 0.0
                
                for hotspot in type_hotspots:
                    distance = self._haversine_distance(
                        country_center_lat, country_center_lng,
                        hotspot['center_latitude'], hotspot['center_longitude']
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        max_intensity = hotspot['hotspot_intensity']
                
                # 距离风险评分 (距离越近风险越高)
                distance_risk = 1.0 / (1.0 + min_distance / 1000.0)  # 1000km为参考距离
                intensity_weighted_risk = distance_risk * max_intensity
                
                proximity_scores[f'distance_to_disaster_{disaster_type}_hotspot'] = distance_risk
                proximity_scores[f'weighted_risk_from_disaster_{disaster_type}'] = intensity_weighted_risk
            
            country_risk_proximity[int(country_id)] = proximity_scores
        
        self.logger.info(f"计算了{len(country_risk_proximity)}个国家的热点风险距离")
        return country_risk_proximity
    
    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """计算两点间的地球距离（公里）"""
        R = 6371  # 地球半径(km)
        
        lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def analyze_disaster_corridors(self, historical_data: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """分析灾害传播走廊（连续发生的空间模式）"""
        self.logger.info("分析灾害传播走廊...")
        
        corridors = {}
        
        # 按时间排序
        sorted_data = historical_data.sort_values('date')
        
        for disaster_type in historical_data['disaster_type_id'].unique():
            if pd.isna(disaster_type):
                continue
                
            type_data = sorted_data[sorted_data['disaster_type_id'] == disaster_type]
            
            if len(type_data) < 5:
                continue
            
            # 分析时间邻近事件的空间关系
            corridors_for_type = []
            
            for i in range(len(type_data) - 1):
                event1 = type_data.iloc[i]
                event2 = type_data.iloc[i + 1]
                
                # 检查时间邻近性（30天内）
                time_diff = (event2['date'] - event1['date']).days
                if 0 < time_diff <= 30:
                    
                    # 计算空间距离
                    if (event1['latitude'] != 0 and event1['longitude'] != 0 and
                        event2['latitude'] != 0 and event2['longitude'] != 0):
                        
                        distance = self._haversine_distance(
                            event1['latitude'], event1['longitude'],
                            event2['latitude'], event2['longitude']
                        )
                        
                        # 如果距离合理（50-2000km），可能是传播走廊
                        if 50 <= distance <= 2000:
                            corridors_for_type.append({
                                'start_location': (event1['latitude'], event1['longitude']),
                                'end_location': (event2['latitude'], event2['longitude']),
                                'distance_km': distance,
                                'time_diff_days': time_diff,
                                'propagation_speed': distance / max(time_diff, 1),  # km/day
                                'impact_change': event2['people_affected'] - event1['people_affected']
                            })
            
            if corridors_for_type:
                corridors[f"disaster_{int(disaster_type)}"] = corridors_for_type
        
        self.logger.info(f"发现了{len(corridors)}种灾害的传播走廊")
        return corridors
    
    def create_spatial_risk_features(self, historical_data: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """基于country_id创建空间风险特征"""
        self.logger.info("基于country_id创建空间风险特征...")
        
        # 1. 识别热点国家
        hotspots = self.identify_disaster_hotspots_by_country(historical_data)
        
        # 2. 计算国家风险特征
        country_risk_features = self.calculate_country_risk_features(historical_data, hotspots)
        
        # 3. 分析区域灾害模式
        regional_patterns = self.identify_regional_disaster_patterns(historical_data)
        
        return country_risk_features
    
    def calculate_country_risk_features(self, historical_data: pd.DataFrame, 
                                      hotspots: Dict[int, List[Dict[str, float]]]) -> Dict[int, Dict[str, float]]:
        """计算每个国家的风险特征"""
        self.logger.info("计算国家风险特征...")
        
        country_risk_features = {}
        all_countries = historical_data['country_id'].dropna().unique()
        
        for country_id in all_countries:
            country_id = int(country_id)
            country_data = historical_data[historical_data['country_id'] == country_id]
            
            if country_data.empty:
                continue
            
            # 基础风险指标
            total_events = len(country_data)
            total_impact = country_data['people_affected'].sum()
            avg_impact = country_data['people_affected'].mean()
            disaster_types_count = len(country_data['disaster_type_id'].unique())
            
            # 计算该国家是否为各类灾害的热点
            hotspot_scores = {}
            total_hotspot_intensity = 0
            
            for disaster_type, hotspot_countries in hotspots.items():
                is_hotspot = False
                hotspot_intensity = 0
                
                for hotspot in hotspot_countries:
                    if hotspot['country_id'] == country_id:
                        is_hotspot = True
                        hotspot_intensity = hotspot['hotspot_intensity']
                        break
                
                hotspot_scores[f'is_hotspot_type_{disaster_type}'] = float(is_hotspot)
                hotspot_scores[f'hotspot_intensity_type_{disaster_type}'] = hotspot_intensity
                total_hotspot_intensity += hotspot_intensity
            
            # 区域风险特征
            region_id = country_data['region_id'].mode().iloc[0] if not country_data['region_id'].empty else 1
            
            # 综合风险特征
            features = {
                'total_disaster_events': float(total_events),
                'total_disaster_impact': float(total_impact),
                'avg_disaster_impact': float(avg_impact),
                'disaster_type_diversity': float(disaster_types_count),
                'total_hotspot_intensity': float(total_hotspot_intensity),
                'region_id': float(region_id),
                'disaster_frequency_score': min(float(total_events) / 10.0, 1.0),  # 标准化到0-1
                'impact_severity_score': min(float(total_impact) / 100000.0, 1.0),  # 标准化到0-1
            }
            
            # 添加热点评分
            features.update(hotspot_scores)
            
            country_risk_features[country_id] = features
        
        self.logger.info(f"计算了{len(country_risk_features)}个国家的风险特征")
        return country_risk_features
    
    def identify_regional_disaster_patterns(self, historical_data: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """识别区域灾害模式"""
        self.logger.info("识别区域灾害模式...")
        
        regional_patterns = {}
        regions = historical_data['region_id'].dropna().unique()
        
        for region_id in regions:
            region_id = int(region_id)
            region_data = historical_data[historical_data['region_id'] == region_id]
            
            if region_data.empty:
                continue
            
            # 区域灾害统计
            disaster_type_freq = region_data['disaster_type_id'].value_counts(normalize=True)
            country_count = len(region_data['country_id'].unique())
            total_events = len(region_data)
            
            regional_patterns[region_id] = {
                'dominant_disaster_types': disaster_type_freq.head(3).to_dict(),
                'total_countries': float(country_count),
                'total_events': float(total_events),
                'avg_events_per_country': float(total_events / country_count) if country_count > 0 else 0.0
            }
        
        return regional_patterns
    
    def _calculate_corridor_risk(self, country_id: int, corridors: Dict[str, List[Dict]], 
                               historical_data: pd.DataFrame) -> float:
        """计算国家的走廊传播风险"""
        country_data = historical_data[historical_data['country_id'] == country_id]
        
        if country_data.empty:
            return 0.0
        
        country_lat = country_data['latitude'].mean()
        country_lng = country_data['longitude'].mean()
        
        if pd.isna(country_lat) or pd.isna(country_lng):
            return 0.0
        
        # 检查该国是否接近任何已知传播走廊
        corridor_risks = []
        
        for disaster_type, type_corridors in corridors.items():
            for corridor in type_corridors:
                # 计算到走廊线段的最短距离
                corridor_distance = self._point_to_line_distance(
                    (country_lat, country_lng),
                    corridor['start_location'],
                    corridor['end_location']
                )
                
                # 距离走廊越近，风险越高
                risk_from_corridor = 1.0 / (1.0 + corridor_distance / 200.0)  # 200km为参考距离
                corridor_risks.append(risk_from_corridor)
        
        return max(corridor_risks) if corridor_risks else 0.0
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                              line_start: Tuple[float, float], 
                              line_end: Tuple[float, float]) -> float:
        """计算点到线段的最短距离（简化版本）"""
        # 简化计算：使用点到线段中点的距离作为近似
        mid_lat = (line_start[0] + line_end[0]) / 2
        mid_lng = (line_start[1] + line_end[1]) / 2
        
        return self._haversine_distance(point[0], point[1], mid_lat, mid_lng)
    
    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """计算两点间的地球距离（公里）"""
        R = 6371  # 地球半径(km)
        
        lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def calculate_geographic_similarity_matrix(self, historical_data: pd.DataFrame) -> np.ndarray:
        """计算国家间的地理灾害模式相似性矩阵"""
        self.logger.info("计算国家地理模式相似性...")
        
        countries = sorted(historical_data['country_id'].unique())
        countries = [c for c in countries if not pd.isna(c)]
        
        n_countries = len(countries)
        similarity_matrix = np.eye(n_countries)  # 对角线为1
        
        for i, country1 in enumerate(countries):
            for j, country2 in enumerate(countries):
                if i < j:  # 只计算上三角
                    similarity = self._calculate_country_similarity(
                        country1, country2, historical_data
                    )
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity  # 对称矩阵
        
        return similarity_matrix, countries
    
    def _calculate_country_similarity(self, country1: int, country2: int, 
                                    historical_data: pd.DataFrame) -> float:
        """计算两个国家的灾害模式相似性"""
        data1 = historical_data[historical_data['country_id'] == country1]
        data2 = historical_data[historical_data['country_id'] == country2]
        
        if data1.empty or data2.empty:
            return 0.0
        
        # 1. 灾害类型分布相似性
        types1 = data1['disaster_type_id'].value_counts(normalize=True)
        types2 = data2['disaster_type_id'].value_counts(normalize=True)
        
        # 计算分布重叠度
        all_types = set(types1.index).union(set(types2.index))
        type_similarity = 0.0
        
        for disaster_type in all_types:
            prob1 = types1.get(disaster_type, 0)
            prob2 = types2.get(disaster_type, 0)
            type_similarity += min(prob1, prob2)  # Overlap coefficient
        
        # 2. 时间模式相似性
        months1 = data1['month'].value_counts(normalize=True)
        months2 = data2['month'].value_counts(normalize=True)
        
        month_similarity = 0.0
        for month in range(1, 13):
            prob1 = months1.get(month, 0)
            prob2 = months2.get(month, 0)
            month_similarity += min(prob1, prob2)
        
        # 3. 影响规模相似性
        impact1_median = data1['people_affected'].median()
        impact2_median = data2['people_affected'].median()
        
        if impact1_median > 0 and impact2_median > 0:
            impact_ratio = min(impact1_median, impact2_median) / max(impact1_median, impact2_median)
        else:
            impact_ratio = 0.5
        
        # 综合相似性
        overall_similarity = (type_similarity * 0.5 + month_similarity * 0.3 + impact_ratio * 0.2)
        return min(overall_similarity, 1.0)
    
    def identify_regional_disaster_patterns(self, historical_data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """识别各地区的灾害模式特征"""
        self.logger.info("识别地区灾害模式...")
        
        regional_patterns = {}
        
        for region_id in historical_data['region_id'].unique():
            if pd.isna(region_id):
                continue
                
            region_data = historical_data[historical_data['region_id'] == region_id]
            
            if region_data.empty:
                continue
            
            # 该地区的主导灾害类型
            dominant_disasters = region_data['disaster_type_id'].value_counts(normalize=True)
            
            # 季节性模式
            seasonal_pattern = region_data['month'].value_counts(normalize=True)
            seasonal_variance = seasonal_pattern.var()
            
            # 影响模式
            impact_distribution = region_data['people_affected'].describe()
            
            # 地理分散程度
            lat_range = region_data['latitude'].max() - region_data['latitude'].min()
            lng_range = region_data['longitude'].max() - region_data['longitude'].min()
            geographic_span = np.sqrt(lat_range**2 + lng_range**2)
            
            regional_patterns[int(region_id)] = {
                'dominant_disaster_types': dominant_disasters.head(3).to_dict(),
                'seasonal_variance': seasonal_variance,
                'most_active_months': seasonal_pattern.nlargest(3).index.tolist(),
                'typical_impact_scale': impact_distribution['50%'],  # 中位数
                'impact_variability': impact_distribution['std'] / max(impact_distribution['mean'], 1) if impact_distribution['mean'] > 0 else 0.0,
                'geographic_span': geographic_span,
                'disaster_diversity': len(dominant_disasters),
                'total_events': len(region_data)
            }
        
        self.logger.info(f"识别了{len(regional_patterns)}个地区的灾害模式")
        return regional_patterns