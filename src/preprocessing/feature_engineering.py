"""
Feature Engineering Module
医疗资源分配特征工程模块 - 基于历史灾害和医疗设施数据创建机器学习特征
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import sys
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.config import ProjectConfig, load_config
except ImportError:
    from config import ProjectConfig, load_config


class MedicalResourceFeatureEngineer:
    """医疗资源分配特征工程器"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 加载处理后的数据
        self.disasters_data = None
        self.facilities_data = None
        self._load_processed_data()
        
    def _load_processed_data(self):
        """加载处理后的灾害和设施数据"""
        try:
            # 加载历史灾害数据
            disasters_file = self.config.paths.processed_data_dir / "historical_disasters" / "processed_historical_disasters.json"
            if disasters_file.exists():
                with open(disasters_file, 'r', encoding='utf-8') as f:
                    disasters_data = json.load(f)
                self.disasters_data = disasters_data.get('disasters', [])
                self.logger.info(f"加载历史灾害数据: {len(self.disasters_data)} 条记录")
            
            # 加载医疗设施数据
            facilities_file = self.config.paths.processed_data_dir / "medical_facilities" / "processed_medical_facilities.json"
            if facilities_file.exists():
                with open(facilities_file, 'r', encoding='utf-8') as f:
                    facilities_data = json.load(f)
                self.facilities_data = facilities_data.get('facilities', [])
                self.logger.info(f"加载医疗设施数据: {len(self.facilities_data)} 条记录")
            
        except Exception as e:
            self.logger.error(f"加载处理后数据失败: {e}")
            self.disasters_data = []
            self.facilities_data = []
    
    def engineer_features(self) -> Dict[str, Any]:
        """主要特征工程方法 - 为医疗资源分配模型创建特征"""
        self.logger.info("开始医疗资源分配特征工程...")
        
        if not self.disasters_data:
            self.logger.error("没有历史灾害数据，无法进行特征工程")
            return self._create_empty_result()
        
        # 1. 创建灾害严重程度特征
        severity_features = self._create_disaster_severity_features()
        
        # 2. 创建医疗需求特征
        medical_demand_features = self._create_medical_demand_features()
        
        # 3. 创建时间特征
        temporal_features = self._create_temporal_features()
        
        # 4. 创建地理特征
        geographic_features = self._create_geographic_features()
        
        # 5. 创建资源供给特征（如果有设施数据）
        supply_features = self._create_resource_supply_features()
        
        # 6. 创建响应历史特征
        response_history_features = self._create_response_history_features()
        
        # 7. 创建交互特征
        interaction_features = self._create_interaction_features(
            severity_features, medical_demand_features, temporal_features
        )
        
        # 8. 整合所有特征
        integrated_features = self._integrate_all_features(
            severity_features, medical_demand_features, temporal_features,
            geographic_features, supply_features, response_history_features,
            interaction_features
        )
        
        # 9. 保存结果
        output_file = self._save_engineered_features(integrated_features)
        
        # 10. 生成报告
        feature_summary = self._generate_feature_summary(integrated_features)
        
        return {
            'engineered_features': integrated_features,
            'feature_summary': feature_summary,
            'output_file': str(output_file),
            'total_records': len(integrated_features)
        }
    
    def _create_disaster_severity_features(self) -> List[Dict[str, Any]]:
        """创建灾害严重程度特征"""
        self.logger.info("创建灾害严重程度特征...")
        severity_features = []
        
        for disaster in self.disasters_data:
            features = {
                'disaster_id': disaster.get('disaster_id'),
                'people_affected': disaster.get('people_affected', 0),
                'people_injured': disaster.get('people_injured', 0),
                'people_dead': disaster.get('people_dead', 0),
                'people_displaced': disaster.get('people_displaced', 0),
                
                # 计算严重程度指标
                'total_casualties': disaster.get('people_injured', 0) + disaster.get('people_dead', 0),
                'casualty_rate': self._calculate_casualty_rate(disaster),
                'displacement_rate': self._calculate_displacement_rate(disaster),
                'severity_score': self._calculate_severity_score(disaster),
                'disaster_scale': self._categorize_disaster_scale(disaster)
            }
            severity_features.append(features)
        
        return severity_features
    
    def _create_medical_demand_features(self) -> List[Dict[str, Any]]:
        """创建医疗需求特征"""
        self.logger.info("创建医疗需求特征...")
        medical_features = []
        
        for disaster in self.disasters_data:
            medical_needs = disaster.get('medical_needs', {})
            
            features = {
                'disaster_id': disaster.get('disaster_id'),
                'field_reports_count': disaster.get('field_reports_count', 0),
                'has_field_assessment': disaster.get('has_field_assessment', False),
                
                # 医疗需求数据
                'total_injured_reported': medical_needs.get('total_injured_reported', 0),
                'total_dead_reported': medical_needs.get('total_dead_reported', 0),
                'total_affected_reported': medical_needs.get('total_affected_reported', 0),
                'total_assisted': medical_needs.get('total_assisted', 0),
                
                # 医疗响应指标
                'has_medical_response': medical_needs.get('has_medical_response', False),
                'medical_facilities_mentioned': medical_needs.get('medical_facilities_mentioned', False),
                'evacuation_mentioned': medical_needs.get('evacuation_mentioned', False),
                
                # 计算医疗需求强度
                'medical_need_intensity': self._calculate_medical_need_intensity(medical_needs),
                'assistance_coverage': self._calculate_assistance_coverage(medical_needs),
                'field_reports_completeness': self._calculate_field_reports_completeness(medical_needs)
            }
            medical_features.append(features)
        
        return medical_features
    
    def _create_temporal_features(self) -> List[Dict[str, Any]]:
        """创建时间特征"""
        self.logger.info("创建时间特征...")
        temporal_features = []
        
        for disaster in self.disasters_data:
            features = {
                'disaster_id': disaster.get('disaster_id'),
                'year': disaster.get('year', 0),
                'month': disaster.get('month', 0),
                'season': disaster.get('season', 'Unknown'),
                'start_date': disaster.get('start_date'),
                
                # 时间特征计算
                'is_weekend': self._is_weekend(disaster.get('start_date')),
                'is_holiday_season': self._is_holiday_season(disaster.get('month', 0)),
                'days_since_year_start': self._days_since_year_start(disaster.get('start_date')),
                'season_encoded': self._encode_season(disaster.get('season')),
                'month_sin': np.sin(2 * np.pi * disaster.get('month', 0) / 12),
                'month_cos': np.cos(2 * np.pi * disaster.get('month', 0) / 12)
            }
            temporal_features.append(features)
        
        return temporal_features
    
    def _create_geographic_features(self) -> List[Dict[str, Any]]:
        """创建地理特征"""
        self.logger.info("创建地理特征...")
        geographic_features = []
        
        for disaster in self.disasters_data:
            location_details = disaster.get('location_details', {})
            
            features = {
                'disaster_id': disaster.get('disaster_id'),
                'country_name': disaster.get('country_name', ''),
                'country_id': disaster.get('country_id', 0),
                'region_name': disaster.get('region_name', ''),
                'iso_code': location_details.get('iso_code', ''),
                'iso3_code': location_details.get('iso3_code', ''),
                'region_id': location_details.get('region_id'),
                'glide_code': location_details.get('glide_code', ''),
                
                # 地理特征计算
                'is_landlocked': self._is_landlocked_country(location_details.get('iso3_code')),
                'development_level': self._get_development_level(location_details.get('iso3_code')),
                'geographic_region': self._get_geographic_region(location_details.get('region_id')),
                'country_risk_level': self._get_country_risk_level(disaster.get('country_name'))
            }
            geographic_features.append(features)
        
        return geographic_features
    
    def _create_resource_supply_features(self) -> List[Dict[str, Any]]:
        """创建资源供给特征（基于医疗设施数据）"""
        self.logger.info("创建资源供给特征...")
        supply_features = []
        
        for disaster in self.disasters_data:
            country_name = disaster.get('country_name', '')
            
            # 查找该国家的医疗设施
            country_facilities = [f for f in (self.facilities_data or []) 
                                if f.get('country_name', '').lower() == country_name.lower()]
            
            features = {
                'disaster_id': disaster.get('disaster_id'),
                'facilities_count_in_country': len(country_facilities),
                'total_bed_capacity': sum(f.get('bed_capacity', 0) for f in country_facilities),
                'total_staff': sum(f.get('total_staff', 0) for f in country_facilities),
                
                # 设施类型分布
                'hospital_count': len([f for f in country_facilities if 'hospital' in f.get('facility_type_name', '').lower()]),
                'clinic_count': len([f for f in country_facilities if 'clinic' in f.get('facility_type_name', '').lower()]),
                
                # 资源密度指标
                'beds_per_affected': self._calculate_beds_per_affected(country_facilities, disaster),
                'staff_per_affected': self._calculate_staff_per_affected(country_facilities, disaster),
                'resource_adequacy_score': self._calculate_resource_adequacy(country_facilities, disaster)
            }
            supply_features.append(features)
        
        return supply_features
    
    def _create_response_history_features(self) -> List[Dict[str, Any]]:
        """创建响应历史特征"""
        self.logger.info("创建响应历史特征...")
        response_features = []
        
        for disaster in self.disasters_data:
            features = {
                'disaster_id': disaster.get('disaster_id'),
                'amount_requested': disaster.get('amount_requested', 0),
                'amount_funded': disaster.get('amount_funded', 0),
                'funding_coverage': disaster.get('funding_coverage', 0),
                'appeals_count': disaster.get('appeals_count', 0),
                'has_active_appeal': disaster.get('has_active_appeal', False),
                'emergency_response': disaster.get('emergency_response', False),
                
                # 响应效果指标
                'funding_per_affected': self._calculate_funding_per_affected(disaster),
                'appeal_effectiveness': self._calculate_appeal_effectiveness(disaster),
                'response_speed_score': self._calculate_response_speed(disaster),
                'funding_gap': disaster.get('amount_requested', 0) - disaster.get('amount_funded', 0)
            }
            response_features.append(features)
        
        return response_features
    
    def _create_interaction_features(self, severity_features: List[Dict], 
                                   medical_demand_features: List[Dict], 
                                   temporal_features: List[Dict]) -> List[Dict[str, Any]]:
        """创建交互特征"""
        self.logger.info("创建交互特征...")
        interaction_features = []
        
        # 创建索引映射
        severity_dict = {f['disaster_id']: f for f in severity_features}
        medical_dict = {f['disaster_id']: f for f in medical_demand_features}
        temporal_dict = {f['disaster_id']: f for f in temporal_features}
        
        for disaster_id in severity_dict.keys():
            severity = severity_dict.get(disaster_id, {})
            medical = medical_dict.get(disaster_id, {})
            temporal = temporal_dict.get(disaster_id, {})
            
            features = {
                'disaster_id': disaster_id,
                
                # 严重程度与医疗需求交互
                'severity_medical_ratio': self._safe_divide(severity.get('severity_score', 0),
                                                          medical.get('medical_need_intensity', 1)),
                'casualties_per_report': self._safe_divide(severity.get('total_casualties', 0),
                                                         medical.get('field_reports_count', 1)),
                
                # 时间与严重程度交互
                'seasonal_severity_impact': severity.get('severity_score', 0) * temporal.get('season_encoded', 1),
                'monthly_pattern_severity': severity.get('severity_score', 0) * temporal.get('month_sin', 0),
                
                # 医疗需求与时间交互
                'seasonal_medical_demand': medical.get('medical_need_intensity', 0) * temporal.get('season_encoded', 1)
            }
            interaction_features.append(features)
        
        return interaction_features
    
    def _integrate_all_features(self, *feature_lists) -> List[Dict[str, Any]]:
        """整合所有特征到单一记录中"""
        self.logger.info("整合所有特征...")
        
        # 创建disaster_id到特征的映射
        feature_dicts = []
        for feature_list in feature_lists:
            if feature_list:  # 确保feature_list不为空
                feature_dict = {f['disaster_id']: f for f in feature_list if f.get('disaster_id')}
                feature_dicts.append(feature_dict)
        
        # 获取所有disaster_id
        all_disaster_ids = set()
        for feature_dict in feature_dicts:
            all_disaster_ids.update(feature_dict.keys())
        
        # 整合特征
        integrated_features = []
        for disaster_id in all_disaster_ids:
            integrated_record = {'disaster_id': disaster_id}
            
            # 合并各类特征
            for feature_dict in feature_dicts:
                if disaster_id in feature_dict:
                    record = feature_dict[disaster_id].copy()
                    record.pop('disaster_id', None)  # 避免重复
                    integrated_record.update(record)
            
            integrated_features.append(integrated_record)
        
        return integrated_features
    
    # 辅助计算方法
    def _calculate_casualty_rate(self, disaster: Dict) -> float:
        """计算伤亡率"""
        total_casualties = disaster.get('people_injured', 0) + disaster.get('people_dead', 0)
        affected = max(disaster.get('people_affected', 1), 1)  # 避免除零
        return total_casualties / affected
    
    def _calculate_displacement_rate(self, disaster: Dict) -> float:
        """计算流离失所率"""
        displaced = disaster.get('people_displaced', 0)
        affected = max(disaster.get('people_affected', 1), 1)
        return displaced / affected
    
    def _calculate_severity_score(self, disaster: Dict) -> float:
        """计算综合严重程度分数"""
        affected = disaster.get('people_affected', 0)
        injured = disaster.get('people_injured', 0)
        dead = disaster.get('people_dead', 0)
        displaced = disaster.get('people_displaced', 0)
        
        # 加权计算严重程度
        score = (affected * 0.1 + injured * 2 + dead * 10 + displaced * 1) / 1000
        return min(score, 10.0)  # 限制在0-10范围内
    
    def _categorize_disaster_scale(self, disaster: Dict) -> str:
        """分类灾害规模"""
        affected = disaster.get('people_affected', 0)
        if affected < 1000:
            return 'Small'
        elif affected < 10000:
            return 'Medium'
        elif affected < 100000:
            return 'Large'
        else:
            return 'Very Large'
    
    def _calculate_medical_need_intensity(self, medical_needs: Dict) -> float:
        """计算医疗需求强度"""
        injured = medical_needs.get('total_injured_reported', 0)
        dead = medical_needs.get('total_dead_reported', 0)
        affected = medical_needs.get('total_affected_reported', 1)
        
        if affected == 0:
            return 0
        
        intensity = (injured + dead * 2) / affected
        return min(intensity, 1.0)  # 限制在0-1范围内
    
    def _calculate_assistance_coverage(self, medical_needs: Dict) -> float:
        """计算救助覆盖率"""
        assisted = medical_needs.get('total_assisted', 0)
        affected = medical_needs.get('total_affected_reported', 1)
        return min(assisted / affected, 1.0) if affected > 0 else 0
    
    def _calculate_field_reports_completeness(self, medical_needs: Dict) -> float:
        """计算现场报告完整性"""
        processed = medical_needs.get('field_reports_processed', 0)
        matched = medical_needs.get('field_reports_matched', 0)
        return matched / processed if processed > 0 else 0
    
    # 时间特征辅助方法
    def _is_weekend(self, date_string: str) -> bool:
        """判断是否为周末"""
        try:
            if not date_string:
                return False
            date = datetime.fromisoformat(date_string.replace('Z', ''))
            return date.weekday() >= 5
        except:
            return False
    
    def _is_holiday_season(self, month: int) -> bool:
        """判断是否为假期季节"""
        return month in [12, 1, 7, 8]  # 冬季和夏季假期
    
    def _days_since_year_start(self, date_string: str) -> int:
        """计算距离年初的天数"""
        try:
            if not date_string:
                return 0
            date = datetime.fromisoformat(date_string.replace('Z', ''))
            year_start = datetime(date.year, 1, 1)
            return (date - year_start).days
        except:
            return 0
    
    def _encode_season(self, season: str) -> int:
        """编码季节"""
        season_map = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Autumn': 4}
        return season_map.get(season, 0)
    
    # 地理特征辅助方法
    def _is_landlocked_country(self, iso3_code: str) -> bool:
        """判断是否为内陆国家（简化版）"""
        landlocked = {'AFG', 'ARM', 'AZE', 'BTN', 'BOL', 'BWA', 'BFA', 'BDI', 'CAF', 'TCD',
                     'PRY', 'KAZ', 'KGZ', 'LAO', 'LSO', 'MKD', 'MWI', 'MLI', 'MNG', 'NPL'}
        return iso3_code in landlocked
    
    def _get_development_level(self, iso3_code: str) -> str:
        """获取发展水平（简化版）"""
        developed = {'USA', 'CAN', 'GBR', 'FRA', 'DEU', 'JPN', 'AUS', 'NZL', 'SWE', 'NOR'}
        if iso3_code in developed:
            return 'Developed'
        else:
            return 'Developing'
    
    def _get_geographic_region(self, region_id: Optional[int]) -> str:
        """获取地理区域"""
        region_map = {1: 'Africa', 2: 'Asia', 3: 'Europe', 4: 'Americas', 5: 'Oceania'}
        return region_map.get(region_id, 'Unknown')
    
    def _get_country_risk_level(self, country_name: str) -> str:
        """获取国家风险等级（简化版）"""
        high_risk_countries = {'Afghanistan', 'Somalia', 'Yemen', 'Syria', 'Haiti'}
        if country_name in high_risk_countries:
            return 'High'
        else:
            return 'Medium'
    
    # 资源供给辅助方法
    def _calculate_beds_per_affected(self, facilities: List[Dict], disaster: Dict) -> float:
        """计算每千名受影响人员的床位数"""
        total_beds = sum(f.get('bed_capacity', 0) for f in facilities)
        affected = max(disaster.get('people_affected', 1), 1)
        return (total_beds * 1000) / affected
    
    def _calculate_staff_per_affected(self, facilities: List[Dict], disaster: Dict) -> float:
        """计算每千名受影响人员的医护人员数"""
        total_staff = sum(f.get('total_staff', 0) for f in facilities)
        affected = max(disaster.get('people_affected', 1), 1)
        return (total_staff * 1000) / affected
    
    def _calculate_resource_adequacy(self, facilities: List[Dict], disaster: Dict) -> float:
        """计算资源充足性分数"""
        beds_ratio = self._calculate_beds_per_affected(facilities, disaster)
        staff_ratio = self._calculate_staff_per_affected(facilities, disaster)
        
        # 基于WHO标准的简化评估
        bed_score = min(beds_ratio / 3.0, 1.0)  # WHO推荐每千人3张床
        staff_score = min(staff_ratio / 2.3, 1.0)  # WHO推荐每千人2.3名医护
        
        return (bed_score + staff_score) / 2
    
    # 响应历史辅助方法
    def _calculate_funding_per_affected(self, disaster: Dict) -> float:
        """计算每名受影响人员的资金"""
        funded = disaster.get('amount_funded', 0)
        affected = max(disaster.get('people_affected', 1), 1)
        return funded / affected
    
    def _calculate_appeal_effectiveness(self, disaster: Dict) -> float:
        """计算呼吁有效性"""
        appeals_count = disaster.get('appeals_count', 0)
        funding_coverage = disaster.get('funding_coverage', 0)
        if appeals_count == 0:
            return 0
        return funding_coverage / appeals_count
    
    def _calculate_response_speed(self, disaster: Dict) -> float:
        """计算响应速度分数（简化版）"""
        has_appeal = disaster.get('has_active_appeal', False)
        emergency = disaster.get('emergency_response', False)
        
        score = 0
        if emergency:
            score += 0.6
        if has_appeal:
            score += 0.4
        
        return score
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """安全除法，避免除零"""
        return numerator / denominator if denominator != 0 else 0
    
    # 保存和报告方法
    def _save_engineered_features(self, features: List[Dict]) -> Path:
        """保存特征工程结果"""
        output_dir = self.config.paths.processed_data_dir / "feature_engineering"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "medical_resource_features.json"
        
        output_data = {
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'total_records': len(features),
                'feature_engineering_version': '2.0',
                'source_disasters': len(self.disasters_data),
                'source_facilities': len(self.facilities_data) if self.facilities_data else 0
            },
            'features': features
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"特征工程结果已保存到: {output_file}")
        return output_file
    
    def _generate_feature_summary(self, features: List[Dict]) -> Dict[str, Any]:
        """生成特征工程摘要"""
        if not features:
            return {'message': 'No features generated'}
        
        sample_record = features[0]
        feature_names = list(sample_record.keys())
        
        # 计算数值特征的统计信息
        numeric_stats = {}
        for feature_name in feature_names:
            if feature_name == 'disaster_id':
                continue
            
            values = [record.get(feature_name, 0) for record in features]
            try:
                numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    numeric_stats[feature_name] = {
                        'mean': np.mean(numeric_values),
                        'std': np.std(numeric_values),
                        'min': np.min(numeric_values),
                        'max': np.max(numeric_values),
                        'non_zero_count': sum(1 for v in numeric_values if v > 0)
                    }
            except:
                pass
        
        return {
            'total_records': len(features),
            'total_features': len(feature_names),
            'feature_categories': {
                'severity': len([f for f in feature_names if 'severity' in f or 'casualty' in f or 'affected' in f]),
                'medical': len([f for f in feature_names if 'medical' in f or 'injured' in f or 'health' in f]),
                'temporal': len([f for f in feature_names if any(t in f for t in ['year', 'month', 'season', 'time'])]),
                'geographic': len([f for f in feature_names if any(g in f for g in ['country', 'region', 'iso'])]),
                'resource': len([f for f in feature_names if any(r in f for r in ['facility', 'bed', 'staff'])]),
                'response': len([f for f in feature_names if any(r in f for r in ['funding', 'appeal', 'response'])]),
                'interaction': len([f for f in feature_names if 'ratio' in f or 'interaction' in f])
            },
            'numeric_feature_stats': numeric_stats
        }
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'engineered_features': [],
            'feature_summary': {'message': 'No data available for feature engineering'},
            'output_file': None,
            'total_records': 0
        }


if __name__ == "__main__":
    from src.config import load_config
    
    config = load_config()
    engineer = MedicalResourceFeatureEngineer(config)
    
    print("开始医疗资源分配特征工程...")
    result = engineer.engineer_features()
    
    summary = result['feature_summary']
    print(f"特征工程完成!")
    print(f"生成记录数: {result['total_records']}")
    print(f"特征总数: {summary.get('total_features', 0)}")
    if result['output_file']:
        print(f"结果文件: {result['output_file']}")