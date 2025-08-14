"""
Data Quality Assessment Module
医疗资源分配项目数据质量评估模块 - 评估历史灾害和医疗设施数据质量
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import sys
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.config import ProjectConfig, load_config
except ImportError:
    from config import ProjectConfig, load_config


class MedicalResourceDataQualityAssessor:
    """医疗资源分配数据质量评估器"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 数据质量评估标准
        self.quality_standards = {
            'completeness_threshold': 0.7,      # 完整性阈值70%
            'accuracy_threshold': 0.8,          # 准确性阈值80%
            'consistency_threshold': 0.85,      # 一致性阈值85%
            'timeliness_months': 60,            # 时效性阈值5年
            'outlier_std_threshold': 3.0,       # 异常值标准差阈值
            'field_reports_match_threshold': 0.6  # Field Reports匹配阈值60%
        }
        
        # 合理数值范围（医疗资源分配相关）
        self.value_ranges = {
            'people_affected': (0, 100_000_000),
            'people_injured': (0, 10_000_000),
            'people_dead': (0, 1_000_000),
            'people_displaced': (0, 50_000_000),
            'amount_requested': (0, 1_000_000_000),
            'amount_funded': (0, 1_000_000_000),
            'bed_capacity': (0, 5000),
            'total_staff': (0, 2000),
            'severity_score': (0, 10),
            'medical_need_intensity': (0, 1)
        }
        
        # 加载数据
        self.disasters_data = None
        self.facilities_data = None
        self.features_data = None
        self._load_all_datasets()
    
    def _load_all_datasets(self):
        """加载所有相关数据集"""
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
            
            # 加载特征工程数据（如果存在）
            features_file = self.config.paths.processed_data_dir / "feature_engineering" / "medical_resource_features.json"
            if features_file.exists():
                with open(features_file, 'r', encoding='utf-8') as f:
                    features_data = json.load(f)
                self.features_data = features_data.get('features', [])
                self.logger.info(f"加载特征工程数据: {len(self.features_data)} 条记录")
                
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            self.disasters_data = []
            self.facilities_data = []
            self.features_data = []
    
    def assess_data_quality(self) -> Dict[str, Any]:
        """主要数据质量评估方法"""
        self.logger.info("开始医疗资源分配数据质量评估...")
        
        if not self.disasters_data:
            self.logger.warning("没有灾害数据，无法进行完整评估")
            return self._create_empty_result()
        
        # 1. 历史灾害数据质量评估
        disasters_quality = self._assess_disasters_quality()
        
        # 2. 医疗设施数据质量评估
        facilities_quality = self._assess_facilities_quality()
        
        # 3. 数据整合质量评估
        integration_quality = self._assess_data_integration_quality()
        
        # 4. 特征工程数据质量评估
        features_quality = self._assess_features_quality()
        
        # 5. 综合质量评估
        overall_quality = self._calculate_overall_quality(
            disasters_quality, facilities_quality, integration_quality, features_quality
        )
        
        # 6. 生成评估报告
        assessment_report = {
            'assessment_metadata': {
                'assessment_date': datetime.now().isoformat(),
                'assessor_version': '2.0',
                'quality_standards': self.quality_standards
            },
            'disasters_quality': disasters_quality,
            'facilities_quality': facilities_quality,
            'integration_quality': integration_quality,
            'features_quality': features_quality,
            'overall_quality': overall_quality,
            'recommendations': self._generate_recommendations(
                disasters_quality, facilities_quality, integration_quality, features_quality
            )
        }
        
        # 7. 保存评估结果
        output_file = self._save_assessment_report(assessment_report)
        assessment_report['output_file'] = str(output_file)
        
        return assessment_report
    
    def _assess_disasters_quality(self) -> Dict[str, Any]:
        """评估历史灾害数据质量"""
        self.logger.info("评估历史灾害数据质量...")
        
        if not self.disasters_data:
            return {'message': 'No disasters data available'}
        
        # 完整性评估
        completeness = self._assess_disasters_completeness()
        
        # 准确性评估
        accuracy = self._assess_disasters_accuracy()
        
        # 一致性评估
        consistency = self._assess_disasters_consistency()
        
        # 时效性评估
        timeliness = self._assess_disasters_timeliness()
        
        # 医疗需求数据质量
        medical_needs_quality = self._assess_medical_needs_quality()
        
        quality_score = np.mean([
            completeness.get('overall_score', 0),
            accuracy.get('overall_score', 0),
            consistency.get('overall_score', 0),
            timeliness.get('overall_score', 0),
            medical_needs_quality.get('overall_score', 0)
        ])
        
        return {
            'total_records': len(self.disasters_data),
            'completeness': completeness,
            'accuracy': accuracy,
            'consistency': consistency,
            'timeliness': timeliness,
            'medical_needs_quality': medical_needs_quality,
            'overall_score': quality_score,
            'quality_level': self._get_quality_level(quality_score)
        }
    
    def _assess_disasters_completeness(self) -> Dict[str, Any]:
        """评估灾害数据完整性"""
        essential_fields = [
            'disaster_id', 'name', 'disaster_type', 'start_date', 
            'country_name', 'people_affected'
        ]
        
        medical_fields = [
            'people_injured', 'people_dead', 'field_reports_count', 'medical_needs'
        ]
        
        completeness_scores = {}
        
        # 基本字段完整性
        for field in essential_fields:
            non_empty_count = sum(1 for record in self.disasters_data 
                                if record.get(field) not in [None, '', 0, {}])
            completeness_scores[field] = non_empty_count / len(self.disasters_data)
        
        # 医疗相关字段完整性
        for field in medical_fields:
            non_empty_count = sum(1 for record in self.disasters_data 
                                if record.get(field) not in [None, '', 0, {}])
            completeness_scores[f'medical_{field}'] = non_empty_count / len(self.disasters_data)
        
        overall_score = np.mean(list(completeness_scores.values()))
        
        return {
            'field_completeness': completeness_scores,
            'overall_score': overall_score,
            'meets_threshold': overall_score >= self.quality_standards['completeness_threshold']
        }
    
    def _assess_disasters_accuracy(self) -> Dict[str, Any]:
        """评估灾害数据准确性"""
        accuracy_issues = {
            'value_range_violations': 0,
            'negative_values': 0,
            'suspicious_zeros': 0,
            'date_format_errors': 0,
            'outliers': 0
        }
        
        total_checks = 0
        
        for record in self.disasters_data:
            total_checks += 1
            
            # 数值范围检查
            for field, (min_val, max_val) in self.value_ranges.items():
                value = record.get(field, 0)
                if isinstance(value, (int, float)):
                    if not (min_val <= value <= max_val):
                        accuracy_issues['value_range_violations'] += 1
                    if value < 0 and field.startswith('people_'):
                        accuracy_issues['negative_values'] += 1
            
            # 日期格式检查
            start_date = record.get('start_date')
            if start_date:
                try:
                    self._parse_iso_date(start_date)
                except:
                    accuracy_issues['date_format_errors'] += 1
            
            # 可疑的零值检查（大规模灾害但无伤亡）
            if (record.get('people_affected', 0) > 10000 and 
                record.get('people_injured', 0) == 0 and 
                record.get('people_dead', 0) == 0):
                accuracy_issues['suspicious_zeros'] += 1
        
        # 异常值检测
        accuracy_issues['outliers'] = self._detect_statistical_outliers()
        
        # 计算准确性分数
        total_issues = sum(accuracy_issues.values())
        accuracy_score = max(0, 1 - (total_issues / (total_checks * 5)))  # 5个检查维度
        
        return {
            'accuracy_issues': accuracy_issues,
            'overall_score': accuracy_score,
            'meets_threshold': accuracy_score >= self.quality_standards['accuracy_threshold']
        }
    
    def _assess_disasters_consistency(self) -> Dict[str, Any]:
        """评估灾害数据一致性"""
        consistency_issues = {
            'duplicate_ids': 0,
            'inconsistent_names': 0,
            'country_mismatch': 0,
            'date_inconsistencies': 0
        }
        
        # 检查重复ID
        disaster_ids = [record.get('disaster_id') for record in self.disasters_data if record.get('disaster_id')]
        consistency_issues['duplicate_ids'] = len(disaster_ids) - len(set(disaster_ids))
        
        # 检查国家名称一致性
        country_names = Counter(record.get('country_name', '').strip() for record in self.disasters_data)
        # 简单的相似度检查（实际项目中可以使用更复杂的算法）
        similar_countries = 0
        country_list = list(country_names.keys())
        for i, country1 in enumerate(country_list):
            for country2 in country_list[i+1:]:
                if country1 and country2 and self._strings_similar(country1, country2):
                    similar_countries += 1
        consistency_issues['inconsistent_names'] = similar_countries
        
        # 计算一致性分数
        total_records = len(self.disasters_data)
        total_issues = sum(consistency_issues.values())
        consistency_score = max(0, 1 - (total_issues / total_records))
        
        return {
            'consistency_issues': consistency_issues,
            'overall_score': consistency_score,
            'meets_threshold': consistency_score >= self.quality_standards['consistency_threshold']
        }
    
    def _assess_disasters_timeliness(self) -> Dict[str, Any]:
        """评估灾害数据时效性"""
        current_date = datetime.now()
        threshold_date = current_date - timedelta(days=self.quality_standards['timeliness_months'] * 30)
        
        recent_count = 0
        date_parseable_count = 0
        
        for record in self.disasters_data:
            start_date = record.get('start_date')
            if start_date:
                try:
                    disaster_date = self._parse_iso_date(start_date)
                    date_parseable_count += 1
                    if disaster_date >= threshold_date:
                        recent_count += 1
                except:
                    continue
        
        timeliness_score = recent_count / max(date_parseable_count, 1)
        
        return {
            'total_records': len(self.disasters_data),
            'parseable_dates': date_parseable_count,
            'recent_records': recent_count,
            'timeliness_score': timeliness_score,
            'threshold_months': self.quality_standards['timeliness_months'],
            'meets_threshold': timeliness_score >= 0.8  # 80%的数据应该是近期的
        }
    
    def _assess_medical_needs_quality(self) -> Dict[str, Any]:
        """评估医疗需求数据质量"""
        total_disasters = len(self.disasters_data)
        
        # 统计医疗需求数据的可用性
        has_medical_data = 0
        has_field_reports = 0
        field_reports_matched = 0
        field_reports_total = 0
        
        for record in self.disasters_data:
            medical_needs = record.get('medical_needs', {})
            
            # 检查是否有医疗需求数据
            if any(medical_needs.get(field, 0) > 0 for field in [
                'total_injured_reported', 'total_dead_reported', 
                'total_affected_reported', 'total_assisted'
            ]):
                has_medical_data += 1
            
            # 检查field reports情况
            if record.get('field_reports_count', 0) > 0:
                has_field_reports += 1
                
                # Field reports匹配情况
                field_reports_total += medical_needs.get('field_reports_processed', 0)
                field_reports_matched += medical_needs.get('field_reports_matched', 0)
        
        # 计算质量指标
        medical_data_coverage = has_medical_data / total_disasters
        field_reports_coverage = has_field_reports / total_disasters
        field_reports_match_rate = field_reports_matched / max(field_reports_total, 1)
        
        overall_score = np.mean([
            medical_data_coverage,
            field_reports_coverage,
            field_reports_match_rate
        ])
        
        return {
            'total_disasters': total_disasters,
            'medical_data_coverage': medical_data_coverage,
            'field_reports_coverage': field_reports_coverage,
            'field_reports_match_rate': field_reports_match_rate,
            'field_reports_matched': field_reports_matched,
            'field_reports_total': field_reports_total,
            'overall_score': overall_score,
            'meets_threshold': field_reports_match_rate >= self.quality_standards['field_reports_match_threshold']
        }
    
    def _assess_facilities_quality(self) -> Dict[str, Any]:
        """评估医疗设施数据质量"""
        self.logger.info("评估医疗设施数据质量...")
        
        if not self.facilities_data:
            return {'message': 'No facilities data available', 'overall_score': 0}
        
        # 完整性评估
        essential_fields = ['facility_id', 'facility_name', 'country_name', 'facility_type_name']
        completeness_scores = {}
        
        for field in essential_fields:
            non_empty_count = sum(1 for record in self.facilities_data 
                                if record.get(field) not in [None, '', 0])
            completeness_scores[field] = non_empty_count / len(self.facilities_data)
        
        # 数值字段准确性
        numeric_fields = ['bed_capacity', 'total_staff']
        accuracy_issues = 0
        total_numeric_checks = 0
        
        for record in self.facilities_data:
            for field in numeric_fields:
                value = record.get(field, 0)
                total_numeric_checks += 1
                if not isinstance(value, (int, float)) or value < 0:
                    accuracy_issues += 1
                elif field in self.value_ranges:
                    min_val, max_val = self.value_ranges[field]
                    if not (min_val <= value <= max_val):
                        accuracy_issues += 1
        
        completeness_score = np.mean(list(completeness_scores.values()))
        accuracy_score = 1 - (accuracy_issues / max(total_numeric_checks, 1))
        overall_score = (completeness_score + accuracy_score) / 2
        
        return {
            'total_records': len(self.facilities_data),
            'completeness': {
                'field_completeness': completeness_scores,
                'overall_score': completeness_score
            },
            'accuracy': {
                'accuracy_issues': accuracy_issues,
                'total_checks': total_numeric_checks,
                'overall_score': accuracy_score
            },
            'overall_score': overall_score,
            'quality_level': self._get_quality_level(overall_score)
        }
    
    def _assess_data_integration_quality(self) -> Dict[str, Any]:
        """评估数据整合质量"""
        self.logger.info("评估数据整合质量...")
        
        integration_quality = {
            'country_coverage': 0,
            'data_correspondence': 0,
            'geographic_consistency': 0
        }
        
        if not self.disasters_data:
            return {'message': 'No data available for integration assessment', 'overall_score': 0}
        
        # 国家覆盖度分析
        disaster_countries = set(record.get('country_name', '').strip() 
                               for record in self.disasters_data if record.get('country_name'))
        
        if self.facilities_data:
            facility_countries = set(record.get('country_name', '').strip() 
                                   for record in self.facilities_data if record.get('country_name'))
            
            # 计算国家覆盖重叠度
            common_countries = disaster_countries.intersection(facility_countries)
            integration_quality['country_coverage'] = len(common_countries) / max(len(disaster_countries), 1)
            
            # 数据对应度（有灾害的国家是否有医疗设施数据）
            countries_with_both = len(common_countries)
            integration_quality['data_correspondence'] = countries_with_both / len(disaster_countries)
        
        # 地理一致性（国家名称标准化程度）
        country_name_variations = len(disaster_countries)
        expected_countries = len(set(name.upper().strip() for name in disaster_countries))
        integration_quality['geographic_consistency'] = expected_countries / max(country_name_variations, 1)
        
        overall_score = np.mean(list(integration_quality.values()))
        
        return {
            'disaster_countries_count': len(disaster_countries),
            'facility_countries_count': len(self.facilities_data) if self.facilities_data else 0,
            'common_countries_count': len(disaster_countries.intersection(
                set(record.get('country_name', '') for record in (self.facilities_data or []))
            )),
            'integration_metrics': integration_quality,
            'overall_score': overall_score,
            'quality_level': self._get_quality_level(overall_score)
        }
    
    def _assess_features_quality(self) -> Dict[str, Any]:
        """评估特征工程数据质量"""
        self.logger.info("评估特征工程数据质量...")
        
        if not self.features_data:
            return {'message': 'No features data available', 'overall_score': 0}
        
        # 特征完整性
        sample_record = self.features_data[0] if self.features_data else {}
        total_features = len(sample_record.keys()) - 1  # 排除disaster_id
        
        # 特征值分布分析
        feature_quality = {}
        numeric_features = []
        
        for feature_name in sample_record.keys():
            if feature_name == 'disaster_id':
                continue
                
            values = [record.get(feature_name, 0) for record in self.features_data]
            
            # 检查数值特征
            try:
                numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    numeric_features.append(feature_name)
                    
                    # 计算特征质量指标
                    non_zero_ratio = sum(1 for v in numeric_values if v != 0) / len(numeric_values)
                    variance = np.var(numeric_values)
                    has_outliers = len([v for v in numeric_values 
                                      if abs(v - np.mean(numeric_values)) > 3 * np.std(numeric_values)]) > 0
                    
                    feature_quality[feature_name] = {
                        'non_zero_ratio': non_zero_ratio,
                        'variance': variance,
                        'has_outliers': has_outliers,
                        'quality_score': min(non_zero_ratio + (1 if variance > 0 else 0), 1.0)
                    }
            except:
                pass
        
        # 计算整体特征质量
        if feature_quality:
            overall_score = np.mean([metrics['quality_score'] for metrics in feature_quality.values()])
        else:
            overall_score = 0
        
        return {
            'total_records': len(self.features_data),
            'total_features': total_features,
            'numeric_features_count': len(numeric_features),
            'feature_quality_details': feature_quality,
            'overall_score': overall_score,
            'quality_level': self._get_quality_level(overall_score)
        }
    
    # 辅助方法
    def _detect_statistical_outliers(self) -> int:
        """检测统计异常值"""
        outlier_count = 0
        numeric_fields = ['people_affected', 'people_injured', 'people_dead', 'amount_requested']
        
        for field in numeric_fields:
            values = [record.get(field, 0) for record in self.disasters_data 
                     if isinstance(record.get(field), (int, float))]
            
            if len(values) > 10:  # 需要足够的数据点
                mean_val = np.mean(values)
                std_val = np.std(values)
                threshold = self.quality_standards['outlier_std_threshold'] * std_val
                
                outliers = [v for v in values if abs(v - mean_val) > threshold]
                outlier_count += len(outliers)
        
        return outlier_count
    
    def _strings_similar(self, s1: str, s2: str, threshold: float = 0.8) -> bool:
        """简单的字符串相似度检查"""
        if not s1 or not s2:
            return False
        
        # 简单的Jaccard相似度
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return (intersection / union) > threshold if union > 0 else False
    
    def _calculate_overall_quality(self, disasters_quality: Dict, facilities_quality: Dict, 
                                  integration_quality: Dict, features_quality: Dict) -> Dict[str, Any]:
        """计算综合数据质量"""
        
        scores = []
        weights = []
        
        # 灾害数据质量（权重最高）
        if disasters_quality.get('overall_score') is not None:
            scores.append(disasters_quality['overall_score'])
            weights.append(0.4)
        
        # 医疗设施数据质量
        if facilities_quality.get('overall_score') is not None:
            scores.append(facilities_quality['overall_score'])
            weights.append(0.2)
        
        # 数据整合质量
        if integration_quality.get('overall_score') is not None:
            scores.append(integration_quality['overall_score'])
            weights.append(0.2)
        
        # 特征工程数据质量
        if features_quality.get('overall_score') is not None:
            scores.append(features_quality['overall_score'])
            weights.append(0.2)
        
        # 计算加权平均
        if scores and weights:
            # 标准化权重
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            overall_score = sum(score * weight for score, weight in zip(scores, normalized_weights))
        else:
            overall_score = 0
        
        return {
            'overall_score': overall_score,
            'quality_level': self._get_quality_level(overall_score),
            'component_scores': {
                'disasters': disasters_quality.get('overall_score', 0),
                'facilities': facilities_quality.get('overall_score', 0),
                'integration': integration_quality.get('overall_score', 0),
                'features': features_quality.get('overall_score', 0)
            },
            'readiness_for_ml': self._assess_ml_readiness(overall_score)
        }
    
    def _get_quality_level(self, score: float) -> str:
        """根据分数获取质量等级"""
        if score >= 0.9:
            return 'Excellent'
        elif score >= 0.8:
            return 'Good'
        elif score >= 0.7:
            return 'Fair'
        elif score >= 0.6:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _assess_ml_readiness(self, overall_score: float) -> Dict[str, Any]:
        """评估机器学习准备度"""
        if overall_score >= 0.8:
            readiness = 'Ready'
            recommendation = '数据质量良好，可以开始机器学习模型训练'
        elif overall_score >= 0.7:
            readiness = 'Nearly Ready'
            recommendation = '数据质量基本满足要求，建议进行小幅改进后开始训练'
        elif overall_score >= 0.6:
            readiness = 'Needs Improvement'
            recommendation = '数据质量需要改进，建议解决主要质量问题后再训练'
        else:
            readiness = 'Not Ready'
            recommendation = '数据质量不足，需要大幅改进才能用于机器学习'
        
        return {
            'readiness_level': readiness,
            'recommendation': recommendation,
            'minimum_score_for_ml': 0.7
        }
    
    def _generate_recommendations(self, disasters_quality: Dict, facilities_quality: Dict, 
                                integration_quality: Dict, features_quality: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 灾害数据建议
        if disasters_quality.get('overall_score', 0) < 0.8:
            completeness = disasters_quality.get('completeness', {})
            if not completeness.get('meets_threshold', True):
                recommendations.append("提高历史灾害数据的完整性，特别是基本字段的填写")
            
            medical_needs = disasters_quality.get('medical_needs_quality', {})
            if medical_needs.get('field_reports_match_rate', 0) < 0.7:
                recommendations.append("改进Field Reports数据匹配，提高医疗需求数据的可用性")
        
        # 设施数据建议
        if facilities_quality.get('overall_score', 0) < 0.7:
            recommendations.append("补充医疗设施数据，确保关键字段的完整性")
        
        # 数据整合建议
        if integration_quality.get('overall_score', 0) < 0.7:
            integration_metrics = integration_quality.get('integration_metrics', {})
            if integration_metrics.get('country_coverage', 0) < 0.5:
                recommendations.append("增加医疗设施数据覆盖的国家，提高与灾害数据的匹配度")
        
        # 特征工程建议
        if features_quality.get('overall_score', 0) < 0.8:
            recommendations.append("优化特征工程过程，提高特征的质量和多样性")
        
        if not recommendations:
            recommendations.append("数据质量整体良好，可以考虑进一步优化以提升模型性能")
        
        return recommendations
    
    def _save_assessment_report(self, assessment_report: Dict) -> Path:
        """保存评估报告"""
        output_dir = self.config.paths.processed_data_dir / "quality_assessment"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"data_quality_report_{timestamp}.json"
        
        # 转换numpy类型为Python原生类型
        serializable_report = self._convert_to_serializable(assessment_report)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"数据质量评估报告已保存到: {output_file}")
        return output_file
    
    def _convert_to_serializable(self, obj):
        """将numpy类型转换为JSON可序列化的类型"""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    def _parse_iso_date(self, date_string: str) -> datetime:
        """兼容Python 3.6的ISO日期解析"""
        # 移除时区标识
        date_str = str(date_string).replace('Z', '').replace('+00:00', '')
        
        # 尝试不同的日期格式
        formats = [
            '%Y-%m-%dT%H:%M:%S',      # 2025-03-23T00:04:00
            '%Y-%m-%dT%H:%M:%S.%f',   # 2025-03-23T00:04:00.123456
            '%Y-%m-%d',               # 2025-03-23
            '%Y-%m-%d %H:%M:%S'       # 2025-03-23 00:04:00
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # 如果所有格式都失败，抛出异常
        raise ValueError(f"无法解析日期格式: {date_string}")
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'message': 'No data available for quality assessment',
            'overall_quality': {'overall_score': 0, 'quality_level': 'No Data'},
            'recommendations': ['请先运行数据预处理步骤生成数据']
        }


if __name__ == "__main__":
    from src.config import load_config
    
    config = load_config()
    assessor = MedicalResourceDataQualityAssessor(config)
    
    print("开始数据质量评估...")
    result = assessor.assess_data_quality()
    
    overall_quality = result.get('overall_quality', {})
    print(f"数据质量评估完成!")
    print(f"整体质量分数: {overall_quality.get('overall_score', 0):.3f}")
    print(f"质量等级: {overall_quality.get('quality_level', 'Unknown')}")
    
    ml_readiness = overall_quality.get('readiness_for_ml', {})
    print(f"机器学习准备度: {ml_readiness.get('readiness_level', 'Unknown')}")
    
    recommendations = result.get('recommendations', [])
    if recommendations:
        print("\n改进建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    if result.get('output_file'):
        print(f"\n详细报告: {result['output_file']}")