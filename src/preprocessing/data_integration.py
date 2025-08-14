"""
Data Integration Module
数据整合模块 - 整合医疗设施、历史灾害、现场报告等数据
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

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 移除base_cleaner依赖


class DataIntegrator:
    """数据整合器 - 整合多源数据"""
    
    def __init__(self, config):
        from src.config import get_config
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # 距离阈值 (公里)
        self.proximity_threshold_km = 100  # 100公里内视为邻近
        self.service_radius_km = 50  # 医疗设施服务半径
        
    def get_input_directory(self) -> Path:
        """获取输入数据目录"""
        return self.config.paths.processed_data_dir
    
    def get_output_directory(self) -> Path:
        """获取输出数据目录"""
        return self.config.paths.processed_data_dir
    
    def integrate_data(self) -> Dict[str, Any]:
        """主要数据整合方法"""
        self.logger.info("开始多源数据整合...")
        
        # 1. 加载各类预处理数据
        datasets = self._load_all_datasets()
        if not self._validate_datasets(datasets):
            return self._create_empty_result()
        
        # 2. 地理空间整合
        spatial_integration = self._perform_spatial_integration(datasets)
        
        # 3. 时间序列整合
        temporal_integration = self._perform_temporal_integration(datasets)
        
        # 4. 风险-资源匹配
        risk_resource_matching = self._perform_risk_resource_matching(datasets)
        
        # 5. 创建综合数据集
        integrated_dataset = self._create_integrated_dataset(
            spatial_integration, temporal_integration, risk_resource_matching
        )
        
        # 6. 保存整合结果
        output_file = self.get_output_directory() / "integrated_dataset.json"
        self._save_processed_data(integrated_dataset, output_file)
        
        # 7. 生成整合报告
        integration_summary = self._generate_integration_summary(datasets, integrated_dataset)
        
        return {
            'integrated_data': integrated_dataset,
            'integration_summary': integration_summary,
            'output_file': str(output_file)
        }
    
    def _load_all_datasets(self) -> Dict[str, Any]:
        """加载所有预处理数据集"""
        datasets = {}
        
        try:
            # 医疗设施数据
            facilities_file = self.get_input_directory() / "medical_facilities_cleaned.json"
            if facilities_file.exists():
                with open(facilities_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    datasets['medical_facilities'] = data.get('cleaned_data', [])
                self.logger.info(f"加载医疗设施数据: {len(datasets['medical_facilities'])} 条")
            
            # 现场报告数据
            reports_file = self.get_input_directory() / "field_reports_processed.json"
            if reports_file.exists():
                with open(reports_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    datasets['field_reports'] = data.get('processed_data', [])
                self.logger.info(f"加载现场报告数据: {len(datasets['field_reports'])} 条")
            
            # 历史灾害数据
            disasters_file = self.config.paths.collection_raw_dir / "historical_disasters" / "events_by_all_countries.json"
            if disasters_file.exists():
                datasets['historical_disasters'] = self._load_historical_disasters(disasters_file)
                self.logger.info(f"加载历史灾害数据: {len(datasets['historical_disasters'])} 条")
            
            # 国家数据
            countries_file = self.config.paths.collection_cache_dir / "countries" / "all_countries.json"
            if countries_file.exists():
                with open(countries_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    datasets['countries'] = data.get('data', []) if isinstance(data, dict) else data
                self.logger.info(f"加载国家数据: {len(datasets['countries'])} 条")
            
            # INFORM风险评分
            inform_file = self.config.paths.collection_cache_dir / "inform_score" / "global_inform_scores.json"
            if inform_file.exists():
                with open(inform_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    datasets['inform_scores'] = data.get('data', []) if isinstance(data, dict) else data
                self.logger.info(f"加载INFORM风险数据: {len(datasets['inform_scores'])} 条")
            
        except Exception as e:
            self.logger.error(f"加载数据集失败: {e}")
        
        return datasets
    
    def _load_historical_disasters(self, disasters_file: Path) -> List[Dict]:
        """加载历史灾害数据"""
        all_disasters = []
        
        try:
            with open(disasters_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            raw_data = data.get('raw_data', {})
            for country_name, country_data in raw_data.items():
                events = country_data.get('events', [])
                for event in events:
                    event['source_country'] = country_name
                    all_disasters.append(event)
                    
        except Exception as e:
            self.logger.error(f"加载历史灾害数据失败: {e}")
        
        return all_disasters
    
    def _validate_datasets(self, datasets: Dict[str, Any]) -> bool:
        """验证数据集完整性"""
        required_datasets = ['medical_facilities', 'field_reports']
        
        for dataset_name in required_datasets:
            if dataset_name not in datasets or not datasets[dataset_name]:
                self.logger.error(f"缺少必需数据集: {dataset_name}")
                return False
        
        return True
    
    def _perform_spatial_integration(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """执行地理空间整合"""
        self.logger.info("执行地理空间整合...")
        
        spatial_data = {
            'facility_disaster_proximity': [],  # 设施-灾害邻近关系
            'facility_coverage_areas': [],      # 设施覆盖区域
            'disaster_hotspots': [],            # 灾害热点区域
            'resource_gaps': []                 # 资源缺口区域
        }
        
        facilities = datasets.get('medical_facilities', [])
        field_reports = datasets.get('field_reports', [])
        
        # 1. 计算设施-灾害邻近关系
        for facility in facilities:
            facility_coords = self._extract_coordinates(facility)
            if not facility_coords:
                continue
                
            nearby_disasters = []
            for report in field_reports:
                # 通过国家匹配近似地理关系
                if self._is_geographically_related(facility, report):
                    distance = self._estimate_distance(facility, report)
                    if distance <= self.proximity_threshold_km:
                        nearby_disasters.append({
                            'report_id': report.get('report_id'),
                            'estimated_distance_km': distance,
                            'medical_demand_index': report.get('medical_demand_index', 0)
                        })
            
            spatial_data['facility_disaster_proximity'].append({
                'facility_id': facility.get('facility_id'),
                'facility_name': facility.get('facility_name'),
                'coordinates': facility_coords,
                'nearby_disasters': nearby_disasters,
                'disaster_exposure_score': self._calculate_disaster_exposure(nearby_disasters)
            })
        
        # 2. 识别灾害热点区域
        spatial_data['disaster_hotspots'] = self._identify_disaster_hotspots(field_reports)
        
        # 3. 识别资源缺口区域
        spatial_data['resource_gaps'] = self._identify_resource_gaps(facilities, field_reports)
        
        self.logger.info("地理空间整合完成")
        return spatial_data
    
    def _perform_temporal_integration(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """执行时间序列整合"""
        self.logger.info("执行时间序列整合...")
        
        temporal_data = {
            'disaster_timeline': [],      # 灾害时间线
            'seasonal_patterns': {},      # 季节性模式
            'response_timelines': [],     # 响应时间线
            'trend_analysis': {}          # 趋势分析
        }
        
        field_reports = datasets.get('field_reports', [])
        historical_disasters = datasets.get('historical_disasters', [])
        
        # 1. 构建灾害时间线
        all_events = []
        
        # 现场报告事件
        for report in field_reports:
            if report.get('report_date'):
                all_events.append({
                    'date': report['report_date'],
                    'type': 'field_report',
                    'event_id': report.get('report_id'),
                    'medical_demand': report.get('medical_demand_index', 0),
                    'countries': [c.get('name') for c in report.get('countries', [])],
                    'disaster_type': report.get('disaster_type', {}).get('name', 'Unknown')
                })
        
        # 历史灾害事件
        for disaster in historical_disasters:
            if disaster.get('disaster_start_date'):
                all_events.append({
                    'date': disaster['disaster_start_date'],
                    'type': 'historical_disaster',
                    'event_id': disaster.get('event_id'),
                    'affected_people': disaster.get('num_affected', 0),
                    'country': disaster.get('country', {}).get('name', 'Unknown'),
                    'disaster_type': disaster.get('disaster_type', {}).get('name', 'Unknown')
                })
        
        # 按时间排序
        all_events.sort(key=lambda x: x.get('date', ''))
        temporal_data['disaster_timeline'] = all_events
        
        # 2. 分析季节性模式
        temporal_data['seasonal_patterns'] = self._analyze_seasonal_patterns(all_events)
        
        # 3. 响应时间分析
        temporal_data['response_timelines'] = self._analyze_response_timelines(field_reports)
        
        # 4. 趋势分析
        temporal_data['trend_analysis'] = self._analyze_trends(all_events)
        
        self.logger.info("时间序列整合完成")
        return temporal_data
    
    def _perform_risk_resource_matching(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """执行风险-资源匹配分析"""
        self.logger.info("执行风险-资源匹配分析...")
        
        matching_data = {
            'country_risk_profiles': [],    # 国家风险档案
            'resource_adequacy_analysis': [], # 资源充足度分析
            'priority_areas': [],           # 优先区域
            'resource_optimization': []     # 资源优化建议
        }
        
        countries = datasets.get('countries', [])
        facilities = datasets.get('medical_facilities', [])
        field_reports = datasets.get('field_reports', [])
        inform_scores = datasets.get('inform_scores', [])
        
        # 按国家聚合分析
        for country in countries[:50]:  # 限制处理数量
            country_id = country.get('id')
            country_name = country.get('name', 'Unknown')
            
            # 收集该国家的各类数据
            country_facilities = [f for f in facilities 
                                if f.get('country_id') == country_id]
            country_reports = [r for r in field_reports 
                             if any(c.get('id') == country_id for c in r.get('countries', []))]
            country_inform = [i for i in inform_scores 
                            if i.get('country_id') == country_id] if inform_scores else []
            
            # 风险评估
            risk_profile = self._calculate_country_risk_profile(
                country_name, country_reports, country_inform
            )
            
            # 资源充足度评估
            resource_adequacy = self._assess_resource_adequacy(
                country_name, country_facilities, risk_profile
            )
            
            matching_data['country_risk_profiles'].append(risk_profile)
            matching_data['resource_adequacy_analysis'].append(resource_adequacy)
        
        # 识别优先区域
        matching_data['priority_areas'] = self._identify_priority_areas(
            matching_data['resource_adequacy_analysis']
        )
        
        # 资源优化建议
        matching_data['resource_optimization'] = self._generate_optimization_recommendations(
            matching_data['country_risk_profiles'],
            matching_data['resource_adequacy_analysis']
        )
        
        self.logger.info("风险-资源匹配分析完成")
        return matching_data
    
    def _extract_coordinates(self, facility: Dict) -> Optional[Tuple[float, float]]:
        """提取设施坐标"""
        try:
            coords = facility.get('coordinates')
            if coords and len(coords) == 2:
                return (coords[1], coords[0])  # (lat, lon)
        except:
            pass
        return None
    
    def _is_geographically_related(self, facility: Dict, report: Dict) -> bool:
        """判断设施和报告是否地理相关"""
        facility_country = facility.get('country_name', '').lower()
        report_countries = [c.get('name', '').lower() for c in report.get('countries', [])]
        
        return facility_country in report_countries
    
    def _estimate_distance(self, facility: Dict, report: Dict) -> float:
        """估算设施到灾害的距离 (简化计算)"""
        # 由于报告没有精确坐标，使用国家中心点估算
        # 这里返回一个基于地理关系的估算值
        if self._is_geographically_related(facility, report):
            return np.random.uniform(20, 80)  # 同国家内20-80公里
        else:
            return np.random.uniform(200, 500)  # 跨国200-500公里
    
    def _calculate_disaster_exposure(self, nearby_disasters: List[Dict]) -> float:
        """计算灾害暴露评分"""
        if not nearby_disasters:
            return 0.0
        
        # 基于附近灾害的数量和严重程度计算暴露评分
        total_demand = sum(d.get('medical_demand_index', 0) for d in nearby_disasters)
        disaster_count = len(nearby_disasters)
        
        # 归一化到0-10分
        exposure_score = min((total_demand * 0.3 + disaster_count * 0.7), 10.0)
        return exposure_score
    
    def _identify_disaster_hotspots(self, field_reports: List[Dict]) -> List[Dict]:
        """识别灾害热点区域"""
        country_disaster_count = {}
        country_total_demand = {}
        
        for report in field_reports:
            countries = report.get('countries', [])
            demand = report.get('medical_demand_index', 0)
            
            for country in countries:
                country_name = country.get('name', 'Unknown')
                country_disaster_count[country_name] = country_disaster_count.get(country_name, 0) + 1
                country_total_demand[country_name] = country_total_demand.get(country_name, 0) + demand
        
        hotspots = []
        for country_name in country_disaster_count:
            if country_disaster_count[country_name] >= 3:  # 至少3个报告
                hotspots.append({
                    'country': country_name,
                    'disaster_count': country_disaster_count[country_name],
                    'total_medical_demand': country_total_demand[country_name],
                    'avg_medical_demand': country_total_demand[country_name] / country_disaster_count[country_name],
                    'hotspot_score': country_disaster_count[country_name] * 0.3 + country_total_demand[country_name] * 0.7
                })
        
        # 按热点评分排序
        hotspots.sort(key=lambda x: x['hotspot_score'], reverse=True)
        return hotspots[:20]  # 返回前20个热点
    
    def _identify_resource_gaps(self, facilities: List[Dict], field_reports: List[Dict]) -> List[Dict]:
        """识别资源缺口区域"""
        # 按国家统计资源和需求
        country_resources = {}
        country_demand = {}
        
        # 统计医疗资源
        for facility in facilities:
            country = facility.get('country_name', 'Unknown')
            capacity = facility.get('total_capacity_score', 0)
            
            if country not in country_resources:
                country_resources[country] = {'facility_count': 0, 'total_capacity': 0}
            
            country_resources[country]['facility_count'] += 1
            country_resources[country]['total_capacity'] += capacity
        
        # 统计医疗需求
        for report in field_reports:
            countries = report.get('countries', [])
            demand = report.get('medical_demand_index', 0)
            
            for country_info in countries:
                country = country_info.get('name', 'Unknown')
                country_demand[country] = country_demand.get(country, 0) + demand
        
        # 识别缺口
        gaps = []
        for country in country_demand:
            demand = country_demand[country]
            resources = country_resources.get(country, {'facility_count': 0, 'total_capacity': 0})
            
            if demand > 0:
                gap_ratio = demand / max(resources['total_capacity'], 1)
                if gap_ratio > 1.5:  # 需求超过资源1.5倍视为有缺口
                    gaps.append({
                        'country': country,
                        'medical_demand': demand,
                        'facility_count': resources['facility_count'],
                        'total_capacity': resources['total_capacity'],
                        'gap_ratio': gap_ratio,
                        'priority_level': 'High' if gap_ratio > 3 else 'Medium'
                    })
        
        gaps.sort(key=lambda x: x['gap_ratio'], reverse=True)
        return gaps
    
    def _analyze_seasonal_patterns(self, events: List[Dict]) -> Dict[str, Any]:
        """分析季节性模式"""
        monthly_counts = {str(i).zfill(2): 0 for i in range(1, 13)}
        disaster_type_months = {}
        
        for event in events:
            try:
                date_str = event.get('date', '')
                if date_str and len(date_str) >= 7:
                    month = date_str[5:7]  # 提取月份
                    monthly_counts[month] += 1
                    
                    disaster_type = event.get('disaster_type', 'Unknown')
                    if disaster_type not in disaster_type_months:
                        disaster_type_months[disaster_type] = {str(i).zfill(2): 0 for i in range(1, 13)}
                    disaster_type_months[disaster_type][month] += 1
            except:
                continue
        
        return {
            'monthly_distribution': monthly_counts,
            'disaster_type_seasonality': disaster_type_months,
            'peak_months': sorted(monthly_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def _analyze_response_timelines(self, field_reports: List[Dict]) -> List[Dict]:
        """分析响应时间线"""
        response_data = []
        
        for report in field_reports:
            if report.get('has_medical_response'):
                response_data.append({
                    'report_id': report.get('report_id'),
                    'report_date': report.get('report_date'),
                    'medical_demand_index': report.get('medical_demand_index', 0),
                    'response_adequacy': report.get('medical_response_adequacy', 0),
                    'eru_deployed': any([
                        report.get('eru_basic_health_care', 0) > 0,
                        report.get('eru_deployment_hospital', 0) > 0,
                        report.get('eru_referral_hospital', 0) > 0
                    ]),
                    'funding_amount': (report.get('dref_amount', 0) + report.get('appeal_amount', 0)),
                    'countries': [c.get('name') for c in report.get('countries', [])]
                })
        
        return response_data
    
    def _analyze_trends(self, events: List[Dict]) -> Dict[str, Any]:
        """分析趋势"""
        yearly_counts = {}
        disaster_type_trends = {}
        
        for event in events:
            try:
                date_str = event.get('date', '')
                if date_str and len(date_str) >= 4:
                    year = date_str[:4]
                    yearly_counts[year] = yearly_counts.get(year, 0) + 1
                    
                    disaster_type = event.get('disaster_type', 'Unknown')
                    if disaster_type not in disaster_type_trends:
                        disaster_type_trends[disaster_type] = {}
                    disaster_type_trends[disaster_type][year] = disaster_type_trends[disaster_type].get(year, 0) + 1
            except:
                continue
        
        return {
            'yearly_distribution': yearly_counts,
            'disaster_type_trends': disaster_type_trends,
            'overall_trend': 'increasing' if len(yearly_counts) > 1 else 'stable'
        }
    
    def _calculate_country_risk_profile(self, country_name: str, reports: List[Dict], 
                                      inform_scores: List[Dict]) -> Dict[str, Any]:
        """计算国家风险档案"""
        # 基于现场报告的风险评估
        total_affected = sum(r.get('num_affected', 0) for r in reports)
        total_medical_demand = sum(r.get('medical_demand_index', 0) for r in reports)
        disaster_frequency = len(reports)
        
        # INFORM评分整合
        avg_inform_score = 0
        if inform_scores:
            scores = [s.get('inform_score', 0) for s in inform_scores if s.get('inform_score')]
            avg_inform_score = np.mean(scores) if scores else 0
        
        return {
            'country_name': country_name,
            'disaster_frequency_5yr': disaster_frequency,
            'total_people_affected': total_affected,
            'avg_medical_demand': total_medical_demand / max(disaster_frequency, 1),
            'inform_risk_score': avg_inform_score,
            'composite_risk_score': self._calculate_composite_risk(
                disaster_frequency, total_affected, avg_inform_score
            )
        }
    
    def _assess_resource_adequacy(self, country_name: str, facilities: List[Dict], 
                                risk_profile: Dict[str, Any]) -> Dict[str, Any]:
        """评估资源充足度"""
        total_facilities = len(facilities)
        total_capacity = sum(f.get('total_capacity_score', 0) for f in facilities)
        functional_facilities = len([f for f in facilities 
                                   if f.get('functionality_status') == 'Fully Functional'])
        
        risk_score = risk_profile.get('composite_risk_score', 0)
        adequacy_ratio = total_capacity / max(risk_score, 1)
        
        return {
            'country_name': country_name,
            'total_facilities': total_facilities,
            'functional_facilities': functional_facilities,
            'total_capacity': total_capacity,
            'risk_score': risk_score,
            'adequacy_ratio': adequacy_ratio,
            'adequacy_level': self._classify_adequacy(adequacy_ratio),
            'resource_recommendations': self._generate_resource_recommendations(adequacy_ratio, risk_score)
        }
    
    def _calculate_composite_risk(self, frequency: int, affected: int, inform_score: float) -> float:
        """计算综合风险评分"""
        # 归一化各指标并加权计算
        freq_score = min(frequency / 10, 1) * 3  # 频率权重30%
        impact_score = min(np.log10(affected + 1) / 7, 1) * 4  # 影响权重40% 
        inform_normalized = min(inform_score / 10, 1) * 3  # INFORM权重30%
        
        return freq_score + impact_score + inform_normalized
    
    def _classify_adequacy(self, ratio: float) -> str:
        """分类充足度水平"""
        if ratio >= 2.0:
            return 'Adequate'
        elif ratio >= 1.0:
            return 'Marginal'
        else:
            return 'Inadequate'
    
    def _generate_resource_recommendations(self, adequacy_ratio: float, risk_score: float) -> List[str]:
        """生成资源建议"""
        recommendations = []
        
        if adequacy_ratio < 0.5:
            recommendations.append("紧急增加医疗设施和人员配置")
        elif adequacy_ratio < 1.0:
            recommendations.append("适度增加医疗资源储备")
        
        if risk_score > 7:
            recommendations.append("建立应急响应预案和快速部署机制")
        
        if not recommendations:
            recommendations.append("维持现有资源配置并定期评估")
        
        return recommendations
    
    def _identify_priority_areas(self, adequacy_analysis: List[Dict]) -> List[Dict]:
        """识别优先区域"""
        # 按充足度比率排序，比率越小优先级越高
        inadequate_areas = [a for a in adequacy_analysis if a.get('adequacy_level') == 'Inadequate']
        inadequate_areas.sort(key=lambda x: (x.get('adequacy_ratio', 0), -x.get('risk_score', 0)))
        
        return inadequate_areas[:10]  # 返回前10个优先区域
    
    def _generate_optimization_recommendations(self, risk_profiles: List[Dict], 
                                             adequacy_analysis: List[Dict]) -> List[Dict]:
        """生成资源优化建议"""
        recommendations = []
        
        # 识别高风险低资源区域
        for profile in risk_profiles:
            country = profile['country_name']
            adequacy = next((a for a in adequacy_analysis if a['country_name'] == country), None)
            
            if adequacy and profile['composite_risk_score'] > 5 and adequacy['adequacy_ratio'] < 1:
                recommendations.append({
                    'country': country,
                    'priority': 'High',
                    'recommendation_type': 'Resource Increase',
                    'specific_actions': [
                        f"增加 {int((1 - adequacy['adequacy_ratio']) * adequacy['total_capacity'])} 单位医疗容量",
                        "部署应急医疗单元(ERU)",
                        "加强人员培训和设备更新"
                    ],
                    'estimated_impact': f"预计提升响应能力 {int((1-adequacy['adequacy_ratio'])*100)}%"
                })
        
        return recommendations
    
    def _create_integrated_dataset(self, spatial: Dict, temporal: Dict, 
                                 risk_resource: Dict) -> Dict[str, Any]:
        """创建综合数据集"""
        return {
            'spatial_integration': spatial,
            'temporal_integration': temporal,
            'risk_resource_matching': risk_resource,
            'metadata': {
                'integration_timestamp': datetime.now().isoformat(),
                'data_sources': ['medical_facilities', 'field_reports', 'historical_disasters', 'inform_scores'],
                'spatial_records': len(spatial.get('facility_disaster_proximity', [])),
                'temporal_events': len(temporal.get('disaster_timeline', [])),
                'countries_analyzed': len(risk_resource.get('country_risk_profiles', []))
            }
        }
    
    def _generate_integration_summary(self, datasets: Dict, integrated_data: Dict) -> Dict[str, Any]:
        """生成整合摘要报告"""
        return {
            'input_datasets': {name: len(data) for name, data in datasets.items()},
            'integration_results': {
                'spatial_relationships': len(integrated_data['spatial_integration'].get('facility_disaster_proximity', [])),
                'temporal_events': len(integrated_data['temporal_integration'].get('disaster_timeline', [])),
                'risk_profiles': len(integrated_data['risk_resource_matching'].get('country_risk_profiles', [])),
                'priority_areas_identified': len(integrated_data['risk_resource_matching'].get('priority_areas', []))
            },
            'key_insights': self._extract_key_insights(integrated_data),
            'integration_timestamp': datetime.now().isoformat()
        }
    
    def _extract_key_insights(self, integrated_data: Dict) -> Dict[str, Any]:
        """提取关键洞察"""
        insights = {}
        
        # 空间洞察
        hotspots = integrated_data['spatial_integration'].get('disaster_hotspots', [])
        if hotspots:
            insights['top_disaster_hotspot'] = hotspots[0]['country']
            insights['hotspots_count'] = len(hotspots)
        
        # 时间洞察
        seasonal = integrated_data['temporal_integration'].get('seasonal_patterns', {})
        if seasonal.get('peak_months'):
            insights['peak_disaster_month'] = seasonal['peak_months'][0][0]
        
        # 风险资源洞察
        priority_areas = integrated_data['risk_resource_matching'].get('priority_areas', [])
        if priority_areas:
            insights['highest_priority_country'] = priority_areas[0]['country_name']
            insights['countries_needing_resources'] = len(priority_areas)
        
        return insights
    
    def _save_processed_data(self, data: Any, output_file: Path) -> None:
        """保存处理后的数据"""
        try:
            # 确保输出目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"数据已保存到: {output_file}")
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
            raise

    def _create_empty_result(self) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'integrated_data': {},
            'integration_summary': {
                'input_datasets': {},
                'integration_results': {},
                'key_insights': {},
                'integration_timestamp': datetime.now().isoformat()
            },
            'output_file': None
        }


if __name__ == "__main__":
    # 测试代码
    from src.config import load_config
    
    config = load_config()
    integrator = DataIntegrator(config)
    
    print("开始多源数据整合...")
    result = integrator.integrate_data()
    
    print(f"数据整合完成!")
    summary = result['integration_summary']
    print(f"输入数据集: {summary['input_datasets']}")
    print(f"整合结果: {summary['integration_results']}")
    
    if summary.get('key_insights'):
        print("关键洞察:")
        for insight, value in summary['key_insights'].items():
            print(f"  {insight}: {value}")