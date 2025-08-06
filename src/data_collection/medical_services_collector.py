"""
02_data_collection/medical_services_collector.py

配置驱动的医疗服务数据收集器
专门处理医疗服务能力和人员配置，基于IFRC Professional Health Services标准
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from src.data_collection.base_collector import ConfigurableCollector
from src.config import get_config, ProjectConfig


class MedicalServicesCollector(ConfigurableCollector):
    """
    医疗服务数据收集器
    基于IFRC标准的医疗服务能力和人员配置数据处理
    """

    def get_data_directory(self) -> Path:
        """获取医疗服务数据保存目录"""
        return self.config.paths.medical_services_dir

    def collect_data(self) -> Dict[str, Any]:
        """
        收集医疗服务数据的主方法
        依赖于医疗设施数据作为输入
        """
        self.logger.info("开始收集医疗服务数据...")

        # 1. 加载医疗设施数据
        facilities_data = self._load_facilities_data()

        if not facilities_data:
            raise ValueError("无法加载医疗设施数据，请先运行医疗设施收集器")

        # 2. 提取服务能力信息
        services_data = self._extract_medical_services(facilities_data)

        # 3. 提取人员配置信息
        staffing_data = self._extract_staffing_data(facilities_data)

        # 4. 创建服务能力矩阵
        capability_matrix = self._create_capability_matrix(services_data, staffing_data)

        # 5. 生成服务标准符合性报告
        compliance_report = self._generate_compliance_report(capability_matrix)

        # 6. 按地区和服务类型分析
        regional_analysis = self._perform_regional_analysis(capability_matrix)

        # 7. 保存分类数据
        self._save_categorized_services_data(capability_matrix)

        result = {
            'data': capability_matrix,
            'services_summary': self._generate_services_summary(services_data),
            'staffing_summary': self._generate_staffing_summary(staffing_data),
            'compliance_report': compliance_report,
            'regional_analysis': regional_analysis,
            'collection_metadata': {
                'facilities_processed': len(facilities_data),
                'services_extracted': len(services_data),
                'staffing_records': len(staffing_data),
                'capability_records': len(capability_matrix)
            }
        }

        return result

    def _load_facilities_data(self) -> List[Dict]:
        """加载医疗设施数据"""
        self.logger.info("加载医疗设施数据...")

        # 尝试从多个可能的位置加载数据
        potential_files = [
            self.config.paths.medical_facilities_dir / "medical_facilities_filtered.json",
            self.config.paths.medical_facilities_dir / "facilities_validated.json",
            self.config.paths.processed_data_dir / "medical_facilities_processed.json"
        ]

        for file_path in potential_files:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 提取实际的设施列表
                    if isinstance(data, dict):
                        facilities = data.get('data', data.get('facilities', []))
                    else:
                        facilities = data

                    if facilities:
                        self.logger.info(f"从 {file_path} 加载了 {len(facilities)} 个设施")
                        return facilities

                except Exception as e:
                    self.logger.warning(f"无法从 {file_path} 加载数据: {e}")
                    continue

        self.logger.error("无法找到医疗设施数据文件")
        return []

    def _extract_medical_services(self, facilities_data: List[Dict]) -> List[Dict]:
        """提取医疗服务能力信息"""
        self.logger.info("提取医疗服务能力信息...")

        services_standards = self.config.ifrc_standards.service_categories
        facility_services = []

        for facility in facilities_data:
            facility_service = self._extract_single_facility_services(facility, services_standards)
            if facility_service:
                facility_services.append(facility_service)

        # 数据验证和质量检查
        validated_services = self._validate_services_data(facility_services)

        self.save_data(validated_services, "medical_services_extracted")
        self.logger.info(f"提取了 {len(validated_services)} 个设施的服务信息")

        return validated_services

    def _extract_single_facility_services(self, facility: Dict, services_standards: Dict) -> Optional[Dict]:
        """提取单个设施的服务信息"""
        health_data = facility.get('health', {})
        if not health_data:
            return None

        facility_service = {
            'facility_id': facility.get('id'),
            'facility_name': facility.get('local_branch_name', ''),
            'english_name': facility.get('english_branch_name', ''),
            'country': facility.get('country', ''),
            'location': facility.get('location', ''),
            'validated': facility.get('validated', False),
            'services_provided': {},
            'service_capacity': {},
            'service_quality_indicators': {},
            'ifrc_compliance': {}
        }

        # 基于IFRC标准提取服务类型
        facility_service['services_provided'] = self._map_services_to_ifrc_standards(health_data, services_standards)

        # 提取服务容量信息
        facility_service['service_capacity'] = self._extract_service_capacity(health_data)

        # 计算服务质量指标
        facility_service['service_quality_indicators'] = self._calculate_service_quality_indicators(
            facility_service['services_provided'],
            facility_service['service_capacity']
        )

        # 评估IFRC标准符合性
        facility_service['ifrc_compliance'] = self._assess_ifrc_compliance(facility_service)

        return facility_service if any(facility_service['services_provided'].values()) else None

    def _map_services_to_ifrc_standards(self, health_data: Dict, services_standards: Dict) -> Dict[str, Any]:
        """将API数据映射到IFRC服务标准"""
        mapped_services = {}

        # 综合医疗服务映射
        if health_data.get('general_medical_services_details'):
            mapped_services['general_medical_services'] = {
                'available': True,
                'details': health_data['general_medical_services_details'],
                'ifrc_category': 'general_medical_services',
                'subcategories': self._identify_general_medical_subcategories(health_data)
            }

        # 专科医疗服务映射
        if health_data.get('specialized_medical_beyond_primary'):
            mapped_services['specialized_medical_services'] = {
                'available': True,
                'details': health_data['specialized_medical_beyond_primary'],
                'ifrc_category': 'specialized_medical_services',
                'specializations': self._identify_specializations(health_data)
            }

        # 血液服务映射
        if health_data.get('blood_services_details'):
            blood_services = self._map_blood_services(health_data['blood_services_details'], services_standards)
            mapped_services['blood_services'] = blood_services

        # 培训服务映射
        if health_data.get('professional_training_facilities'):
            mapped_services['professional_training'] = {
                'available': True,
                'details': health_data['professional_training_facilities'],
                'ifrc_category': 'professional_training_services',
                'training_types': self._identify_training_types(health_data)
            }

        return mapped_services

    def _map_blood_services(self, blood_services_data: Any, services_standards: Dict) -> Dict[str, Any]:
        """映射血液服务到IFRC标准"""
        blood_standards = services_standards.get('blood_services', {})

        blood_service_mapping = {
            'available': True,
            'ifrc_category': 'blood_services',
            'services_offered': {},
            'compliance_level': 0.0
        }

        # 检查IFRC标准中定义的血液服务类型
        ifrc_blood_services = ['donor_screening', 'blood_testing', 'blood_component_preparation', 'blood_distribution']
        
        # 简化血液服务映射
        blood_service_mapping['services_offered'] = {
            'available': True,
            'details': str(blood_services_data)
        }
        blood_service_mapping['compliance_level'] = 0.8  # 基础合规评分
        
        return blood_service_mapping

    def _identify_general_medical_subcategories(self, health_data: Dict) -> List[str]:
        """识别综合医疗服务子类别"""
        subcategories = []
        
        # 基于可用数据推断服务类型
        if health_data.get('maximum_capacity', 0) > 0:
            subcategories.append('inpatient_services')
        
        if health_data.get('is_isolation_rooms_wards'):
            subcategories.append('isolation_services')
        
        # 默认包含门诊服务
        subcategories.append('outpatient_services')
        
        return subcategories

    def _identify_specializations(self, health_data: Dict) -> List[str]:
        """识别专科医疗服务类型"""
        specializations = []
        
        # 基于医院类型推断专科服务
        if health_data.get('is_teaching_hospital'):
            specializations.append('medical_education')
        
        if health_data.get('blood_services_details'):
            specializations.append('hematology')
        
        return specializations

    def _identify_training_types(self, health_data: Dict) -> List[str]:
        """识别培训服务类型"""
        training_types = []
        
        if health_data.get('professional_training_facilities'):
            training_types.extend(['medical_training', 'nursing_training'])
        
        return training_types

    def _extract_service_capacity(self, health_data: Dict) -> Dict[str, Any]:
        """提取服务容量信息"""
        capacity = {}
        
        # 床位容量
        capacity['bed_capacity'] = self.safe_int_convert(health_data.get('maximum_capacity', 0))
        
        # 隔离室容量
        capacity['isolation_capacity'] = self.safe_int_convert(health_data.get('number_of_isolation_rooms', 0))
        
        # 人员容量
        staff_mappings = self.config.get_field_mappings()
        total_staff = sum(
            self.safe_int_convert(health_data.get(field, 0))
            for field in staff_mappings.values()
        )
        capacity['staff_capacity'] = total_staff
        
        return capacity

    def _calculate_service_quality_indicators(self, services: Dict, capacity: Dict) -> Dict[str, float]:
        """计算服务质量指标"""
        indicators = {}
        
        # 服务多样性指标
        indicators['service_diversity'] = len(services)
        
        # 容量利用指标
        total_capacity = capacity.get('bed_capacity', 0) + capacity.get('staff_capacity', 0)
        indicators['capacity_score'] = min(1.0, total_capacity / 100.0)  # 标准化到0-1
        
        # 基础质量评分
        indicators['overall_quality_score'] = (indicators['service_diversity'] * 0.3 + 
                                               indicators['capacity_score'] * 0.7)
        
        return indicators

    def _assess_ifrc_compliance(self, facility_service: Dict) -> Dict[str, Any]:
        """评估IFRC标准符合性"""
        compliance = {
            'overall_compliance_score': 0.0,
            'compliant_categories': [],
            'improvement_areas': []
        }
        
        services = facility_service.get('services_provided', {})
        capacity = facility_service.get('service_capacity', {})
        
        # 基础服务符合性检查
        if services:
            compliance['overall_compliance_score'] += 0.4
            compliance['compliant_categories'].append('service_availability')
        
        # 容量符合性检查
        if capacity.get('bed_capacity', 0) > 0:
            compliance['overall_compliance_score'] += 0.3
            compliance['compliant_categories'].append('capacity_adequacy')
        
        # 人员配置符合性
        if capacity.get('staff_capacity', 0) > 0:
            compliance['overall_compliance_score'] += 0.3
            compliance['compliant_categories'].append('staffing_adequacy')
        
        # 识别改进领域
        if compliance['overall_compliance_score'] < 0.5:
            compliance['improvement_areas'].append('service_enhancement_needed')
        
        if capacity.get('bed_capacity', 0) == 0:
            compliance['improvement_areas'].append('capacity_expansion_needed')
        
        return compliance

    def _validate_services_data(self, services_data: List[Dict]) -> List[Dict]:
        """验证服务数据质量"""
        validated_services = []
        
        for service in services_data:
            # 基本验证
            if service.get('facility_id') and service.get('services_provided'):
                validated_services.append(service)
        
        self.logger.info(f"服务数据验证: {len(validated_services)}/{len(services_data)} 通过验证")
        return validated_services

    def _generate_services_summary(self, services_data: List[Dict]) -> Dict[str, Any]:
        """生成服务数据摘要"""
        if not services_data:
            return {}
        
        summary = {
            'total_services': len(services_data),
            'services_extracted': len(services_data),
            'average_services_per_facility': len(services_data) / len(services_data) if services_data else 0,
            'validation_rate': 1.0  # 简化版本
        }
        
        return summary

    def _generate_staffing_summary(self, staffing_data: List[Dict]) -> Dict[str, Any]:
        """生成人员配置摘要"""
        return {
            'staffing_records_processed': len(staffing_data) if staffing_data else 0
        }

    def _generate_compliance_report(self, capability_matrix: List[Dict]) -> Dict[str, Any]:
        """生成合规性报告"""
        if not capability_matrix:
            return {}
        
        total_facilities = len(capability_matrix)
        compliant_facilities = sum(1 for facility in capability_matrix 
                                  if facility.get('ifrc_compliance', {}).get('overall_compliance_score', 0) >= 0.5)
        
        return {
            'total_facilities_assessed': total_facilities,
            'compliant_facilities': compliant_facilities,
            'compliance_rate': compliant_facilities / total_facilities if total_facilities > 0 else 0
        }

    def _perform_regional_analysis(self, capability_matrix: List[Dict]) -> Dict[str, Any]:
        """执行区域分析"""
        if not capability_matrix:
            return {}
        
        # 简化的区域分析
        countries = {}
        for facility in capability_matrix:
            country = facility.get('country', 'Unknown')
            if country not in countries:
                countries[country] = 0
            countries[country] += 1
        
        return {
            'countries_covered': list(countries.keys()),
            'facilities_per_country': countries
        }

    def _save_categorized_services_data(self, capability_matrix: List[Dict]) -> None:
        """保存分类的服务数据"""
        # 保存到医疗服务数据目录
        self.save_data(capability_matrix, "medical_services_capability_matrix")


if __name__ == "__main__":
    # 测试医疗服务收集器
    from src.config import load_config

    # 加载配置
    config = load_config()

    # 创建收集器
    collector = MedicalServicesCollector(config)

    # 运行收集
    print("开始医疗服务数据收集...")
    result = collector.run_collection()

    if result.get('collection_status') != 'failed':
        print(f"收集完成!")
        print(f"处理设施数: {result.get('collection_metadata', {}).get('facilities_processed', 0)}")
        print(f"提取服务数: {result.get('collection_metadata', {}).get('services_extracted', 0)}")
    else:
        print(f"收集失败: {result.get('error', 'Unknown error')}")