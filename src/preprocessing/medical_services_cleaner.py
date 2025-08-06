"""
03_preprocessing/medical_services_cleaner.py

医疗服务数据清洗模块 - 专门处理医疗服务数据的清洗
基于IFRC医疗服务标准处理服务能力和合规性数据
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from src.preprocessing.base_cleaner import BaseCleaner, DataQualityAssessor
from src.config import get_config


class MedicalServicesCleaner(BaseCleaner):
    """
    医疗服务数据清洗器
    专门处理医疗服务能力矩阵和合规性数据
    """

    def get_data_files(self) -> List[Path]:
        """获取医疗服务数据文件列表"""
        potential_files = [
            self.config.paths.medical_services_dir / "collected_data.json",
            self.config.paths.medical_services_dir / "medical_services_capability_matrix.json",
            self.config.paths.raw_data_dir / "medical_services.json"
        ]

        existing_files = [f for f in potential_files if f.exists()]
        return existing_files

    def get_output_filename(self) -> str:
        """获取输出文件名"""
        return "medical_services_cleaned.json"

    def clean_data(self) -> Dict[str, Any]:
        """清洗医疗服务数据"""
        self.logger.info("开始清洗医疗服务数据...")

        # 加载数据
        all_services = []
        data_files = self.get_data_files()

        if not data_files:
            self.logger.warning("未找到医疗服务数据文件")
            return {"services": [], "cleaning_summary": {"error": "No data files found"}}

        for file_path in data_files:
            data = self.load_data_file(file_path)
            if data:
                if isinstance(data, list):
                    all_services.extend(data)
                else:
                    all_services.append(data)

        if not all_services:
            self.logger.warning("加载的医疗服务数据为空")
            return {"services": [], "cleaning_summary": {"error": "No service data loaded"}}

        self.logger.info(f"加载了 {len(all_services)} 个医疗服务记录")

        # 清洗数据
        cleaned_services = []
        cleaning_issues = []

        for i, service in enumerate(all_services):
            try:
                cleaned_service = self._clean_single_service(service)
                if cleaned_service:
                    cleaned_services.append(cleaned_service)
                else:
                    cleaning_issues.append(f"服务记录 {i} 未通过清洗验证")
            except Exception as e:
                cleaning_issues.append(f"服务记录 {i} 清洗失败: {str(e)}")
                self.logger.warning(f"清洗服务记录 {i} 时出错: {e}")

        # 生成清洗报告
        cleaning_report = self.generate_cleaning_report(
            len(all_services), len(cleaned_services), cleaning_issues
        )

        # 数据质量评估
        quality_report = self._assess_services_quality(cleaned_services)

        # 服务能力分析
        capability_analysis = self._analyze_service_capabilities(cleaned_services)

        # 合规性分析
        compliance_analysis = self._analyze_ifrc_compliance(cleaned_services)

        self.log_cleaning_stats("医疗服务", len(all_services), len(cleaned_services))

        result = {
            "services": cleaned_services,
            "cleaning_report": cleaning_report,
            "quality_report": quality_report,
            "capability_analysis": capability_analysis,
            "compliance_analysis": compliance_analysis,
            "summary": {
                "total_services": len(cleaned_services),
                "services_by_country": self._count_by_country(cleaned_services),
                "services_by_type": self._count_by_service_type(cleaned_services)
            }
        }

        # 保存清洗后的数据
        self.save_cleaned_data(result)

        return result

    def _clean_single_service(self, service: Dict) -> Optional[Dict]:
        """清洗单个医疗服务记录"""
        if not service or not isinstance(service, dict):
            return None

        cleaned = service.copy()

        # 验证必要字段
        required_fields = ['facility_id']
        if not self.validate_required_fields(cleaned, required_fields):
            return None

        # 清洗基本信息字段
        cleaned['facility_id'] = self.safe_int_convert(cleaned.get('facility_id'))
        cleaned['facility_name'] = self.clean_string_field(cleaned.get('facility_name'))
        cleaned['english_name'] = self.clean_string_field(cleaned.get('english_name'))
        cleaned['country'] = self.clean_string_field(cleaned.get('country'))
        cleaned['location'] = self.clean_string_field(cleaned.get('location'))

        # 处理验证状态
        if 'validated' in cleaned:
            cleaned['validated'] = bool(cleaned['validated'])

        # 清洗服务提供信息
        services_provided = cleaned.get('services_provided', {})
        if services_provided and isinstance(services_provided, dict):
            cleaned['services_provided'] = self._clean_services_provided(services_provided)

        # 清洗服务容量信息
        service_capacity = cleaned.get('service_capacity', {})
        if service_capacity and isinstance(service_capacity, dict):
            cleaned['service_capacity'] = self._clean_service_capacity(service_capacity)

        # 清洗服务质量指标
        quality_indicators = cleaned.get('service_quality_indicators', {})
        if quality_indicators and isinstance(quality_indicators, dict):
            cleaned['service_quality_indicators'] = self._clean_quality_indicators(quality_indicators)

        # 清洗IFRC合规性信息
        ifrc_compliance = cleaned.get('ifrc_compliance', {})
        if ifrc_compliance and isinstance(ifrc_compliance, dict):
            cleaned['ifrc_compliance'] = self._clean_ifrc_compliance(ifrc_compliance)

        # 计算整体数据质量评分
        cleaned['_data_quality'] = self._calculate_service_quality_score(cleaned)
        cleaned['_cleaned_at'] = self.standardize_date('now')

        return cleaned

    def _clean_services_provided(self, services: Dict) -> Dict:
        """清洗服务提供信息"""
        cleaned_services = {}

        for service_category, service_data in services.items():
            if isinstance(service_data, dict):
                cleaned_category = {
                    'available': bool(service_data.get('available', False)),
                    'details': self.clean_string_field(service_data.get('details')),
                    'ifrc_category': self.clean_string_field(service_data.get('ifrc_category'))
                }

                # 处理子类别
                if 'subcategories' in service_data:
                    subcategories = service_data['subcategories']
                    if isinstance(subcategories, list):
                        cleaned_category['subcategories'] = [
                            self.clean_string_field(sub) for sub in subcategories
                        ]

                # 处理专科服务
                if 'specializations' in service_data:
                    specializations = service_data['specializations']
                    if isinstance(specializations, list):
                        cleaned_category['specializations'] = [
                            self.clean_string_field(spec) for spec in specializations
                        ]

                cleaned_services[service_category] = cleaned_category

        return cleaned_services

    def _clean_service_capacity(self, capacity: Dict) -> Dict:
        """清洗服务容量信息"""
        cleaned_capacity = {}

        # 清洗数值型容量字段
        numeric_fields = ['bed_capacity', 'isolation_capacity', 'staff_capacity']
        for field in numeric_fields:
            if field in capacity:
                value = self.safe_int_convert(capacity[field])
                # 应用合理性检查
                if field == 'bed_capacity' and value > self.cleaning_config['max_bed_capacity']:
                    value = self.cleaning_config['max_bed_capacity']
                elif field == 'staff_capacity' and value > self.cleaning_config['max_staff_count']:
                    value = self.cleaning_config['max_staff_count']
                cleaned_capacity[field] = value

        # 清洗其他容量字段
        for field, value in capacity.items():
            if field not in numeric_fields:
                if isinstance(value, (int, float)):
                    cleaned_capacity[field] = self.safe_numeric_convert(value)
                else:
                    cleaned_capacity[field] = self.clean_string_field(value)

        return cleaned_capacity

    def _clean_quality_indicators(self, indicators: Dict) -> Dict:
        """清洗服务质量指标"""
        cleaned_indicators = {}

        for indicator, value in indicators.items():
            if isinstance(value, (int, float)):
                # 确保质量指标在合理范围内 (0-1 或 0-100)
                numeric_value = float(value)
                if indicator.endswith('_score') or indicator.endswith('_rate'):
                    # 评分和比率应该在0-1之间
                    if numeric_value > 1 and numeric_value <= 100:
                        numeric_value = numeric_value / 100.0
                    cleaned_indicators[indicator] = max(0.0, min(1.0, numeric_value))
                else:
                    cleaned_indicators[indicator] = max(0.0, numeric_value)
            else:
                cleaned_indicators[indicator] = self.clean_string_field(value)

        return cleaned_indicators

    def _clean_ifrc_compliance(self, compliance: Dict) -> Dict:
        """清洗IFRC合规性信息"""
        cleaned_compliance = {}

        # 清洗合规性评分
        if 'overall_compliance_score' in compliance:
            score = self.safe_numeric_convert(compliance['overall_compliance_score'])
            cleaned_compliance['overall_compliance_score'] = max(0.0, min(1.0, score))

        # 清洗合规类别列表
        if 'compliant_categories' in compliance:
            categories = compliance['compliant_categories']
            if isinstance(categories, list):
                cleaned_compliance['compliant_categories'] = [
                    self.clean_string_field(cat) for cat in categories if cat
                ]

        # 清洗改进领域列表
        if 'improvement_areas' in compliance:
            areas = compliance['improvement_areas']
            if isinstance(areas, list):
                cleaned_compliance['improvement_areas'] = [
                    self.clean_string_field(area) for area in areas if area
                ]

        return cleaned_compliance

    def _calculate_service_quality_score(self, service: Dict) -> float:
        """计算服务数据质量评分"""
        criteria = {
            'basic_fields': ['facility_id', 'facility_name', 'country', 'services_provided'],
            'numeric_fields': []
        }

        # 添加容量相关数值字段
        capacity = service.get('service_capacity', {})
        if capacity:
            criteria['numeric_fields'].extend(['bed_capacity', 'staff_capacity'])

        assessor = DataQualityAssessor(self.config)
        quality_scores = assessor.assess_record_quality(service, criteria)

        return quality_scores.get('overall_quality_score', 0.0)

    def _assess_services_quality(self, services: List[Dict]) -> Dict[str, Any]:
        """评估服务数据整体质量"""
        if not services:
            return {}

        criteria = {
            'basic_fields': ['facility_id', 'facility_name', 'services_provided'],
            'numeric_fields': ['service_capacity.bed_capacity', 'service_capacity.staff_capacity']
        }

        assessor = DataQualityAssessor(self.config)
        return assessor.generate_quality_report(services, criteria)

    def _analyze_service_capabilities(self, services: List[Dict]) -> Dict[str, Any]:
        """分析服务能力"""
        if not services:
            return {}

        analysis = {
            'total_facilities_analyzed': len(services),
            'service_types_available': set(),
            'capacity_statistics': {
                'total_bed_capacity': 0,
                'total_staff_capacity': 0,
                'average_bed_capacity': 0,
                'average_staff_capacity': 0
            },
            'service_coverage': {}
        }

        bed_capacities = []
        staff_capacities = []

        for service in services:
            # 收集服务类型
            services_provided = service.get('services_provided', {})
            for service_type in services_provided.keys():
                analysis['service_types_available'].add(service_type)

            # 收集容量数据
            capacity = service.get('service_capacity', {})
            bed_capacity = capacity.get('bed_capacity', 0)
            staff_capacity = capacity.get('staff_capacity', 0)

            bed_capacities.append(bed_capacity)
            staff_capacities.append(staff_capacity)

            analysis['capacity_statistics']['total_bed_capacity'] += bed_capacity
            analysis['capacity_statistics']['total_staff_capacity'] += staff_capacity

        # 计算平均值
        if services:
            analysis['capacity_statistics']['average_bed_capacity'] = (
                analysis['capacity_statistics']['total_bed_capacity'] / len(services)
            )
            analysis['capacity_statistics']['average_staff_capacity'] = (
                analysis['capacity_statistics']['total_staff_capacity'] / len(services)
            )

        # 转换set为list以便JSON序列化
        analysis['service_types_available'] = list(analysis['service_types_available'])

        # 服务覆盖分析
        for service_type in analysis['service_types_available']:
            count = sum(1 for service in services
                       if service_type in service.get('services_provided', {}))
            analysis['service_coverage'][service_type] = {
                'facilities_providing': count,
                'coverage_rate': count / len(services) if services else 0
            }

        return analysis

    def _analyze_ifrc_compliance(self, services: List[Dict]) -> Dict[str, Any]:
        """分析IFRC合规性"""
        if not services:
            return {}

        compliance_scores = []
        compliant_count = 0
        all_categories = set()
        all_improvement_areas = set()

        for service in services:
            compliance = service.get('ifrc_compliance', {})
            score = compliance.get('overall_compliance_score', 0)
            compliance_scores.append(score)

            if score >= 0.5:  # 50%以上认为合规
                compliant_count += 1

            # 收集合规类别
            categories = compliance.get('compliant_categories', [])
            all_categories.update(categories)

            # 收集改进领域
            areas = compliance.get('improvement_areas', [])
            all_improvement_areas.update(areas)

        analysis = {
            'total_services_assessed': len(services),
            'compliant_services': compliant_count,
            'compliance_rate': compliant_count / len(services) if services else 0,
            'average_compliance_score': sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0,
            'compliance_distribution': {
                'excellent': sum(1 for score in compliance_scores if score >= 0.9),
                'good': sum(1 for score in compliance_scores if 0.7 <= score < 0.9),
                'satisfactory': sum(1 for score in compliance_scores if 0.5 <= score < 0.7),
                'needs_improvement': sum(1 for score in compliance_scores if score < 0.5)
            },
            'common_compliant_categories': list(all_categories),
            'common_improvement_areas': list(all_improvement_areas)
        }

        return analysis

    def _count_by_country(self, services: List[Dict]) -> Dict[str, int]:
        """按国家统计服务数量"""
        country_counts = {}
        for service in services:
            country = service.get('country', 'Unknown')
            country_counts[country] = country_counts.get(country, 0) + 1
        return country_counts

    def _count_by_service_type(self, services: List[Dict]) -> Dict[str, int]:
        """按服务类型统计"""
        type_counts = {}
        for service in services:
            services_provided = service.get('services_provided', {})
            for service_type in services_provided.keys():
                type_counts[service_type] = type_counts.get(service_type, 0) + 1
        return type_counts


if __name__ == "__main__":
    # 测试医疗服务清洗模块
    from src.config import load_config

    config = load_config()
    cleaner = MedicalServicesCleaner(config)

    print("开始医疗服务数据清洗...")
    result = cleaner.clean_data()

    print(f"清洗完成!")
    print(f"服务记录数量: {result['summary']['total_services']}")
    print(f"覆盖国家: {len(result['summary']['services_by_country'])}")
    print(f"服务类型: {list(result['summary']['services_by_type'].keys())}")

    if result.get('compliance_analysis'):
        compliance = result['compliance_analysis']
        print(f"IFRC合规率: {compliance.get('compliance_rate', 0):.2%}")
        print(f"平均合规评分: {compliance.get('average_compliance_score', 0):.2f}")

