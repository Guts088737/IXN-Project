"""
03_preprocessing/medical_facilities_cleaner.py

医疗设施数据清洗模块 - 专门处理医疗设施数据的清洗
基于IFRC标准处理医疗设施数据质量问题
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from src.preprocessing.base_cleaner import BaseCleaner, DataQualityAssessor
from src.config import get_config


class MedicalFacilitiesCleaner(BaseCleaner):
    """
    医疗设施数据清洗器
    专门处理IFRC Local Units API收集的医疗设施数据
    """

    def get_data_files(self) -> List[Path]:
        """获取医疗设施数据文件列表"""
        potential_files = [
            self.config.paths.medical_facilities_dir / "collected_data.json",
            self.config.paths.medical_facilities_dir / "medical_facilities_filtered.json",
            self.config.paths.raw_data_dir / "medical_facilities.json"
        ]

        existing_files = [f for f in potential_files if f.exists()]
        return existing_files

    def get_output_filename(self) -> str:
        """获取输出文件名"""
        return "medical_facilities_cleaned.json"

    def clean_data(self) -> Dict[str, Any]:
        """清洗医疗设施数据"""
        self.logger.info("开始清洗医疗设施数据...")

        # 加载数据
        all_facilities = []
        data_files = self.get_data_files()

        if not data_files:
            self.logger.warning("未找到医疗设施数据文件")
            return {"facilities": [], "cleaning_summary": {"error": "No data files found"}}

        for file_path in data_files:
            data = self.load_data_file(file_path)
            if data:
                if isinstance(data, list):
                    all_facilities.extend(data)
                else:
                    all_facilities.append(data)

        if not all_facilities:
            self.logger.warning("加载的医疗设施数据为空")
            return {"facilities": [], "cleaning_summary": {"error": "No facility data loaded"}}

        self.logger.info(f"加载了 {len(all_facilities)} 个医疗设施记录")

        # 清洗数据
        cleaned_facilities = []
        cleaning_issues = []

        for i, facility in enumerate(all_facilities):
            try:
                cleaned_facility = self._clean_single_facility(facility)
                if cleaned_facility:
                    cleaned_facilities.append(cleaned_facility)
                else:
                    cleaning_issues.append(f"设施记录 {i} 未通过清洗验证")
            except Exception as e:
                cleaning_issues.append(f"设施记录 {i} 清洗失败: {str(e)}")
                self.logger.warning(f"清洗设施记录 {i} 时出错: {e}")

        # 生成清洗报告
        cleaning_report = self.generate_cleaning_report(
            len(all_facilities), len(cleaned_facilities), cleaning_issues
        )

        # 数据质量评估
        quality_report = self._assess_facilities_quality(cleaned_facilities)

        # 分类数据
        categorized_data = self._categorize_facilities(cleaned_facilities)

        self.log_cleaning_stats("医疗设施", len(all_facilities), len(cleaned_facilities))

        result = {
            "facilities": cleaned_facilities,
            "cleaning_report": cleaning_report,
            "quality_report": quality_report,
            "categorized_data": categorized_data,
            "summary": {
                "total_facilities": len(cleaned_facilities),
                "facilities_by_country": self._count_by_country(cleaned_facilities),
                "facilities_by_type": self._count_by_type(cleaned_facilities)
            }
        }

        # 保存清洗后的数据
        self.save_cleaned_data(result)

        return result

    def _clean_single_facility(self, facility: Dict) -> Optional[Dict]:
        """清洗单个医疗设施记录"""
        if not facility or not isinstance(facility, dict):
            return None

        cleaned = facility.copy()

        # 验证必要字段
        required_fields = ['id', 'country']
        if not self.validate_required_fields(cleaned, required_fields):
            return None

        # 清洗基本信息字段
        cleaned['id'] = self.safe_int_convert(cleaned.get('id'))
        cleaned['country'] = self.clean_string_field(cleaned.get('country'))
        cleaned['local_branch_name'] = self.clean_string_field(cleaned.get('local_branch_name'))
        cleaned['english_branch_name'] = self.clean_string_field(cleaned.get('english_branch_name'))

        # 处理健康数据
        health_data = cleaned.get('health', {})
        if health_data and isinstance(health_data, dict):
            cleaned_health = self._clean_health_data(health_data)
            cleaned['health'] = cleaned_health
        else:
            return None  # 没有健康数据的设施不符合要求

        # 清洗布尔字段
        boolean_fields = ['validated']
        for field in boolean_fields:
            if field in cleaned:
                cleaned[field] = bool(cleaned[field])

        # 清洗日期字段
        date_fields = ['created_at', 'updated_at']
        for field in date_fields:
            if field in cleaned:
                cleaned[field] = self.standardize_date(cleaned[field])

        # 添加数据质量标记
        cleaned['_data_quality'] = self._calculate_facility_quality_score(cleaned)
        cleaned['_cleaned_at'] = self.standardize_date('now')

        return cleaned

    def _clean_health_data(self, health_data: Dict) -> Dict:
        """清洗健康数据部分"""
        cleaned_health = health_data.copy()

        # 清理容量数据
        capacity_fields = ['maximum_capacity', 'number_of_isolation_rooms']
        for field in capacity_fields:
            if field in cleaned_health:
                value = self.safe_int_convert(cleaned_health[field])

                # 检查异常值
                if field == 'maximum_capacity' and value > self.cleaning_config['max_bed_capacity']:
                    self.logger.warning(f"异常床位容量: {value}, 设置为最大值")
                    value = self.cleaning_config['max_bed_capacity']

                cleaned_health[field] = value

        # 清理人员数据
        staff_mappings = self.config.get_field_mappings()
        for staff_type, api_field in staff_mappings.items():
            if api_field in cleaned_health:
                value = self.safe_int_convert(cleaned_health[api_field])

                # 检查异常值
                if value > self.cleaning_config['max_staff_count']:
                    self.logger.warning(f"异常人员数量 {api_field}: {value}, 设置为最大值")
                    value = self.cleaning_config['max_staff_count']

                cleaned_health[api_field] = value

        # 标准化布尔字段
        boolean_fields = [
            'is_teaching_hospital', 'is_in_patient_capacity',
            'is_isolation_rooms_wards', 'is_warehousing', 'is_cold_chain'
        ]
        for field in boolean_fields:
            if field in cleaned_health:
                cleaned_health[field] = bool(cleaned_health[field])

        # 清理文本字段
        text_fields = [
            'health_facility_type_details', 'primary_health_care_center_details',
            'hospital_type_details', 'general_medical_services_details',
            'blood_services_details', 'professional_training_facilities'
        ]
        for field in text_fields:
            if field in cleaned_health:
                cleaned_health[field] = self.clean_string_field(cleaned_health[field])

        return cleaned_health

    def _calculate_facility_quality_score(self, facility: Dict) -> float:
        """计算设施数据质量评分"""
        criteria = {
            'basic_fields': ['id', 'country', 'local_branch_name'],
            'numeric_fields': []
        }

        # 添加健康数据中的数值字段
        health_data = facility.get('health', {})
        if health_data:
            criteria['numeric_fields'].extend(['maximum_capacity', 'number_of_isolation_rooms'])
            staff_fields = list(self.config.get_field_mappings().values())
            criteria['numeric_fields'].extend(staff_fields)

        assessor = DataQualityAssessor(self.config)
        quality_scores = assessor.assess_record_quality(facility, criteria)

        return quality_scores.get('overall_quality_score', 0.0)

    def _assess_facilities_quality(self, facilities: List[Dict]) -> Dict[str, Any]:
        """评估设施数据整体质量"""
        if not facilities:
            return {}

        criteria = {
            'basic_fields': ['id', 'country', 'local_branch_name', 'health'],
            'numeric_fields': ['health.maximum_capacity']
        }

        assessor = DataQualityAssessor(self.config)
        return assessor.generate_quality_report(facilities, criteria)

    def _categorize_facilities(self, facilities: List[Dict]) -> Dict[str, Any]:
        """对设施进行分类"""
        categories = {
            'by_type': {},
            'by_country': {},
            'by_validation_status': {'validated': 0, 'unvalidated': 0},
            'by_capacity': {'small': 0, 'medium': 0, 'large': 0}
        }

        for facility in facilities:
            # 按类型分类
            facility_type = self._classify_facility_type(facility.get('health', {}))
            categories['by_type'][facility_type] = categories['by_type'].get(facility_type, 0) + 1

            # 按国家分类
            country = facility.get('country', 'Unknown')
            categories['by_country'][country] = categories['by_country'].get(country, 0) + 1

            # 按验证状态分类
            if facility.get('validated', False):
                categories['by_validation_status']['validated'] += 1
            else:
                categories['by_validation_status']['unvalidated'] += 1

            # 按容量分类
            capacity = facility.get('health', {}).get('maximum_capacity', 0)
            if capacity == 0:
                categories['by_capacity']['small'] += 1
            elif capacity <= 50:
                categories['by_capacity']['medium'] += 1
            else:
                categories['by_capacity']['large'] += 1

        return categories

    def _classify_facility_type(self, health_data: Dict) -> str:
        """根据健康数据分类设施类型"""
        if not health_data:
            return 'unknown'

        if health_data.get('hospital_type_details'):
            return 'hospital'
        elif health_data.get('primary_health_care_center_details'):
            return 'primary_care'
        elif health_data.get('blood_services_details'):
            return 'blood_center'
        elif health_data.get('professional_training_facilities'):
            return 'training_facility'
        elif health_data.get('maximum_capacity', 0) > 0:
            return 'clinic_with_beds'
        else:
            return 'basic_clinic'

    def _count_by_country(self, facilities: List[Dict]) -> Dict[str, int]:
        """按国家统计设施数量"""
        country_counts = {}
        for facility in facilities:
            country = facility.get('country', 'Unknown')
            country_counts[country] = country_counts.get(country, 0) + 1
        return country_counts

    def _count_by_type(self, facilities: List[Dict]) -> Dict[str, int]:
        """按类型统计设施数量"""
        type_counts = {}
        for facility in facilities:
            facility_type = self._classify_facility_type(facility.get('health', {}))
            type_counts[facility_type] = type_counts.get(facility_type, 0) + 1
        return type_counts

    def get_cleaning_statistics(self) -> Dict[str, Any]:
        """获取清洗统计信息"""
        # 读取最新的清洗结果
        output_file = self.config.paths.processed_data_dir / self.get_output_filename()
        if output_file.exists():
            data = self.load_data_file(output_file)
            if data and 'cleaning_report' in data:
                return data['cleaning_report']

        return {"error": "No cleaning statistics available"}


if __name__ == "__main__":
    # 测试医疗设施清洗模块
    from src.config import load_config

    config = load_config()
    cleaner = MedicalFacilitiesCleaner(config)

    print("开始医疗设施数据清洗...")
    result = cleaner.clean_data()

    print(f"清洗完成!")
    print(f"设施数量: {result['summary']['total_facilities']}")
    print(f"覆盖国家: {len(result['summary']['facilities_by_country'])}")
    print(f"设施类型: {list(result['summary']['facilities_by_type'].keys())}")

    if result.get('quality_report'):
        quality = result['quality_report'].get('quality_assessment', {})
        print(f"数据质量评分: {quality.get('average_quality_score', 0):.2f}")

