"""
02_data_collection/medical_facilities_collector.py

配置驱动的医疗设施数据收集器
使用统一的基础类和配置管理
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from src.data_collection.base_collector import ConfigurableCollector
from src.config import get_config, ProjectConfig


class MedicalFacilitiesCollector(ConfigurableCollector):
    """
    医疗设施数据收集器
    基于配置的IFRC Local Units API数据收集
    """

    def get_data_directory(self) -> Path:
        """获取医疗设施数据保存目录"""
        return self.config.paths.medical_facilities_dir

    def collect_data(self) -> Dict[str, Any]:
        """
        收集医疗设施数据的主方法
        """
        self.logger.info("开始收集医疗设施数据...")

        # 1. 收集所有本地单位数据
        all_units = self._collect_all_local_units()

        # 2. 筛选医疗设施
        medical_facilities = self._filter_medical_facilities(all_units)

        # 3. 按国家分组收集详细数据
        country_data = self._collect_by_countries(medical_facilities)

        # 4. 数据验证和清理
        validated_data = self._validate_and_clean_data(medical_facilities)

        # 5. 保存分类数据
        self._save_categorized_data(validated_data)

        result = {
            'data': validated_data,
            'raw_units_count': len(all_units),
            'medical_facilities_count': len(medical_facilities),
            'countries_data': country_data,
            'collection_metadata': {
                'total_processed': len(all_units),
                'medical_facilities_found': len(medical_facilities),
                'data_quality_passed': len(validated_data)
            }
        }

        return result

    def _collect_all_local_units(self) -> List[Dict]:
        """收集所有本地单位数据"""
        self.logger.info("收集所有本地单位数据...")

        return self.make_paginated_request(
            'local_units_list',
            max_records=self.config.collection.max_records_per_collection
        )

    def _filter_medical_facilities(self, all_units: List[Dict]) -> List[Dict]:
        """筛选医疗设施"""
        self.logger.info("筛选医疗设施...")

        medical_facilities = []
        config = self.config.collection.medical_facilities

        for i, unit in enumerate(all_units):
            try:
                if self._is_medical_facility(unit, config):
                    medical_facilities.append(unit)
                if (i + 1) % 100 == 0:
                    self.logger.info(f"已处理 {i + 1}/{len(all_units)} 个单位，找到 {len(medical_facilities)} 个医疗设施")
            except Exception as e:
                self.logger.error(f"处理第 {i} 个单位时出错: {e}, 单位ID: {unit.get('id', 'unknown')}")
                continue

        # 按国家过滤
        filtered_facilities = self.filter_by_countries(medical_facilities)

        self.logger.info(f"找到 {len(filtered_facilities)} 个医疗设施")
        return filtered_facilities

    def _safe_get_field(self, data: Dict, field: str) -> Any:
        """
        安全获取字段值，处理可能的类型错误
        """
        try:
            field_data = data.get(field)
            if isinstance(field_data, dict):
                return field_data
            elif field_data is not None:
                # 如果字段值不是字典但存在，返回一个包含该值的字典
                return {'value': field_data}
            return None
        except (AttributeError, TypeError):
            return None

    def _is_medical_facility(self, unit: Dict, config: Dict) -> bool:
        """
        判断是否为医疗设施（基于配置）
        """
        health_data = unit.get('health', {})
        
        # 确保health_data是字典类型
        if not isinstance(health_data, dict):
            self.logger.debug(f"单位 {unit.get('id', 'unknown')} 的health字段不是字典类型: {type(health_data)}")
            return False

        if not health_data:
            return False

        # 检查必需字段
        required_fields = config.get('required_fields', [])
        has_required_fields = any(
            self._safe_get_field(health_data, field) is not None
            for field in required_fields
        )

        # 检查人员配置
        field_mappings = self.config.get_field_mappings()
        has_medical_staff = False
        for field in field_mappings.values():
            try:
                if isinstance(field, str) and field in health_data:
                    staff_count = self.safe_int_convert(health_data.get(field, 0))
                    if staff_count >= config.get('minimum_staff_threshold', 1):
                        has_medical_staff = True
                        break
            except (AttributeError, TypeError):
                continue

        # 检查医疗容量
        has_capacity = (
                self.safe_int_convert(health_data.get('maximum_capacity', 0)) >=
                config.get('minimum_capacity_threshold', 0)
        )

        # 检查验证状态
        if not config.get('include_unvalidated', True):
            if not unit.get('validated', False):
                return False

        return has_required_fields or has_medical_staff or has_capacity

    def _collect_by_countries(self, facilities: List[Dict]) -> Dict[str, Any]:
        """按国家收集详细数据"""
        self.logger.info("按国家组织医疗设施数据...")

        country_data = {}
        countries = set(f.get('country', 'Unknown') for f in facilities)

        for country in countries:
            country_facilities = [f for f in facilities if f.get('country') == country]

            country_summary = self._analyze_country_facilities(country_facilities)
            country_data[country] = {
                'facilities_count': len(country_facilities),
                'facilities': country_facilities,
                'summary': country_summary
            }

        return country_data

    def _analyze_country_facilities(self, facilities: List[Dict]) -> Dict[str, Any]:
        """分析国家医疗设施概况"""
        if not facilities:
            return {}

        # 统计设施类型
        facility_types = {}
        total_capacity = 0
        total_staff = 0
        validated_count = 0

        field_mappings = self.config.get_field_mappings()

        for facility in facilities:
            health_data = facility.get('health', {})
            
            # 确保health_data是字典类型
            if not isinstance(health_data, dict):
                self.logger.debug(f"跳过单位 {facility.get('id', 'unknown')}，health字段类型错误: {type(health_data)}")
                continue

            # 容量统计
            capacity = self.safe_int_convert(health_data.get('maximum_capacity', 0))
            total_capacity += capacity

            # 人员统计
            facility_staff = 0
            for field in field_mappings.values():
                try:
                    if isinstance(field, str) and field in health_data:
                        facility_staff += self.safe_int_convert(health_data.get(field, 0))
                except (AttributeError, TypeError):
                    continue
            total_staff += facility_staff

            # 验证状态
            if facility.get('validated', False):
                validated_count += 1

            # 设施类型统计
            facility_type = self._classify_facility_type(health_data)
            facility_types[facility_type] = facility_types.get(facility_type, 0) + 1

        return {
            'total_facilities': len(facilities),
            'total_bed_capacity': total_capacity,
            'total_medical_staff': total_staff,
            'validated_facilities': validated_count,
            'validation_rate': validated_count / len(facilities),
            'average_capacity_per_facility': total_capacity / len(facilities),
            'average_staff_per_facility': total_staff / len(facilities),
            'facility_types_distribution': facility_types
        }

    def _classify_facility_type(self, health_data: Dict) -> str:
        """
        基于IFRC标准分类设施类型
        """
        # 确保health_data是字典类型
        if not isinstance(health_data, dict):
            return 'unknown'
            
        # 根据设施特征分类
        if health_data.get('hospital_type_details'):
            return 'hospital'
        elif health_data.get('primary_health_care_center_details'):
            return 'primary_care'
        elif health_data.get('blood_services_details'):
            return 'blood_center'
        elif health_data.get('professional_training_facilities'):
            return 'training_facility'
        elif self.safe_int_convert(health_data.get('maximum_capacity', 0)) > 0:
            return 'clinic_with_beds'
        else:
            return 'basic_clinic'

    def _validate_and_clean_data(self, facilities: List[Dict]) -> List[Dict]:
        """验证和清理数据"""
        self.logger.info("验证和清理医疗设施数据...")

        validated_facilities = []

        for facility in facilities:
            # 数据质量检查
            validation_result = self.validate_collected_data([facility])

            if validation_result['is_valid']:
                # 清理数据
                cleaned_facility = self._clean_facility_data(facility)
                # 添加元数据
                enriched_facility = self.enrich_data_with_metadata([cleaned_facility])[0]
                validated_facilities.append(enriched_facility)

        self.logger.info(f"数据验证完成：{len(validated_facilities)}/{len(facilities)} 通过验证")
        return validated_facilities

    def _clean_facility_data(self, facility: Dict) -> Dict:
        """清理单个设施数据"""
        cleaned = facility.copy()
        health_data = cleaned.get('health', {})

        if health_data:
            # 处理数值字段的异常值
            numeric_fields = ['maximum_capacity', 'number_of_isolation_rooms']
            numeric_fields.extend(self.config.get_field_mappings().values())

            for field in numeric_fields:
                if field in health_data:
                    health_data[field] = self.safe_int_convert(health_data[field])

            # 标准化布尔字段
            boolean_fields = [
                'is_teaching_hospital', 'is_in_patient_capacity',
                'is_isolation_rooms_wards', 'is_warehousing', 'is_cold_chain'
            ]

            for field in boolean_fields:
                if field in health_data:
                    health_data[field] = bool(health_data[field])

        return cleaned

    def _save_categorized_data(self, validated_data: List[Dict]) -> None:
        """保存分类数据"""
        # 按设施类型保存
        by_type = {}
        for facility in validated_data:
            facility_type = self._classify_facility_type(facility.get('health', {}))
            if facility_type not in by_type:
                by_type[facility_type] = []
            by_type[facility_type].append(facility)

        for facility_type, facilities in by_type.items():
            self.save_data(facilities, f"facilities_by_type_{facility_type}")

        # 按国家保存
        by_country = {}
        for facility in validated_data:
            country = facility.get('country', 'Unknown')
            if country not in by_country:
                by_country[country] = []
            by_country[country].append(facility)

        for country, facilities in by_country.items():
            country_safe = country.replace(' ', '_').replace('/', '_')
            self.save_data(facilities, f"facilities_by_country_{country_safe}")

    def _get_required_fields(self) -> List[str]:
        """获取医疗设施必需字段"""
        return ['id', 'country', 'health']

    def _apply_custom_filters(self, data: List[Dict]) -> List[Dict]:
        """应用医疗设施特定的过滤器"""
        config = self.config.collection.medical_facilities

        # 过滤掉质量太低的数据
        filtered_data = []
        for item in data:
            quality_score = self.calculate_data_quality_score(item, self._get_required_fields())
            if quality_score >= 0.3:  # 最低质量阈值
                filtered_data.append(item)

        return filtered_data

    def collect_facility_details_batch(self, facility_ids: List[int]) -> Dict[str, Any]:
        """
        批量收集设施详细信息
        """
        self.logger.info(f"批量收集 {len(facility_ids)} 个设施的详细信息...")

        detailed_facilities = []
        failed_ids = []

        for facility_id in facility_ids:
            try:
                # 使用缓存
                cache_key = f"facility_detail_{facility_id}"
                detail = self.make_api_request(
                    f"local_unit_detail",
                    use_cache=True,
                    cache_key=cache_key
                )

                if detail:
                    detailed_facilities.append(detail)
                else:
                    failed_ids.append(facility_id)

            except Exception as e:
                self.logger.warning(f"获取设施详情失败 ID {facility_id}: {e}")
                failed_ids.append(facility_id)

        result = {
            'detailed_facilities': detailed_facilities,
            'successful_count': len(detailed_facilities),
            'failed_ids': failed_ids,
            'success_rate': len(detailed_facilities) / len(facility_ids) if facility_ids else 0
        }

        self.save_data(result, "facility_details_batch")
        return result


if __name__ == "__main__":
    # 使用示例
    from src.config import load_config

    # 加载配置
    config = load_config()  # 可以传入配置文件路径

    # 创建收集器
    collector = MedicalFacilitiesCollector(config)

    # 运行收集
    print("开始医疗设施数据收集...")
    result = collector.run_collection()

    print(f"收集完成!")
    print(f"原始单位数: {result.get('raw_units_count', 0)}")
    print(f"医疗设施数: {result.get('medical_facilities_count', 0)}")
    print(f"通过验证的设施数: {len(result.get('data', []))}")

    # 数据质量报告
    validation = collector.validate_collected_data(result.get('data', []))
    print(f"数据质量评分: {validation['quality_score']:.2f}")

    if validation['issues']:
        print("数据质量问题:")
        for issue in validation['issues'][:5]:  # 显示前5个问题
            print(f"  - {issue}")