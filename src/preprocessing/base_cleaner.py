"""
03_preprocessing/base_cleaner.py

基础数据清洗类 - 提供通用的数据清洗功能
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
from src.config import get_config


class BaseCleaner(ABC):
    """
    基础数据清洗器抽象类
    提供通用的数据清洗功能和模板方法
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = self._setup_logger()
        self.cleaning_config = self.config.processing.data_cleaning

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(getattr(logging, self.config.logging.level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(self.config.logging.format)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @abstractmethod
    def clean_data(self) -> Dict[str, Any]:
        """子类必须实现的清洗数据方法"""
        pass

    @abstractmethod
    def get_data_files(self) -> List[Path]:
        """子类必须实现：获取需要清洗的数据文件列表"""
        pass

    @abstractmethod
    def get_output_filename(self) -> str:
        """子类必须实现：获取输出文件名"""
        pass

    def load_data_file(self, file_path: Path) -> Optional[List[Dict]]:
        """加载数据文件的通用方法"""
        if not file_path.exists():
            self.logger.warning(f"数据文件不存在: {file_path}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 处理不同的数据结构
            if isinstance(data, dict):
                if 'data' in data:
                    return data['data']
                elif 'metadata' in data and 'data' in data:
                    return data['data']
                else:
                    # 如果是单个记录，包装成列表
                    return [data]
            else:
                return data

        except Exception as e:
            self.logger.error(f"加载数据文件失败 {file_path}: {e}")
            return None

    def save_cleaned_data(self, cleaned_data: Dict[str, Any], output_filename: str = None) -> None:
        """保存清洗后的数据"""
        output_dir = self.config.paths.processed_data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = output_filename or self.get_output_filename()
        output_file = output_dir / filename

        # 添加清洗元数据
        final_data = {
            'cleaned_data': cleaned_data,
            'cleaning_metadata': {
                'cleaned_at': datetime.now().isoformat(),
                'cleaner_class': self.__class__.__name__,
                'cleaning_config': self.cleaning_config,
                'config_version': '1.0'
            }
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"清洗后的数据已保存: {output_file}")
        except Exception as e:
            self.logger.error(f"保存清洗后的数据失败: {e}")

    def standardize_date(self, date_value: Any) -> Optional[str]:
        """标准化日期格式的通用方法"""
        if not date_value:
            return None

        if isinstance(date_value, str):
            # 尝试解析不同日期格式
            date_formats = ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%fZ']
            for fmt in date_formats:
                try:
                    # 处理不同长度的日期字符串
                    date_str = date_value[:19] if 'T' in date_value else date_value[:10]
                    if fmt.startswith('%Y-%m-%dT') and 'T' not in date_value:
                        continue
                    parsed_date = datetime.strptime(date_str, fmt[:len(date_str)])
                    return parsed_date.strftime('%Y-%m-%d')
                except:
                    continue

        return str(date_value) if date_value else None

    def safe_numeric_convert(self, value: Any, default: float = 0.0) -> float:
        """安全的数值转换"""
        if value is None:
            return default

        try:
            # 处理负值
            numeric_value = float(value)
            if numeric_value < 0:
                if self.cleaning_config.get('negative_value_handling') == 'set_zero':
                    return 0.0
            return numeric_value
        except (ValueError, TypeError):
            return default

    def safe_int_convert(self, value: Any, default: int = 0) -> int:
        """安全的整数转换"""
        return int(self.safe_numeric_convert(value, default))

    def clean_string_field(self, value: Any) -> Optional[str]:
        """清理字符串字段"""
        if not value:
            return None

        if isinstance(value, str):
            # 去除前后空格，标准化
            cleaned = value.strip()
            return cleaned if cleaned else None

        return str(value)

    def validate_required_fields(self, record: Dict, required_fields: List[str]) -> bool:
        """验证必要字段"""
        return all(record.get(field) is not None for field in required_fields)

    def calculate_data_completeness(self, record: Dict, all_fields: List[str]) -> float:
        """计算数据完整性评分"""
        if not all_fields:
            return 1.0

        available_fields = sum(1 for field in all_fields if record.get(field) is not None)
        return available_fields / len(all_fields)

    def detect_outliers_iqr(self, values: List[float], multiplier: float = 1.5) -> Tuple[float, float]:
        """使用IQR方法检测异常值边界"""
        if len(values) < 4:
            return float('-inf'), float('inf')

        sorted_values = sorted(values)
        n = len(sorted_values)
        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        return lower_bound, upper_bound

    def generate_cleaning_report(self, original_count: int, cleaned_count: int,
                                 issues: List[str] = None) -> Dict[str, Any]:
        """生成清洗报告"""
        issues = issues or []

        report = {
            'cleaning_summary': {
                'original_records': original_count,
                'cleaned_records': cleaned_count,
                'records_removed': original_count - cleaned_count,
                'cleaning_rate': cleaned_count / original_count if original_count > 0 else 0,
                'issues_found': len(issues),
                'issues_list': issues[:10]  # 只保留前10个问题
            },
            'cleaning_config_used': self.cleaning_config,
            'timestamp': datetime.now().isoformat()
        }

        return report

    def log_cleaning_stats(self, data_type: str, original_count: int, cleaned_count: int):
        """记录清洗统计信息"""
        removal_rate = (original_count - cleaned_count) / original_count if original_count > 0 else 0
        self.logger.info(f"{data_type} 清洗完成:")
        self.logger.info(f"  原始记录: {original_count}")
        self.logger.info(f"  清洗后记录: {cleaned_count}")
        self.logger.info(f"  移除率: {removal_rate:.2%}")


class DataQualityAssessor:
    """数据质量评估器"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.quality_weights = self.config.processing.quality_assessment

    def assess_record_quality(self, record: Dict, assessment_criteria: Dict) -> Dict[str, float]:
        """评估单条记录的质量"""
        scores = {}

        # 基础信息完整性
        basic_fields = assessment_criteria.get('basic_fields', [])
        if basic_fields:
            basic_score = sum(1 for field in basic_fields if record.get(field)) / len(basic_fields)
            scores['basic_info_score'] = basic_score

        # 数据准确性（基于数值合理性）
        numeric_fields = assessment_criteria.get('numeric_fields', [])
        if numeric_fields:
            valid_numerics = 0
            for field in numeric_fields:
                value = record.get(field)
                if value is not None:
                    try:
                        num_val = float(value)
                        if num_val >= 0:  # 简单的合理性检查
                            valid_numerics += 1
                    except:
                        pass
            accuracy_score = valid_numerics / len(numeric_fields) if numeric_fields else 1.0
            scores['accuracy_score'] = accuracy_score

        # 计算综合质量评分
        overall_score = 0
        for score_type, score_value in scores.items():
            weight_key = score_type.replace('_score', '_weight')
            weight = self.quality_weights.get(weight_key, 0.5)
            overall_score += score_value * weight

        scores['overall_quality_score'] = overall_score
        return scores

    def generate_quality_report(self, records: List[Dict], assessment_criteria: Dict) -> Dict[str, Any]:
        """生成数据质量报告"""
        if not records:
            return {'error': 'No records to assess'}

        quality_scores = []
        for record in records:
            scores = self.assess_record_quality(record, assessment_criteria)
            quality_scores.append(scores['overall_quality_score'])

        avg_quality = sum(quality_scores) / len(quality_scores)
        high_quality_count = sum(1 for score in quality_scores if score >= 0.7)

        return {
            'quality_assessment': {
                'total_records': len(records),
                'average_quality_score': avg_quality,
                'high_quality_records': high_quality_count,
                'high_quality_rate': high_quality_count / len(records),
                'quality_distribution': {
                    'excellent': sum(1 for score in quality_scores if score >= 0.9),
                    'good': sum(1 for score in quality_scores if 0.7 <= score < 0.9),
                    'fair': sum(1 for score in quality_scores if 0.5 <= score < 0.7),
                    'poor': sum(1 for score in quality_scores if score < 0.5)
                }
            },
            'assessment_criteria': assessment_criteria,
            'timestamp': datetime.now().isoformat()
        }

