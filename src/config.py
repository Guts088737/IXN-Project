"""
config.py - 项目配置管理

通过配置文件统一管理所有设置，简化代码并提高可维护性
"""

import os
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
import json
import yaml


@dataclass
class APIConfig:
    """API相关配置"""
    base_url_v1: str = "https://goadmin.ifrc.org/api/v1"
    base_url_v2: str = "https://goadmin.ifrc.org/api/v2"
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 0.5
    rate_limit_delay: float = 0.5
    headers: Dict[str, str] = field(default_factory=lambda: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    })


@dataclass
class DataPathConfig:
    """数据路径配置"""
    root_dir: Path = field(default_factory=lambda: Path("data"))
    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    models_dir: Path = field(default_factory=lambda: Path("data/models"))
    results_dir: Path = field(default_factory=lambda: Path("data/results"))
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))

    # 细分的原始数据路径
    medical_facilities_dir: Path = field(default_factory=lambda: Path("data/raw/medical_facilities"))
    historical_disasters_dir: Path = field(default_factory=lambda: Path("data/raw/historical_disasters"))
    predicted_disasters_dir: Path = field(default_factory=lambda: Path("data/raw/predicted_disasters"))
    medical_services_dir: Path = field(default_factory=lambda: Path("data/raw/medical_services"))


@dataclass
class CollectionConfig:
    """数据收集配置"""
    max_records_per_collection: int = 500  # 减少到500条进行测试
    batch_size: int = 50  # 减少批次大小
    countries_of_interest: List[str] = field(default_factory=lambda: [
        'Turkey', 'Pakistan', 'Philippines', 'Indonesia', 'Bangladesh',
        'India', 'Nepal', 'Myanmar', 'Afghanistan', 'Syria'
    ])

    # 医疗设施收集配置
    medical_facilities: Dict[str, Any] = field(default_factory=lambda: {
        'include_unvalidated': True,
        'minimum_staff_threshold': 1,
        'minimum_capacity_threshold': 0,
        'required_fields': [
            'health_facility_type_details',
            'primary_health_care_center_details',
            'hospital_type_details',
            'general_medical_services_details'
        ]
    })

    # 历史灾害收集配置
    historical_disasters: Dict[str, Any] = field(default_factory=lambda: {
        'years_back': 5,
        'minimum_funding_threshold': 1000,  # 最小资金阈值
        'disaster_types_filter': [],  # 空列表表示包含所有类型
        'include_appeals': True,
        'include_operations': True
    })


@dataclass
class IFRCStandardsConfig:
    """IFRC标准配置 - 基于你提供的文档"""

    # 设施分类标准
    facility_types: Dict[str, str] = field(default_factory=lambda: {
        'primary_health_care_center': '初级卫生保健中心',
        'general_hospital': '综合医院',
        'specialized_hospital': '专科医院',
        'mental_health_hospital': '精神健康医院',
        'pharmacy_dispensary': '药房/配药中心',
        'long_term_nursing_facility': '长期护理机构',
        'professional_training_facility': '专业培训设施',
        'blood_centre': '血液中心',
        'ambulance_station': '救护车站'
    })

    # 服务分类标准
    service_categories: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        'general_medical_services': {
            'outpatient_services': '门诊服务',
            'minor_trauma_management': '轻伤处理和小手术',
            'basic_life_support': '基础生命支持和紧急稳定',
            'basic_laboratory': '基础实验室服务',
            'referral_capacity': '转诊能力'
        },
        'blood_services': {
            'donor_screening': '供血者筛查和血液采集',
            'blood_testing': '供血者血液检测',
            'blood_component_preparation': '血液成分制备',
            'blood_distribution': '血液产品分发'
        }
    })

    # 人员配置标准
    staffing_categories: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'doctors_generalist': {
            'api_field': 'general_practitioner',
            'definition': '全科医生，包括家庭和初级护理医生'
        },
        'doctors_specialist': {
            'api_field': 'specialist',
            'definition': '具有特定专业领域重点的医疗专业人员'
        },
        'nurses_professional': {
            'api_field': 'nurse',
            'definition': '为需要护理的人员提供治疗、支持和护理服务'
        },
        'midwife_professional': {
            'api_field': 'midwife',
            'definition': '规划、管理、提供和评估助产护理服务'
        },
        'dentist': {
            'api_field': 'dentist',
            'definition': '专门从事口腔、牙齿和牙龈诊断、预防、管理和治疗'
        }
    })


@dataclass
class ProcessingConfig:
    """数据预处理配置"""

    # 数据清洗参数
    data_cleaning: Dict[str, Any] = field(default_factory=lambda: {
        'max_bed_capacity': 2000,  # 超过此数值视为异常
        'max_staff_count': 1000,  # 超过此数值视为异常
        'negative_value_handling': 'set_zero',  # 处理API中的负值
        'missing_value_strategy': 'fill_zero',  # 缺失值策略
        'outlier_detection_method': 'iqr',  # 异常值检测方法
        'outlier_threshold': 3.0  # 异常值阈值
    })

    # 特征工程参数
    feature_engineering: Dict[str, Any] = field(default_factory=lambda: {
        'capacity_normalization': True,
        'calculate_ratios': True,
        'create_categorical_features': True,
        'create_interaction_features': False
    })

    # 数据质量评估
    quality_assessment: Dict[str, float] = field(default_factory=lambda: {
        'basic_info_weight': 0.3,  # 基础信息完整性权重
        'capacity_info_weight': 0.3,  # 容量信息权重
        'staffing_info_weight': 0.2,  # 人员信息权重
        'validation_weight': 0.2  # 验证状态权重
    })


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Path = field(default_factory=lambda: Path("logs/ifrc_medical_resources.log"))
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True


class ProjectConfig:
    """项目主配置类"""

    def __init__(self, config_file: str = None):
        self.api = APIConfig()
        self.paths = DataPathConfig()
        self.collection = CollectionConfig()
        self.ifrc_standards = IFRCStandardsConfig()
        self.processing = ProcessingConfig()
        self.logging = LoggingConfig()

        # 为main.py兼容性添加的属性
        self.DATA_DIR = self.paths.root_dir
        self.RAW_DATA_DIR = self.paths.raw_data_dir
        self.PROCESSED_DATA_DIR = self.paths.processed_data_dir
        self.MODELS_DIR = self.paths.models_dir
        self.RESULTS_DIR = self.paths.results_dir
        self.CACHE_DIR = self.paths.cache_dir
        
        self.RESOURCE_TYPES = ['medical_staff', 'bed_capacity', 'equipment', 'supplies']
        
        self.LOG_CONFIG = {
            'level': self.logging.level,
            'format': self.logging.format,
            'date_format': '%Y-%m-%d %H:%M:%S',
            'max_log_files': self.logging.backup_count
        }
        
        self.IFRC_API_CONFIG = {
            'base_url': self.api.base_url_v2,
            'timeout': self.api.timeout,
            'retry_attempts': self.api.retry_attempts
        }

        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)

        # 创建必要的目录
        self.create_directories()

    def create_directories(self):
        """创建所有必要的目录"""
        directories = [
            self.paths.root_dir,
            self.paths.raw_data_dir,
            self.paths.processed_data_dir,
            self.paths.models_dir,
            self.paths.results_dir,
            self.paths.cache_dir,
            self.paths.medical_facilities_dir,
            self.paths.historical_disasters_dir,
            self.paths.predicted_disasters_dir,
            self.paths.medical_services_dir,
            self.logging.file_path.parent
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def load_from_file(self, config_file: str):
        """从配置文件加载设置"""
        config_path = Path(config_file)

        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

        self._update_from_dict(config_data)

    def _update_from_dict(self, config_data: Dict[str, Any]):
        """从字典更新配置"""
        for section, values in config_data.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

    def save_to_file(self, config_file: str):
        """保存配置到文件"""
        config_data = {
            'api': self.api.__dict__,
            'paths': {k: str(v) for k, v in self.paths.__dict__.items()},
            'collection': self.collection.__dict__,
            'ifrc_standards': self.ifrc_standards.__dict__,
            'processing': self.processing.__dict__,
            'logging': {k: str(v) if isinstance(v, Path) else v
                        for k, v in self.logging.__dict__.items()}
        }

        config_path = Path(config_file)

        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2, default=str)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

    def get_api_endpoints(self) -> Dict[str, str]:
        """获取所有API端点 - 基于您提供的正确API端点"""
        return {
            # 医疗设施相关API端点
            'local_units_list': f"{self.api.base_url_v2}/public-local-units/",
            'local_unit_detail': f"{self.api.base_url_v2}/public-local-units/{{id}}/",
            'local_units_options': f"{self.api.base_url_v2}/local-units-options/",
            
            # 历史灾害数据
            'country_historical_disasters': f"{self.api.base_url_v2}/country/{{id}}/historical-disaster/",
            
            # 其他有用端点
            'disaster_types': f"{self.api.base_url_v2}/disaster_type/",
            'appeals': f"{self.api.base_url_v2}/appeal/",  # 添加appeals端点
            'countries': f"{self.api.base_url_v2}/country/",

            # V1 API端点 (如果需要预测数据)
            'earthquake': f"{self.api.base_url_v1}/earthquake/",
            'global_exposure': f"{self.api.base_url_v1}/global-exposure-data/",
            'risk_score': f"{self.api.base_url_v1}/risk-score/",
            'seasonal': f"{self.api.base_url_v1}/seasonal/",
            'country_seasonal': f"{self.api.base_url_v1}/country-seasonal/",
            'early_actions': f"{self.api.base_url_v1}/early-actions/",
            'inform_score': f"{self.api.base_url_v1}/inform-score/"
        }

    def get_field_mappings(self) -> Dict[str, str]:
        """获取API字段映射"""
        return {
            staff_type: config['api_field']
            for staff_type, config in self.ifrc_standards.staffing_categories.items()
        }

    def validate_config(self) -> List[str]:
        """验证配置的有效性"""
        issues = []

        # 验证必要的目录
        if not self.paths.root_dir.exists():
            issues.append(f"根目录不存在: {self.paths.root_dir}")

        # 验证API配置
        if not self.api.base_url_v1.startswith('http'):
            issues.append("API v1 URL格式无效")

        if not self.api.base_url_v2.startswith('http'):
            issues.append("API v2 URL格式无效")

        # 验证数据收集配置
        if self.collection.max_records_per_collection <= 0:
            issues.append("最大记录数必须大于0")

        if not self.collection.countries_of_interest:
            issues.append("至少需要指定一个关注国家")

        return issues


# 全局配置实例
config = ProjectConfig()


def load_config(config_file: str = None) -> ProjectConfig:
    """加载配置"""
    global config
    if config_file:
        config = ProjectConfig(config_file)
    return config


def get_config(env: str = None) -> ProjectConfig:
    """获取全局配置"""
    if env:
        load_config_from_env()
    return config

def validate_config(config_obj: ProjectConfig) -> List[str]:
    """验证配置的有效性"""
    return config_obj.validate_config()


# 环境变量支持
def load_config_from_env():
    """从环境变量加载配置覆盖"""
    global config

    # API配置
    if 'IFRC_API_BASE_URL_V1' in os.environ:
        config.api.base_url_v1 = os.environ['IFRC_API_BASE_URL_V1']

    if 'IFRC_API_BASE_URL_V2' in os.environ:
        config.api.base_url_v2 = os.environ['IFRC_API_BASE_URL_V2']

    if 'IFRC_API_TIMEOUT' in os.environ:
        config.api.timeout = int(os.environ['IFRC_API_TIMEOUT'])

    # 数据路径配置
    if 'IFRC_DATA_ROOT' in os.environ:
        root = Path(os.environ['IFRC_DATA_ROOT'])
        config.paths.root_dir = root
        config.paths.raw_data_dir = root / "01_raw"
        config.paths.processed_data_dir = root / "02_processed"
        # ... 更新其他路径

    # 日志配置
    if 'IFRC_LOG_LEVEL' in os.environ:
        config.logging.level = os.environ['IFRC_LOG_LEVEL']


if __name__ == "__main__":
    # 生成默认配置文件
    config = ProjectConfig()

    # 保存为YAML格式
    config.save_to_file("config.yaml")
    print("默认配置已保存到 config.yaml")

    # 保存为JSON格式
    config.save_to_file("config.json")
    print("默认配置已保存到 config.json")

    # 验证配置
    issues = config.validate_config()
    if issues:
        print("配置验证发现问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("配置验证通过")