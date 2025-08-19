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
    
    # 预测API基础URL
    prediction_base_url: str = "http://go-risk-staging.northeurope.cloudapp.azure.com/api/v1"
    
    # Risk API endpoints
    risk_api_endpoints: Dict[str, str] = field(default_factory=lambda: {
        'earthquake': 'http://go-risk-staging.northeurope.cloudapp.azure.com/api/v1/earthquake/',
        'global_exposure': 'http://go-risk-staging.northeurope.cloudapp.azure.com/api/v1/global-exposure-data/',
        'seasonal': 'http://go-risk-staging.northeurope.cloudapp.azure.com/api/v1/seasonal/',
        'country_seasonal': 'http://go-risk-staging.northeurope.cloudapp.azure.com/api/v1/country-seasonal/',
        'pdc': 'http://go-risk-staging.northeurope.cloudapp.azure.com/api/v1/pdc/',
        'early_actions': 'http://go-risk-staging.northeurope.cloudapp.azure.com/api/v1/early-actions/',
        'risk_score': 'http://go-risk-staging.northeurope.cloudapp.azure.com/api/v1/risk-score/',
        'adam_exposure': 'http://go-risk-staging.northeurope.cloudapp.azure.com/api/v1/adam-exposure/',
        'inform_score': 'http://go-risk-staging.northeurope.cloudapp.azure.com/api/v1/inform-score/',
        'gdacs': 'http://go-risk-staging.northeurope.cloudapp.azure.com/api/v1/gdacs/',
        'meteoswiss': 'http://go-risk-staging.northeurope.cloudapp.azure.com/api/v1/meteoswiss/',
        'gwis': 'http://go-risk-staging.northeurope.cloudapp.azure.com/api/v1/gwis/'
    })
    
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 0.5
    rate_limit_delay: float = 0.5
    headers: Dict[str, str] = field(default_factory=lambda: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    })
    
    # API缓存配置
    cache_timeout: int = 3600  # 1小时缓存


class DataPathConfig:
    """数据路径配置 - 使用绝对路径避免创建多余目录"""
    
    def __init__(self):
        # 获取项目根目录（config.py的上级目录）
        self.project_root = Path(__file__).parent.parent
        
        # 基础路径
        self.root_dir = self.project_root / "data"
        
        # 主要数据目录（基于项目根目录的绝对路径）
        self.src_dir = self.project_root / "src"
        self.data_collection_dir = self.src_dir / "data_collection"
        self.preprocessing_dir = self.src_dir / "preprocessing" 
        self.models_dir = self.src_dir / "models"
        self.output_dir = self.src_dir / "output"
        
        # data_collection模块的数据路径
        self.collection_data_dir = self.data_collection_dir / "data"
        self.collection_cache_dir = self.collection_data_dir / "cache"
        self.collection_raw_dir = self.collection_data_dir / "raw"
        
        # 保持向后兼容的原始路径
        self.raw_data_dir = self.collection_raw_dir
        self.cache_dir = self.collection_cache_dir
        
        # preprocessing模块的数据路径
        self.processed_data_dir = self.preprocessing_dir / "data"
        
        # 其他模块的数据路径  
        self.models_data_dir = self.models_dir / "data"
        self.results_dir = self.output_dir / "data"
        
        # 细分的原始数据收集路径
        self.historical_disasters_dir = self.collection_raw_dir / "historical_disasters" 
        self.predicted_disasters_dir = self.collection_raw_dir / "predicted_disasters"
        self.countries_dir = self.collection_raw_dir / "countries"
        self.disaster_types_dir = self.collection_raw_dir / "disaster_types"
        
        # 细分的缓存路径（按数据类别分类）
        self.countries_cache_dir = self.collection_cache_dir / "countries"
        self.disaster_types_cache_dir = self.collection_cache_dir / "disaster_types"
        self.historical_disasters_cache_dir = self.collection_cache_dir / "historical_disasters"
        self.predicted_disasters_cache_dir = self.collection_cache_dir / "predicted_disasters"


@dataclass
class CollectionConfig:
    """数据收集配置"""
    max_records_per_collection: int = 20000  # 减少到500条进行测试
    batch_size: int = 200  # 减少批次大小
    countries_of_interest: List[str] = field(default_factory=lambda: [
        'Turkey', 'Pakistan', 'Philippines', 'Indonesia', 'Bangladesh',
        'India', 'Nepal', 'Myanmar', 'Afghanistan', 'Syria'
    ])


    # 历史灾害收集配置
    historical_disasters: Dict[str, Any] = field(default_factory=lambda: {
        'years_back': 5,  # 修改为近5年
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
class PredictionConfig:
    """灾害预测配置"""
    
    # 可预测的灾害类型 (基于历史数据分析结果)
    predictable_disasters: Dict[int, str] = field(default_factory=lambda: {
        15: "Fire", 12: "Flood", 21: "Food Insecurity", 19: "Heat Wave", 
        24: "Landslide", 13: "Other", 27: "Pluvial/Flash Flood", 
        5: "Population Movement", 23: "Storm Surge", 54: "Transport Accident", 
        11: "Tsunami", 8: "Volcanic Eruption", 14: "Cold Wave", 
        6: "Complex Emergency", 4: "Cyclone", 20: "Drought", 
        2: "Earthquake", 7: "Civil Unrest", 62: "Insect Infestation", 
        68: "Transport Emergency", 66: "Biological Emergency"
    })
    
    # 国家ISO3代码映射
    country_iso3_mapping: Dict[str, str] = field(default_factory=lambda: {
        'Bangladesh': 'BGD', 'Philippines': 'PHL', 'Indonesia': 'IDN',
        'India': 'IND', 'Nepal': 'NPL', 'Pakistan': 'PAK',
        'Turkey': 'TUR', 'Afghanistan': 'AFG', 'Myanmar': 'MMR',
        'Syria': 'SYR', 'Japan': 'JPN', 'China': 'CHN',
        'United States': 'USA', 'Australia': 'AUS', 'Brazil': 'BRA',
        'Mexico': 'MEX', 'Iran': 'IRN', 'Thailand': 'THA'
    })
    
    # 预测模型参数
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        'default_time_horizon_months': 12,
        'confidence_thresholds': {
            'high': 50,     # 50+个历史事件
            'medium': 20,   # 20-49个历史事件  
            'low': 5,       # 5-19个历史事件
            'very_low': 0   # <5个历史事件
        },
        'enhancement_limits': {
            'min_multiplier': 0.1,  # 最小增强系数
            'max_multiplier': 3.0,  # 最大增强系数
            'api_weight': 0.3,      # API数据权重
            'historical_weight': 0.7 # 历史数据权重
        }
    })
    
    # 灾害类型关键词映射 (用于API数据匹配)
    disaster_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        'flood': ['flood', 'flooding', 'inundation'],
        'drought': ['drought', 'dry', 'arid'],
        'cyclone': ['cyclone', 'hurricane', 'typhoon', 'storm', 'tropical'],
        'earthquake': ['earthquake', 'seismic', 'tremor'],
        'fire': ['fire', 'wildfire', 'forest fire'],
        'heat wave': ['heat', 'hot', 'temperature', 'heatwave'],
        'cold wave': ['cold', 'freeze', 'winter', 'frost'],
        'landslide': ['landslide', 'slope', 'landslip'],
        'tsunami': ['tsunami', 'tidal'],
        'volcanic eruption': ['volcanic', 'volcano', 'eruption']
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
        self.prediction = PredictionConfig()
        self.logging = LoggingConfig()

        # 为main.py兼容性添加的属性
        self.DATA_DIR = self.paths.root_dir
        self.RAW_DATA_DIR = self.paths.raw_data_dir
        self.PROCESSED_DATA_DIR = self.paths.processed_data_dir
        self.MODELS_DIR = self.paths.models_data_dir
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
            # 原始数据目录
            self.paths.historical_disasters_dir,
            self.paths.predicted_disasters_dir,
            self.paths.countries_dir,
            self.paths.disaster_types_dir,
            # 分类缓存目录
            self.paths.countries_cache_dir,
            self.paths.disaster_types_cache_dir,
            self.paths.historical_disasters_cache_dir,
            self.paths.predicted_disasters_cache_dir,
            # 日志目录
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
            'appeals': f"{self.api.base_url_v2}/appeal/",
            'countries': f"{self.api.base_url_v2}/country/",
            'regions': f"{self.api.base_url_v2}/region/",

            # V1 API端点 (基础数据)
            'pdc': f"{self.api.base_url_v1}/pdc/",
            'publish_report': f"{self.api.base_url_v1}/publish-report/",
            'risk_score': f"{self.api.base_url_v1}/risk-score/",
            'adam_exposure': f"{self.api.base_url_v1}/adam-exposure/",
            'inform_score': f"{self.api.base_url_v1}/inform-score/",
            'gdacs': f"{self.api.base_url_v1}/gdacs/",
            
            # 灾害预测API端点 (来自预测服务器)
            'earthquake': f"{self.api.prediction_base_url}/earthquake/",
            'global_exposure_data': f"{self.api.prediction_base_url}/global-exposure-data/",
            'seasonal': f"{self.api.prediction_base_url}/seasonal/",
            'country_seasonal': f"{self.api.prediction_base_url}/country-seasonal/",
            'early_actions': f"{self.api.prediction_base_url}/early-actions/",
            'meteosiss': f"{self.api.prediction_base_url}/meteosiss/",
            'gvfs': f"{self.api.prediction_base_url}/gvfs/"
        }
    
    def get_prediction_endpoints(self) -> Dict[str, str]:
        """获取专门的灾害预测API端点"""
        return {
            'earthquake': f"{self.api.prediction_base_url}/earthquake/",
            'global_exposure': f"{self.api.prediction_base_url}/global-exposure-data/",
            'seasonal': f"{self.api.prediction_base_url}/seasonal/",
            'country_seasonal': f"{self.api.prediction_base_url}/country-seasonal/",
            'risk_score': f"{self.api.prediction_base_url}/risk-score/",
            'early_actions': f"{self.api.prediction_base_url}/early-actions/",
            'inform_score': f"{self.api.prediction_base_url}/inform-score/",
            'meteosiss': f"{self.api.prediction_base_url}/meteosiss/",
            'gvfs': f"{self.api.prediction_base_url}/gvfs/"
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


# 全局配置实例（延迟初始化）
config = None


def load_config(config_file: str = None) -> ProjectConfig:
    """加载配置"""
    global config
    if config_file:
        config = ProjectConfig(config_file)
    elif config is None:
        config = ProjectConfig()
    return config


def get_config(env: str = None) -> ProjectConfig:
    """获取全局配置"""
    global config
    if config is None:
        config = ProjectConfig()
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