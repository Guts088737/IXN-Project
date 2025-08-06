
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()


class Config:
    """基础配置类"""

    # ========== 项目基本信息 ==========
    PROJECT_NAME = "IFRC Medical Resource Allocation System Based on predicted Risk Events"
    VERSION = "1.0.0"
    AUTHOR = "Desheng He"

    # ========== 目录配置 ==========
    # 数据目录
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "01_raw"
    PROCESSED_DATA_DIR = DATA_DIR / "02_processed"
    MODELS_DIR = DATA_DIR / "03_models"
    RESULTS_DIR = DATA_DIR / "04_results"
    CACHE_DIR = DATA_DIR / "05_cache"

    # 源代码目录
    SRC_DIR = PROJECT_ROOT / "src"

    # 确保目录存在
    @classmethod
    def ensure_directories(cls):
        """确保所有必要的目录存在"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.CACHE_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

            # 创建 .gitkeep 文件保持空目录
            gitkeep_file = directory / ".gitkeep"
            if not gitkeep_file.exists():
                gitkeep_file.touch()

    # ========== IFRC API 配置 ==========
    IFRC_API_CONFIG = {
        "base_url": "https://goadmin.ifrc.org/api/v2/",
        "timeout": 30,
        "max_retries": 3,
        "retry_delay": 1,  # 秒
        "rate_limit": {
            "requests_per_minute": 60,
            "requests_per_hour": 1000
        }
    }

    # API端点配置
    API_ENDPOINTS = {
        "disaster_types": "disaster_type/",
        "events": "event/",
        "local_units": "local-units/",
        "countries": "country/",
        "appeals": "appeal/"
    }

    # 请求头配置
    API_HEADERS = {
        "User-Agent": f"{PROJECT_NAME}/{VERSION}",
        "Accept": "application/json",
        "Accept-Language": "en"
    }

    # ========== 数据收集配置 ==========
    DATA_COLLECTION_CONFIG = {
        "upcoming_events_days": 30,  # 获取未来30天的事件
        "historical_events_years": 5,  # 获取过去5年的历史数据
        "batch_size": 50,  # 批量获取数据的大小
        "update_interval": 3600,  # 数据更新间隔（秒）
    }

    # ========== 数据预处理配置 ==========
    PREPROCESSING_CONFIG = {
        "missing_value_strategy": "median",  # 缺失值处理策略
        "outlier_method": "iqr",  # 异常值检测方法
        "scaling_method": "standard",  # 数据标准化方法
        "encoding_method": "one_hot",  # 分类变量编码方法
        "feature_selection": True,  # 是否进行特征选择
        "train_test_split": 0.8,  # 训练测试集分割比例
        "random_state": 42  # 随机种子
    }

    # ========== 机器学习模型配置 ==========
    MODEL_CONFIG = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": -1
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        "neural_network": {
            "hidden_layer_sizes": (100, 50),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 1000,
            "random_state": 42
        }
    }

    # ========== 医疗资源类型配置 ==========
    RESOURCE_TYPES = [
        "doctors",
        "nurses",
        "specialists",
        "hospital_beds",
        "isolation_rooms",
        "ambulances",
        "medical_supplies",
        "medicine",
        "surgical_equipment"
    ]

    # ========== 灾害类型配置 ==========
    DISASTER_TYPES = {
        1: "Epidemic",
        2: "Earthquake",
        4: "Cyclone",
        5: "Population Movement",
        6: "Complex Emergency",
        7: "Civil Unrest",
        8: "Volcanic Eruption",
        11: "Tsunami",
        12: "Flood",
        13: "Other",
        14: "Cold Wave",
        15: "Fire",
        19: "Heat Wave",
        20: "Drought",
        21: "Food Insecurity",
        23: "Storm Surge",
        24: "Landslide",
        27: "Pluvial/Flash Flood"
    }

    # ========== 预测配置 ==========
    PREDICTION_CONFIG = {
        "confidence_threshold": 0.8,  # 置信度阈值
        "max_allocation_distance": 500,  # 最大调配距离（公里）
        "severity_weights": {  # 严重程度权重
            1: 1.0,
            2: 1.5,
            3: 2.0,
            4: 2.5
        },
        "resource_multipliers": {  # 资源需求倍数
            "doctors": 0.001,  # 每1000人需要1个医生
            "nurses": 0.003,  # 每1000人需要3个护士
            "hospital_beds": 0.005,  # 每1000人需要5个床位
            "ambulances": 0.0002  # 每1000人需要0.2个救护车
        }
    }

    # ========== 输出配置 ==========
    OUTPUT_CONFIG = {
        "export_formats": ["json", "csv", "excel"],
        "include_confidence": True,
        "include_metadata": True,
        "date_format": "%Y-%m-%d %H:%M:%S",
        "decimal_places": 2,
        "ifrc_integration": {
            "api_endpoint": "/api/medical-predictions",
            "update_frequency": "hourly",
            "data_retention_days": 30
        }
    }

    # ========== 缓存配置 ==========
    CACHE_CONFIG = {
        "api_cache_timeout": 1800,  # API缓存超时时间（秒）
        "model_cache_timeout": 3600,  # 模型缓存超时时间（秒）
        "max_cache_size": 100,  # 最大缓存大小（MB）
        "cache_cleanup_interval": 86400,  # 缓存清理间隔（秒）
        "enable_cache": True
    }

    # ========== 日志配置 ==========
    LOG_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "file_rotation": "daily",
        "max_log_files": 30,
        "max_file_size": "10MB"
    }

    # ========== 验证配置 ==========
    VALIDATION_CONFIG = {
        "cross_validation_folds": 5,
        "validation_metrics": ["mae", "mse", "r2", "mape"],
        "model_selection_metric": "mae",
        "hyperparameter_tuning": True,
        "tuning_iterations": 100
    }


class DevelopmentConfig(Config):
    """开发环境配置"""

    DEBUG = True

    # 开发环境使用更详细的日志
    LOG_CONFIG = {
        **Config.LOG_CONFIG,
        "level": "DEBUG"
    }

    # 开发环境使用更小的数据集进行快速测试
    MODEL_CONFIG = {
        **Config.MODEL_CONFIG,
        "random_forest": {
            **Config.MODEL_CONFIG["random_forest"],
            "n_estimators": 50  # 减少估计器数量以加快训练
        }
    }

    # 开发环境缓存时间更短
    CACHE_CONFIG = {
        **Config.CACHE_CONFIG,
        "api_cache_timeout": 300,  # 5分钟
        "model_cache_timeout": 600  # 10分钟
    }


class ProductionConfig(Config):
    """生产环境配置"""

    DEBUG = False

    # 生产环境使用更少的日志
    LOG_CONFIG = {
        **Config.LOG_CONFIG,
        "level": "WARNING"
    }

    # 生产环境使用更强的模型
    MODEL_CONFIG = {
        **Config.MODEL_CONFIG,
        "random_forest": {
            **Config.MODEL_CONFIG["random_forest"],
            "n_estimators": 200  # 增加估计器数量以提高性能
        }
    }

    # 生产环境更长的缓存时间
    CACHE_CONFIG = {
        **Config.CACHE_CONFIG,
        "api_cache_timeout": 3600,  # 1小时
        "model_cache_timeout": 7200  # 2小时
    }


class TestingConfig(Config):
    """测试环境配置"""

    DEBUG = True
    TESTING = True

    # 测试环境使用临时目录
    DATA_DIR = PROJECT_ROOT / "test_data"
    RAW_DATA_DIR = DATA_DIR / "01_raw"
    PROCESSED_DATA_DIR = DATA_DIR / "02_processed"
    MODELS_DIR = DATA_DIR / "03_models"
    RESULTS_DIR = DATA_DIR / "04_results"
    CACHE_DIR = DATA_DIR / "05_cache"

    # 测试环境使用最小配置
    MODEL_CONFIG = {
        "random_forest": {
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42
        }
    }

    # 测试环境关闭缓存
    CACHE_CONFIG = {
        **Config.CACHE_CONFIG,
        "enable_cache": False
    }


# ========== 配置选择 ==========
config_mapping = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name=None):
    """获取配置对象"""
    if config_name is None:
        config_name = os.getenv('IFRC_ENV', 'default')

    config_class = config_mapping.get(config_name, config_mapping['default'])
    config_instance = config_class()

    # 确保目录存在
    config_instance.ensure_directories()

    return config_instance


# 当前配置实例
current_config = get_config()


# ========== 配置验证函数 ==========
def validate_config(config):
    """验证配置的有效性"""
    errors = []

    # 检查必要的目录
    required_dirs = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
        config.RESULTS_DIR,
        config.CACHE_DIR
    ]

    for directory in required_dirs:
        if not directory.exists():
            errors.append(f"Directory does not exist: {directory}")

    # 检查API配置
    if not config.IFRC_API_CONFIG.get("base_url"):
        errors.append("IFRC API base URL is not configured")

    # 检查模型配置
    if not config.RESOURCE_TYPES:
        errors.append("No resource types configured")

    return errors


if __name__ == "__main__":
    # 测试配置
    config = get_config()
    print(f"Project: {config.PROJECT_NAME}")
    print(f"Version: {config.VERSION}")
    print(f"Data Directory: {config.DATA_DIR}")
    print(f"API Base URL: {config.IFRC_API_CONFIG['base_url']}")

    # 验证配置
    errors = validate_config(config)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")