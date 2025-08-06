
import sys
import os
import argparse
from pathlib import Path


from src.config import get_config, validate_config


def setup_logging(config):

    import logging
    from logging.handlers import RotatingFileHandler


    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)


    log_format = logging.Formatter(
        config.LOG_CONFIG["format"],
        datefmt=config.LOG_CONFIG["date_format"]
    )


    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.LOG_CONFIG["level"]))


    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)


    file_handler = RotatingFileHandler(
        log_dir / "ifrc_medical_allocation.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=config.LOG_CONFIG["max_log_files"]
    )
    file_handler.setLevel(getattr(logging, config.LOG_CONFIG["level"]))
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


def print_banner():

    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        IFRC Medical Resource Allocation System              ║
    ║                                                              ║
    ║        Author: Your Name                                     ║
    ║        Version: 1.0.0                                        ║
    ║        Description: AI-powered medical resource allocation   ║
    ║                     for disaster response                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():

    required_packages = [
        'requests',
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False

    return True


def run_data_collection():

    try:
        # 导入医疗设施和服务收集器
        from src.data_collection.medical_facilities_collector import MedicalFacilitiesCollector
        from src.data_collection.medical_services_collector import MedicalServicesCollector

        print("🔄 Starting medical data collection...")

        config = get_config()
        
        # 收集医疗设施数据
        facilities_collector = MedicalFacilitiesCollector(config)
        print("  🏥 Collecting medical facilities...")
        facilities_result = facilities_collector.run_collection()
        
        if facilities_result.get('collection_status') == 'failed':
            print(f"❌ Medical facilities collection failed: {facilities_result.get('error', 'Unknown error')}")
            return False
        
        print(f"  ✅ Medical facilities: {facilities_result.get('medical_facilities_count', 0)} facilities collected")
        
        # 收集医疗服务数据
        services_collector = MedicalServicesCollector(config)
        print("  🩺 Collecting medical services...")
        services_result = services_collector.run_collection()
        
        if services_result.get('collection_status') == 'failed':
            print(f"⚠️ Medical services collection had issues: {services_result.get('error', 'Unknown error')}")
            # 服务收集失败不影响整体流程，因为它依赖于设施数据
        else:
            print(f"  ✅ Medical services: {services_result.get('services_summary', {}).get('services_extracted', 0)} services processed")

        print("✅ Medical data collection completed successfully!")
        return True

    except ImportError as e:
        print(f"❌ Data collection failed - Import error: {str(e)}")
        print("Please ensure all required modules are properly implemented.")
        return False
    except Exception as e:
        print(f"❌ Data collection failed: {str(e)}")
        return False


def run_preprocessing():
    """运行医疗数据预处理流程"""
    try:
        from src.preprocessing.medical_facilities_cleaner import MedicalFacilitiesCleaner
        from src.preprocessing.medical_services_cleaner import MedicalServicesCleaner

        print("🔄 Starting medical data preprocessing...")

        config = get_config()
        
        # 清洗医疗设施数据
        facilities_cleaner = MedicalFacilitiesCleaner(config)
        print("  🏥 Cleaning medical facilities data...")
        facilities_result = facilities_cleaner.clean_data()
        
        if facilities_result.get('cleaning_summary', {}).get('error'):
            print(f"  ⚠️ Facilities cleaning issues: {facilities_result['cleaning_summary']['error']}")
        else:
            facilities_count = facilities_result.get('summary', {}).get('total_facilities', 0)
            print(f"  ✅ Medical facilities cleaned: {facilities_count} facilities")
        
        # 清洗医疗服务数据
        services_cleaner = MedicalServicesCleaner(config)
        print("  🩺 Cleaning medical services data...")
        services_result = services_cleaner.clean_data()
        
        if services_result.get('cleaning_summary', {}).get('error'):
            print(f"  ⚠️ Services cleaning issues: {services_result['cleaning_summary']['error']}")
        else:
            services_count = services_result.get('summary', {}).get('total_services', 0)
            print(f"  ✅ Medical services cleaned: {services_count} service records")
        
        # 数据质量报告摘要
        print("  📊 Data quality summary:")
        
        # 设施质量评估
        if facilities_result.get('quality_report'):
            facilities_quality = facilities_result['quality_report'].get('quality_assessment', {})
            avg_quality = facilities_quality.get('average_quality_score', 0)
            print(f"    📈 Facilities avg quality: {avg_quality:.2f}")
        
        # 服务质量评估
        if services_result.get('quality_report'):
            services_quality = services_result['quality_report'].get('quality_assessment', {})
            avg_quality = services_quality.get('average_quality_score', 0)
            print(f"    📈 Services avg quality: {avg_quality:.2f}")
        
        # IFRC合规性摘要
        if services_result.get('compliance_analysis'):
            compliance = services_result['compliance_analysis']
            compliance_rate = compliance.get('compliance_rate', 0)
            print(f"    📋 IFRC compliance rate: {compliance_rate:.2%}")

        print("✅ Medical data preprocessing completed successfully!")
        print("  📁 Check data/02_processed/ for cleaned data files")
        return True

    except ImportError as e:
        print(f"❌ Data preprocessing failed - Import error: {str(e)}")
        print("Please ensure data cleaner modules are properly implemented.")
        return False
    except Exception as e:
        print(f"❌ Data preprocessing failed: {str(e)}")
        return False


def run_model_training():
    """运行模型训练（暂未实现）"""
    print("🔄 Model training...")
    print("  ⚠️ Model training module is not yet implemented")
    print("  📋 Focus: Currently prioritizing medical data collection and cleaning")
    print("✅ Model training step skipped for now")
    return True


def run_prediction():
    """运行预测（暂未实现）"""
    print("🔄 Prediction...")
    print("  ⚠️ Prediction module is not yet implemented")
    print("  📋 Focus: Currently prioritizing medical data collection and cleaning")
    print("✅ Prediction step skipped for now")
    return True


def run_full_pipeline():
    """运行完整的医疗数据处理流程"""
    print("🚀 Starting IFRC Medical Resource Data Pipeline...")
    print("📋 Focus: Medical facilities and services data collection & cleaning")

    steps = [
        ("Medical Data Collection", run_data_collection),
        ("Medical Data Preprocessing", run_preprocessing),
        ("Model Training (Placeholder)", run_model_training),
        ("Prediction (Placeholder)", run_prediction)
    ]

    for step_name, step_function in steps:
        print(f"\n{'=' * 60}")
        print(f"Step: {step_name}")
        print('=' * 60)

        success = step_function()
        if not success:
            print(f"❌ Pipeline failed at step: {step_name}")
            return False

    print("\n" + "=" * 60)
    print("🎉 Medical data pipeline completed successfully!")
    print("📊 Check data/02_processed/ for cleaned medical data")
    print("📈 Check data/04_results/ for data quality reports")
    print("=" * 60)
    return True


def show_status():

    config = get_config()

    print("\n📊 System Status:")
    print("-" * 50)


    data_dirs = [
        ("Raw Data", config.RAW_DATA_DIR),
        ("Processed Data", config.PROCESSED_DATA_DIR),
        ("Models", config.MODELS_DIR),
        ("Results", config.RESULTS_DIR),
        ("Cache", config.CACHE_DIR)
    ]

    for name, path in data_dirs:
        if path.exists():
            file_count = len(list(path.glob("*")))
            print(f"  {name}: ✅ ({file_count} files)")
        else:
            print(f"  {name}: ❌ (not found)")


    errors = validate_config(config)
    if errors:
        print(f"\n⚠️  Configuration Issues:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"\n✅ Configuration: OK")

    print("-" * 50)


def main():

    parser = argparse.ArgumentParser(
        description="IFRC Medical Resource Allocation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --run-all             
  python main.py --collect-data         
  python main.py --train-models         
  python main.py --predict             
  python main.py --status               
        """
    )

    # 添加命令行参数
    parser.add_argument('--run-all', action='store_true',
                        help='Run the complete pipeline')
    parser.add_argument('--collect-data', action='store_true',
                        help='Run data collection only')
    parser.add_argument('--preprocess', action='store_true',
                        help='Run data preprocessing only')
    parser.add_argument('--train-models', action='store_true',
                        help='Run model training only')
    parser.add_argument('--predict', action='store_true',
                        help='Run prediction only')
    parser.add_argument('--status', action='store_true',
                        help='Show system status')
    parser.add_argument('--config', default='development',
                        choices=['development', 'production', 'testing'],
                        help='Configuration environment')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()


    os.environ['IFRC_ENV'] = args.config


    config = get_config(args.config)


    if args.verbose:
        config.LOG_CONFIG["level"] = "DEBUG"

    logger = setup_logging(config)


    print_banner()


    if not check_dependencies():
        sys.exit(1)


    errors = validate_config(config)
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print(f"🔧 Environment: {args.config}")
    print(f"📁 Data Directory: {config.DATA_DIR}")
    print(f"🌐 API Base URL: {config.IFRC_API_CONFIG['base_url']}")


    try:
        if args.status:
            show_status()
        elif args.run_all:
            success = run_full_pipeline()
            sys.exit(0 if success else 1)
        elif args.collect_data:
            success = run_data_collection()
            sys.exit(0 if success else 1)
        elif args.preprocess:
            success = run_preprocessing()
            sys.exit(0 if success else 1)
        elif args.train_models:
            success = run_model_training()
            sys.exit(0 if success else 1)
        elif args.predict:
            success = run_prediction()
            sys.exit(0 if success else 1)
        else:

            parser.print_help()
            show_status()

    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()