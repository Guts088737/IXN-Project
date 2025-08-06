import sys
import os
import argparse
from pathlib import Path


src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))


from config import get_config, validate_config


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
        from src.data_collection.disaster_collector import DisasterCollector
        from src.data_collection.resource_collector import ResourceCollector

        print("🔄 Starting data collection...")


        disaster_collector = DisasterCollector()
        print("  📊 Collecting disaster events...")
        disaster_collector.collect_upcoming_events()
        disaster_collector.collect_historical_events()


        resource_collector = ResourceCollector()
        print("  🏥 Collecting medical resources...")
        resource_collector.collect_local_units()

        print("✅ Data collection completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Data collection failed: {str(e)}")
        return False


def run_preprocessing():

    try:
        from src.preprocessing.data_cleaner import DataCleaner
        from src.preprocessing.feature_engineer import FeatureEngineer

        print("🔄 Starting data preprocessing...")

        # 数据清洗
        cleaner = DataCleaner()
        print("  🧹 Cleaning data...")
        cleaner.clean_disaster_data()
        cleaner.clean_resource_data()

        # 特征工程
        engineer = FeatureEngineer()
        print("  ⚙️ Engineering features...")
        engineer.create_features()

        print("✅ Data preprocessing completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Data preprocessing failed: {str(e)}")
        return False


def run_model_training():

    try:
        from src.training.trainer import Trainer

        print("🔄 Starting model training...")

        trainer = Trainer()
        print("  🤖 Training machine learning models...")


        config = get_config()
        for resource_type in config.RESOURCE_TYPES:
            print(f"    📈 Training {resource_type} prediction model...")
            trainer.train_resource_model(resource_type)

        print("✅ Model training completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        return False


def run_prediction():

    try:
        from src.models.resource_predictor import ResourcePredictor
        from src.output.prediction_service import PredictionService

        print("🔄 Starting prediction...")


        predictor = ResourcePredictor()
        prediction_service = PredictionService()

        print("  🔮 Generating predictions...")
        predictions = prediction_service.generate_all_predictions()

        print("  📤 Exporting results...")
        prediction_service.export_predictions(predictions)

        print("✅ Prediction completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return False


def run_full_pipeline():

    print("🚀 Starting full pipeline...")

    steps = [
        ("Data Collection", run_data_collection),
        ("Data Preprocessing", run_preprocessing),
        ("Model Training", run_model_training),
        ("Prediction", run_prediction)
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
    print("🎉 Full pipeline completed successfully!")
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