"""
运行多样化灾害预测系统
数据源: src/preprocessing/data/historical_disasters/processed_historical_disasters.json
输出: src/output/data/
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 简化导入，直接使用相对路径
try:
    from src.preprocessing.feature_reversal_engineer import FeatureReversalEngineer
    from src.models.spatial_analysis import DisasterSpatialAnalyzer  
    from src.models.mixed_expert_predictor import MixedExpertPredictor
    from src.models.random_forest_expert_ensemble import RandomForestExpertEnsemble
    from src.preprocessing.intelligent_district_selector import IntelligentDistrictSelector
    from src.validation.prediction_validator import PredictionValidator
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"模块导入失败: {e}")
    MODULES_AVAILABLE = False


def load_reference_data():
    """加载所有参考数据"""
    reference_data = {}
    
    # 1. 加载国家数据
    countries_path = project_root / 'src' / 'data_collection' / 'data' / 'cache' / 'countries' / 'all_countries.json'
    if countries_path.exists():
        with open(countries_path, 'r', encoding='utf-8') as f:
            countries_json = json.load(f)
        
        countries_map = {}
        for country in countries_json.get('data', []):
            country_id = country.get('id')
            countries_map[country_id] = {
                'name': country.get('name', ''),
                'iso': country.get('iso', ''),
                'region': country.get('region', 1)
            }
        reference_data['countries'] = countries_map
        print(f" 加载{len(countries_map)}个国家信息")
    
    # 2. 加载灾害类型数据
    disaster_types_path = project_root / 'src' / 'data_collection' / 'data' / 'cache' / 'disaster_types' / 'basic_disaster_types.json'
    if disaster_types_path.exists():
        with open(disaster_types_path, 'r', encoding='utf-8') as f:
            disaster_types_json = json.load(f)
        
        disaster_types_map = {}
        for dt in disaster_types_json.get('data', {}).get('results', []):
            disaster_types_map[dt.get('id')] = dt.get('name', '')
        reference_data['disaster_types'] = disaster_types_map
        print(f" 加载{len(disaster_types_map)}种灾害类型")
    
    # 3. 加载regions数据
    regions_path = project_root / 'src' / 'data_collection' / 'data' / 'cache' / 'regions' / 'regions_list.json'
    if regions_path.exists():
        with open(regions_path, 'r', encoding='utf-8') as f:
            regions_json = json.load(f)
        
        regions_map = {}
        for region in regions_json.get('data', {}).get('results', []):
            region_id = region.get('id')
            regions_map[region_id] = {
                'name': region.get('region_name', ''),
                'label': region.get('label', '')
            }
        reference_data['regions'] = regions_map
        print(f" 加载{len(regions_map)}个region信息")
    
    # 4. 加载district数据
    districts_path = project_root / 'src' / 'data_collection' / 'data' / 'cache' / 'districts' / 'districts_data.json'
    if districts_path.exists():
        try:
            with open(districts_path, 'r', encoding='utf-8') as f:
                districts_json = json.load(f)
            
            districts_map = {}
            districts_by_country = {}
            
            if isinstance(districts_json, list):
                for page_data in districts_json:
                    if not page_data or 'data' not in page_data:
                        continue
                    
                    page_data_section = page_data.get('data')
                    if not page_data_section or 'results' not in page_data_section:
                        continue
                    
                    for district in page_data_section.get('results', []):
                        if not district or not isinstance(district, dict):
                            continue
                        
                        district_id = district.get('id')
                        if district_id is None:
                            continue
                        
                        country_name = district.get('country_name', '')
                        district_info = {
                            'name': district.get('name', ''),
                            'code': district.get('code', ''),
                            'country_name': country_name,
                            'country_iso': district.get('country_iso', ''),
                        }
                        
                        # 优先使用centroid坐标，其次使用bbox中心点
                        centroid = district.get('centroid')
                        if centroid and isinstance(centroid, dict) and 'coordinates' in centroid:
                            coords = centroid['coordinates']
                            if isinstance(coords, list) and len(coords) >= 2:
                                district_info['latitude'] = float(coords[1])
                                district_info['longitude'] = float(coords[0])
                            else:
                                district_info['latitude'] = 0.0
                                district_info['longitude'] = 0.0
                        elif district.get('bbox') and isinstance(district['bbox'], dict) and district['bbox'].get('coordinates'):
                            bbox_coords = district['bbox']['coordinates']
                            if isinstance(bbox_coords, list) and len(bbox_coords) > 0 and isinstance(bbox_coords[0], list):
                                coords = bbox_coords[0]
                                if coords:
                                    lats = [coord[1] for coord in coords if isinstance(coord, list) and len(coord) >= 2]
                                    lngs = [coord[0] for coord in coords if isinstance(coord, list) and len(coord) >= 2]
                                    if lats and lngs:
                                        district_info['latitude'] = sum(lats) / len(lats)
                                        district_info['longitude'] = sum(lngs) / len(lngs)
                                    else:
                                        district_info['latitude'] = 0.0
                                        district_info['longitude'] = 0.0
                                else:
                                    district_info['latitude'] = 0.0
                                    district_info['longitude'] = 0.0
                            else:
                                district_info['latitude'] = 0.0
                                district_info['longitude'] = 0.0
                        else:
                            district_info['latitude'] = 0.0
                            district_info['longitude'] = 0.0
                        
                        districts_map[district_id] = district_info
                        
                        # 按国家分组districts
                        if country_name not in districts_by_country:
                            districts_by_country[country_name] = []
                        districts_by_country[country_name].append(district_id)
            
            reference_data['districts'] = districts_map
            reference_data['districts_by_country'] = districts_by_country
            print(f" 加载{len(districts_map)}个district信息")
        except Exception as e:
            print(f"districts数据加载失败: {e}")
            reference_data['districts'] = {}
    
    return reference_data


def load_historical_data():
    """从正确路径加载历史数据"""
    data_path = project_root / 'src' / 'preprocessing' / 'data' / 'historical_disasters' / 'processed_historical_disasters.json'
    
    print(f"数据源: {data_path}")
    
    if not data_path.exists():
        print(f" 历史数据文件不存在")
        return None
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        
        # 转换为DataFrame
        if 'disasters' in data_json:
            events = data_json['disasters']
        elif 'events' in data_json:
            events = data_json['events']
        elif isinstance(data_json, list):
            events = data_json
        else:
            events = [data_json]
        
        df = pd.DataFrame(events)
        
        # 数据字段映射和清理
        if len(df) > 0:
            # 确保必要字段存在
            required_fields = ['disaster_type_id', 'country_id', 'year', 'month', 'people_affected']
            missing_fields = [field for field in required_fields if field not in df.columns]
            
            if missing_fields:
                print(f" 缺少必要字段: {missing_fields}")
                return None
            
            # 提取地理信息
            if 'location_details' in df.columns:
                df['region_id'] = df['location_details'].apply(
                    lambda x: x.get('region_id') if isinstance(x, dict) else 1
                )
            elif 'region_id' not in df.columns:
                df['region_id'] = 1
            
            # 从countries数据获取坐标（数据驱动方式）
            # 暂时设置为0，后续通过district数据获取具体坐标
            df['latitude'] = 0.0
            df['longitude'] = 0.0
            
            # 添加date字段（从start_date转换）
            if 'start_date' in df.columns:
                df['date'] = pd.to_datetime(df['start_date'], errors='coerce')
            else:
                # 如果没有start_date，从year和month构造
                df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
            
            # 清理数据
            df = df.dropna(subset=['disaster_type_id', 'country_id'])
            df['disaster_type_id'] = df['disaster_type_id'].astype(int)
            df['country_id'] = df['country_id'].astype(int)
        
        print(f" 加载{len(df)}条历史记录")
        print(f"   灾害类型: {len(df['disaster_type_id'].unique()) if 'disaster_type_id' in df.columns else 0}种")
        print(f"   国家数量: {len(df['country_id'].unique()) if 'country_id' in df.columns else 0}个")
        
        return df
        
    except Exception as e:
        print(f" 加载失败: {e}")
        return None


def create_global_test_requests(historical_data, countries_map):
    """基于历史数据创建全球测试请求"""
    test_requests = []
    
    # 从历史数据选择不同国家和地区，增加数据完整性验证
    if 'country_id' in historical_data.columns:
        countries = historical_data['country_id'].dropna().unique()
        
        # 数据完整性验证：过滤无效country_id
        valid_countries = []
        for country_id in countries:
            if country_id in countries_map:  # 确保country_id在参考数据中存在
                valid_countries.append(country_id)
        
        if len(valid_countries) == 0:
            print("警告：所有country_id都无效，使用原始数据")
            valid_countries = countries
        
        selected_countries = np.random.choice(valid_countries, size=min(20, len(valid_countries)), replace=False)
        
        for country_id in selected_countries:
            country_data = historical_data[historical_data['country_id'] == country_id]
            
            avg_lat = country_data['latitude'].mean() if 'latitude' in country_data.columns else 0
            avg_lng = country_data['longitude'].mean() if 'longitude' in country_data.columns else 0
            region_id = country_data['region_id'].mode().iloc[0] if 'region_id' in country_data.columns and not country_data['region_id'].empty else 1
            
            # 测试不同月份
            for month in [6, 9, 12]:
                test_requests.append({
                    'country_id': int(country_id),
                    'region_id': int(region_id) if not pd.isna(region_id) else 1,
                    'month': month,
                    'latitude': float(avg_lat) if not pd.isna(avg_lat) else 0,
                    'longitude': float(avg_lng) if not pd.isna(avg_lng) else 0
                })
    
    return test_requests


def run_prediction_system():
    """运行预测系统"""
    if not MODULES_AVAILABLE:
        print(" 必要模块未导入，无法运行")
        return None
        
    print("启动混合专家+随机森林预测系统...")
    
    # 1. 加载数据
    historical_data = load_historical_data()
    if historical_data is None:
        return None
    
    # 2. 直接使用组件而不是pipeline
    print("\n初始化预测组件...")
    
    try:
        feature_engineer = FeatureReversalEngineer()
        spatial_analyzer = DisasterSpatialAnalyzer()
        expert_predictor = MixedExpertPredictor()
        rf_ensemble = RandomForestExpertEnsemble()
        district_selector = IntelligentDistrictSelector()
        prediction_validator = PredictionValidator()
        
        print(" 组件初始化完成")
        
        # 3. 特征工程
        print("\n执行特征工程...")
        feature_results = feature_engineer.generate_data_driven_expert_features(historical_data)
        expert_specializations = feature_results['expert_specializations']
        country_risk_factors = feature_results['country_risk_factors']
        
        # 4. 空间分析
        print("执行空间分析...")
        try:
            spatial_features = spatial_analyzer.create_spatial_risk_features(historical_data)
            print("空间分析完成")
        except ZeroDivisionError as e:
            print(f"空间分析出现除零错误: {e}")
            import traceback
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"空间分析失败: {e}")
            return None
        
        # 5. 训练随机森林
        print("训练随机森林模型...")
        try:
            rf_results = rf_ensemble.train_random_forest_models(historical_data, country_risk_factors, spatial_features)
            print("随机森林训练成功")
        except ZeroDivisionError as e:
            print(f"随机森林训练出现除零错误: {e}")
            return None
        except Exception as e:
            print(f"随机森林训练失败: {e}")
            return None
        
        # 6. 加载参考数据用于验证组件
        print("加载参考数据...")
        reference_data = load_reference_data()
        countries_map = reference_data.get('countries', {})
        disaster_types_map = reference_data.get('disaster_types', {})
        regions_map = reference_data.get('regions', {})
        districts_map = reference_data.get('districts', {})
        
        # 7. 训练专家模型
        print("训练专家模型...")
        expert_predictor.set_historical_data_reference(historical_data)  # 设置历史数据引用
        expert_predictor.train_expert_models(historical_data, expert_specializations, country_risk_factors)
        
        # 8. 初始化验证组件
        print("初始化验证组件...")
        district_selector.build_district_disaster_profiles(historical_data, districts_map, countries_map)
        prediction_validator.initialize_validation_rules(historical_data, districts_map, countries_map)
        
        print(" 所有模型和验证组件初始化完成")
        
    except Exception as e:
        print(f" 组件初始化或训练失败: {e}")
        return None
    
    # 9. 执行预测（接下来一个月）
    print(f"\n执行接下来一个月的灾害预测...")
    test_requests = create_global_test_requests(historical_data, countries_map)
    
    predictions = []
    disaster_types = set()
    regions = set()
    
    # 去重机制 - 避免重复样本
    seen_combinations = set()
    
    # 获取下个月的月份
    next_month = (datetime.now().month % 12) + 1
    
    for i, request in enumerate(test_requests):
        try:
            # 设置预测时间为下个月
            request['month'] = next_month
            
            # 使用专家模型预测
            expert_pred = expert_predictor.predict_disasters(request, spatial_features, districts_map)
            
            # 使用随机森林集成
            ensemble_pred = rf_ensemble.predict_with_ensemble(
                request, expert_pred, country_risk_factors, spatial_features
            )
            
            # 提取主要威胁并增加多样性
            disaster_probs = ensemble_pred.get('disaster_probabilities', {})
            if disaster_probs:
                # 添加调试信息
                if i < 3:
                    print(f"   调试: 灾害概率分布 {dict(list(disaster_probs.items())[:5])}")
                
                # 使用多样性采样而不是总是选择最高概率
                disaster_types_list = list(disaster_probs.keys())
                probabilities_list = list(disaster_probs.values())
                
                # NaN检查和处理
                probabilities_list = [p if not np.isnan(p) else 0.001 for p in probabilities_list]
                
                # 归一化概率用于采样
                total_prob = sum(probabilities_list)
                if total_prob > 0 and not np.isnan(total_prob):
                    normalized_probs = [p / total_prob for p in probabilities_list]
                    # 检查归一化后的概率
                    if any(np.isnan(p) for p in normalized_probs):
                        # 如果出现NaN，使用等概率分布
                        normalized_probs = [1.0 / len(disaster_types_list)] * len(disaster_types_list)
                    
                    # 使用概率采样，增加多样性
                    selected_idx = np.random.choice(len(disaster_types_list), p=normalized_probs)
                    disaster_type = disaster_types_list[selected_idx]
                    probability = probabilities_list[selected_idx]
                else:
                    # 如果所有概率都是0或NaN，随机选择一种灾害类型
                    disaster_type = np.random.choice(disaster_types_list)
                    probability = 0.001  # 设置最小概率
                
                # 数据完整性验证：跳过无效的country_id
                if request['country_id'] not in countries_map:
                    continue  # 跳过无效的country_id
                
                # 使用智能district选择器选择最优位置
                country_name = countries_map.get(request['country_id'], {}).get('name', '')
                districts_by_country = reference_data.get('districts_by_country', {})
                
                selected_district = None
                selected_district_id = None
                district_coords = {'latitude': request.get('latitude', 0), 'longitude': request.get('longitude', 0)}
                selection_info = {}
                
                # 使用智能选择器选择最优district
                optimal_district_id, selection_score, selection_info = district_selector.select_optimal_district_for_disaster(
                    request['country_id'], disaster_type, districts_by_country, districts_map, countries_map, next_month
                )
                
                if optimal_district_id and optimal_district_id in districts_map:
                    selected_district_id = optimal_district_id
                    selected_district = districts_map[selected_district_id]
                    
                    # 使用智能选择的district的真实坐标
                    district_lat = selected_district.get('latitude', 0)
                    district_lng = selected_district.get('longitude', 0)
                    
                    if district_lat != 0 or district_lng != 0:
                        district_coords = {
                            'latitude': float(district_lat),
                            'longitude': float(district_lng)
                        }
                elif country_name in districts_by_country:
                    # 智能选择失败时的后备方案
                    country_district_ids = districts_by_country[country_name]
                    if country_district_ids:
                        selected_district_id = np.random.choice(country_district_ids)
                        selected_district = districts_map.get(selected_district_id, {})
                        
                        district_lat = selected_district.get('latitude', 0)
                        district_lng = selected_district.get('longitude', 0)
                        
                        if district_lat != 0 or district_lng != 0:
                            district_coords = {
                                'latitude': float(district_lat),
                                'longitude': float(district_lng)
                            }
                        selection_info = {'method': 'random_fallback'}
                
                # 去重检查 - 避免同一district重复预测
                if selected_district_id in seen_combinations:
                    continue  # 跳过已使用的district
                seen_combinations.add(selected_district_id)
                
                disaster_types.add(disaster_type)
                regions.add(request['region_id'])
                
                # 生成事件发生时间（下个月内的随机日期）
                import random
                next_year = datetime.now().year if next_month > datetime.now().month else datetime.now().year + 1
                max_day = 28 if next_month == 2 else (30 if next_month in [4,6,9,11] else 31)
                predicted_day = random.randint(1, max_day)
                predicted_hour = random.randint(0, 23)
                predicted_minute = random.randint(0, 59)
                
                predicted_datetime = datetime(next_year, next_month, predicted_day, predicted_hour, predicted_minute)
                
                # 格式化预测结果
                prediction_entry = {
                    'prediction_id': f"NEXT_MONTH_{i+1}_{datetime.now().strftime('%H%M%S')}",
                    'country_id': int(request['country_id']),
                    'country_name': countries_map.get(request['country_id'], {}).get('name', f"Country_{request['country_id']}"),
                    'country_iso': countries_map.get(request['country_id'], {}).get('iso', ''),
                    'region_id': int(request['region_id']),
                    'region_name': regions_map.get(request['region_id'], {}).get('name', f"Region_{request['region_id']}"),
                    'district_id': int(selected_district_id) if selected_district_id else None,
                    'district_name': selected_district.get('name') if selected_district else None,
                    'district_code': selected_district.get('code') if selected_district else None,
                    'predicted_month': next_month,
                    'predicted_date': predicted_datetime.isoformat(),
                    'predicted_year': next_year,
                    'disaster_type_id': int(disaster_type),
                    'disaster_type_name': disaster_types_map.get(int(disaster_type), f"Disaster_Type_{disaster_type}"),
                    'probability': float(probability),
                    'estimated_impact': int(ensemble_pred.get('impact_estimates', {}).get(disaster_type, 1000)),
                    'latitude': float(district_coords['latitude']),
                    'longitude': float(district_coords['longitude']),
                    'coordinate_source': 'district_centroid' if selected_district and district_coords['latitude'] != 0 else 'country_approximation',
                    'prediction_method': 'expert_rf_ensemble',
                    'prediction_timestamp': datetime.now().isoformat(),
                    'district_selection_info': selection_info
                }
                
                predictions.append(prediction_entry)
                
                if i < 10:
                    country_name = countries_map.get(request['country_id'], {}).get('name', f"国家{request['country_id']}")
                    disaster_name = disaster_types_map.get(disaster_type, f"灾害{disaster_type}")
                    print(f"   {country_name}: {disaster_name}, 概率{probability:.1%}")
        
        except Exception as e:
            if i < 3:
                print(f"   预测失败: {e}")
            continue
    
    # 10. 验证预测合理性并过滤不合理预测
    print(f"\n验证预测地理合理性...")
    reasonable_predictions, validation_report = prediction_validator.filter_and_redistribute_predictions(
        predictions, districts_map, countries_map
    )
    
    print(f"验证结果: {len(reasonable_predictions)}/{len(predictions)}通过验证")
    if validation_report['filtered_predictions'] > 0:
        print(f"主要过滤原因: {list(validation_report['common_rejection_reasons'].keys())[:3]}")
    
    # 11. 保存结果到 src/output/data
    output_dir = project_root / 'src' / 'output' / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'next_month_predictions_{timestamp}.json'
    
    # 转换所有numpy类型为Python原生类型
    clean_predictions = []
    for pred in reasonable_predictions:
        clean_pred = {}
        for key, value in pred.items():
            if isinstance(value, (np.integer, np.int32, np.int64)):
                clean_pred[key] = int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                clean_pred[key] = float(value)
            else:
                clean_pred[key] = value
        clean_predictions.append(clean_pred)
    
    # 重新计算多样性指标（基于验证后的数据）
    validated_disaster_types = set(pred['disaster_type_id'] for pred in clean_predictions)
    validated_regions = set(pred['region_id'] for pred in clean_predictions)
    
    # 简化数据结构 - 仅保留指定字段
    simplified_predictions = []
    for pred in clean_predictions:
        simplified_pred = {
            'country_name': pred['country_name'],
            'region_name': pred['region_name'], 
            'district_name': pred['district_name'],
            'predicted_date': f"{pred['predicted_year']}-{pred['predicted_month']:02d}",
            'disaster_type_name': pred['disaster_type_name'],
            'probability': pred['probability'],
            'estimated_impact': pred['estimated_impact'],
            'latitude': pred['latitude'],
            'longitude': pred['longitude'],
            'validation_confidence': pred.get('validation_confidence', 1.0)
        }
        simplified_predictions.append(simplified_pred)
    
    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'prediction_period': 'Next month',
            'total_predictions': len(clean_predictions),
            'validation_enabled': True
        },
        'diversity_metrics': {
            'disaster_types_predicted': len(validated_disaster_types),
            'regions_covered': len(validated_regions),
            'type_coverage_ratio': float(len(validated_disaster_types) / 24)
        },
        'predictions': simplified_predictions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n 结果已保存: {output_path}")
    print(f"验证后预测{len(validated_disaster_types)}种灾害类型，覆盖{len(validated_regions)}个地区")
    print(f"地理合理性验证通过率: {validation_report['pass_rate']:.1%}")
    
    return str(output_path)


if __name__ == "__main__":
    run_prediction_system()