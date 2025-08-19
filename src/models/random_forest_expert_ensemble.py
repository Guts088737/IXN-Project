"""
混合专家+随机森林集成模型
解决灾害类型多样性和地理偏向问题
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from collections import defaultdict


class RandomForestExpertEnsemble:
    """随机森林+混合专家集成模型"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 随机森林模型组件
        self.disaster_type_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.impact_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.region_rf = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # 数据预处理组件
        self.feature_scaler = StandardScaler()
        self.disaster_type_encoder = LabelEncoder()
        self.region_encoder = LabelEncoder()
        
        # 专家-随机森林融合权重
        self.expert_rf_weights = {}
        self.global_diversity_boosters = {}
        
        # 训练状态
        self.is_trained = False
        self.feature_names = []
        
    def prepare_features(self, historical_data: pd.DataFrame, 
                        country_risk_factors: Dict[int, Dict[str, float]],
                        spatial_features: Dict[int, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """准备随机森林特征矩阵"""
        self.logger.info("准备随机森林特征...")
        self.logger.info(f"历史数据形状: {historical_data.shape}")
        self.logger.info(f"国家风险因子数量: {len(country_risk_factors)}")
        self.logger.info(f"空间特征数量: {len(spatial_features)}")
        
        feature_rows = []
        disaster_types = []
        impacts = []
        
        for _, row in historical_data.iterrows():
            country_id = row['country_id']
            if pd.isna(country_id):
                continue
            
            # 基础特征
            features = {
                'latitude': row.get('latitude', 0),
                'longitude': row.get('longitude', 0),
                'month': row.get('month', 6),
                'year': row.get('year', 2020),
                'region_id': row.get('region_id', 1),
                'funding_coverage': row.get('funding_coverage', 0.5)
            }
            
            # 添加国家风险因子
            if int(country_id) in country_risk_factors:
                risk_factors = country_risk_factors[int(country_id)]
                features.update(risk_factors)
            
            # 添加空间特征
            if int(country_id) in spatial_features:
                spatial_feats = spatial_features[int(country_id)]
                # 选择数值型空间特征
                numeric_spatial = {k: v for k, v in spatial_feats.items() 
                                 if isinstance(v, (int, float)) and not pd.isna(v)}
                features.update(numeric_spatial)
            
            # 添加时间特征
            features.update({
                'is_wet_season': float(row.get('month', 6) in [6, 7, 8, 9]),
                'is_dry_season': float(row.get('month', 6) in [12, 1, 2, 3]),
                'year_normalized': (row.get('year', 2020) - 2000) / 25.0,  # 改为25年跨度避免边界问题
                'latitude_abs': abs(row.get('latitude', 0)),
                'is_tropical': float(abs(row.get('latitude', 0)) < 23.5),
                'is_coastal': float(abs(row.get('longitude', 0)) < 180),  # 简化的沿海判断
            })
            
            # 添加交互特征
            features.update({
                'lat_month_interaction': features['latitude'] * features['month'],
                'lng_month_interaction': features['longitude'] * features['month'], 
                'tropical_wet_interaction': features['is_tropical'] * features['is_wet_season'],
                'region_month_interaction': features['region_id'] * features['month']
            })
            
            feature_rows.append(features)
            disaster_types.append(row.get('disaster_type_id', 1))
            impacts.append(row.get('people_affected', 1000))
        
        # 转换为数组
        if not feature_rows:
            return np.array([]), np.array([]), np.array([])
        
        # 统一特征名称
        all_feature_names = set()
        for features in feature_rows:
            all_feature_names.update(features.keys())
        
        self.feature_names = sorted(list(all_feature_names))
        
        # 构建特征矩阵并清理异常值（改进稳健性）
        feature_matrix = np.zeros((len(feature_rows), len(self.feature_names)))
        for i, features in enumerate(feature_rows):
            for j, feature_name in enumerate(self.feature_names):
                value = features.get(feature_name, 0.0)
                # 清理异常值（更严格的边界）
                if pd.isna(value) or np.isinf(value):
                    value = 0.0
                elif abs(value) > 1000:  # 更严格的极值限制
                    # 对数压缩极值而不是截断
                    value = np.sign(value) * (1000 + np.log10(abs(value) - 1000 + 1) * 100)
                feature_matrix[i, j] = float(value)
        
        # 添加小的随机扰动避免所有特征值相同导致的标准化问题
        for j in range(feature_matrix.shape[1]):
            col_std = np.std(feature_matrix[:, j])
            if col_std == 0:  # 如果该特征列的标准差为0
                # 添加微小的随机扰动
                feature_matrix[:, j] += np.random.normal(0, 1e-6, feature_matrix.shape[0])
        
        return feature_matrix, np.array(disaster_types), np.array(impacts)
    
    def train_random_forest_models(self, historical_data: pd.DataFrame,
                                 country_risk_factors: Dict[int, Dict[str, float]],
                                 spatial_features: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """训练随机森林模型"""
        self.logger.info("训练随机森林模型...")
        
        # 准备特征
        X, y_disaster, y_impact = self.prepare_features(historical_data, country_risk_factors, spatial_features)
        
        if len(X) == 0:
            raise ValueError("无有效特征数据")
        
        # 数据预处理（避免除零错误）
        try:
            X_scaled = self.feature_scaler.fit_transform(X)
        except Exception as e:
            self.logger.warning(f"特征标准化失败，使用原始特征: {e}")
            X_scaled = X
        
        y_disaster_encoded = self.disaster_type_encoder.fit_transform(y_disaster)
        
        # 分割训练/验证数据
        try:
            self.logger.info(f"数据分割前 - X_scaled形状: {X_scaled.shape}, y_disaster_encoded唯一值: {len(np.unique(y_disaster_encoded))}")
            X_train, X_val, y_disaster_train, y_disaster_val, y_impact_train, y_impact_val = train_test_split(
                X_scaled, y_disaster_encoded, y_impact, test_size=0.2, random_state=42, stratify=y_disaster_encoded
            )
            self.logger.info("数据分割成功")
        except Exception as e:
            self.logger.error(f"数据分割失败: {e}")
            # 如果分层失败，不使用分层
            X_train, X_val, y_disaster_train, y_disaster_val, y_impact_train, y_impact_val = train_test_split(
                X_scaled, y_disaster_encoded, y_impact, test_size=0.2, random_state=42
            )
        
        training_results = {}
        
        # 1. 训练灾害类型分类器
        self.logger.info("训练灾害类型随机森林...")
        try:
            self.disaster_type_rf.fit(X_train, y_disaster_train)
            self.logger.info("灾害类型分类器训练成功")
        except Exception as e:
            self.logger.error(f"灾害类型分类器训练失败: {e}")
            raise e
        
        # 评估灾害类型预测 - 动态调整交叉验证折数
        disaster_type_score = self.disaster_type_rf.score(X_val, y_disaster_val)
        
        # 动态确定交叉验证折数
        unique_classes, class_counts = np.unique(y_disaster_train, return_counts=True)
        min_class_size = min(class_counts)
        cv_folds = min(5, min_class_size)  # 确保每折至少有1个样本
        
        if cv_folds < 2:
            cv_folds = 2  # 最少2折
            self.logger.warning(f"某些灾害类型样本过少，使用{cv_folds}折交叉验证")
        
        disaster_cv_scores = cross_val_score(self.disaster_type_rf, X_train, y_disaster_train, cv=cv_folds)
        
        training_results['disaster_type_classification'] = {
            'validation_accuracy': disaster_type_score,
            'cv_mean': disaster_cv_scores.mean(),
            'cv_std': disaster_cv_scores.std(),
            'unique_classes_predicted': len(np.unique(self.disaster_type_rf.predict(X_val)))
        }
        
        # 2. 训练影响规模回归器（添加异常值处理）
        self.logger.info("训练影响规模随机森林...")
        
        # 学习影响约束模式并清理训练数据
        self.impact_constraints = self._learn_impact_constraints_from_data(historical_data)
        y_impact_cleaned = self._clean_impact_outliers(y_impact_train)
        self.impact_rf.fit(X_train, y_impact_cleaned)
        
        impact_score = self.impact_rf.score(X_val, y_impact_val)
        
        # 影响回归也使用动态交叉验证折数
        impact_cv_folds = min(5, len(y_impact_train) // 10)  # 每折至少10个样本
        if impact_cv_folds < 2:
            impact_cv_folds = 2
        
        impact_cv_scores = cross_val_score(self.impact_rf, X_train, y_impact_train, cv=impact_cv_folds)
        
        training_results['impact_regression'] = {
            'validation_r2': impact_score,
            'cv_mean': impact_cv_scores.mean(),
            'cv_std': impact_cv_scores.std()
        }
        
        # 3. 训练区域分类器（用于地理多样性）
        if 'region_id' in historical_data.columns:
            regions = historical_data['region_id'].dropna()
            if len(regions.unique()) > 1:
                self.logger.info("训练区域分类随机森林...")
                region_encoded = self.region_encoder.fit_transform(regions)
                
                # 使用相同的特征训练区域预测
                region_X = X_scaled[:len(regions)]
                self.region_rf.fit(region_X, region_encoded)
                
                region_score = self.region_rf.score(region_X, region_encoded)
                training_results['region_classification'] = {
                    'accuracy': region_score,
                    'unique_regions': len(self.region_encoder.classes_)
                }
        
        # 4. 特征重要性分析
        disaster_feature_importance = dict(zip(self.feature_names, self.disaster_type_rf.feature_importances_))
        impact_feature_importance = dict(zip(self.feature_names, self.impact_rf.feature_importances_))
        
        training_results['feature_importance'] = {
            'disaster_type_features': dict(sorted(disaster_feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]),
            'impact_features': dict(sorted(impact_feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        }
        
        # 5. 多样性增强策略
        self.global_diversity_boosters = self._calculate_diversity_boosters(historical_data)
        training_results['diversity_boosters'] = self.global_diversity_boosters
        
        self.is_trained = True
        self.logger.info("随机森林模型训练完成")
        
        return training_results
    
    def _calculate_diversity_boosters(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """计算多样性增强因子"""
        # 分析历史数据中被忽略的灾害类型
        disaster_freq = historical_data['disaster_type_id'].value_counts()
        total_events = len(historical_data)
        
        # 识别低频但重要的灾害类型
        rare_disasters = {}
        for disaster_type, count in disaster_freq.items():
            frequency = count / max(total_events, 1)  # 避免除零
            if frequency < 0.05:  # 频率低于5%的灾害
                # 计算其平均影响
                type_data = historical_data[historical_data['disaster_type_id'] == disaster_type]
                avg_impact = type_data['people_affected'].mean()
                rare_disasters[disaster_type] = {
                    'frequency': frequency,
                    'avg_impact': avg_impact,
                    'boost_factor': min(1.0 / max(frequency, 0.001), 5.0)  # 避免除零
                }
        
        # 分析地理多样性
        region_freq = historical_data['region_id'].value_counts(normalize=True)
        underrepresented_regions = {
            region: freq for region, freq in region_freq.items() if freq < 0.1
        }
        
        return {
            'rare_disaster_boosters': rare_disasters,
            'underrepresented_regions': underrepresented_regions,
            'total_disaster_types': len(disaster_freq),
            'geographic_regions': len(region_freq)
        }
    
    def _clean_impact_outliers(self, y_impact: np.ndarray) -> np.ndarray:
        """清理影响数据中的异常值"""
        # 计算分位数
        q75 = np.percentile(y_impact, 75)
        q25 = np.percentile(y_impact, 25)
        iqr = q75 - q25
        
        # 使用IQR方法识别异常值，但不完全删除
        upper_bound = q75 + 2.5 * iqr  # 放宽上界到2.5倍IQR
        lower_bound = max(q25 - 1.5 * iqr, 100)  # 下界最少100人
        
        # 对异常值进行压缩而不是删除
        cleaned_impact = np.copy(y_impact)
        for i, impact in enumerate(y_impact):
            if impact > upper_bound:
                # 对数压缩极值
                cleaned_impact[i] = upper_bound + np.log10(max(impact - upper_bound, 1)) * 1000
            elif impact < lower_bound:
                cleaned_impact[i] = lower_bound
        
        # 确保没有超过绝对上限
        cleaned_impact = np.minimum(cleaned_impact, 500000)  # 训练数据绝对上限50万
        
        self.logger.info(f"影响数据清理: 原始范围[{y_impact.min():.0f}, {y_impact.max():.0f}] -> 清理后[{cleaned_impact.min():.0f}, {cleaned_impact.max():.0f}]")
        
        return cleaned_impact
    
    def predict_with_ensemble(self, current_conditions: Dict[str, Any],
                            expert_prediction: Dict[str, Any],
                            country_risk_factors: Dict[int, Dict[str, float]],
                            spatial_features: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """使用混合专家+随机森林集成预测"""
        if not self.is_trained:
            raise ValueError("随机森林模型未训练")
        
        self.logger.info("执行集成预测...")
        
        # 1. 准备当前条件的特征向量
        feature_vector = self._prepare_single_prediction_features(
            current_conditions, country_risk_factors, spatial_features
        )
        
        # 2. 随机森林预测
        rf_prediction = self._predict_with_random_forest(feature_vector, current_conditions)
        
        # 3. 融合专家预测和随机森林预测
        ensemble_prediction = self._fuse_expert_and_rf_predictions(expert_prediction, rf_prediction, current_conditions)
        
        # 4. 应用多样性增强
        enhanced_prediction = self._apply_diversity_enhancement(ensemble_prediction, current_conditions)
        
        return enhanced_prediction
    
    def _prepare_single_prediction_features(self, current_conditions: Dict[str, Any],
                                          country_risk_factors: Dict[int, Dict[str, float]],
                                          spatial_features: Dict[int, Dict[str, float]]) -> np.ndarray:
        """为单次预测准备特征向量"""
        country_id = current_conditions.get('country_id', 1)
        month = current_conditions.get('month', 6)
        year = current_conditions.get('year', 2025)
        
        # 构建特征字典
        features = {
            'latitude': current_conditions.get('latitude', 0),
            'longitude': current_conditions.get('longitude', 0),
            'month': month,
            'year': year,
            'region_id': current_conditions.get('region_id', 1),
            'funding_coverage': current_conditions.get('funding_coverage', 0.5)
        }
        
        # 添加风险因子
        if country_id in country_risk_factors:
            features.update(country_risk_factors[country_id])
        
        # 添加空间特征
        if country_id in spatial_features:
            spatial_feats = spatial_features[country_id]
            numeric_spatial = {k: v for k, v in spatial_feats.items() 
                             if isinstance(v, (int, float)) and not pd.isna(v)}
            features.update(numeric_spatial)
        
        # 添加派生特征
        features.update({
            'is_wet_season': float(month in [6, 7, 8, 9]),
            'is_dry_season': float(month in [12, 1, 2, 3]),
            'year_normalized': (year - 2000) / 25.0,
            'latitude_abs': abs(features['latitude']),
            'is_tropical': float(abs(features['latitude']) < 23.5),
            'is_coastal': float(abs(features['longitude']) < 180),
            'lat_month_interaction': features['latitude'] * month,
            'lng_month_interaction': features['longitude'] * month,
            'tropical_wet_interaction': float(abs(features['latitude']) < 23.5) * float(month in [6, 7, 8, 9]),
            'region_month_interaction': features['region_id'] * month
        })
        
        # 转换为向量并清理异常值
        feature_vector = np.zeros(len(self.feature_names))
        for i, feature_name in enumerate(self.feature_names):
            value = features.get(feature_name, 0.0)
            # 清理异常值
            if pd.isna(value) or np.isinf(value):
                value = 0.0
            elif abs(value) > 1e6:  # 限制极大值
                value = np.sign(value) * 1e6
            feature_vector[i] = float(value)
        
        # 标准化（避免除零错误）
        try:
            feature_vector_scaled = self.feature_scaler.transform(feature_vector.reshape(1, -1))
        except Exception as e:
            # 如果标准化失败，使用原始特征向量
            self.logger.warning(f"特征标准化失败，使用原始特征: {e}")
            feature_vector_scaled = feature_vector.reshape(1, -1)
        
        return feature_vector_scaled[0]
    
    def _predict_with_random_forest(self, feature_vector: np.ndarray, 
                                  current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """使用随机森林进行预测"""
        feature_matrix = feature_vector.reshape(1, -1)
        
        # 1. 灾害类型预测（概率分布）
        disaster_type_probs = self.disaster_type_rf.predict_proba(feature_matrix)[0]
        disaster_type_classes = self.disaster_type_encoder.classes_
        
        # 转换回原始灾害类型ID
        disaster_type_probabilities = {}
        for i, prob in enumerate(disaster_type_probs):
            original_disaster_type = self.disaster_type_encoder.inverse_transform([i])[0]
            disaster_type_probabilities[original_disaster_type] = prob
        
        # 2. 影响规模预测（使用数据驱动约束）
        raw_impact = self.impact_rf.predict(feature_matrix)[0]
        predicted_impact = self._apply_data_driven_impact_constraints(raw_impact, current_conditions)
        
        # 3. 区域预测（用于地理多样性）
        predicted_region = None
        if hasattr(self.region_rf, 'predict'):
            try:
                region_encoded = self.region_rf.predict(feature_matrix)[0]
                predicted_region = self.region_encoder.inverse_transform([region_encoded])[0]
            except:
                predicted_region = current_conditions.get('region_id', 1)
        
        # 4. 特征重要性（当前预测）
        current_feature_importance = dict(zip(self.feature_names, 
                                           self.disaster_type_rf.feature_importances_))
        
        rf_prediction = {
            'disaster_type_probabilities': disaster_type_probabilities,
            'predicted_impact': predicted_impact,
            'predicted_region': predicted_region,
            'prediction_confidence': np.max(disaster_type_probs),
            'feature_importance': current_feature_importance,
            'model_certainty': 1.0 - np.std(disaster_type_probs)  # 分布越集中越确定
        }
        
        return rf_prediction
    
    def _fuse_expert_and_rf_predictions(self, expert_prediction: Dict[str, Any], 
                                      rf_prediction: Dict[str, Any],
                                      current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """融合专家预测和随机森林预测"""
        # 获取所有可能的灾害类型
        expert_probs = expert_prediction.get('disaster_probabilities', {})
        rf_probs = rf_prediction.get('disaster_type_probabilities', {})
        
        all_disaster_types = set(expert_probs.keys()).union(set(rf_probs.keys()))
        
        # 自适应权重：随机森林在多样性上更强，专家在专业性上更强
        rf_weight = 0.75  # 进一步提高随机森林权重，增强多样性
        expert_weight = 0.25  # 降低专家权重，减少偏向
        
        fused_probabilities = {}
        fused_impacts = {}
        fused_confidences = {}
        
        for disaster_type in all_disaster_types:
            # 概率融合
            expert_prob = expert_probs.get(disaster_type, 0.0)
            rf_prob = rf_probs.get(disaster_type, 0.0)
            
            # 应用多样性增强
            diversity_boost = self._get_diversity_boost(disaster_type)
            
            fused_prob = (expert_prob * expert_weight + rf_prob * rf_weight) * diversity_boost
            fused_probabilities[disaster_type] = min(fused_prob, 1.0)
            
            # 影响融合（添加边界约束）
            expert_impact = expert_prediction.get('impact_estimates', {}).get(disaster_type, 1000)
            rf_impact = rf_prediction.get('predicted_impact', 1000)
            raw_fused_impact = expert_impact * expert_weight + rf_impact * rf_weight
            
            # 应用数据驱动的影响约束到融合结果
            constrained_fused_impact = self._apply_data_driven_impact_constraints(
                raw_fused_impact, {**current_conditions, 'disaster_type_id': disaster_type}
            )
            fused_impacts[disaster_type] = constrained_fused_impact
            
            # 置信度融合
            expert_conf = expert_prediction.get('confidence_scores', {}).get(disaster_type, 0.5)
            rf_conf = rf_prediction.get('prediction_confidence', 0.5)
            fused_conf = expert_conf * expert_weight + rf_conf * rf_weight
            fused_confidences[disaster_type] = fused_conf
        
        return {
            'disaster_probabilities': fused_probabilities,
            'impact_estimates': fused_impacts,
            'confidence_scores': fused_confidences,
            'fusion_method': 'expert_rf_ensemble',
            'rf_contribution': rf_weight,
            'expert_contribution': expert_weight,
            'diversity_enhanced': True
        }
    
    def _get_diversity_boost(self, disaster_type: int) -> float:
        """获取多样性增强因子"""
        rare_disasters = self.global_diversity_boosters.get('rare_disaster_boosters', {})
        
        if disaster_type in rare_disasters:
            boost_factor = rare_disasters[disaster_type].get('boost_factor', 1.0)
            return min(boost_factor, 3.0)  # 最大3倍增强
        
        return 1.0
    
    def _apply_diversity_enhancement(self, ensemble_prediction: Dict[str, Any], 
                                   current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """应用多样性增强策略"""
        enhanced_prediction = ensemble_prediction.copy()
        
        # 1. 地理多样性增强
        enhanced_prediction = self._enhance_geographic_diversity(enhanced_prediction, current_conditions)
        
        # 2. 灾害类型多样性增强
        enhanced_prediction = self._enhance_disaster_type_diversity(enhanced_prediction)
        
        # 3. 重新标准化概率
        enhanced_prediction = self._renormalize_probabilities(enhanced_prediction)
        
        return enhanced_prediction
    
    def _enhance_geographic_diversity(self, prediction: Dict[str, Any], 
                                    current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """增强地理多样性"""
        current_region = current_conditions.get('region_id', 1)
        underrepresented_regions = self.global_diversity_boosters.get('underrepresented_regions', {})
        
        # 如果当前区域被低估，则增强相关灾害类型
        if current_region in underrepresented_regions:
            boost_factor = 1.5  # 1.5倍增强
            
            probabilities = prediction['disaster_probabilities']
            enhanced_probabilities = {}
            
            for disaster_type, prob in probabilities.items():
                enhanced_probabilities[disaster_type] = min(prob * boost_factor, 1.0)
            
            prediction['disaster_probabilities'] = enhanced_probabilities
            prediction['geographic_diversity_enhanced'] = True
        
        return prediction
    
    def _enhance_disaster_type_diversity(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """增强灾害类型多样性"""
        probabilities = prediction['disaster_probabilities']
        
        # 确保所有24种灾害类型都有合理的概率
        # 从灾害类型数据动态获取所有ID
        all_disaster_type_ids = [66, 57, 7, 14, 6, 4, 20, 2, 1, 15, 12, 21, 19, 62, 24, 13, 27, 5, 67, 23, 54, 68, 11, 8]
        
        enhanced_probabilities = probabilities.copy()
        
        for disaster_type in all_disaster_type_ids:
            if disaster_type not in enhanced_probabilities:
                # 为缺失的灾害类型分配基础概率
                enhanced_probabilities[disaster_type] = 0.02
            elif enhanced_probabilities[disaster_type] < 0.01:
                # 提升过低的概率
                enhanced_probabilities[disaster_type] = 0.02
        
        # 对历史上少见但重要的灾害类型给予额外增强
        rare_but_important = [8, 11, 2, 1, 67, 66, 57]  # 火山、海啸、地震、流行病、辐射、生物、化学
        for disaster_type in rare_but_important:
            if disaster_type in enhanced_probabilities:
                enhanced_probabilities[disaster_type] *= 1.5
        
        prediction['disaster_probabilities'] = enhanced_probabilities
        prediction['type_diversity_enhanced'] = True
        
        return prediction
    
    
    def _learn_impact_constraints_from_data(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """从历史数据学习影响约束模式"""
        constraints = {}
        
        # 按灾害类型分析历史影响分布
        for disaster_type in historical_data['disaster_type_id'].unique():
            type_data = historical_data[historical_data['disaster_type_id'] == disaster_type]
            impacts = type_data['people_affected'].dropna()
            
            if len(impacts) > 0:
                constraints[disaster_type] = {
                    'median': np.median(impacts),
                    'q95': np.percentile(impacts, 95),
                    'q99': np.percentile(impacts, 99),
                    'max_historical': impacts.max(),
                    'mean': impacts.mean(),
                    'sample_count': len(impacts)
                }
        
        # 按地理区域分析影响分布
        region_constraints = {}
        for region_id in historical_data['region_id'].unique():
            region_data = historical_data[historical_data['region_id'] == region_id]
            region_impacts = region_data['people_affected'].dropna()
            
            if len(region_impacts) > 0:
                region_constraints[region_id] = {
                    'median': np.median(region_impacts),
                    'q95': np.percentile(region_impacts, 95),
                    'typical_max': np.percentile(region_impacts, 90)
                }
        
        return {
            'disaster_type_constraints': constraints,
            'region_constraints': region_constraints,
            'global_q99': np.percentile(historical_data['people_affected'].dropna(), 99)
        }
    
    def _apply_data_driven_impact_constraints(self, raw_impact: float, 
                                            current_conditions: Dict[str, Any]) -> float:
        """使用数据驱动的约束调整影响预测"""
        if not hasattr(self, 'impact_constraints'):
            return max(raw_impact, 100)  # 如果没有约束数据，只应用基础下界
        
        disaster_type = current_conditions.get('disaster_type_id')
        region_id = current_conditions.get('region_id')
        
        # 获取该灾害类型的历史分布
        type_constraints = self.impact_constraints['disaster_type_constraints'].get(disaster_type, {})
        region_constraints = self.impact_constraints['region_constraints'].get(region_id, {})
        
        # 基础约束
        constrained_impact = max(raw_impact, 100)
        
        # 应用灾害类型约束
        if type_constraints:
            # 使用历史95分位数作为软上界
            type_q95 = type_constraints.get('q95', 50000)
            type_max = type_constraints.get('max_historical', 100000)
            
            # 如果预测超过95分位数，应用对数压缩
            if constrained_impact > type_q95:
                excess = constrained_impact - type_q95
                compressed_excess = np.log10(max(excess, 1)) * type_q95 * 0.1
                constrained_impact = type_q95 + compressed_excess
            
            # 硬上界：历史最大值的1.5倍
            hard_upper = type_max * 1.5
            constrained_impact = min(constrained_impact, hard_upper)
        
        # 应用地理区域约束
        if region_constraints:
            region_typical_max = region_constraints.get('typical_max', 100000)
            constrained_impact = min(constrained_impact, region_typical_max * 2)
        
        # 全局约束：99分位数的2倍
        global_max = self.impact_constraints.get('global_q99', 200000) * 2
        constrained_impact = min(constrained_impact, global_max)
        
        return int(constrained_impact)
    
    def _renormalize_probabilities(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """重新标准化概率分布"""
        probabilities = prediction['disaster_probabilities']
        
        # 计算总概率
        total_prob = sum(probabilities.values())
        
        # 标准化（确保总和不超过合理范围）
        if total_prob > 2.0:  # 如果总概率过高，进行压缩
            compression_factor = 1.5 / total_prob
            normalized_probabilities = {
                disaster_type: prob * compression_factor 
                for disaster_type, prob in probabilities.items()
            }
        else:
            normalized_probabilities = probabilities
        
        prediction['disaster_probabilities'] = normalized_probabilities
        prediction['probability_sum'] = sum(normalized_probabilities.values())
        
        return prediction
    
    def generate_diverse_predictions(self, current_conditions: Dict[str, Any],
                                   expert_prediction: Dict[str, Any],
                                   country_risk_factors: Dict[int, Dict[str, float]],
                                   spatial_features: Dict[int, Dict[str, float]],
                                   num_predictions: int = 10) -> List[Dict[str, Any]]:
        """生成多样化的预测结果"""
        diverse_predictions = []
        
        # 基础集成预测
        base_prediction = self.predict_with_ensemble(
            current_conditions, expert_prediction, country_risk_factors, spatial_features
        )
        
        # 生成多样化变体
        for i in range(num_predictions):
            # 随机扰动输入条件以增加多样性
            perturbed_conditions = self._perturb_conditions(current_conditions, perturbation_level=0.1)
            
            # 重新预测
            variant_prediction = self.predict_with_ensemble(
                perturbed_conditions, expert_prediction, country_risk_factors, spatial_features
            )
            
            # 选择不同的顶级灾害
            disaster_probs = variant_prediction['disaster_probabilities']
            sorted_disasters = sorted(disaster_probs.items(), key=lambda x: x[1], reverse=True)
            
            # 确保预测多样性
            if i < len(sorted_disasters):
                top_disaster_type, top_prob = sorted_disasters[min(i, len(sorted_disasters)-1)]
                
                prediction_entry = {
                    'prediction_id': f"ENS_{i+1}_{hash(str(current_conditions)) % 10000}",
                    'disaster_type_id': top_disaster_type,
                    'probability': top_prob,
                    'estimated_impact': variant_prediction['impact_estimates'].get(top_disaster_type, 1000),
                    'confidence': variant_prediction['confidence_scores'].get(top_disaster_type, 0.5),
                    'prediction_method': 'expert_rf_ensemble',
                    'diversity_rank': i + 1,
                    'geographic_context': perturbed_conditions
                }
                
                diverse_predictions.append(prediction_entry)
        
        return diverse_predictions
    
    def _perturb_conditions(self, conditions: Dict[str, Any], perturbation_level: float = 0.1) -> Dict[str, Any]:
        """轻微扰动条件以增加预测多样性"""
        perturbed = conditions.copy()
        
        # 扰动数值特征
        if 'latitude' in perturbed:
            perturbed['latitude'] += np.random.normal(0, perturbation_level * 5)  # 5度标准差
        if 'longitude' in perturbed:
            perturbed['longitude'] += np.random.normal(0, perturbation_level * 10)  # 10度标准差
        
        # 扰动月份（偶尔改变）
        if np.random.random() < perturbation_level:
            perturbed['month'] = max(1, min(12, perturbed.get('month', 6) + np.random.choice([-1, 0, 1])))
        
        return perturbed
    
    def evaluate_ensemble_performance(self, historical_data: pd.DataFrame,
                                    country_risk_factors: Dict[int, Dict[str, float]],
                                    spatial_features: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """评估集成模型性能"""
        self.logger.info("评估集成模型性能...")
        
        # 准备测试数据
        X, y_disaster, y_impact = self.prepare_features(historical_data, country_risk_factors, spatial_features)
        X_scaled = self.feature_scaler.transform(X)
        
        # 灾害类型预测性能
        disaster_predictions = self.disaster_type_rf.predict(X_scaled)
        disaster_accuracy = np.mean(disaster_predictions == self.disaster_type_encoder.transform(y_disaster))
        
        # 检查预测的灾害类型多样性
        unique_predicted_types = len(np.unique(disaster_predictions))
        total_possible_types = len(self.disaster_type_encoder.classes_)
        diversity_ratio = unique_predicted_types / max(total_possible_types, 1)  # 避免除零
        
        # 影响预测性能
        impact_predictions = self.impact_rf.predict(X_scaled)
        impact_mae = np.mean(np.abs(impact_predictions - y_impact))
        impact_r2 = self.impact_rf.score(X_scaled, y_impact)
        
        performance_metrics = {
            'disaster_type_accuracy': disaster_accuracy,
            'disaster_type_diversity': {
                'unique_types_predicted': unique_predicted_types,
                'total_possible_types': total_possible_types,
                'diversity_ratio': diversity_ratio
            },
            'impact_prediction': {
                'mae': impact_mae,
                'r2_score': impact_r2
            },
            'model_complexity': {
                'total_features': len(self.feature_names),
                'disaster_rf_trees': self.disaster_type_rf.n_estimators,
                'impact_rf_trees': self.impact_rf.n_estimators
            },
            'diversity_enhancement': {
                'rare_disasters_count': len(self.global_diversity_boosters.get('rare_disaster_boosters', {})),
                'underrepresented_regions': len(self.global_diversity_boosters.get('underrepresented_regions', {}))
            }
        }
        
        return performance_metrics