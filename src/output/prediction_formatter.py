"""
预测结果格式化器 - 数据驱动的预测输出处理
完全基于数据模式自动格式化和优化预测结果
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import json


class PredictionFormatter:
    """数据驱动的预测结果格式化器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_templates = {}
        self.risk_thresholds = {}
        
    def format_prediction_results(self, prediction_results: Dict[str, Any], 
                                current_conditions: Dict[str, Any],
                                historical_context: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """格式化预测结果为标准输出格式"""
        self.logger.info("格式化预测结果...")
        
        # 自动确定风险等级阈值
        if historical_context is not None:
            self._auto_calibrate_risk_thresholds(historical_context)
        
        formatted_output = {
            'prediction_summary': self._create_prediction_summary(prediction_results),
            'detailed_forecasts': self._create_detailed_forecasts(prediction_results),
            'risk_assessment': self._create_risk_assessment(prediction_results, current_conditions),
            'confidence_analysis': self._create_confidence_analysis(prediction_results),
            'expert_insights': self._extract_expert_insights(prediction_results),
            'metadata': self._create_output_metadata(current_conditions)
        }
        
        return formatted_output
    
    def _auto_calibrate_risk_thresholds(self, historical_data: pd.DataFrame) -> None:
        """基于历史数据自动校准风险等级阈值"""
        if 'people_affected' not in historical_data.columns:
            self.risk_thresholds = {'low': 1000, 'medium': 10000, 'high': 100000}
            return
        
        impacts = historical_data['people_affected'].dropna()
        if len(impacts) == 0:
            self.risk_thresholds = {'low': 1000, 'medium': 10000, 'high': 100000}
            return
        
        # 基于数据分布自动设定阈值
        self.risk_thresholds = {
            'low': impacts.quantile(0.33),
            'medium': impacts.quantile(0.67),
            'high': impacts.quantile(0.9),
            'extreme': impacts.quantile(0.99)
        }
        
        self.logger.info(f"自动校准风险阈值: {self.risk_thresholds}")
    
    def _create_prediction_summary(self, prediction_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建预测摘要"""
        disaster_probs = prediction_results.get('disaster_probabilities', {})
        
        if not disaster_probs:
            return {'status': 'no_predictions', 'message': '无可用预测'}
        
        # 找出最高概率的灾害
        top_disaster = max(disaster_probs.items(), key=lambda x: x[1])
        disaster_type, probability = top_disaster
        
        # 影响估计
        impact_estimates = prediction_results.get('impact_estimates', {})
        estimated_impact = impact_estimates.get(disaster_type, 0)
        
        # 风险等级
        risk_level = self._categorize_risk_level(estimated_impact)
        
        summary = {
            'primary_threat': {
                'disaster_type_id': disaster_type,
                'probability': round(probability, 3),
                'estimated_impact': int(estimated_impact),
                'risk_level': risk_level
            },
            'prediction_confidence': round(prediction_results.get('final_confidence', 0.0), 3),
            'total_disaster_types_considered': len(disaster_probs),
            'high_probability_threats': [
                {'disaster_type': dt, 'probability': round(prob, 3)} 
                for dt, prob in disaster_probs.items() if prob > 0.3
            ]
        }
        
        return summary
    
    def _categorize_risk_level(self, estimated_impact: float) -> str:
        """基于影响规模分类风险等级"""
        if estimated_impact >= self.risk_thresholds.get('extreme', 1000000):
            return 'extreme'
        elif estimated_impact >= self.risk_thresholds.get('high', 100000):
            return 'high'
        elif estimated_impact >= self.risk_thresholds.get('medium', 10000):
            return 'medium'
        else:
            return 'low'
    
    def _create_detailed_forecasts(self, prediction_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建详细预测"""
        detailed_forecasts = []
        
        disaster_probs = prediction_results.get('disaster_probabilities', {})
        impact_estimates = prediction_results.get('impact_estimates', {})
        confidence_scores = prediction_results.get('confidence_scores', {})
        expert_contributions = prediction_results.get('expert_contributions', {})
        
        # 按概率排序
        sorted_disasters = sorted(disaster_probs.items(), key=lambda x: x[1], reverse=True)
        
        for disaster_type, probability in sorted_disasters:
            if probability > 0.05:  # 只包含概率>5%的预测
                forecast = {
                    'disaster_type_id': disaster_type,
                    'probability': round(probability, 4),
                    'estimated_impact': int(impact_estimates.get(disaster_type, 0)),
                    'risk_level': self._categorize_risk_level(impact_estimates.get(disaster_type, 0)),
                    'confidence': round(confidence_scores.get(disaster_type, 0.0), 3),
                    'contributing_experts': self._format_expert_contributions(
                        expert_contributions.get(disaster_type, [])
                    )
                }
                detailed_forecasts.append(forecast)
        
        return detailed_forecasts
    
    def _format_expert_contributions(self, contributions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化专家贡献信息"""
        formatted_contributions = []
        
        # 按权重排序
        sorted_contributions = sorted(contributions, key=lambda x: x.get('weight', 0), reverse=True)
        
        for contrib in sorted_contributions:
            if contrib.get('weight', 0) > 0.1:  # 只显示权重>10%的专家
                formatted_contrib = {
                    'expert_name': contrib.get('expert', 'unknown'),
                    'contribution_weight': round(contrib.get('weight', 0), 3),
                    'expert_probability': round(contrib.get('probability', 0), 3),
                    'expert_confidence': round(contrib.get('confidence', 0), 3)
                }
                formatted_contributions.append(formatted_contrib)
        
        return formatted_contributions
    
    def _create_risk_assessment(self, prediction_results: Dict[str, Any], 
                              current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """创建风险评估"""
        disaster_probs = prediction_results.get('disaster_probabilities', {})
        impact_estimates = prediction_results.get('impact_estimates', {})
        
        # 计算综合风险评分
        total_risk_score = 0.0
        for disaster_type, probability in disaster_probs.items():
            impact = impact_estimates.get(disaster_type, 0)
            risk_contribution = probability * np.log10(max(impact, 1))
            total_risk_score += risk_contribution
        
        # 标准化风险评分
        normalized_risk = min(total_risk_score / 10.0, 1.0)
        
        # 风险等级
        if normalized_risk > 0.8:
            overall_risk_level = 'critical'
        elif normalized_risk > 0.6:
            overall_risk_level = 'high'
        elif normalized_risk > 0.4:
            overall_risk_level = 'medium'
        else:
            overall_risk_level = 'low'
        
        risk_assessment = {
            'overall_risk_score': round(normalized_risk, 3),
            'overall_risk_level': overall_risk_level,
            'immediate_threats': [
                dt for dt, prob in disaster_probs.items() if prob > 0.5
            ],
            'potential_threats': [
                dt for dt, prob in disaster_probs.items() if 0.2 <= prob <= 0.5
            ],
            'low_probability_events': [
                dt for dt, prob in disaster_probs.items() if 0.05 <= prob < 0.2
            ],
            'risk_diversification': len([p for p in disaster_probs.values() if p > 0.1]),
            'country_context': {
                'country_id': current_conditions.get('country_id'),
                'assessment_month': current_conditions.get('month'),
                'assessment_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        return risk_assessment
    
    def _create_confidence_analysis(self, prediction_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建置信度分析"""
        confidence_scores = prediction_results.get('confidence_scores', {})
        final_confidence = prediction_results.get('final_confidence', 0.0)
        
        if not confidence_scores:
            return {'overall_confidence': 'unknown', 'confidence_distribution': {}}
        
        # 置信度分布
        confidence_values = list(confidence_scores.values())
        confidence_distribution = {
            'mean': round(np.mean(confidence_values), 3),
            'std': round(np.std(confidence_values), 3),
            'min': round(min(confidence_values), 3),
            'max': round(max(confidence_values), 3),
            'median': round(np.median(confidence_values), 3)
        }
        
        # 置信度等级
        if final_confidence > 0.8:
            confidence_level = 'very_high'
            reliability_message = '预测高度可信'
        elif final_confidence > 0.6:
            confidence_level = 'high'
            reliability_message = '预测较为可信'
        elif final_confidence > 0.4:
            confidence_level = 'medium'
            reliability_message = '预测中等可信度'
        else:
            confidence_level = 'low'
            reliability_message = '预测置信度较低，建议谨慎参考'
        
        confidence_analysis = {
            'overall_confidence_level': confidence_level,
            'overall_confidence_score': round(final_confidence, 3),
            'reliability_message': reliability_message,
            'confidence_distribution': confidence_distribution,
            'high_confidence_predictions': [
                dt for dt, conf in confidence_scores.items() if conf > 0.7
            ],
            'low_confidence_predictions': [
                dt for dt, conf in confidence_scores.items() if conf < 0.4
            ]
        }
        
        return confidence_analysis
    
    def _extract_expert_insights(self, prediction_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """提取专家见解"""
        expert_contributions = prediction_results.get('expert_contributions', {})
        insights = defaultdict(list)
        
        for disaster_type, contributions in expert_contributions.items():
            for contrib in contributions:
                expert_name = contrib.get('expert', '')
                weight = contrib.get('weight', 0)
                probability = contrib.get('probability', 0)
                
                if weight > 0.2:  # 权重>20%的专家
                    if probability > 0.5:
                        insight = f"{expert_name}高度关注灾害{disaster_type}(概率{probability:.2f})"
                    elif probability > 0.3:
                        insight = f"{expert_name}认为灾害{disaster_type}有中等风险(概率{probability:.2f})"
                    else:
                        insight = f"{expert_name}检测到灾害{disaster_type}的潜在风险"
                    
                    insights[expert_name].append(insight)
        
        return dict(insights)
    
    def _create_output_metadata(self, current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """创建输出元数据"""
        metadata = {
            'prediction_timestamp': pd.Timestamp.now().isoformat(),
            'input_conditions': current_conditions,
            'model_version': 'mixed_expert_v1.0',
            'prediction_scope': 'disaster_type_and_impact',
            'geographical_context': current_conditions.get('country_id'),
            'temporal_context': current_conditions.get('month'),
            'data_driven_features': True,
            'expert_fusion_method': 'dynamic_weighted_ensemble'
        }
        
        return metadata
    
    def export_to_json(self, formatted_results: Dict[str, Any], output_path: str) -> None:
        """导出为JSON格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"预测结果已导出至: {output_path}")
    
    def export_to_csv(self, formatted_results: Dict[str, Any], output_path: str) -> None:
        """导出为CSV格式"""
        # 将详细预测转换为DataFrame
        detailed_forecasts = formatted_results.get('detailed_forecasts', [])
        if detailed_forecasts:
            df = pd.DataFrame(detailed_forecasts)
            
            # 展开专家贡献信息
            expert_data = []
            for forecast in detailed_forecasts:
                for contrib in forecast.get('contributing_experts', []):
                    expert_row = {
                        'disaster_type_id': forecast['disaster_type_id'],
                        'overall_probability': forecast['probability'],
                        'estimated_impact': forecast['estimated_impact'],
                        'risk_level': forecast['risk_level'],
                        'expert_name': contrib['expert_name'],
                        'expert_weight': contrib['contribution_weight'],
                        'expert_probability': contrib['expert_probability'],
                        'expert_confidence': contrib['expert_confidence']
                    }
                    expert_data.append(expert_row)
            
            # 保存主预测表
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            # 保存专家贡献表
            if expert_data:
                expert_df = pd.DataFrame(expert_data)
                expert_output_path = output_path.replace('.csv', '_expert_contributions.csv')
                expert_df.to_csv(expert_output_path, index=False, encoding='utf-8')
                self.logger.info(f"专家贡献数据已导出至: {expert_output_path}")
        
        self.logger.info(f"预测结果已导出至: {output_path}")
    
    def create_human_readable_report(self, formatted_results: Dict[str, Any]) -> str:
        """创建人类可读的报告"""
        report_lines = []
        
        # 标题
        report_lines.append("=== 灾害预测报告 ===")
        report_lines.append("")
        
        # 预测摘要
        summary = formatted_results.get('prediction_summary', {})
        primary_threat = summary.get('primary_threat', {})
        
        if primary_threat:
            report_lines.append("【主要威胁】")
            report_lines.append(f"  预测灾害类型: {primary_threat.get('disaster_type_id')}")
            report_lines.append(f"  发生概率: {primary_threat.get('probability', 0):.1%}")
            report_lines.append(f"  预计影响人数: {primary_threat.get('estimated_impact', 0):,}")
            report_lines.append(f"  风险等级: {primary_threat.get('risk_level', 'unknown').upper()}")
            report_lines.append("")
        
        # 置信度分析
        confidence = formatted_results.get('confidence_analysis', {})
        report_lines.append("【置信度评估】")
        report_lines.append(f"  整体可信度: {confidence.get('overall_confidence_level', 'unknown').upper()}")
        report_lines.append(f"  置信度评分: {confidence.get('overall_confidence_score', 0):.1%}")
        report_lines.append(f"  可信度说明: {confidence.get('reliability_message', '')}")
        report_lines.append("")
        
        # 风险评估
        risk_assessment = formatted_results.get('risk_assessment', {})
        report_lines.append("【风险评估】")
        report_lines.append(f"  综合风险等级: {risk_assessment.get('overall_risk_level', 'unknown').upper()}")
        report_lines.append(f"  风险评分: {risk_assessment.get('overall_risk_score', 0):.3f}")
        
        immediate_threats = risk_assessment.get('immediate_threats', [])
        if immediate_threats:
            report_lines.append(f"  紧急威胁: {', '.join(map(str, immediate_threats))}")
        
        potential_threats = risk_assessment.get('potential_threats', [])
        if potential_threats:
            report_lines.append(f"  潜在威胁: {', '.join(map(str, potential_threats))}")
        
        report_lines.append("")
        
        # 专家见解
        expert_insights = formatted_results.get('expert_insights', {})
        if expert_insights:
            report_lines.append("【专家分析】")
            for expert_name, insights in expert_insights.items():
                if insights:
                    report_lines.append(f"  {expert_name}:")
                    for insight in insights:
                        report_lines.append(f"    - {insight}")
            report_lines.append("")
        
        # 详细预测列表
        detailed_forecasts = formatted_results.get('detailed_forecasts', [])
        if detailed_forecasts:
            report_lines.append("【详细预测】")
            for i, forecast in enumerate(detailed_forecasts[:10], 1):  # 最多显示10个
                report_lines.append(f"  {i}. 灾害类型{forecast['disaster_type_id']}: "
                                  f"概率{forecast['probability']:.1%}, "
                                  f"影响{forecast['estimated_impact']:,}人, "
                                  f"风险{forecast['risk_level']}")
        
        # 元数据
        metadata = formatted_results.get('metadata', {})
        report_lines.append("")
        report_lines.append("【报告信息】")
        report_lines.append(f"  生成时间: {metadata.get('prediction_timestamp', '')}")
        report_lines.append(f"  预测国家: {metadata.get('geographical_context', '')}")
        report_lines.append(f"  预测月份: {metadata.get('temporal_context', '')}")
        report_lines.append(f"  模型版本: {metadata.get('model_version', '')}")
        
        return "\n".join(report_lines)
    
    def create_alert_system_output(self, formatted_results: Dict[str, Any], 
                                 alert_threshold: float = 0.7) -> Dict[str, Any]:
        """创建告警系统输出"""
        summary = formatted_results.get('prediction_summary', {})
        primary_threat = summary.get('primary_threat', {})
        
        alert_output = {
            'alert_triggered': False,
            'alert_level': 'none',
            'alert_message': '',
            'recommended_actions': [],
            'monitoring_suggestions': []
        }
        
        threat_probability = primary_threat.get('probability', 0.0)
        threat_impact = primary_threat.get('estimated_impact', 0)
        risk_level = primary_threat.get('risk_level', 'low')
        
        # 触发告警条件
        if threat_probability >= alert_threshold or risk_level in ['high', 'extreme']:
            alert_output['alert_triggered'] = True
            
            if threat_probability >= 0.9 or risk_level == 'extreme':
                alert_output['alert_level'] = 'critical'
                alert_output['alert_message'] = f"紧急警报：灾害类型{primary_threat.get('disaster_type_id')}高概率发生({threat_probability:.1%})"
                alert_output['recommended_actions'] = [
                    '立即启动应急响应机制',
                    '通知相关救援部门',
                    '准备救灾物资和人员',
                    '发布公众预警信息'
                ]
            elif threat_probability >= alert_threshold or risk_level == 'high':
                alert_output['alert_level'] = 'warning'
                alert_output['alert_message'] = f"警告：灾害类型{primary_threat.get('disaster_type_id')}存在较高风险({threat_probability:.1%})"
                alert_output['recommended_actions'] = [
                    '加强监测相关指标',
                    '准备应急预案',
                    '关注天气和环境变化',
                    '检查救援物资储备'
                ]
        
        # 监测建议
        risk_assessment = formatted_results.get('risk_assessment', {})
        potential_threats = risk_assessment.get('potential_threats', [])
        if potential_threats:
            alert_output['monitoring_suggestions'] = [
                f'持续监测潜在威胁：{", ".join(map(str, potential_threats))}',
                '建议增加预测频率',
                '关注专家模型置信度变化'
            ]
        
        return alert_output
    
    def generate_expert_performance_summary(self, expert_predictions: Dict[str, Dict[str, Any]],
                                          fusion_weights: Dict[str, float]) -> Dict[str, Any]:
        """生成专家性能摘要"""
        performance_summary = {
            'expert_rankings': [],
            'weight_distribution': fusion_weights,
            'contribution_analysis': {},
            'specialization_effectiveness': {}
        }
        
        # 专家排名
        sorted_experts = sorted(fusion_weights.items(), key=lambda x: x[1], reverse=True)
        for rank, (expert_name, weight) in enumerate(sorted_experts, 1):
            expert_prediction = expert_predictions.get(expert_name, {})
            avg_confidence = np.mean(list(expert_prediction.get('confidence_scores', {}).values()))
            
            performance_summary['expert_rankings'].append({
                'rank': rank,
                'expert_name': expert_name,
                'fusion_weight': round(weight, 3),
                'average_confidence': round(avg_confidence, 3),
                'prediction_count': len(expert_prediction.get('disaster_probabilities', {}))
            })
        
        return performance_summary