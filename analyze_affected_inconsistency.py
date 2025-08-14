#!/usr/bin/env python3
"""
分析people_affected字段的不一致问题
"""

import json

def analyze_affected_inconsistency():
    with open('src/preprocessing/data/historical_disasters/processed_historical_disasters.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    disasters = data['disasters']
    print('=== people_affected 不一致详细分析 ===')

    inconsistent_records = []
    
    for d in disasters:
        affected_main = d.get('people_affected', 0)
        affected_medical = d.get('medical_needs', {}).get('total_affected_reported', 0)
        
        if affected_main != affected_medical and affected_medical > 0:
            inconsistent_records.append({
                'disaster_id': d.get('disaster_id'),
                'name': d.get('name', ''),
                'people_affected': affected_main,
                'medical_affected': affected_medical,
                'ratio': affected_medical / max(affected_main, 1),
                'field_reports_matched': d.get('medical_needs', {}).get('field_reports_matched', 0),
                'field_reports_processed': d.get('medical_needs', {}).get('field_reports_processed', 0)
            })

    print(f'总共发现 {len(inconsistent_records)} 个不一致的记录:')
    print()
    
    # 按比例排序显示前10个
    inconsistent_records.sort(key=lambda x: x['ratio'], reverse=True)
    
    for i, item in enumerate(inconsistent_records[:10]):
        print(f'{i+1}. 灾害ID: {item["disaster_id"]}')
        print(f'   名称: {item["name"][:80]}{"..." if len(item["name"]) > 80 else ""}')
        print(f'   people_affected (主字段): {item["people_affected"]:,}')
        print(f'   total_affected_reported (medical): {item["medical_affected"]:,}')
        print(f'   比例 (medical/main): {item["ratio"]:.2f}')
        print(f'   Field Reports 匹配: {item["field_reports_matched"]}/{item["field_reports_processed"]}')
        print()
    
    # 统计分析
    print('=== 统计分析 ===')
    zero_main_positive_medical = [r for r in inconsistent_records if r['people_affected'] == 0]
    positive_both_different = [r for r in inconsistent_records if r['people_affected'] > 0]
    
    print(f'- 主字段为0，medical字段为正数的记录: {len(zero_main_positive_medical)}')
    print(f'- 两个字段都为正数但不相等的记录: {len(positive_both_different)}')
    
    if positive_both_different:
        ratios = [r['ratio'] for r in positive_both_different]
        avg_ratio = sum(ratios) / len(ratios)
        print(f'- 不相等记录的平均比例: {avg_ratio:.2f}')
        higher_medical = [r for r in positive_both_different if r['ratio'] > 1]
        print(f'- medical字段数值更高的记录: {len(higher_medical)}')

if __name__ == "__main__":
    analyze_affected_inconsistency()