#!/usr/bin/env python3
"""
分析processed_historical_disasters中的数据问题
"""

import json

def analyze_data_issues():
    with open('src/preprocessing/data/historical_disasters/processed_historical_disasters.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    disasters = data['disasters']
    print('=== 数据不一致性分析 ===')
    print(f'总记录数: {len(disasters)}')
    print()

    # 1. 检查people_affected与medical_needs.total_affected_reported的不一致
    print('1. people_affected 与 medical_affected 不一致分析:')
    inconsistent_affected = []
    zero_affected_but_medical_data = []
    
    for d in disasters:
        affected_main = d.get('people_affected', 0)
        affected_medical = d.get('medical_needs', {}).get('total_affected_reported', 0)
        
        if affected_main == 0 and affected_medical > 0:
            zero_affected_but_medical_data.append({
                'disaster_id': d.get('disaster_id'),
                'name': d.get('name', '')[:50] + '...' if len(d.get('name', '')) > 50 else d.get('name', ''),
                'people_affected': affected_main,
                'medical_affected': affected_medical
            })
        elif affected_main != affected_medical and affected_medical > 0:
            inconsistent_affected.append({
                'disaster_id': d.get('disaster_id'),
                'name': d.get('name', '')[:50] + '...' if len(d.get('name', '')) > 50 else d.get('name', ''),
                'people_affected': affected_main,
                'medical_affected': affected_medical,
                'ratio': affected_medical / max(affected_main, 1)
            })

    print(f'   - people_affected=0但medical中有数据的记录: {len(zero_affected_but_medical_data)}')
    print(f'   - 数值不一致的记录: {len(inconsistent_affected)}')
    
    # 显示前5个people_affected=0但medical中有数据的记录
    if zero_affected_but_medical_data:
        print('   前5个people_affected=0但medical有数据的记录:')
        for i, item in enumerate(zero_affected_but_medical_data[:5]):
            print(f'   {i+1}. ID:{item["disaster_id"]} - {item["name"]}')
            print(f'      people_affected: {item["people_affected"]:,}')
            print(f'      medical_affected: {item["medical_affected"]:,}')
            print()
    
    # 2. 检查field_reports匹配情况
    print('2. Field Reports 匹配情况分析:')
    total_reports_processed = 0
    total_reports_matched = 0
    reports_with_data = 0
    
    for d in disasters:
        medical_needs = d.get('medical_needs', {})
        processed = medical_needs.get('field_reports_processed', 0)
        matched = medical_needs.get('field_reports_matched', 0)
        
        total_reports_processed += processed
        total_reports_matched += matched
        
        if processed > 0:
            reports_with_data += 1
    
    match_rate = total_reports_matched / max(total_reports_processed, 1)
    print(f'   - 总处理的报告数: {total_reports_processed:,}')
    print(f'   - 总匹配的报告数: {total_reports_matched:,}')
    print(f'   - 匹配率: {match_rate:.2%}')
    print(f'   - 有field reports的灾害数: {reports_with_data:,}')
    print()
    
    # 3. 检查日期格式问题
    print('3. 日期格式分析:')
    valid_dates = 0
    invalid_dates = 0
    
    for d in disasters:
        start_date = d.get('start_date')
        if start_date and 'T' in start_date and len(start_date) >= 19:
            valid_dates += 1
        else:
            invalid_dates += 1
    
    print(f'   - 有效日期格式: {valid_dates:,}')
    print(f'   - 无效日期格式: {invalid_dates:,}')
    print()
    
    # 4. 检查伤亡数据
    print('4. 伤亡数据分析:')
    injured_from_main = sum(1 for d in disasters if d.get('people_injured', 0) > 0)
    dead_from_main = sum(1 for d in disasters if d.get('people_dead', 0) > 0)
    injured_from_medical = sum(1 for d in disasters if d.get('medical_needs', {}).get('total_injured_reported', 0) > 0)
    dead_from_medical = sum(1 for d in disasters if d.get('medical_needs', {}).get('total_dead_reported', 0) > 0)
    
    print(f'   - 主字段中有injured数据的记录: {injured_from_main:,}')
    print(f'   - medical字段中有injured数据的记录: {injured_from_medical:,}')
    print(f'   - 主字段中有dead数据的记录: {dead_from_main:,}')
    print(f'   - medical字段中有dead数据的记录: {dead_from_medical:,}')

if __name__ == "__main__":
    analyze_data_issues()