"""
临时脚本：测试IFRC API连接
"""

import requests
import json
from src.config import get_config

def test_api_connection():
    """测试API连接"""
    config = get_config()
    
    print("Testing IFRC API connection...")
    print(f"API V1 URL: {config.api.base_url_v1}")
    print(f"API V2 URL: {config.api.base_url_v2}")
    
    # 测试您提供的正确API端点
    test_urls = [
        f"{config.api.base_url_v2}/public-local-units/?limit=1",
        f"{config.api.base_url_v2}/local-units-options/",
        f"{config.api.base_url_v2}/country/1/historical-disaster/?limit=1", 
        f"{config.api.base_url_v2}/disaster_type/?limit=1",
    ]
    
    for url in test_urls:
        print(f"\nTesting: {url}")
        try:
            response = requests.get(
                url, 
                headers=config.api.headers, 
                timeout=config.api.timeout
            )
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"SUCCESS: Data retrieved")
                print(f"Response fields: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                
                if isinstance(data, dict) and 'results' in data:
                    results = data['results']
                    print(f"Results count: {len(results)}")
                    if results:
                        print(f"First result fields: {list(results[0].keys())}")
            else:
                print(f"FAILED: {response.status_code}")
                print(f"Response content: {response.text[:200]}...")
                
        except requests.exceptions.Timeout:
            print("ERROR: Request timeout")
        except requests.exceptions.ConnectionError:
            print("ERROR: Connection error")
        except Exception as e:
            print(f"ERROR: Other error: {e}")

if __name__ == "__main__":
    test_api_connection()