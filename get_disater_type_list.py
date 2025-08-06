import requests
from requests.exceptions import ConnectionError, Timeout, RequestException
import time
import json


def get_disaster_types():
    """
    获取IFRC灾害类型数据
    正确的API端点：https://goadmin.ifrc.org/api/v2/disaster_type/
    """
    # 正确的URL
    url = "https://goadmin.ifrc.org/api/v2/disaster_type/"

    headers = {
        'User-Agent': 'Mozilla/03_models.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en'
    }

    try:
        print("正在连接IFRC GO API...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        print("✓ 连接成功！")
        data = response.json()

        return data

    except ConnectionError as e:
        print(f"✗ 连接错误: {e}")
        return None
    except Timeout as e:
        print(f"✗ 超时错误: {e}")
        return None
    except RequestException as e:
        print(f"✗ 请求错误: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"✗ JSON解析错误: {e}")
        return None


def display_disaster_types(data):
    """显示灾害类型数据"""
    if not data:
        print("没有获取到数据")
        return

    print(f"\n=== IFRC 灾害类型数据 ===")
    print(f"总数: {data.get('count', 'N/A')}")
    print(f"下一页: {data.get('next', 'N/A')}")
    print(f"上一页: {data.get('previous', 'N/A')}")

    results = data.get('results', [])
    if results:
        print(f"\n灾害类型列表 ({len(results)} 项):")
        print("-" * 50)
        for item in results:
            print(f"ID: {item.get('id', 'N/A')}")
            print(f"名称: {item.get('name', 'N/A')}")
            print(f"摘要: {item.get('summary', 'N/A') or '无'}")
            print(f"语言: {item.get('translation_module_original_language', 'N/A')}")
            print("-" * 30)
    else:
        print("没有找到灾害类型数据")


def save_to_file(data, filename="disaster_types.json"):
    """保存数据到JSON文件"""
    if data:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✓ 数据已保存到 {filename}")
        except Exception as e:
            print(f"✗ 保存文件失败: {e}")


def main():
    """主函数"""
    print("IFRC 灾害类型数据获取工具")
    print("=" * 40)

    # 获取数据
    data = get_disaster_types()

    if data:
        # 显示数据
        display_disaster_types(data)

        # 保存到文件
        save_to_file(data)

        # 提供一些额外的信息
        print(f"\n=== 额外信息 ===")
        print("✓ 该API是公开的，不需要认证")
        print("✓ 数据来源：IFRC GO平台")
        print("✓ 文档：https://go-wiki.ifrc.org/")
        print("✓ 更多API端点：https://goadmin.ifrc.org/api-docs/swagger-ui/")

    else:
        print("✗ 获取数据失败")
        print("\n建议检查：")
        print("02_processed. 网络连接是否正常")
        print("01_raw. 防火墙是否阻止了访问")
        print("05_cache. 是否需要使用代理")


if __name__ == "__main__":
    main()