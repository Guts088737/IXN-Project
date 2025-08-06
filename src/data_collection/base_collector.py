"""
02_data_collection/base_collector.py

基础收集器类，使用配置驱动的方式统一API调用和数据保存
"""

import requests
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from src.config import get_config, ProjectConfig


class BaseCollector(ABC):
    """
    基础收集器抽象类
    提供统一的API调用、错误处理、数据保存等功能
    """

    def __init__(self, config: ProjectConfig = None):
        self.config = config or get_config()
        self.session = self._setup_session()
        self.logger = self._setup_logger()
        self.api_endpoints = self.config.get_api_endpoints()

        # 创建数据保存目录
        self.data_dir = self.get_data_directory()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _setup_session(self) -> requests.Session:
        """设置HTTP会话"""
        session = requests.Session()
        session.headers.update(self.config.api.headers)
        return session

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(getattr(logging, self.config.logging.level))

        if not logger.handlers:
            # 文件处理器
            if self.config.logging.file_path:
                file_handler = logging.FileHandler(
                    self.config.logging.file_path,
                    encoding='utf-8'
                )
                file_handler.setFormatter(
                    logging.Formatter(self.config.logging.format)
                )
                logger.addHandler(file_handler)

            # 控制台处理器
            if self.config.logging.console_output:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(
                    logging.Formatter(self.config.logging.format)
                )
                logger.addHandler(console_handler)

        return logger

    @abstractmethod
    def get_data_directory(self) -> Path:
        """获取数据保存目录 - 子类必须实现"""
        pass

    def make_api_request(self, endpoint: str, params: Dict = None,
                         use_cache: bool = True, cache_key: str = None) -> Optional[Dict]:
        """
        统一的API请求方法

        Args:
            endpoint: API端点名称或完整URL
            params: 请求参数
            use_cache: 是否使用缓存
            cache_key: 缓存键名
        """
        # 确定完整的URL
        if endpoint.startswith('http'):
            url = endpoint
        else:
            url = self.api_endpoints.get(endpoint, endpoint)
            if not url.startswith('http'):
                self.logger.error(f"无效的API端点: {endpoint}")
                return None

        # 检查缓存
        if use_cache and cache_key:
            cached_data = self._load_from_cache(cache_key)
            if cached_data:
                self.logger.info(f"从缓存加载数据: {cache_key}")
                return cached_data

        # 发起API请求
        for attempt in range(self.config.api.retry_attempts):
            try:
                self.logger.debug(f"API请求 (尝试 {attempt + 1}): {url}")

                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.api.timeout
                )
                response.raise_for_status()

                data = response.json()

                # 保存到缓存
                if use_cache and cache_key:
                    self._save_to_cache(data, cache_key)

                self.logger.info(f"API请求成功: {url}")
                return data

            except requests.exceptions.Timeout:
                self.logger.warning(f"API请求超时 (尝试 {attempt + 1}): {url}")
                if attempt < self.config.api.retry_attempts - 1:
                    time.sleep(self.config.api.retry_delay)

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"API请求失败 (尝试 {attempt + 1}): {url}, 错误: {e}")
                if attempt < self.config.api.retry_attempts - 1:
                    time.sleep(self.config.api.retry_delay)

        self.logger.error(f"API请求最终失败: {url}")
        return None

    def make_paginated_request(self, endpoint: str, params: Dict = None,
                               max_records: int = None) -> List[Dict]:
        """
        分页API请求

        Args:
            endpoint: API端点
            params: 基础参数
            max_records: 最大记录数
        """
        if params is None:
            params = {}

        max_records = max_records or self.config.collection.max_records_per_collection
        batch_size = self.config.collection.batch_size

        all_results = []
        offset = 0

        self.logger.info(f"开始分页请求: {endpoint}, 最大记录数: {max_records}")

        while len(all_results) < max_records:
            # 设置分页参数
            current_params = params.copy()
            current_params.update({
                'limit': min(batch_size, max_records - len(all_results)),
                'offset': offset,
                'format': 'json'
            })

            # 发起请求
            response = self.make_api_request(endpoint, current_params, use_cache=False)
            if not response:
                break

            results = response.get('results', [])
            if not results:
                self.logger.info("没有更多数据")
                break

            all_results.extend(results)
            offset += batch_size

            self.logger.info(f"已获取 {len(all_results)} 条记录...")

            # API调用间隔
            time.sleep(self.config.api.rate_limit_delay)

        self.logger.info(f"分页请求完成，共获取 {len(all_results)} 条记录")
        return all_results

    def save_data(self, data: Any, filename: str, include_metadata: bool = True) -> Path:
        """
        保存数据到文件

        Args:
            data: 要保存的数据
            filename: 文件名（不含扩展名）
            include_metadata: 是否包含元数据
        """
        if include_metadata and isinstance(data, dict) and 'metadata' not in data:
            # 添加元数据
            data = {
                'metadata': {
                    'collection_date': datetime.now().isoformat(),
                    'collector': self.__class__.__name__,
                    'config_version': '1.0'
                },
                'data': data
            }

        filepath = self.data_dir / f"{filename}.json"

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"数据已保存: {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
            raise

    def load_data(self, filename: str) -> Optional[Any]:
        """
        从文件加载数据
        """
        filepath = self.data_dir / f"{filename}.json"

        if not filepath.exists():
            self.logger.warning(f"文件不存在: {filepath}")
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.logger.info(f"数据已加载: {filepath}")
            return data

        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return None

    def _parse_iso_datetime(self, date_string: str) -> datetime:
        """
        解析ISO格式日期时间字符串，兼容Python 3.6
        """
        try:
            # 去掉时区信息简化处理
            date_string = date_string.replace('Z', '').replace('+00:00', '')
            if 'T' in date_string:
                return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S')
            else:
                return datetime.strptime(date_string, '%Y-%m-%d')
        except:
            # 如果解析失败，返回很久之前的时间，强制重新获取
            return datetime(1970, 1, 1)

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """从缓存加载数据"""
        cache_file = self.config.paths.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            # 检查缓存是否过期（可配置）
            try:
                cache_time_str = cached_data.get('cache_time', '1970-01-01T00:00:00')
                cache_time = self._parse_iso_datetime(cache_time_str)
                
                if (datetime.now() - cache_time).days > 1:  # 1天后过期
                    return None
            except Exception as e:
                self.logger.warning(f"解析缓存时间失败: {e}，忽略缓存")
                return None

            return cached_data.get('data')

        except Exception as e:
            self.logger.warning(f"加载缓存失败: {e}")
            return None

    def _save_to_cache(self, data: Any, cache_key: str) -> None:
        """保存数据到缓存"""
        cache_file = self.config.paths.cache_dir / f"{cache_key}.json"

        try:
            cache_data = {
                'cache_time': datetime.now().isoformat(),
                'cache_key': cache_key,
                'data': data
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            self.logger.warning(f"保存缓存失败: {e}")

    def filter_by_countries(self, data: List[Dict], country_field: str = 'country') -> List[Dict]:
        """
        按配置的关注国家过滤数据
        """
        countries_of_interest = self.config.collection.countries_of_interest
        if not countries_of_interest:
            return data

        filtered = [
            item for item in data
            if item.get(country_field) in countries_of_interest
        ]

        self.logger.info(f"按国家过滤: {len(data)} → {len(filtered)} 条记录")
        return filtered

    def safe_int_convert(self, value: Any, default: int = 0) -> int:
        """
        安全的整数转换（处理API异常值）
        """
        if value is None:
            return default

        try:
            # 处理API中的负数异常值
            if isinstance(value, (int, float)) and value < 0:
                return 0
            return int(float(value))
        except (ValueError, TypeError):
            return default

    def calculate_data_quality_score(self, item: Dict, required_fields: List[str]) -> float:
        """
        计算数据质量评分
        """
        if not required_fields:
            return 1.0

        available_fields = sum(1 for field in required_fields if item.get(field) is not None)
        return available_fields / len(required_fields)

    def generate_collection_summary(self, collected_data: List[Dict],
                                    data_type: str) -> Dict[str, Any]:
        """
        生成数据收集总结
        """
        if not collected_data:
            return {
                'collection_summary': {
                    'data_type': data_type,
                    'total_records': 0,
                    'collection_status': 'failed',
                    'timestamp': datetime.now().isoformat()
                }
            }

        # 基础统计
        countries = set()
        for item in collected_data:
            if 'country' in item:
                countries.add(item['country'])

        summary = {
            'collection_summary': {
                'data_type': data_type,
                'total_records': len(collected_data),
                'countries_covered': list(countries),
                'countries_count': len(countries),
                'collection_status': 'success',
                'timestamp': datetime.now().isoformat(),
                'collector_class': self.__class__.__name__,
                'config_version': '1.0'
            }
        }

class ConfigurableCollector(BaseCollector):
    """
    可配置收集器：提供默认实现模板，供子类继承并覆盖核心数据逻辑
    """

    @abstractmethod
    def collect_data(self) -> Any:
        """子类必须实现：定义如何收集数据"""
        pass

    def run_collection(self) -> Dict[str, Any]:
        """
        执行数据收集完整流程
        """
        self.logger.info("开始运行数据收集器...")
        try:
            collected_data = self.collect_data()
            self.logger.info("数据收集成功")

            # 可以选择保存数据
            self.save_data(collected_data, "collected_data")
            return collected_data
        except Exception as e:
            self.logger.error(f"收集过程中发生错误: {e}", exc_info=True)
            return {
                "data": [],
                "collection_status": "failed",
                "error": str(e)
            }

    def validate_collected_data(self, data: List[Dict]) -> Dict[str, Any]:
        """
        验证收集到的数据质量
        """
        self.logger.info("开始验证数据质量...")

        if not data:
            return {
                "is_valid": False,
                "quality_score": 0.0,
                "issues": ["没有数据"]
            }

        required_fields = self._get_required_fields() if hasattr(self, '_get_required_fields') else []
        quality_scores = [
            self.calculate_data_quality_score(item, required_fields)
            for item in data
        ]
        average_score = sum(quality_scores) / len(quality_scores)

        issues = []
        for i, score in enumerate(quality_scores):
            if score < 0.3:
                issues.append(f"记录 {i} 的质量评分过低: {score:.2f}")

        return {
            "is_valid": average_score >= 0.3,
            "quality_score": average_score,
            "issues": issues
        }

    def enrich_data_with_metadata(self, data: List[Dict]) -> List[Dict]:
        """
        为每条数据添加收集元信息
        """
        for item in data:
            item['metadata'] = {
                "collected_at": datetime.now().isoformat(),
                "collector": self.__class__.__name__,
                "config_version": "1.0"
            }
        return data