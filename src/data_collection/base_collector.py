"""
Base collector classes with configuration-driven API calls and data saving
"""

import requests
import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from src.config import get_config, ProjectConfig


class BaseCollector(ABC):
    """
    Base collector abstract class
    Provides unified API calls, error handling, data saving and other functions
    """
    def __init__(self, config: ProjectConfig = None):
        self.config = config or get_config()
        self.session = self._setup_session()
        self.logger = self._setup_logger()
        self.api_endpoints = self.config.get_api_endpoints()

        # Create data save directory
        self.data_dir = self.get_data_directory()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _setup_session(self) -> requests.Session:
        """Setup HTTP session"""
        session = requests.Session()
        session.headers.update(self.config.api.headers)
        return session

    def _setup_logger(self) -> logging.Logger:
        """Setup logging handler"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(getattr(logging, self.config.logging.level))

        if not logger.handlers:
            # File handler
            if self.config.logging.file_path:
                file_handler = logging.FileHandler(
                    self.config.logging.file_path,
                    encoding='utf-8'
                )
                file_handler.setFormatter(
                    logging.Formatter(self.config.logging.format)
                )
                logger.addHandler(file_handler)

            # Console handler
            if self.config.logging.console_output:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(
                    logging.Formatter(self.config.logging.format)
                )
                logger.addHandler(console_handler)

        return logger

    @abstractmethod
    def get_data_directory(self) -> Path:
        """Get data save directory """
        pass

    def make_api_request(self, endpoint: str, params: Dict = None,
                         use_cache: bool = True, cache_key: str = None) -> Optional[Dict]:
        """Unified API"""
        # Check the complete URL
        if endpoint.startswith('http'):
            url = endpoint
        else:
            url = self.api_endpoints.get(endpoint, endpoint)
            if not url.startswith('http'):
                self.logger.error(f"Invalid API endpoint: {endpoint}")
                return None

        if use_cache and cache_key:
            cached_data = self._load_from_cache(cache_key)
            if cached_data:
                self.logger.info(f"Load data from cache: {cache_key}")
                return cached_data

        for attempt in range(self.config.api.retry_attempts):
            try:
                self.logger.debug(f"API request attempts {attempt + 1}): {url}")

                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.api.timeout
                )
                response.raise_for_status()

                data = response.json()

                if use_cache and cache_key:
                    self._save_to_cache(data, cache_key)

                self.logger.info(f"API request successful: {url}")
                return data

            except requests.exceptions.Timeout:
                self.logger.warning(f"API request timeout attempt {attempt + 1}): {url}")
                if attempt < self.config.api.retry_attempts - 1:
                    time.sleep(self.config.api.retry_delay)

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"API request failed attempt {attempt + 1}): {url}, error: {e}")
                if attempt < self.config.api.retry_attempts - 1:
                    time.sleep(self.config.api.retry_delay)

        self.logger.error(f"API request ultimately failed:{url}")
        return None

    def make_paginated_request(self, endpoint: str, params: Dict = None,
                               max_records: int = None) -> List[Dict]:
        if params is None:
            params = {}

        max_records = max_records or self.config.collection.max_records_per_collection
        batch_size = self.config.collection.batch_size

        all_results = []
        offset = 0

        self.logger.info(f"Start pagination request: {endpoint}, Maximum number of records: {max_records}")

        while len(all_results) < max_records:
            current_params = params.copy()
            current_params.update({
                'limit': min(batch_size, max_records - len(all_results)),
                'offset': offset,
                'format': 'json'
            })

            response = self.make_api_request(endpoint, current_params, use_cache=False)
            if not response:
                break

            results = response.get('results', [])
            if not results:
                self.logger.info("No more data")
                break

            all_results.extend(results)
            offset += batch_size

            self.logger.info(f"{len(all_results)} records retrieved...")

            time.sleep(self.config.api.rate_limit_delay)

        self.logger.info(f"Pagination request completed, a total of {len(all_results)} records retrieved.")
        return all_results

    def save_data(self, data: Any, filename: str, include_metadata: bool = True) -> Path:
        """save data """
        if include_metadata and isinstance(data, dict) and 'metadata' not in data:
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

            self.logger.info(f"Data saved: {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Failure to save: {e}")
            raise

    def load_data(self, filename: str) -> Optional[Any]:

        filepath = self.data_dir / f"{filename}.json"

        if not filepath.exists():
            self.logger.warning(f"File does not exist: {filepath}")
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.logger.info(f"Data loaded: {filepath}")
            return data

        except Exception as e:
            self.logger.error(f"Failure to load data: {e}")
            return None

    def _parse_iso_datetime(self, date_string: str) -> datetime:
        try:

            date_string = date_string.replace('Z', '').replace('+00:00', '')
            if 'T' in date_string:

                if '.' in date_string:
                    date_string = date_string.split('.')[0]
                return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S')
            else:
                return datetime.strptime(date_string, '%Y-%m-%d')
        except Exception as e:
            self.logger.warning(f"Failed to parse time string '{date_string}': {e}")
            return datetime(1970, 1, 1)

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache with classification support"""
        cache_file = self._get_cache_file_path(cache_key)
        
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            try:
                cache_time_str = cached_data.get('cache_time', '1970-01-01T00:00:00')
                cache_time = self._parse_iso_datetime(cache_time_str)
                
                if (datetime.now() - cache_time).days > 1:
                    return None
            except Exception as e:
                self.logger.warning(f"Parse cache timeout failed: {e}, ignoring cache")
                return None

            return cached_data.get('data')

        except Exception as e:
            self.logger.warning(f"Failure to save to cache: {e}")
            return None

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get cache file path and ensure directory exists"""
        if '/' in cache_key:
            category, key_name = cache_key.split('/', 1)
            cache_file = self.config.paths.collection_cache_dir / category / f"{key_name}.json"
        else:
            cache_file = self.config.paths.collection_cache_dir / f"{cache_key}.json"
        
        # Ensure directory exists
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        return cache_file

    def _save_to_cache(self, data: Any, cache_key: str) -> None:
        """Save data to cache with classification support"""
        cache_file = self._get_cache_file_path(cache_key)

        try:
            cache_data = {
                'cache_time': datetime.now().isoformat(),
                'cache_key': cache_key,
                'data': data
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            self.logger.warning(f"Failure to save data to cache: {e}")


class ConfigurableCollector(BaseCollector):
    @abstractmethod
    def collect_data(self) -> Any:
        pass

    def run_collection(self) -> Dict[str, Any]:

        self.logger.info("Start running the data collector...")
        try:
            collected_data = self.collect_data()
            self.logger.info("Data collection is successful.")

            return collected_data
        except Exception as e:
            self.logger.error(f"An error occurred during the collection process: {e}", exc_info=True)
            return {
                "data": [],
                "collection_status": "failed",
                "error": str(e)
            }
