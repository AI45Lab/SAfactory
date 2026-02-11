import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from evaluator.config import CacheConfig

log = logging.getLogger("evaluator.cache")


class CacheManager:
    """统一管理 Parquet 缓存的读写和过期策略。"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache_dir = Path(config.cache_dir)
        if config.enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_key(self, key: str) -> str:
        return re.sub(r'[^\w\-]', '_', key)

    def _get_path(self, key: str) -> Path:
        return self._cache_dir / f"{self._sanitize_key(key)}.parquet"

    def _is_expired(self, path: Path) -> bool:
        if not path.exists():
            return True
        return (time.time() - path.stat().st_mtime) > self.config.ttl_seconds

    def load(self, key: str) -> Optional[pd.DataFrame]:
        """尝试从缓存加载，命中且未过期返回 DataFrame，否则返回 None。"""
        if not self.config.enabled:
            return None

        path = self._get_path(key)

        if not path.exists():
            log.debug(f"Cache miss (not found): {path}")
            return None

        if self._is_expired(path):
            log.info(f"Cache expired, removing: {path}")
            path.unlink(missing_ok=True)
            return None

        try:
            df = pd.read_parquet(path)
            log.info(f"Cache hit [{key}]: loaded {len(df)} records")
            return df
        except Exception as e:
            log.warning(f"Cache read failed [{key}], will re-fetch: {e}")
            path.unlink(missing_ok=True)
            return None

    def save(self, key: str, df: pd.DataFrame) -> None:
        """将 DataFrame 写入缓存。"""
        if not self.config.enabled or df.empty:
            return
        path = self._get_path(key)
        try:
            df.to_parquet(path, index=False)
            log.info(f"Cache saved [{key}]: {len(df)} records -> {path}")
        except Exception as e:
            log.error(f"Cache write failed [{key}]: {e}")

    def invalidate(self, key: str = None) -> None:
        """清除指定 key 或全部缓存。"""
        if key:
            self._get_path(key).unlink(missing_ok=True)
            log.info(f"Cache invalidated: {key}")
        else:
            for f in self._cache_dir.glob("*.parquet"):
                f.unlink()
            log.info("All cache invalidated.")