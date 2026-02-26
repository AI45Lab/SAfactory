from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheConfig:
    """缓存相关配置。"""
    cache_dir: str = ".cache"
    ttl_seconds: int = 3600
    enabled: bool = True


@dataclass
class ReportConfig:
    """报告生成配置。"""
    title: str
    description: str = ""
    output_format: str = "text"  # text, json, html
    include_timestamp: bool = True
    metadata: dict = field(default_factory=dict)
    cache: CacheConfig = field(default_factory=CacheConfig)


@dataclass
class AnalysisResult:
    """分析结果容器。"""
    name: str
    data: Any
    summary: str = ""
    metrics: dict = field(default_factory=dict)