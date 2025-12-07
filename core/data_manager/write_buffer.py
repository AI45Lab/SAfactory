"""
通用数据库写入缓冲器

解决并发场景下频繁小批量写入导致的效率问题。
支持任意 Tortoise Model 的 create 和 update 操作缓冲。
通过内存缓冲 + 定时/阈值批量写入来优化性能。
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Type, TypeVar, Set, Optional, Any
from datetime import datetime
import logging

from tortoise import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Model)


@dataclass
class UpdateEntry:
    """更新操作条目"""
    instance: Model
    update_fields: Set[str]
    timestamp: datetime = field(default_factory=datetime.now)


class WriteBuffer:
    """
    通用数据库写入缓冲器

    功能：
    1. 支持任意 Tortoise Model 的 create/update 操作缓冲
    2. 按模型类型分组缓冲
    3. 达到阈值或定时触发批量写入
    4. 并发安全：使用锁保护缓冲区操作
    5. 优雅关闭：确保程序退出时数据不丢失
    """

    def __init__(
        self,
        buffer_size: int = 100,
        flush_interval: float = 5.0,
        auto_start: bool = True
    ):
        """
        初始化写入缓冲器

        Args:
            buffer_size: 每个模型的缓冲区大小阈值，达到后触发写入
            flush_interval: 定时刷新间隔（秒）
            auto_start: 是否在首次操作时自动启动后台刷新任务
        """
        # 按模型类型分组的创建缓冲区
        self._create_buffers: Dict[Type[Model], List[Model]] = {}
        # 按模型类型分组的更新缓冲区
        self._update_buffers: Dict[Type[Model], List[UpdateEntry]] = {}

        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._auto_start = auto_start

        # 统计信息
        self._stats = {
            "total_create_buffered": 0,
            "total_create_flushed": 0,
            "total_update_buffered": 0,
            "total_update_flushed": 0,
            "flush_count": 0,
        }

    async def start(self):
        """启动后台刷新任务"""
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info(
            f"WriteBuffer started: buffer_size={self._buffer_size}, "
            f"flush_interval={self._flush_interval}s"
        )

    async def stop(self):
        """停止缓冲器并确保所有数据写入"""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # 最终刷新，确保数据不丢失
        await self.flush()
        logger.info(f"WriteBuffer stopped: stats={self._stats}")

    async def buffer_create(self, instance: T) -> T:
        """
        缓冲创建操作

        Args:
            instance: 待创建的模型实例（尚未保存到数据库）

        Returns:
            传入的实例（供后续引用，注意此时 pk 可能为 None）
        """
        if self._auto_start and not self._running:
            await self.start()

        model_class = type(instance)
        should_flush = False

        async with self._lock:
            if model_class not in self._create_buffers:
                self._create_buffers[model_class] = []
            self._create_buffers[model_class].append(instance)
            self._stats["total_create_buffered"] += 1
            buffer_len = len(self._create_buffers[model_class])
            if buffer_len >= self._buffer_size:
                should_flush = True

        # 达到阈值时触发该模型的写入
        if should_flush:
            asyncio.create_task(self.flush_model(model_class, operation="create"))

        return instance

    async def buffer_update(
        self,
        instance: T,
        update_fields: Optional[Set[str]] = None
    ) -> None:
        """
        缓冲更新操作

        Args:
            instance: 待更新的模型实例（已修改但尚未保存）
            update_fields: 需要更新的字段集合，为 None 时更新所有字段
        """
        if self._auto_start and not self._running:
            await self.start()

        model_class = type(instance)
        should_flush = False

        async with self._lock:
            if model_class not in self._update_buffers:
                self._update_buffers[model_class] = []

            entry = UpdateEntry(
                instance=instance,
                update_fields=update_fields or set()
            )
            self._update_buffers[model_class].append(entry)
            self._stats["total_update_buffered"] += 1
            buffer_len = len(self._update_buffers[model_class])
            if buffer_len >= self._buffer_size:
                should_flush = True

        if should_flush:
            asyncio.create_task(self.flush_model(model_class, operation="update"))

    async def flush(self) -> Dict[str, int]:
        """
        批量写入所有缓冲区数据

        Returns:
            各操作写入的记录数统计
        """
        results = {"created": 0, "updated": 0}

        # 获取所有需要刷新的模型
        async with self._lock:
            create_models = list(self._create_buffers.keys())
            update_models = list(self._update_buffers.keys())

        # 刷新所有创建缓冲
        for model_class in create_models:
            count = await self.flush_model(model_class, operation="create")
            results["created"] += count

        # 刷新所有更新缓冲
        for model_class in update_models:
            count = await self.flush_model(model_class, operation="update")
            results["updated"] += count

        if results["created"] > 0 or results["updated"] > 0:
            self._stats["flush_count"] += 1
            logger.debug(f"WriteBuffer flushed: {results}")

        return results

    async def flush_model(
        self,
        model_class: Type[Model],
        operation: str = "all"
    ) -> int:
        """
        刷新指定模型的缓冲区

        Args:
            model_class: 模型类
            operation: "create", "update", 或 "all"

        Returns:
            写入的记录数
        """
        total = 0

        if operation in ("create", "all"):
            total += await self._flush_creates(model_class)

        if operation in ("update", "all"):
            total += await self._flush_updates(model_class)

        return total

    async def _flush_creates(self, model_class: Type[Model]) -> int:
        """刷新指定模型的创建缓冲区"""
        async with self._lock:
            if model_class not in self._create_buffers:
                return 0
            to_create = self._create_buffers.pop(model_class, [])

        if not to_create:
            return 0

        try:
            await model_class.bulk_create(to_create)
            count = len(to_create)
            self._stats["total_create_flushed"] += count
            logger.debug(f"Bulk created {count} {model_class.__name__} records")
            return count
        except Exception as e:
            logger.error(f"Bulk create failed for {model_class.__name__}: {e}")
            # 降级逐条插入
            return await self._fallback_create(to_create)

    async def _fallback_create(self, instances: List[Model]) -> int:
        """降级逐条创建"""
        success = 0
        for instance in instances:
            try:
                await instance.save()
                success += 1
            except Exception as e:
                logger.error(f"Failed to create {type(instance).__name__}: {e}")
        self._stats["total_create_flushed"] += success
        return success

    async def _flush_updates(self, model_class: Type[Model]) -> int:
        """刷新指定模型的更新缓冲区"""
        async with self._lock:
            if model_class not in self._update_buffers:
                return 0
            to_update = self._update_buffers.pop(model_class, [])

        if not to_update:
            return 0

        # 按 update_fields 分组，相同字段的可以批量更新
        grouped = self._group_updates_by_fields(to_update)
        total = 0

        for fields, entries in grouped.items():
            instances = [e.instance for e in entries]
            try:
                if fields:
                    # 有指定字段时使用 bulk_update
                    await model_class.bulk_update(instances, fields=list(fields))
                else:
                    # 无指定字段时逐个 save
                    for inst in instances:
                        await inst.save()
                total += len(instances)
            except Exception as e:
                logger.error(f"Bulk update failed for {model_class.__name__}: {e}")
                # 降级逐条更新
                for inst in instances:
                    try:
                        await inst.save()
                        total += 1
                    except Exception as inner_e:
                        logger.error(f"Failed to update {model_class.__name__}: {inner_e}")

        self._stats["total_update_flushed"] += total
        logger.debug(f"Bulk updated {total} {model_class.__name__} records")
        return total

    def _group_updates_by_fields(
        self,
        entries: List[UpdateEntry]
    ) -> Dict[frozenset, List[UpdateEntry]]:
        """按更新字段分组"""
        grouped: Dict[frozenset, List[UpdateEntry]] = {}
        for entry in entries:
            key = frozenset(entry.update_fields)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(entry)
        return grouped

    async def _periodic_flush(self):
        """定时刷新任务"""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic flush error: {e}")

    @property
    def pending_count(self) -> Dict[str, int]:
        """当前各缓冲区待写入的记录数"""
        create_count = sum(len(v) for v in self._create_buffers.values())
        update_count = sum(len(v) for v in self._update_buffers.values())
        return {"create": create_count, "update": update_count}

    @property
    def stats(self) -> dict:
        """获取统计信息"""
        return {
            **self._stats,
            "pending": self.pending_count,
            "running": self._running
        }
