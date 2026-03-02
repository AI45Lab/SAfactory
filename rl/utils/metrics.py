import statistics
from enum import Enum
from typing import Dict, List

import wandb


class AggType(Enum):
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    STD = "std"


class MetricsRecorder:
    """Record metrics during generate_rollout and push to wandb."""

    def __init__(self):
        self._data: Dict[str, List[float]] = {}
        self._agg_types: Dict[str, AggType] = {}
        self._defined_metrics: set = set()

    def record(self, tag: str, value: float, agg: AggType = AggType.MEAN):
        """Record a value for a tag with specified aggregation type."""
        if tag not in self._data:
            self._data[tag] = []
            self._agg_types[tag] = agg
        self._data[tag].append(value)

    def aggregate(self) -> Dict[str, float]:
        """Aggregate all recorded values."""
        result = {}
        for tag, values in self._data.items():
            if not values:
                continue
            agg = self._agg_types[tag]
            if agg == AggType.SUM:
                result[tag] = sum(values)
            elif agg == AggType.MEAN:
                result[tag] = sum(values) / len(values)
            elif agg == AggType.MAX:
                result[tag] = max(values)
            elif agg == AggType.STD:
                result[tag] = statistics.stdev(values) if len(values) > 1 else 0.0
        return result

    def push(self, step: int):
        """Push aggregated metrics to wandb and clear."""
        metrics = self.aggregate()
        if wandb.run is not None:
            for key in metrics:
                if key not in self._defined_metrics:
                    wandb.define_metric(key, step_metric="rollout/step")
                    self._defined_metrics.add(key)
            if metrics:
                metrics["rollout/step"] = step
                wandb.log(metrics)
        self.clear()

    def clear(self):
        """Clear all recorded data."""
        self._data.clear()
        self._agg_types.clear()
