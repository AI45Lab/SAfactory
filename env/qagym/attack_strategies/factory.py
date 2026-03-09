from typing import TYPE_CHECKING, Dict, Type

from .base import AttackStrategy

if TYPE_CHECKING:
    from env.qagym.qa_env import QAGym

ATTACK_STRATEGY_REGISTRY: Dict[str, Type[AttackStrategy]] = {}


def register_attack_strategy(name: str):
    def _decorator(cls: Type[AttackStrategy]) -> Type[AttackStrategy]:
        ATTACK_STRATEGY_REGISTRY[name] = cls
        return cls

    return _decorator


def build_attack_strategy(env: "QAGym", attack_method: str) -> AttackStrategy:
    strategy_cls = ATTACK_STRATEGY_REGISTRY.get(attack_method)
    if not strategy_cls:
        raise ValueError(f"Unsupported attack_method: {attack_method}")
    return strategy_cls(env)
