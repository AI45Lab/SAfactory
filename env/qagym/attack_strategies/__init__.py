import importlib
import pkgutil

from .base import AttackStrategy
from .factory import build_attack_strategy, register_attack_strategy


# find and load all default attack strategies automatically
def load_default_strategies() -> None:
    implementations_pkg = importlib.import_module(f"{__name__}.implementations")
    for module_info in pkgutil.iter_modules(implementations_pkg.__path__):
        module_name = module_info.name
        if module_name.startswith("_"):
            continue
        importlib.import_module(f"{implementations_pkg.__name__}.{module_name}")


load_default_strategies()

__all__ = [
    "AttackStrategy",
    "build_attack_strategy",
    "register_attack_strategy",
]
