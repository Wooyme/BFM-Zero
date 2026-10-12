"""G1 motion queries, discovered automatically from the modules in this package.

Add a module containing a concrete QueryFunction subclass (a dataclass); select it from YAML by
class name, with the remaining keys as its fields:

    type: LocomotionQuery
    min_speed: 0.5
    max_speed: 1.0

A class that needs to translate the YAML keys can define a from_config(**cfg) classmethod.
No registration or changes to the query script are required.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any, Mapping

import yaml

from .base import QueryFunction, in_bounds, parse_bounds
from .locomotion import LocomotionQuery
from .metrics import METRICS
from .range import RangeQuery
from .walk import WalkQuery

__all__ = [
    "METRICS",
    "QueryFunction",
    "LocomotionQuery",
    "RangeQuery",
    "WalkQuery",
    "in_bounds",
    "parse_bounds",
    "get_query_classes",
    "make_query",
    "load_query",
]


def get_query_classes() -> dict[str, type[QueryFunction]]:
    """Load all query modules and return their concrete classes by class name."""
    query_classes = {}
    modules = sorted(pkgutil.walk_packages(__path__, prefix=__name__ + "."), key=lambda module: module.name)
    for module_info in modules:
        module = importlib.import_module(module_info.name)
        for _, query_cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(query_cls, QueryFunction) and not inspect.isabstract(query_cls):
                other = query_classes.setdefault(query_cls.__name__, query_cls)
                if other is not query_cls:
                    raise ValueError(f"Two query classes named {query_cls.__name__}: {other} and {query_cls}")
    return query_classes


def make_query(cfg: Mapping[str, Any]) -> QueryFunction:
    """Build a query from a config dict: 'type' is the class name, other keys are its arguments."""
    cfg = dict(cfg)
    if "type" not in cfg:
        raise ValueError(f"Query config needs a 'type' key, got {cfg}")
    name = cfg.pop("type")
    classes = get_query_classes()
    if name not in classes:
        raise ValueError(f"Unknown query type '{name}', available: {list(classes)}")
    query_cls = classes[name]
    from_config = getattr(query_cls, "from_config", None)
    return from_config(**cfg) if from_config is not None else query_cls(**cfg)


def load_query(path: str | Path) -> QueryFunction:
    with open(path) as f:
        return make_query(yaml.safe_load(f))
