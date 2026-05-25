from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any


@dataclass
class ObjectiveSpec:
    channel: str
    direction: str


@dataclass
class OptimizeAlgorithmSpec:
    kind: str


@dataclass
class OptimizeAcquisitionSpec:
    num_repeats_per_point: int = 1
    averaging_method: str = "mean"
    max_evals: int = 100


@dataclass
class OptimizeAxis:
    param_schema: dict[str, Any]
    path: str
    param_store: Any
    lower: float
    upper: float
    initial: float


@dataclass
class OptimizeSpec:
    axes: list[OptimizeAxis]
    objective: ObjectiveSpec
    algorithm: OptimizeAlgorithmSpec
    acquisition: OptimizeAcquisitionSpec


@dataclass
class AlgorithmParameter:
    """Metadata for an algorithm parameter."""

    name: str
    label: str
    minimum: float
    maximum: float
    default: float
    step: float | None = None
    tooltip: str = ""


AlgorithmRegistryEntry = dict[str, Any]
ALGORITHM_REGISTRY: dict[str, AlgorithmRegistryEntry] = {}


def register_algorithm(
    kind: str,
    *,
    display_name: str,
    description: str,
    parameters: list[AlgorithmParameter],
    spec_cls: type[OptimizeAlgorithmSpec],
    optimizer_cls: type[Optimizer],
) -> None:
    ALGORITHM_REGISTRY[kind] = {
        "display_name": display_name,
        "description": description,
        "parameters": parameters,
        "spec_cls": spec_cls,
        "optimizer_cls": optimizer_cls,
    }


def build_algorithm_spec(algorithm: dict[str, Any]) -> OptimizeAlgorithmSpec:
    algorithm_kind = algorithm.get("kind", "nelder_mead")
    algo_info = ALGORITHM_REGISTRY.get(algorithm_kind)
    if algo_info is None:
        raise ValueError(f"Unsupported optimisation algorithm '{algorithm_kind}'")

    spec_cls = algo_info["spec_cls"]
    params_by_name = {param.name: param for param in algo_info["parameters"]}
    kwargs: dict[str, Any] = {"kind": algorithm_kind}

    for field in fields(spec_cls):
        if field.name == "kind":
            continue
        param = params_by_name.get(field.name)
        if param is None:
            raise ValueError(
                f"Algorithm spec '{spec_cls.__name__}' has no registry entry for field '{field.name}'"
            )

        raw_value = algorithm.get(field.name, param.default)
        if isinstance(param.default, int) and not isinstance(param.default, bool):
            value = int(raw_value)
        else:
            value = float(raw_value)

        if value < param.minimum or value > param.maximum:
            raise ValueError(
                f"Optimisation {field.name} must be between {param.minimum} and {param.maximum}"
            )
        kwargs[field.name] = value

    return spec_cls(**kwargs)


class Optimizer(ABC):
    """Common ask/tell interface for sequential optimisers.

    Implementations propose one point at a time via :meth:`ask`, then consume the
    corresponding objective value via :meth:`tell`.
    """

    @abstractmethod
    def ask(self) -> tuple[float, ...] | None:
        """Return the next point to evaluate, or ``None`` if no more are needed."""

    @abstractmethod
    def tell(
        self, point: tuple[float, ...], value: float, std_dev: float = 0.0
    ) -> None:
        """Report the measured objective value for a point returned by :meth:`ask`."""

    @abstractmethod
    def is_done(self) -> bool:
        """Return whether the optimiser has terminated."""

    @abstractmethod
    def best(self) -> tuple[tuple[float, ...], float] | None:
        """Return the best point/value pair seen so far, if any evaluations completed."""

    @abstractmethod
    def best_std(self) -> float | None:
        """Return the standard deviation for the current best point, if any."""

    @abstractmethod
    def termination_reason(self) -> str | None:
        """Return the termination reason, or ``None`` while the optimiser is active."""
