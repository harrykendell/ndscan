from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from artiq.language import HasEnvironment

from .result_channels import NumericChannel, ResultSink
from .scan_runner import ScanAxis, select_runner_class

__all__ = [
    "ObjectiveSpec",
    "NelderMeadSpec",
    "OptimizeAxis",
    "OptimizeSpec",
    "Optimizer",
    "NelderMeadOptimizer",
    "OptimizeRunner",
    "describe_optimise",
]


@dataclass
class ObjectiveSpec:
    channel: str
    direction: str


@dataclass
class NelderMeadSpec:
    kind: str = "nelder_mead"
    max_evals: int = 100
    xatol: float = 1e-3
    fatol: float = 1e-3


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
    algorithm: NelderMeadSpec


class Optimizer:
    def ask(self) -> tuple[float, ...] | None:
        raise NotImplementedError

    def tell(self, point: tuple[float, ...], value: float) -> None:
        raise NotImplementedError

    def is_done(self) -> bool:
        raise NotImplementedError

    def best(self) -> tuple[tuple[float, ...], float] | None:
        raise NotImplementedError

    def termination_reason(self) -> str | None:
        raise NotImplementedError


class NelderMeadOptimizer(Optimizer):
    """Sequential ask/tell Nelder-Mead implementation."""

    def __init__(
        self,
        initial: tuple[float, ...],
        lower_bounds: tuple[float, ...],
        upper_bounds: tuple[float, ...],
        max_evals: int,
        xatol: float,
        fatol: float,
    ):
        self._initial = np.array(initial, dtype=float)
        self._lower = np.array(lower_bounds, dtype=float)
        self._upper = np.array(upper_bounds, dtype=float)
        self._max_evals = max_evals
        self._xatol = xatol
        self._fatol = fatol

        self._alpha = 1.0
        self._gamma = 2.0
        self._rho = 0.5
        self._sigma = 0.5

        self._num_asked = 0
        self._queue = deque[np.ndarray]()
        self._termination_reason: str | None = None

        self._stage = "initial"
        self._simplex = self._build_initial_simplex()
        self._values: list[float | None] = [None] * len(self._simplex)
        self._pending_candidate: tuple[np.ndarray, float] | None = None
        self._shrink_order: list[int] = []
        self._shrink_completed = 0

        for point in self._simplex:
            self._enqueue(point)

    def ask(self) -> tuple[float, ...] | None:
        if self._termination_reason is not None:
            return None

        if self._queue:
            point = self._queue.popleft()
            self._num_asked += 1
            return tuple(point.tolist())

        if self._num_asked >= self._max_evals:
            self._termination_reason = "max_evals_reached"
        return None

    def tell(self, point: tuple[float, ...], value: float) -> None:
        del point

        if self._termination_reason is not None:
            return

        if self._stage == "initial":
            idx = self._first_unset_index()
            self._values[idx] = value
            if all(v is not None for v in self._values):
                self._sort_simplex()
                self._maybe_terminate()
                if self._termination_reason is None:
                    self._start_iteration()
            return

        if self._stage == "reflect":
            self._handle_reflect(value)
            return

        if self._stage == "expand":
            self._handle_expand(value)
            return

        if self._stage in {"outside_contract", "inside_contract"}:
            self._handle_contract(value)
            return

        if self._stage == "shrink":
            self._handle_shrink(value)
            return

        raise RuntimeError(f"Unexpected optimizer stage '{self._stage}'")

    def is_done(self) -> bool:
        return self._termination_reason is not None

    def best(self) -> tuple[tuple[float, ...], float] | None:
        if any(v is None for v in self._values):
            return None
        best_idx = int(np.argmin(self._values))
        return tuple(self._simplex[best_idx].tolist()), float(self._values[best_idx])

    def termination_reason(self) -> str | None:
        return self._termination_reason

    def _build_initial_simplex(self) -> list[np.ndarray]:
        if len(self._initial) == 0:
            raise ValueError("Need at least one optimization axis")

        simplex = [self._clip(self._initial.copy())]
        for i in range(len(self._initial)):
            point = self._initial.copy()
            delta = 0.05 * (self._upper[i] - self._lower[i])
            if delta <= 0:
                raise ValueError("Optimization bounds must satisfy min < max")
            if point[i] + delta <= self._upper[i]:
                point[i] += delta
            elif point[i] - delta >= self._lower[i]:
                point[i] -= delta
            elif point[i] != self._lower[i]:
                point[i] = self._lower[i]
            else:
                point[i] = self._upper[i]
            simplex.append(self._clip(point))
        return simplex

    def _clip(self, point: np.ndarray) -> np.ndarray:
        return np.clip(point, self._lower, self._upper)

    def _enqueue(self, point: np.ndarray) -> bool:
        if self._num_asked + len(self._queue) >= self._max_evals:
            return False
        self._queue.append(self._clip(point))
        return True

    def _remaining_budget(self) -> int:
        return self._max_evals - self._num_asked - len(self._queue)

    def _first_unset_index(self) -> int:
        for i, value in enumerate(self._values):
            if value is None:
                return i
        raise RuntimeError("No unset simplex values left")

    def _sort_simplex(self) -> None:
        pairs = sorted(zip(self._simplex, self._values), key=lambda p: p[1])
        self._simplex = [point for point, _ in pairs]
        self._values = [float(value) for _, value in pairs]

    def _maybe_terminate(self) -> None:
        if self._num_asked >= self._max_evals and not self._queue:
            self._termination_reason = "max_evals_reached"
            return

        best = self._simplex[0]
        best_value = self._values[0]
        max_x_delta = max(np.max(np.abs(point - best)) for point in self._simplex[1:])
        max_f_delta = max(abs(value - best_value) for value in self._values[1:])
        if max_x_delta <= self._xatol and max_f_delta <= self._fatol:
            self._termination_reason = "converged"

    def _start_iteration(self) -> None:
        if self._termination_reason is not None:
            return
        if self._num_asked >= self._max_evals:
            self._termination_reason = "max_evals_reached"
            return

        centroid = np.mean(self._simplex[:-1], axis=0)
        worst = self._simplex[-1]
        reflected = centroid + self._alpha * (centroid - worst)
        self._pending_candidate = (self._clip(reflected), 0.0)
        self._stage = "reflect"
        if not self._enqueue(self._pending_candidate[0]):
            self._termination_reason = "max_evals_reached"

    def _replace_worst(self, point: np.ndarray, value: float) -> None:
        self._simplex[-1] = self._clip(point)
        self._values[-1] = float(value)
        self._sort_simplex()
        self._maybe_terminate()
        if self._termination_reason is None:
            self._start_iteration()

    def _handle_reflect(self, reflected_value: float) -> None:
        reflected_point = self._pending_candidate[0]
        best_value = self._values[0]
        second_worst_value = self._values[-2]
        worst_value = self._values[-1]

        if reflected_value < best_value:
            centroid = np.mean(self._simplex[:-1], axis=0)
            expanded = centroid + self._gamma * (reflected_point - centroid)
            self._pending_candidate = (self._clip(expanded), reflected_value)
            self._stage = "expand"
            if not self._enqueue(self._pending_candidate[0]):
                self._stage = "idle"
                self._replace_worst(reflected_point, reflected_value)
            return

        if reflected_value < second_worst_value:
            self._stage = "idle"
            self._replace_worst(reflected_point, reflected_value)
            return

        centroid = np.mean(self._simplex[:-1], axis=0)
        if reflected_value < worst_value:
            contracted = centroid + self._rho * (reflected_point - centroid)
            self._pending_candidate = (self._clip(contracted), reflected_value)
            self._stage = "outside_contract"
            if not self._enqueue(self._pending_candidate[0]):
                self._stage = "idle"
                self._replace_worst(reflected_point, reflected_value)
            return

        contracted = centroid - self._rho * (centroid - self._simplex[-1])
        self._pending_candidate = (self._clip(contracted), reflected_value)
        self._stage = "inside_contract"
        if not self._enqueue(self._pending_candidate[0]):
            self._stage = "idle"
            self._maybe_terminate()

    def _handle_expand(self, expanded_value: float) -> None:
        expanded_point, reflected_value = self._pending_candidate
        self._stage = "idle"
        if expanded_value < reflected_value:
            self._replace_worst(expanded_point, expanded_value)
            return
        reflected_point = self._clip(
            np.mean(self._simplex[:-1], axis=0)
            + self._alpha * (np.mean(self._simplex[:-1], axis=0) - self._simplex[-1])
        )
        self._replace_worst(reflected_point, reflected_value)

    def _handle_contract(self, contracted_value: float) -> None:
        contracted_point, reflected_value = self._pending_candidate
        should_accept = (
            contracted_value <= reflected_value
            if self._stage == "outside_contract"
            else contracted_value < self._values[-1]
        )
        if should_accept:
            self._stage = "idle"
            self._replace_worst(contracted_point, contracted_value)
            return

        self._stage = "shrink"
        self._shrink_order = list(range(1, len(self._simplex)))
        self._shrink_completed = 0
        if self._remaining_budget() < len(self._shrink_order):
            self._termination_reason = "max_evals_reached"
            return
        best = self._simplex[0]
        for idx in self._shrink_order:
            shrunk = best + self._sigma * (self._simplex[idx] - best)
            assert self._enqueue(self._clip(shrunk))

    def _handle_shrink(self, shrunk_value: float) -> None:
        idx = self._shrink_order[self._shrink_completed]
        best = self._simplex[0]
        shrunk = self._clip(best + self._sigma * (self._simplex[idx] - best))
        self._simplex[idx] = shrunk
        self._values[idx] = shrunk_value
        self._shrink_completed += 1
        if self._shrink_completed != len(self._shrink_order):
            return

        self._sort_simplex()
        self._stage = "idle"
        self._maybe_terminate()
        if self._termination_reason is None:
            self._start_iteration()


class OptimizeRunner(HasEnvironment):
    def build(
        self,
        max_rtio_underflow_retries: int = 3,
        max_transitory_error_retries: int = 10,
        skip_on_persistent_transitory_error: bool = False,
    ):
        self.max_rtio_underflow_retries = max_rtio_underflow_retries
        self.max_transitory_error_retries = max_transitory_error_retries
        self.skip_on_persistent_transitory_error = skip_on_persistent_transitory_error
        self.setattr_device("core")
        self.setattr_device("scheduler")

    def run(
        self,
        fragment,
        spec: OptimizeSpec,
        axis_sinks: list[ResultSink],
        objective_channel: NumericChannel,
        on_best_updated: Callable[[tuple[float, ...], float], None] | None = None,
        on_terminated: Callable[[str], None] | None = None,
    ) -> None:
        optimizer = NelderMeadOptimizer(
            tuple(axis.initial for axis in spec.axes),
            tuple(axis.lower for axis in spec.axes),
            tuple(axis.upper for axis in spec.axes),
            spec.algorithm.max_evals,
            spec.algorithm.xatol,
            spec.algorithm.fatol,
        )

        point_runner = select_runner_class(fragment)(
            self,
            max_rtio_underflow_retries=self.max_rtio_underflow_retries,
            max_transitory_error_retries=self.max_transitory_error_retries,
            skip_on_persistent_transitory_error=self.skip_on_persistent_transitory_error,
        )
        scan_axes = [
            ScanAxis(axis.param_schema, axis.path, axis.param_store) for axis in spec.axes
        ]
        point_runner.setup(fragment, scan_axes, axis_sinks)

        current_point: tuple[float, ...] | None = None
        point_loaded = False
        num_points_recorded = 0
        try:
            while not optimizer.is_done():
                fragment.recompute_param_defaults()
                try:
                    fragment.host_setup()
                    while not optimizer.is_done():
                        if current_point is None:
                            current_point = optimizer.ask()
                            if current_point is None:
                                break
                            point_loaded = False
                        if not point_loaded:
                            point_runner.set_points(iter([current_point]))
                            point_loaded = True

                        completed = point_runner.acquire(device_cleanup=False)

                        new_count = len(axis_sinks[0].get_all())
                        if new_count == num_points_recorded:
                            if completed:
                                optimizer.tell(current_point, float("inf"))
                                current_point = None
                                point_loaded = False
                        else:
                            num_points_recorded = new_count
                            objective_value = objective_channel.sink.get_last()
                            transformed = (
                                objective_value
                                if spec.objective.direction == "min"
                                else -objective_value
                            )
                            optimizer.tell(current_point, transformed)
                            if on_best_updated is not None:
                                best = optimizer.best()
                                if best is not None:
                                    best_point, best_value = best
                                    actual_value = (
                                        best_value
                                        if spec.objective.direction == "min"
                                        else -best_value
                                    )
                                    on_best_updated(best_point, actual_value)

                            current_point = None
                            point_loaded = False

                        if not completed:
                            break

                    if optimizer.is_done():
                        return
                finally:
                    fragment.host_cleanup()
                    if hasattr(self.core, "close"):
                        self.core.close()
                self.scheduler.pause()
        finally:
            fragment.device_cleanup()
            if on_terminated is not None and optimizer.termination_reason() is not None:
                on_terminated(optimizer.termination_reason())


def describe_optimise(
    spec: OptimizeSpec,
    fragment,
    short_result_names: dict[Any, str],
) -> dict[str, Any]:
    desc = {
        "fragment_fqn": fragment.fqn,
        "axes": [
            {
                "param": axis.param_schema,
                "path": axis.path,
                "min": axis.lower,
                "max": axis.upper,
                "initial": axis.initial,
            }
            for axis in spec.axes
        ],
        "channels": {
            name: channel.describe()
            for channel, name in short_result_names.items()
            if channel.save_by_default
        },
    }
    return desc
