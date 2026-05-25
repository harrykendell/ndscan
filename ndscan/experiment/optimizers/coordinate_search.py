from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import (
    AlgorithmParameter,
    OptimizeAlgorithmSpec,
    Optimizer,
    register_algorithm,
)


@dataclass
class CoordinateSearchOptimizeAlgorithmSpec(OptimizeAlgorithmSpec):
    xatol: float = 1e-3
    fatol: float = 1e-3
    step_reduction_factor: float = 0.5


class CoordinateSearchOptimizer(Optimizer):
    """Sequential bounded coordinate-search optimizer.

    Coordinate search probes one axis at a time by stepping forward and backward within
    the configured bounds, accepting any improving move and halving the step sizes when a
    full pass yields no improvement. It is a simple derivative-free local search method
    that is easy to reason about and often robust on separable or moderately coupled
    objectives, though it can be slower on strongly rotated valleys.
    """

    def __init__(
        self,
        initial: tuple[float, ...],
        lower_bounds: tuple[float, ...],
        upper_bounds: tuple[float, ...],
        xatol: float,
        fatol: float,
        step_reduction_factor: float = 0.5,
    ):
        self._initial = np.array(initial, dtype=float)
        self._lower = np.array(lower_bounds, dtype=float)
        self._upper = np.array(upper_bounds, dtype=float)
        self._span = self._upper - self._lower
        self._xatol = xatol
        self._fatol = fatol
        self._step_reduction_factor = step_reduction_factor

        if len(self._initial) == 0:
            raise ValueError("Need at least one optimization axis")
        if np.any(self._span <= 0.0):
            raise ValueError("Optimization bounds must satisfy min < max")

        self._num_asked = 0
        self._termination_reason: str | None = None

        self._current_point = self._clip(self._initial.copy())
        self._current_value: float | None = None
        self._best_point: np.ndarray | None = None
        self._best_value: float | None = None
        self._best_std: float | None = None

        self._step_sizes = 0.25 * self._span
        self._minimum_step = np.maximum(
            self._xatol * self._span,
            np.finfo(float).eps * np.maximum(1.0, self._span),
        )

        self._axis_index = 0
        self._direction_index = 0
        self._trial_candidates: list[tuple[np.ndarray, float]] = []
        self._cycle_start_value: float | None = None
        self._cycle_improved = False

        self._pending_point: np.ndarray | None = None

    def ask(self) -> tuple[float, ...] | None:
        if self._termination_reason is not None:
            return None

        if self._current_value is None:
            self._pending_point = self._current_point.copy()
        else:
            self._pending_point = None
            self._prepare_next_point()
            if self._pending_point is None:
                return None

        self._num_asked += 1
        return tuple(self._pending_point.tolist())

    def tell(
        self, point: tuple[float, ...], value: float, std_dev: float = 0.0
    ) -> None:
        del point

        if self._termination_reason is not None or self._pending_point is None:
            return

        value = float(value)
        std_dev = float(std_dev)
        pending_point = self._pending_point
        self._pending_point = None

        if self._current_value is None:
            self._current_value = value
            self._best_point = pending_point.copy()
            self._best_value = value
            self._best_std = std_dev
            self._cycle_start_value = value
            return

        if value < self._best_value:
            self._best_point = pending_point.copy()
            self._best_value = value
            self._best_std = std_dev

        self._trial_candidates.append((pending_point.copy(), value))
        self._direction_index += 1

        if self._direction_index != 2:
            return

        best_candidate, best_value = min(self._trial_candidates, key=lambda p: p[1])
        if best_value < self._current_value:
            self._current_point = best_candidate
            self._current_value = best_value
            self._cycle_improved = True

        self._trial_candidates.clear()
        self._direction_index = 0
        self._axis_index += 1
        if self._axis_index == len(self._current_point):
            self._finish_cycle()

    def is_done(self) -> bool:
        return self._termination_reason is not None

    def best(self) -> tuple[tuple[float, ...], float] | None:
        if self._best_point is None or self._best_value is None:
            return None
        return tuple(self._best_point.tolist()), float(self._best_value)

    def best_std(self) -> float | None:
        return self._best_std

    def termination_reason(self) -> str | None:
        return self._termination_reason

    def _clip(self, point: np.ndarray) -> np.ndarray:
        return np.clip(point, self._lower, self._upper)

    def _prepare_next_point(self) -> None:
        while self._termination_reason is None:
            if self._axis_index == len(self._current_point):
                self._finish_cycle()
                if self._termination_reason is not None:
                    return
                continue

            axis = self._axis_index
            direction = 1.0 if self._direction_index == 0 else -1.0
            candidate = self._current_point.copy()
            candidate[axis] = np.clip(
                candidate[axis] + direction * self._step_sizes[axis],
                self._lower[axis],
                self._upper[axis],
            )

            if np.array_equal(candidate, self._current_point):
                self._direction_index += 1
                if self._direction_index == 2:
                    self._direction_index = 0
                    self._axis_index += 1
                continue

            self._pending_point = candidate
            return

    def _finish_cycle(self) -> None:
        if self._current_value is None:
            return

        improvement = self._cycle_start_value - self._current_value
        if self._cycle_improved:
            if improvement <= self._fatol and np.all(
                self._step_sizes <= self._minimum_step
            ):
                self._termination_reason = "converged"
                return
        else:
            if np.all(self._step_sizes <= self._minimum_step):
                self._termination_reason = "converged"
                return
            self._step_sizes *= self._step_reduction_factor

        self._axis_index = 0
        self._direction_index = 0
        self._trial_candidates.clear()
        self._cycle_start_value = self._current_value
        self._cycle_improved = False


register_algorithm(
    "coordinate_search",
    display_name="Coordinate search",
    description="Simple bounded coordinate-search optimizer",
    parameters=[
        AlgorithmParameter(
            name="fatol",
            label="fatol",
            minimum=0.0,
            maximum=10**9,
            default=1e-3,
            tooltip="Terminate when the objective value changes by at most this amount.",
        ),
        AlgorithmParameter(
            name="xatol",
            label="xatol",
            minimum=0.0,
            maximum=1.0,
            default=1e-3,
            step=1e-4,
            tooltip="Terminate when each axis moves by at most this fraction of its configured bounds span.",
        ),
        AlgorithmParameter(
            name="step_reduction_factor",
            label="step_reduction_factor",
            minimum=0.1,
            maximum=0.9,
            default=0.5,
            step=0.05,
            tooltip="Fraction to reduce step sizes by when no improvement is found in a cycle.",
        ),
    ],
    spec_cls=CoordinateSearchOptimizeAlgorithmSpec,
    optimizer_cls=CoordinateSearchOptimizer,
)

__all__ = [
    "CoordinateSearchOptimizeAlgorithmSpec",
    "CoordinateSearchOptimizer",
]
