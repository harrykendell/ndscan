from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .base import (
    AlgorithmParameter,
    OptimizeAlgorithmSpec,
    Optimizer,
    register_algorithm,
)


@dataclass
class NelderMeadOptimizeAlgorithmSpec(OptimizeAlgorithmSpec):
    xatol: float = 1e-3
    fatol: float = 1e-3
    simplex_step_fraction: float = 0.5


class NelderMeadOptimizer(Optimizer):
    """Sequential ask/tell Nelder-Mead implementation.

    Nelder-Mead maintains a simplex of ``n + 1`` points for ``n`` parameters and improves
    it through reflection, expansion, contraction, and shrink steps. It is a local,
    derivative-free method that can work well on smooth objectives, but may need more
    evaluations than simpler search methods and is sensitive to local minima.
    """

    def __init__(
        self,
        initial: tuple[float, ...],
        lower_bounds: tuple[float, ...],
        upper_bounds: tuple[float, ...],
        xatol: float,
        fatol: float,
        simplex_step_fraction: float = 0.5,
    ):
        self._initial = np.array(initial, dtype=float)
        self._lower = np.array(lower_bounds, dtype=float)
        self._upper = np.array(upper_bounds, dtype=float)
        self._span = self._upper - self._lower
        self._xatol = xatol
        self._fatol = fatol
        self._simplex_step_fraction = simplex_step_fraction

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
        self._value_std_devs: list[float | None] = [None] * len(self._simplex)
        self._pending_candidate: tuple[np.ndarray, float, float] | None = None
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

        return None

    def tell(
        self, point: tuple[float, ...], value: float, std_dev: float = 0.0
    ) -> None:
        # When we are told the value for a point we need to update the simplex
        del point

        if self._termination_reason is not None:
            return

        std_dev = float(std_dev)

        if self._stage == "initial":
            idx = self._first_unset_index()
            self._values[idx] = value
            self._value_std_devs[idx] = std_dev
            if all(v is not None for v in self._values):
                self._sort_simplex()
                self._maybe_terminate()
                if self._termination_reason is None:
                    self._start_iteration()
            return

        if self._stage == "reflect":
            self._handle_reflect(value, std_dev)
            return

        if self._stage == "expand":
            self._handle_expand(value, std_dev)
            return

        if self._stage in {"outside_contract", "inside_contract"}:
            self._handle_contract(value, std_dev)
            return

        if self._stage == "shrink":
            self._handle_shrink(value, std_dev)
            return

        raise RuntimeError(f"Unexpected optimizer stage '{self._stage}'")

    def is_done(self) -> bool:
        return self._termination_reason is not None

    def best(self) -> tuple[tuple[float, ...], float] | None:
        if any(v is None for v in self._values):
            return None
        best_idx = int(np.argmin(self._values))
        return tuple(self._simplex[best_idx].tolist()), float(self._values[best_idx])

    def best_std(self) -> float | None:
        if any(v is None for v in self._values):
            return None
        best_idx = int(np.argmin(self._values))
        std_dev = self._value_std_devs[best_idx]
        return None if std_dev is None else float(std_dev)

    def termination_reason(self) -> str | None:
        return self._termination_reason

    def _build_initial_simplex(self) -> list[np.ndarray]:
        if len(self._initial) == 0:
            raise ValueError("Need at least one optimization axis")
        if np.any(self._span <= 0.0):
            raise ValueError("Optimization bounds must satisfy min < max")

        # Start with the initial point, then add one point per axis by stepping.
        simplex = [self._clip(self._initial.copy())]
        for i in range(len(self._initial)):
            point = self._initial.copy()
            delta = self._simplex_step_fraction * self._span[i]
            # Try to perturb in positive direction, otherwise negative
            point[i] += delta if point[i] + delta <= self._upper[i] else -delta
            simplex.append(self._clip(point))
        return simplex

    def _clip(self, point: np.ndarray) -> np.ndarray:
        return np.clip(point, self._lower, self._upper)

    def _enqueue(self, point: np.ndarray) -> bool:
        self._queue.append(self._clip(point))
        return True

    def _remaining_budget(self) -> int:
        return len(self._queue)

    def _first_unset_index(self) -> int:
        for i, value in enumerate(self._values):
            if value is None:
                return i
        raise RuntimeError("No unset simplex values left")

    def _sort_simplex(self) -> None:
        pairs = sorted(
            zip(self._simplex, self._values, self._value_std_devs), key=lambda p: p[1]
        )
        self._simplex = [point for point, _, _ in pairs]
        self._values = [float(value) for _, value, _ in pairs]
        self._value_std_devs = [
            None if std_dev is None else float(std_dev) for _, _, std_dev in pairs
        ]

    def _maybe_terminate(self) -> None:
        best = self._simplex[0]
        best_value = self._values[0]
        max_x_delta = max(
            np.max(np.abs((point - best) / self._span)) for point in self._simplex[1:]
        )
        max_f_delta = max(abs(value - best_value) for value in self._values[1:])
        if max_x_delta <= self._xatol and max_f_delta <= self._fatol:
            self._termination_reason = "converged"

    def _start_iteration(self) -> None:
        if self._termination_reason is not None:
            return

        centroid = np.mean(self._simplex[:-1], axis=0)
        worst = self._simplex[-1]
        reflected = centroid + self._alpha * (centroid - worst)
        self._pending_candidate = (self._clip(reflected), 0.0, 0.0)
        self._stage = "reflect"
        self._enqueue(self._pending_candidate[0])

    def _replace_worst(self, point: np.ndarray, value: float, std_dev: float) -> None:
        self._simplex[-1] = self._clip(point)
        self._values[-1] = float(value)
        self._value_std_devs[-1] = float(std_dev)
        self._sort_simplex()
        self._maybe_terminate()
        if self._termination_reason is None:
            self._start_iteration()

    def _handle_reflect(self, reflected_value: float, reflected_std_dev: float) -> None:
        reflected_point = self._pending_candidate[0]
        best_value = self._values[0]
        second_worst_value = self._values[-2]
        worst_value = self._values[-1]

        if reflected_value < best_value:
            centroid = np.mean(self._simplex[:-1], axis=0)
            expanded = centroid + self._gamma * (reflected_point - centroid)
            self._pending_candidate = (
                self._clip(expanded),
                reflected_value,
                reflected_std_dev,
            )
            self._stage = "expand"
            self._enqueue(self._pending_candidate[0])
            return

        if reflected_value < second_worst_value:
            self._stage = "idle"
            self._replace_worst(reflected_point, reflected_value, reflected_std_dev)
            return

        centroid = np.mean(self._simplex[:-1], axis=0)
        if reflected_value < worst_value:
            contracted = centroid + self._rho * (reflected_point - centroid)
            self._pending_candidate = (
                self._clip(contracted),
                reflected_value,
                reflected_std_dev,
            )
            self._stage = "outside_contract"
            self._enqueue(self._pending_candidate[0])
            return

        contracted = centroid - self._rho * (centroid - self._simplex[-1])
        self._pending_candidate = (
            self._clip(contracted),
            reflected_value,
            reflected_std_dev,
        )
        self._stage = "inside_contract"
        self._enqueue(self._pending_candidate[0])

    def _handle_expand(self, expanded_value: float, expanded_std_dev: float) -> None:
        expanded_point, reflected_value, reflected_std_dev = self._pending_candidate
        self._stage = "idle"
        if expanded_value < reflected_value:
            self._replace_worst(expanded_point, expanded_value, expanded_std_dev)
            return
        reflected_point = self._clip(
            np.mean(self._simplex[:-1], axis=0)
            + self._alpha * (np.mean(self._simplex[:-1], axis=0) - self._simplex[-1])
        )
        self._replace_worst(reflected_point, reflected_value, reflected_std_dev)

    def _handle_contract(
        self, contracted_value: float, contracted_std_dev: float
    ) -> None:
        contracted_point, reflected_value, _ = self._pending_candidate
        should_accept = (
            contracted_value <= reflected_value
            if self._stage == "outside_contract"
            else contracted_value < self._values[-1]
        )
        if should_accept:
            self._stage = "idle"
            self._replace_worst(contracted_point, contracted_value, contracted_std_dev)
            return

        self._stage = "shrink"
        self._shrink_order = list(range(1, len(self._simplex)))
        self._shrink_completed = 0
        best = self._simplex[0]
        for idx in self._shrink_order:
            shrunk = best + self._sigma * (self._simplex[idx] - best)
            self._enqueue(self._clip(shrunk))

    def _handle_shrink(self, shrunk_value: float, shrunk_std_dev: float) -> None:
        idx = self._shrink_order[self._shrink_completed]
        best = self._simplex[0]
        shrunk = self._clip(best + self._sigma * (self._simplex[idx] - best))
        self._simplex[idx] = shrunk
        self._values[idx] = shrunk_value
        self._value_std_devs[idx] = shrunk_std_dev
        self._shrink_completed += 1
        if self._shrink_completed != len(self._shrink_order):
            return

        self._sort_simplex()
        self._stage = "idle"
        self._maybe_terminate()
        if self._termination_reason is None:
            self._start_iteration()


register_algorithm(
    "nelder_mead",
    display_name="Nelder-Mead",
    description="Local derivative-free simplex method",
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
            name="simplex_step_fraction",
            label="simplex_step_fraction",
            minimum=0.01,
            maximum=1.0,
            default=0.25,
            step=0.01,
            tooltip="Initial simplex step size as a fraction of the parameter bounds span.",
        ),
    ],
    spec_cls=NelderMeadOptimizeAlgorithmSpec,
    optimizer_cls=NelderMeadOptimizer,
)

__all__ = [
    "NelderMeadOptimizeAlgorithmSpec",
    "NelderMeadOptimizer",
]
