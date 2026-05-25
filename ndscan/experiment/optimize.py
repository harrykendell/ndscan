from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from itertools import repeat
from typing import Any

import numpy as np
from artiq.language import HasEnvironment

from .optimizers import (
    ALGORITHM_REGISTRY,
    CoordinateSearchOptimizeAlgorithmSpec,
    CoordinateSearchOptimizer,
    NelderMeadOptimizeAlgorithmSpec,
    NelderMeadOptimizer,
    ObjectiveSpec,
    Optimizer,
    OptimizeAcquisitionSpec,
    OptimizeAlgorithmSpec,
    OptimizeAxis,
    OptimizeSpec,
    build_algorithm_spec,
)
from .result_channels import NumericChannel, ResultSink
from .scan_runner import ScanAxis, select_runner_class

__all__ = [
    "ObjectiveSpec",
    "OptimizeAlgorithmSpec",
    "NelderMeadOptimizeAlgorithmSpec",
    "CoordinateSearchOptimizeAlgorithmSpec",
    "OptimizeAcquisitionSpec",
    "OptimizeAxis",
    "OptimizeSpec",
    "Optimizer",
    "NelderMeadOptimizer",
    "CoordinateSearchOptimizer",
    "OptimizeRunner",
    "describe_optimise",
    "build_algorithm_spec",
    "ALGORITHM_REGISTRY",
]


def create_optimizer(spec: OptimizeSpec) -> Optimizer:
    initial = tuple(axis.initial for axis in spec.axes)
    lower_bounds = tuple(axis.lower for axis in spec.axes)
    upper_bounds = tuple(axis.upper for axis in spec.axes)

    algorithm_kind = spec.algorithm.kind
    algo_info = ALGORITHM_REGISTRY.get(algorithm_kind)
    if algo_info is None:
        raise ValueError(f"Unsupported optimisation algorithm '{algorithm_kind}'")

    optimizer_cls = algo_info["optimizer_cls"]

    # Extract optimizer-specific parameters from the spec (all fields except 'kind')
    kwargs = {}
    for field in fields(spec.algorithm):
        if field.name != "kind":
            kwargs[field.name] = getattr(spec.algorithm, field.name)

    return optimizer_cls(initial, lower_bounds, upper_bounds, **kwargs)


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
        on_best_updated: Callable[[tuple[float, ...], float, float], None]
        | None = None,
        on_terminated: Callable[[str], None] | None = None,
    ) -> None:
        optimizer = create_optimizer(spec)

        point_runner = select_runner_class(fragment)(
            self,
            max_rtio_underflow_retries=self.max_rtio_underflow_retries,
            max_transitory_error_retries=self.max_transitory_error_retries,
            skip_on_persistent_transitory_error=(
                self.skip_on_persistent_transitory_error
            ),
        )
        scan_axes = [
            ScanAxis(axis.param_schema, axis.path, axis.param_store)
            for axis in spec.axes
        ]
        point_runner.setup(fragment, scan_axes, axis_sinks)

        repeats_per_point = spec.acquisition.num_repeats_per_point
        max_evals = spec.acquisition.max_evals
        num_evals_used = 0
        current_point: tuple[float, ...] | None = None
        current_objective_samples: list[float] = []
        point_loaded = False
        num_points_recorded = 0
        termination_reason: str | None = None
        try:
            while not optimizer.is_done():
                fragment.recompute_param_defaults()
                try:
                    fragment.host_setup()
                    while not optimizer.is_done() and num_evals_used < max_evals:
                        if current_point is None:
                            current_point = optimizer.ask()
                            if current_point is None:
                                break
                            current_objective_samples.clear()
                            point_loaded = False
                        if not point_loaded:
                            point_runner.set_points(
                                repeat(current_point, repeats_per_point)
                            )
                            point_loaded = True

                        completed = point_runner.acquire(device_cleanup=False)

                        new_count = len(axis_sinks[0].get_all())
                        if new_count != num_points_recorded:
                            current_objective_samples.extend(
                                float(v)
                                for v in objective_channel.sink.get_all()[
                                    num_points_recorded:new_count
                                ]
                            )
                            num_points_recorded = new_count

                        if len(current_objective_samples) >= repeats_per_point:
                            objective_value = _aggregate_objective_samples(
                                current_objective_samples[:repeats_per_point],
                                spec.acquisition.averaging_method,
                            )
                            objective_std_dev = float(
                                np.std(current_objective_samples[:repeats_per_point])
                            )
                            transformed = (
                                objective_value
                                if spec.objective.direction == "min"
                                else -objective_value
                            )
                            optimizer.tell(
                                current_point, transformed, objective_std_dev
                            )
                            num_evals_used += repeats_per_point
                            if on_best_updated is not None:
                                best = optimizer.best()
                                if best is not None:
                                    best_point, best_value = best
                                    best_std = optimizer.best_std()
                                    actual_value = (
                                        best_value
                                        if spec.objective.direction == "min"
                                        else -best_value
                                    )
                                    on_best_updated(
                                        best_point,
                                        actual_value,
                                        0.0 if best_std is None else best_std,
                                    )

                            current_objective_samples.clear()
                            current_point = None
                            point_loaded = False
                        elif completed:
                            optimizer.tell(current_point, float("inf"), 0.0)
                            num_evals_used += repeats_per_point
                            current_objective_samples.clear()
                            current_point = None
                            point_loaded = False

                        if not completed:
                            break

                    if optimizer.is_done():
                        termination_reason = optimizer.termination_reason()
                        return
                    if num_evals_used >= max_evals:
                        termination_reason = "max_evals_reached"
                        return
                finally:
                    fragment.host_cleanup()
                    if hasattr(self.core, "close"):
                        self.core.close()
                self.scheduler.pause()
        finally:
            fragment.device_cleanup()
            if termination_reason is None:
                termination_reason = optimizer.termination_reason()
            if on_terminated is not None and termination_reason is not None:
                on_terminated(termination_reason)


def _aggregate_objective_samples(samples: list[float], method: str) -> float:
    if method == "mean":
        return float(np.mean(samples))
    if method == "median":
        return float(np.median(samples))
    raise ValueError(f"Unsupported optimisation averaging method '{method}'")


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
        "acquisition": {
            "num_repeats_per_point": spec.acquisition.num_repeats_per_point,
            "averaging_method": spec.acquisition.averaging_method,
            "max_evals": spec.acquisition.max_evals,
        },
        "objective": {
            "channel": spec.objective.channel,
            "direction": spec.objective.direction,
        },
        "algorithm": {k: getattr(spec.algorithm, k) for k in vars(spec.algorithm)},
        "channels": {
            name: channel.describe()
            for channel, name in short_result_names.items()
            if channel.save_by_default
        },
    }
    return desc
