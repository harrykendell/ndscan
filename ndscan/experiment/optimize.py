from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import fields
from itertools import repeat
from typing import Any

import numpy as np
from artiq.language import HasEnvironment

from .optimizers import (
    ALGORITHM_REGISTRY,
    BayesianOptimizer,
    BayesianOptimizerOptimizeAlgorithmSpec,
    CoordinateSearchOptimizeAlgorithmSpec,
    CoordinateSearchOptimizer,
    NelderMeadOptimizeAlgorithmSpec,
    NelderMeadOptimizer,
    ObjectiveSpec,
    OptimizeAcquisitionSpec,
    OptimizeAlgorithmSpec,
    OptimizeAxis,
    Optimizer,
    OptimizeSpec,
    build_algorithm_spec,
)
from .result_channels import AppendingDatasetSink, NumericChannel, ResultSink
from .scan_runner import ScanAxis, select_runner_class

logger = logging.getLogger(__name__)

__all__ = [
    "ObjectiveSpec",
    "BayesianOptimizer",
    "BayesianOptimizerOptimizeAlgorithmSpec",
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
    "OptimizeResultPublisher",
    "default_optimise_params",
    "make_optimise_spec",
    "select_objective_channel",
    "format_best_result",
    "describe_optimise",
    "build_algorithm_spec",
    "ALGORITHM_REGISTRY",
]


def default_optimise_params() -> dict[str, Any]:
    return {
        "parameters": [],
        "objective": {"channel": "", "direction": "min"},
        "algorithm": {
            "kind": "nelder_mead",
            "xatol": 1e-3,
            "fatol": 1e-3,
            "simplex_step_fraction": 0.5,
            "user_seed": -1,
        },
        "num_repeats_per_point": 1,
        "averaging_method": "mean",
        "max_evals": 100,
        "reference_normalisation": "none",
        "reference_resample_interval": 1,
        "skip_on_persistent_transitory_error": False,
    }


def make_optimise_spec(
    params: dict[str, Any],
    schemata: dict[str, dict[str, Any]],
    sample_instances: dict[str, Any],
) -> tuple[OptimizeSpec, bool]:
    optimise = params.get("optimise", {})
    axes = []
    for parameter in optimise.get("parameters", []):
        fqn = parameter["fqn"]
        path = parameter["path"]
        try:
            schema = schemata[fqn]
            store_type = sample_instances[fqn].StoreType
        except KeyError as error:
            raise ValueError(
                "Experiment does not have a parameter matching optimisation "
                f"parameter FQN '{fqn}'"
            ) from error
        if schema["type"] != "float":
            raise ValueError(
                f"Optimisation currently only supports float parameters, got '{fqn}'"
            )

        lower = float(parameter["min"])
        upper = float(parameter["max"])
        initial = float(parameter["initial"])
        if not lower < upper:
            raise ValueError(f"Optimisation bounds for '{fqn}' must satisfy min < max")
        if not lower <= initial <= upper:
            raise ValueError(
                f"Optimisation initial value for '{fqn}' must lie within bounds"
            )
        axes.append(
            OptimizeAxis(
                schema,
                path,
                store_type((fqn, path), store_type.value_from_pyon(initial)),
                lower,
                upper,
                initial,
            )
        )

    objective = optimise.get("objective", {})
    repeats = int(optimise.get("num_repeats_per_point", 1))
    averaging_method = optimise.get("averaging_method", "mean")
    max_evals = int(optimise.get("max_evals", 100))
    reference_normalisation = optimise.get("reference_normalisation", "none")
    reference_resample_interval = int(optimise.get("reference_resample_interval", 1))
    if repeats < 1:
        raise ValueError("Optimisation num_repeats_per_point must be positive")
    if averaging_method not in {"mean", "median"}:
        raise ValueError("Optimisation averaging_method must be 'mean' or 'median'")
    if max_evals < 1:
        raise ValueError("Optimisation max_evals must be positive")
    if reference_normalisation not in {"none", "subtract", "divide"}:
        raise ValueError(
            "Optimisation reference_normalisation must be "
            "'none', 'subtract', or 'divide'"
        )
    if reference_resample_interval < 1:
        raise ValueError("Optimisation reference_resample_interval must be positive")

    spec = OptimizeSpec(
        axes,
        ObjectiveSpec(objective.get("channel", ""), objective.get("direction", "min")),
        build_algorithm_spec(optimise.get("algorithm", {})),
        OptimizeAcquisitionSpec(
            repeats,
            averaging_method,
            max_evals,
            reference_normalisation,
            reference_resample_interval,
        ),
    )
    return spec, optimise.get("skip_on_persistent_transitory_error", False)


def select_objective_channel(
    spec: OptimizeSpec, channels: dict[str, Any]
) -> NumericChannel:
    path = spec.objective.channel
    if not path:
        raise ValueError("Optimisation objective channel must be specified")
    try:
        channel = channels[path]
    except KeyError as error:
        raise ValueError(
            f"Optimisation objective channel '{path}' does not exist"
        ) from error
    if not channel.save_by_default:
        raise ValueError("Optimisation objective channel must be saved by default")
    if not isinstance(channel, NumericChannel):
        raise ValueError(
            "Optimisation objective channel must be numeric (float or int)"
        )
    if spec.objective.direction not in {"min", "max"}:
        raise ValueError("Optimisation objective direction must be 'min' or 'max'")
    if not spec.axes:
        raise ValueError("Optimisation requires at least one parameter")
    return channel


def _format_numeric_value(value: float, spec: dict[str, Any]) -> str:
    unit = spec.get("unit", "")
    unit_suffix = f" {unit}" if unit else ""
    display_scale = 1 / spec.get("scale", 1)
    limits = spec.get("min"), spec.get("max")
    if limits[0] is None or limits[1] is None:
        limits = value, value

    scaled_span = abs(display_scale)
    if limits[1] > limits[0]:
        scaled_span *= abs(limits[1] - limits[0])
    elif abs(value) > 0:
        scaled_span *= abs(value)
    if not math.isfinite(scaled_span) or scaled_span <= 0:
        precision = 0
    else:
        smallest_digit = math.floor(math.log10(scaled_span)) - 3
        precision = int(-smallest_digit) if smallest_digit < 0 else 0
    return f"{value * display_scale:.{precision}f}{unit_suffix}"


def format_best_result(
    spec: OptimizeSpec,
    objective_channel: NumericChannel,
    point: tuple[float, ...],
    value: float,
) -> str:
    entries = [
        f"Objective: {_format_numeric_value(value, objective_channel.describe())}"
    ]
    for axis, axis_value in zip(spec.axes, point):
        name = axis.param_schema["fqn"].split(".")[-1]
        entries.append(
            f"{name}: {_format_numeric_value(axis_value, axis.param_schema['spec'])}"
        )
    return ", ".join(entries)


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
        on_evaluation: Callable[[tuple[float, ...], float, float, int], None]
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
        reference_normalisation = spec.acquisition.reference_normalisation
        use_reference = reference_normalisation != "none"
        reference_point = tuple(axis.initial for axis in spec.axes)
        reference_value: float | None = None
        reference_std_dev = 0.0
        candidates_since_reference = 0

        def publish_best() -> None:
            if on_best_updated is None:
                return
            best = optimizer.best()
            if best is None:
                return
            best_point, best_value = best
            best_std = optimizer.best_std()
            actual_value = (
                best_value if spec.objective.direction == "min" else -best_value
            )
            on_best_updated(
                best_point,
                actual_value,
                0.0 if best_std is None else best_std,
            )

        num_evals_used = 0
        current_point: tuple[float, ...] | None = None
        current_point_kind: str | None = None
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
                            if use_reference and (
                                reference_value is None
                                or candidates_since_reference
                                >= spec.acquisition.reference_resample_interval
                            ):
                                current_point = reference_point
                                current_point_kind = "reference"
                            else:
                                current_point = optimizer.ask()
                                if current_point is None:
                                    break
                                current_point_kind = "candidate"
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
                            num_evals_used += repeats_per_point

                            if current_point_kind == "reference":
                                reference_value = objective_value
                                reference_std_dev = objective_std_dev
                                candidates_since_reference = 0
                            else:
                                normalised = _normalise_objective_value(
                                    objective_value,
                                    objective_std_dev,
                                    reference_value,
                                    reference_std_dev,
                                    reference_normalisation,
                                )
                                if normalised is None:
                                    optimizer.tell(current_point, float("inf"), 0.0)
                                    if on_evaluation is not None:
                                        on_evaluation(
                                            current_point,
                                            float("nan"),
                                            float("nan"),
                                            num_points_recorded - 1,
                                        )
                                else:
                                    objective_value, objective_std_dev = normalised
                                    if on_evaluation is not None:
                                        on_evaluation(
                                            current_point,
                                            objective_value,
                                            objective_std_dev,
                                            num_points_recorded - 1,
                                        )
                                    transformed = (
                                        objective_value
                                        if spec.objective.direction == "min"
                                        else -objective_value
                                    )
                                    optimizer.tell(
                                        current_point,
                                        transformed,
                                        objective_std_dev,
                                    )
                                candidates_since_reference += 1
                                publish_best()

                            current_objective_samples.clear()
                            current_point = None
                            current_point_kind = None
                            point_loaded = False
                        elif completed:
                            num_evals_used += repeats_per_point
                            if current_point_kind == "reference":
                                termination_reason = "reference_measurement_failed"
                                return

                            optimizer.tell(current_point, float("inf"), 0.0)
                            if on_evaluation is not None:
                                on_evaluation(
                                    current_point,
                                    float("nan"),
                                    float("nan"),
                                    max(num_points_recorded - 1, 0),
                                )
                            candidates_since_reference += 1
                            current_objective_samples.clear()
                            current_point = None
                            current_point_kind = None
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


class OptimizeResultPublisher:
    """Own optimisation-specific result datasets and termination reporting."""

    def __init__(
        self, owner, dataset_prefix: str, spec: OptimizeSpec, objective_channel
    ):
        self._owner = owner
        self._dataset_prefix = dataset_prefix
        self._spec = spec
        self._objective_channel = objective_channel
        self._best = None
        self._evaluation_axis_sinks = [
            AppendingDatasetSink(
                owner, dataset_prefix + f"optimizer.evaluations.axis_{index}"
            )
            for index in range(len(spec.axes))
        ]
        self._evaluation_objective_sink = AppendingDatasetSink(
            owner, dataset_prefix + "optimizer.evaluations.objective"
        )
        self._evaluation_std_sink = AppendingDatasetSink(
            owner, dataset_prefix + "optimizer.evaluations.objective_std"
        )
        self._evaluation_point_index_sink = AppendingDatasetSink(
            owner, dataset_prefix + "optimizer.evaluations.point_index"
        )

    def append_evaluation(
        self,
        point: tuple[float, ...],
        value: float,
        std_dev: float,
        point_index: int,
    ) -> None:
        """Publish one value actually supplied to the optimiser.

        Reference acquisitions are deliberately omitted. ``point_index`` links each
        aggregated candidate evaluation back to the last raw acquisition contributing
        to it, allowing history views to hide future evaluations correctly.
        """
        for sink, axis_value in zip(self._evaluation_axis_sinks, point):
            sink.push(axis_value)
        self._evaluation_objective_sink.push(value)
        self._evaluation_std_sink.push(std_dev)
        # Push this completion marker last so subscribers never see a nominally
        # complete evaluation before its axes and objective are available.
        self._evaluation_point_index_sink.push(point_index)

    def update_best(
        self, point: tuple[float, ...], value: float, std_dev: float
    ) -> None:
        self._best = (point, value, std_dev)
        self._owner.set_dataset(
            self._dataset_prefix + "optimizer.best_value", value, broadcast=True
        )
        self._owner.set_dataset(
            self._dataset_prefix + "optimizer.best_std", std_dev, broadcast=True
        )
        for index, axis_value in enumerate(point):
            self._owner.set_dataset(
                self._dataset_prefix + f"optimizer.best_axis_{index}",
                axis_value,
                broadcast=True,
            )

    def set_termination_reason(self, reason: str) -> None:
        self._owner.set_dataset(
            self._dataset_prefix + "optimizer.termination_reason",
            reason,
            broadcast=True,
        )
        message = f"Optimizer terminated: {reason}"
        if self._best is not None:
            point, value, _std_dev = self._best
            message += "; best result: " + format_best_result(
                self._spec, self._objective_channel, point, value
            )
        logger.info(message)

    def broadcast_metadata(self, push: Callable[[str, Any], None]) -> None:
        push("optimizer.kind", self._spec.algorithm.kind)
        push("optimizer.objective_channel", self._spec.objective.channel)
        push("optimizer.objective_direction", self._spec.objective.direction)
        push("optimizer.max_evals", self._spec.acquisition.max_evals)
        push(
            "optimizer.num_repeats_per_point",
            self._spec.acquisition.num_repeats_per_point,
        )
        push("optimizer.averaging_method", self._spec.acquisition.averaging_method)
        push(
            "optimizer.reference_normalisation",
            self._spec.acquisition.reference_normalisation,
        )
        push(
            "optimizer.reference_resample_interval",
            self._spec.acquisition.reference_resample_interval,
        )


def _aggregate_objective_samples(samples: list[float], method: str) -> float:
    if method == "mean":
        return float(np.mean(samples))
    if method == "median":
        return float(np.median(samples))
    raise ValueError(f"Unsupported optimisation averaging method '{method}'")


def _normalise_objective_value(
    value: float,
    std_dev: float,
    reference_value: float | None,
    reference_std_dev: float,
    method: str,
) -> tuple[float, float] | None:
    if method == "none":
        return value, std_dev
    if reference_value is None:
        return None
    if method == "subtract":
        return value - reference_value, float(np.hypot(std_dev, reference_std_dev))
    if method == "divide":
        if abs(reference_value) <= np.finfo(float).eps:
            return None
        normalised = value / reference_value
        normalised_std_dev = float(
            np.hypot(
                std_dev / reference_value,
                value * reference_std_dev / reference_value**2,
            )
        )
        return normalised, normalised_std_dev
    raise ValueError(f"Unsupported reference normalisation method '{method}'")


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
            "reference_normalisation": spec.acquisition.reference_normalisation,
            "reference_resample_interval": (
                spec.acquisition.reference_resample_interval
            ),
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
