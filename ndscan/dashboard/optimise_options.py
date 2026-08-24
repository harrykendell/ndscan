"""Optimisation-specific controls for the ndscan argument editor."""

from __future__ import annotations

from typing import Any

from artiq.gui.scientific_spinbox import ScientificSpinBox
from artiq.gui.tools import disable_scroll_wheel

from .._qt import QtCore, QtWidgets
from ..experiment.optimize import ALGORITHM_REGISTRY
from ..utils import ExecutionMode, shorten_to_unambiguous_suffixes
from .scan_options import NumericScanOption, SyncValue, make_divider

_OPTIMISE_AXIS_FIELDS = (
    (
        QtWidgets.QStyle.StandardPixmap.SP_ArrowDown,
        "Lower bound",
        "box_min",
    ),
    (
        QtWidgets.QStyle.StandardPixmap.SP_MediaPlay,
        "Initial value",
        "box_initial",
    ),
    (
        QtWidgets.QStyle.StandardPixmap.SP_ArrowUp,
        "Upper bound",
        "box_max",
    ),
)


def _make_icon_label(
    icon: QtWidgets.QStyle.StandardPixmap, tooltip: str
) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel()
    label.setPixmap(QtWidgets.QApplication.style().standardIcon(icon).pixmap(16, 16))
    label.setToolTip(tooltip)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    label.setFixedWidth(20)
    return label


def _make_value_box(minimum: float, maximum: float, value: float, step=None):
    box = ScientificSpinBox()
    disable_scroll_wheel(box)
    box.setDecimals(8)
    if hasattr(box, "setPrecision"):
        box.setPrecision()
    else:
        box.setSigFigs()
    box.setMinimum(minimum)
    box.setMaximum(maximum)
    if step is not None:
        box.setSingleStep(step)
    box.setRelativeStep()
    box.setValue(value)
    return box


class ExecutionModeSelector(QtWidgets.QWidget):
    mode_changed = QtCore.pyqtSignal(str)

    def __init__(self, current_mode: str):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(QtWidgets.QLabel("Execution mode:"))
        self.box = QtWidgets.QComboBox()
        self.box.addItems([mode.value for mode in ExecutionMode])
        try:
            mode = ExecutionMode[current_mode]
        except KeyError:
            mode = ExecutionMode.scan
        self.box.setCurrentText(mode.value)
        self.box.currentTextChanged.connect(
            lambda text: self.mode_changed.emit(ExecutionMode(text).name)
        )
        layout.addWidget(self.box)
        layout.addStretch()

    def current_mode(self) -> str:
        return ExecutionMode(self.box.currentText()).name


class OptimiseOptions:
    """Global controls that apply only to optimisation execution."""

    _averaging_methods = {"Mean": "mean", "Median": "median"}
    _reference_methods = {
        "None": "none",
        "Subtract": "subtract",
        "Divide": "divide",
    }

    def __init__(self, params: dict[str, Any]):
        current = params.get("optimise", {})
        self._objective_paths = {}
        self.objective_container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(self.objective_container)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(QtWidgets.QLabel("Objective:"))
        self.direction_box = QtWidgets.QComboBox()
        self.direction_box.addItems(["Minimise", "Maximise"])
        self.direction_box.setToolTip(
            "Choose whether lower or higher objective values are better."
        )
        layout.addWidget(self.direction_box)
        self.objective_box = QtWidgets.QComboBox()
        self.objective_box.setToolTip("Select the numeric result channel to optimise.")
        self._add_objective("<Select objective channel>", "")
        result_channels = params.get("result_channels", {})
        shortened = shorten_to_unambiguous_suffixes(
            result_channels, lambda path, n: "/".join(path.split("/")[-n:])
        )
        for path, description in result_channels.items():
            if description["type"] not in {"float", "int"}:
                continue
            short_name = shortened[path]
            label = description.get("description") or short_name
            if label != short_name:
                label += f" ({short_name})"
            self._add_objective(label, path)
        layout.addWidget(self.objective_box, 1)

        objective = current.get("objective", {})
        current_path = objective.get("channel", "")
        label = next(
            (
                label
                for label, path in self._objective_paths.items()
                if path == current_path
            ),
            None,
        )
        if label is None and current_path:
            label = f"<Missing channel: {current_path}>"
            self._add_objective(label, current_path)
        self.objective_box.setCurrentText(label or "<Select objective channel>")
        self.direction_box.setCurrentText(
            "Maximise" if objective.get("direction", "min") == "max" else "Minimise"
        )

        self.acquisition_container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(self.acquisition_container)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(QtWidgets.QLabel("Repeat each point"))
        self.repeats_box = QtWidgets.QSpinBox()
        self.repeats_box.setRange(1, 10**6)
        self.repeats_box.setValue(current.get("num_repeats_per_point", 1))
        layout.addWidget(self.repeats_box)
        layout.addWidget(QtWidgets.QLabel("times, take the"))
        self.averaging_box = QtWidgets.QComboBox()
        self.averaging_box.addItems(self._averaging_methods)
        averaging = current.get("averaging_method", "mean")
        self.averaging_box.setCurrentText(
            next(
                (
                    label
                    for label, value in self._averaging_methods.items()
                    if value == averaging
                ),
                "Mean",
            )
        )
        layout.addWidget(self.averaging_box)
        layout.addStretch()

        self.reference_container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(self.reference_container)
        layout.setContentsMargins(5, 5, 5, 5)
        self.reference_box = QtWidgets.QComboBox()
        self.reference_box.addItems(self._reference_methods)
        reference_method = current.get("reference_normalisation", "none")
        self.reference_box.setCurrentText(
            next(
                (
                    label
                    for label, value in self._reference_methods.items()
                    if value == reference_method
                ),
                "None",
            )
        )
        layout.addWidget(self.reference_box)
        layout.addWidget(QtWidgets.QLabel("(resampled every"))
        self.reference_interval_box = QtWidgets.QSpinBox()
        self.reference_interval_box.setRange(1, 10**7)
        self.reference_interval_box.setValue(
            current.get("reference_resample_interval", 1)
        )
        layout.addWidget(self.reference_interval_box)
        layout.addWidget(QtWidgets.QLabel("candidate points)"))
        layout.addStretch()

        self.error_container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(self.error_container)
        layout.setContentsMargins(5, 5, 5, 5)
        self.skip_errors_box = QtWidgets.QCheckBox(
            "Apply a maximally bad result if transitory errors persist"
        )
        self.skip_errors_box.setChecked(
            current.get("skip_on_persistent_transitory_error", False)
        )
        self.skip_errors_box.setToolTip(
            "Continue by reporting infinite cost to the optimiser after the configured "
            "transitory-error retry limit is exhausted."
        )
        layout.addWidget(self.skip_errors_box)
        layout.addStretch()

        self.algorithm_container = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(self.algorithm_container)
        outer.setContentsMargins(5, 5, 5, 5)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Algorithm:"))
        self.algorithm_box = QtWidgets.QComboBox()
        self._algorithm_kinds = {}
        for kind, info in ALGORITHM_REGISTRY.items():
            label = info["display_name"]
            self._algorithm_kinds[label] = kind
            self.algorithm_box.addItem(label)
        row.addWidget(self.algorithm_box)
        row.addStretch()
        row.addWidget(QtWidgets.QLabel("Max evaluations:"))
        self.max_evals_box = QtWidgets.QSpinBox()
        self.max_evals_box.setRange(1, 10**7)
        self.max_evals_box.setValue(current.get("max_evals", 100))
        self.max_evals_box.setToolTip(
            "Maximum number of objective evaluations before stopping."
        )
        row.addWidget(self.max_evals_box)
        outer.addLayout(row)

        algorithm = current.get("algorithm", {})
        current_kind = algorithm.get("kind", "nelder_mead")
        label = next(
            (
                label
                for label, kind in self._algorithm_kinds.items()
                if kind == current_kind
            ),
            None,
        )
        if label is None:
            label = f"<Unknown algorithm: {current_kind}>"
            self._algorithm_kinds[label] = current_kind
            self.algorithm_box.addItem(label)
        self.algorithm_box.setCurrentText(label)

        self.algorithm_parameter_widgets = {}
        self._algorithm_parameter_containers = {}
        for kind, info in ALGORITHM_REGISTRY.items():
            container = QtWidgets.QWidget()
            parameters_layout = QtWidgets.QFormLayout(container)
            parameters_layout.setContentsMargins(0, 0, 0, 0)
            widgets = {}
            for parameter in info["parameters"]:
                widget = _make_value_box(
                    parameter.minimum,
                    parameter.maximum,
                    algorithm.get(parameter.name, parameter.default),
                    parameter.step,
                )
                widget.setToolTip(parameter.tooltip)
                parameters_layout.addRow(parameter.label + ":", widget)
                widgets[parameter.name] = widget
            self.algorithm_parameter_widgets[kind] = widgets
            self._algorithm_parameter_containers[kind] = container
            outer.addWidget(container)
        self.algorithm_box.currentIndexChanged.connect(self._update_algorithm)
        self.repeats_box.valueChanged.connect(self._update_averaging)
        self.reference_box.currentIndexChanged.connect(self._update_reference)
        self._update_algorithm()
        self._update_averaging()
        self._update_reference()

    def _add_objective(self, label: str, path: str) -> None:
        self._objective_paths[label] = path
        self.objective_box.addItem(label)

    def _update_algorithm(self, *_args) -> None:
        selected = self._algorithm_kinds.get(self.algorithm_box.currentText())
        for kind, container in self._algorithm_parameter_containers.items():
            container.setVisible(kind == selected)

    def _update_averaging(self, *_args) -> None:
        self.averaging_box.setEnabled(self.repeats_box.value() > 1)

    def _update_reference(self, *_args) -> None:
        self.reference_interval_box.setEnabled(
            self.reference_box.currentText() != "None"
        )

    def get_widgets(self) -> list[QtWidgets.QWidget]:
        return [
            self.objective_container,
            self.acquisition_container,
            self.reference_container,
            self.error_container,
            self.algorithm_container,
        ]

    def set_visible(self, visible: bool) -> None:
        for widget in self.get_widgets():
            widget.setVisible(visible)

    def connect_change_signal(self, callback) -> None:
        widgets = [
            self.objective_box,
            self.direction_box,
            self.repeats_box,
            self.averaging_box,
            self.max_evals_box,
            self.reference_box,
            self.reference_interval_box,
            self.skip_errors_box,
            self.algorithm_box,
        ]
        widgets.extend(
            widget
            for parameters in self.algorithm_parameter_widgets.values()
            for widget in parameters.values()
        )
        for widget in widgets:
            for signal_name in ("currentIndexChanged", "valueChanged", "stateChanged"):
                signal = getattr(widget, signal_name, None)
                if signal is not None:
                    signal.connect(callback)

    def write_to_params(self, params: dict[str, Any]) -> None:
        optimise = params.setdefault("optimise", {})
        optimise["objective"] = {
            "channel": self._objective_paths.get(self.objective_box.currentText(), ""),
            "direction": "max"
            if self.direction_box.currentText() == "Maximise"
            else "min",
        }
        kind = self._algorithm_kinds.get(
            self.algorithm_box.currentText(), "nelder_mead"
        )
        optimise["algorithm"] = {"kind": kind}
        for name, widget in self.algorithm_parameter_widgets.get(kind, {}).items():
            optimise["algorithm"][name] = widget.value()
        optimise["num_repeats_per_point"] = self.repeats_box.value()
        optimise["averaging_method"] = self._averaging_methods[
            self.averaging_box.currentText()
        ]
        optimise["max_evals"] = self.max_evals_box.value()
        optimise["reference_normalisation"] = self._reference_methods[
            self.reference_box.currentText()
        ]
        optimise["reference_resample_interval"] = self.reference_interval_box.value()
        optimise["skip_on_persistent_transitory_error"] = (
            self.skip_errors_box.isChecked()
        )


class OptimiseAxisOption(NumericScanOption):
    """Min/initial/max controls for one floating-point optimisation parameter."""

    def _default_values(self) -> tuple[float, float, float]:
        lower, upper = self._default_numeric_range_values()
        initial = self._default_numeric_centre_value((lower + upper) / 2.0)
        return lower, min(max(initial, lower), upper), upper

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        lower, initial, upper = self._default_values()
        for (icon, tooltip, attr), value in zip(
            _OPTIMISE_AXIS_FIELDS, (lower, initial, upper)
        ):
            label = _make_icon_label(icon, tooltip)
            layout.addWidget(label)
            layout.setStretchFactor(label, 0)
            box = self._make_spin_box()
            box.setToolTip(tooltip)
            box.setValue(value / self.scale)
            setattr(self, attr, box)
            layout.addWidget(box, 1)
            if attr != "box_max":
                layout.addWidget(make_divider())

    def write_to_params(self, params: dict) -> None:
        params.setdefault("optimise", {}).setdefault("parameters", []).append(
            {
                "fqn": self.schema["fqn"],
                "path": self.path,
                "min": self.box_min.value() * self.scale,
                "initial": self.box_initial.value() * self.scale,
                "max": self.box_max.value() * self.scale,
            }
        )

    def read_sync_values(self, sync_values: dict) -> None:
        lower, initial, upper = self._default_values()
        self.box_min.setValue(lower / self.scale)
        value = self._current_numeric_sync_value(sync_values)
        self.box_initial.setValue(initial / self.scale if value is None else value)
        self.box_max.setValue(upper / self.scale)

    def write_sync_values(self, sync_values: dict) -> None:
        sync_values[SyncValue.initial] = self.box_initial.value()
        sync_values[SyncValue.centre] = self.box_initial.value()

    def attempt_read_from_optimise_parameter(self, parameter: dict) -> bool:
        lower, initial, upper = self._default_values()
        self.box_min.setValue(parameter.get("min", lower) / self.scale)
        self.box_initial.setValue(parameter.get("initial", initial) / self.scale)
        self.box_max.setValue(parameter.get("max", upper) / self.scale)
        return True
