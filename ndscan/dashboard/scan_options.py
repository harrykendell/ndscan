"""Handling of the different override type widgets (fixed and various scans).

Notably, this is _not_ related to the list of global "Scan options" at the bottom of the
argument editor (as mirrored by ndscan.experiment.scan_generator).
"""

import logging
import math
from collections import OrderedDict
from enum import Enum, unique
from typing import Any, Optional

from artiq.gui.scientific_spinbox import ScientificSpinBox
from artiq.gui.tools import disable_scroll_wheel
from sipyco import pyon

from .._qt import QtCore, QtGui, QtWidgets
from ..utils import eval_param_default
from .utils import format_override_identity, load_icon_cached

logger = logging.getLogger(__name__)


def parse_list_pyon(values: str) -> list[float]:
    return pyon.decode("[" + values + "]")


def _raise_missing_default_dataset(key, default=None):
    raise KeyError(key)


def make_divider():
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.Shape.VLine)
    f.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
    f.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding
    )
    return f


def make_icon_label(
    sp: QtWidgets.QStyle.StandardPixmap, tooltip: str
) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel()
    icon = QtWidgets.QApplication.style().standardIcon(sp)
    label.setPixmap(icon.pixmap(16, 16))
    label.setToolTip(tooltip)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    label.setFixedWidth(20)
    return label


@unique
class SyncValue(Enum):
    """Equivalent values to be synchronised between similar scan types.

    Not all values will have a meaning for all scan types; they should just be left
    alone so that arguments for like scans are synchronised between each other.
    """

    centre = "centre"
    lower = "lower"
    upper = "upper"
    initial = "initial"
    num_points = "num_points"


class ScanOption(QtCore.QObject):
    """One "line" of scan options (the widgets specific to it, not including the
    selection box), and the code for serialising/deserialising it to the scan schema,
    plus synchronisation of the SyncValues between options.
    """

    value_changed = QtCore.pyqtSignal()
    execution_modes = frozenset({"scan", "optimise"})
    option_tooltip = ""

    def __init__(self, schema: dict[str, Any], path: str):
        super().__init__()
        self.schema = schema
        self.path = path

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        raise NotImplementedError

    def write_to_params(self, params: dict) -> None:
        raise NotImplementedError

    def read_sync_values(self, sync_values: dict) -> None:
        pass

    def write_sync_values(self, sync_values: dict) -> None:
        pass

    def attempt_read_from_axis(self, axis: dict) -> bool:
        return False

    def attempt_read_from_optimise_parameter(self, parameter: dict) -> bool:
        return False

    def make_randomise_box(self):
        box = QtWidgets.QCheckBox()
        box.setToolTip("Randomise scan point order")
        box.setIcon(load_icon_cached("media-playlist-shuffle-32.svg"))
        box.setChecked(True)
        box.stateChanged.connect(self.value_changed)
        return box


class StringFixedScanOption(ScanOption):
    option_tooltip = "Hold this parameter fixed at a string value."

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box = QtWidgets.QLineEdit()
        self.box.setToolTip("Fixed string value")
        layout.addWidget(self.box)

    def write_to_params(self, params: dict) -> None:
        o = {"path": self.path, "value": self.box.text()}
        params["overrides"].setdefault(self.schema["fqn"], []).append(o)

    def set_value(self, value) -> None:
        self.box.setText(value)


class BoolFixedScanOption(ScanOption):
    option_tooltip = "Hold this parameter fixed at a boolean value."

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box = QtWidgets.QCheckBox()
        self.box.setToolTip("Fixed boolean value")
        layout.addWidget(self.box)

    def write_to_params(self, params: dict) -> None:
        o = {"path": self.path, "value": self.box.isChecked()}
        params["overrides"].setdefault(self.schema["fqn"], []).append(o)

    def set_value(self, value) -> None:
        self.box.setChecked(value)


class EnumFixedScanOption(ScanOption):
    option_tooltip = "Hold this parameter fixed at one enum member."

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box = QtWidgets.QComboBox()
        self._members = self.schema["spec"]["members"]
        self._member_values_to_keys = {val: key for key, val in self._members.items()}
        self.box.addItems(self._members.values())
        self.box.setToolTip("Fixed enum value")
        layout.addWidget(self.box)

    def write_to_params(self, params: dict) -> None:
        o = {
            "path": self.path,
            "value": self._member_values_to_keys[self.box.currentText()],
        }
        params["overrides"].setdefault(self.schema["fqn"], []).append(o)

    def set_value(self, value) -> None:
        try:
            text = self._members[value]
        except KeyError:
            text = next(iter(self._members.values()))
            identity = format_override_identity(self.schema["fqn"], self.path)
            logger.warning(
                f"Stored value '{value}' not in schema for enum parameter "
                f"'{identity}', setting to '{text}'"
            )
        self.box.setCurrentText(text)


class NumericScanOption(ScanOption):
    def __init__(self, schema: dict[str, Any], path: str):
        super().__init__(schema, path)
        spec = schema.get("spec", {})
        self.scale = spec.get("scale", 1.0)
        self.min = spec.get("min", float("-inf"))
        self.max = spec.get("max", float("inf"))

    def _make_spin_box(self, set_limits_from_spec=True):
        box = ScientificSpinBox()
        disable_scroll_wheel(box)
        box.valueChanged.connect(self.value_changed)

        spec = self.schema.get("spec", {})
        step = spec.get("step", 1.0)

        box.setDecimals(8)
        # setPrecision() was renamed in ARTIQ 8.
        if hasattr(box, "setPrecision"):
            box.setPrecision()
        else:
            box.setSigFigs()
        box.setSingleStep(step / self.scale)
        box.setRelativeStep()

        if set_limits_from_spec:
            box.setMinimum(self.min / self.scale)
            box.setMaximum(self.max / self.scale)

        unit = spec.get("unit", "")
        if unit:
            box.setSuffix(" " + unit)
        return box

    def _default_numeric_spec_value(self, key: str, fallback: float = 0.0) -> float:
        value = self.schema.get("spec", {}).get(key)
        if value is None:
            return fallback
        if isinstance(value, (int, float)) and not math.isfinite(value):
            return fallback
        return float(value)

    def _default_numeric_step_value(self, fallback: float = 1.0) -> float:
        value = self.schema.get("spec", {}).get("step")
        if value is None:
            return fallback
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            return fallback
        return value

    def _default_numeric_param_value(self) -> Optional[float]:
        try:
            return float(
                eval_param_default(
                    self.schema["default"],
                    _raise_missing_default_dataset,
                )
            )
        except Exception:
            return None

    def _default_numeric_centre_value(self, fallback: float = 0.0) -> float:
        value = self._default_numeric_param_value()
        if value is None or not math.isfinite(value):
            return fallback
        return value

    def _default_numeric_range_values(self) -> tuple[float, float]:
        lower = self.schema.get("spec", {}).get("min")
        upper = self.schema.get("spec", {}).get("max")
        if isinstance(lower, (int, float)) and not math.isfinite(lower):
            lower = None
        if isinstance(upper, (int, float)) and not math.isfinite(upper):
            upper = None

        centre = self._default_numeric_param_value()
        step = self._default_numeric_step_value()

        if lower is not None:
            lower = float(lower)
        if upper is not None:
            upper = float(upper)

        if lower is None and upper is None:
            if centre is None:
                return 0.0, step
            return centre - step, centre + step

        if lower is None:
            if centre is None:
                lower = upper - step
            else:
                lower = min(centre - step, upper)
        if upper is None:
            if centre is None:
                upper = lower + step
            else:
                upper = max(centre + step, lower)

        if upper < lower:
            lower, upper = upper, lower
        if upper == lower:
            upper = lower + step
        return lower, upper

    def _default_numeric_half_span_value(self) -> float:
        lower, upper = self._default_numeric_range_values()
        centre = self._default_numeric_centre_value((lower + upper) / 2.0)

        half_span = min(max(centre - lower, 0.0), max(upper - centre, 0.0))
        if half_span > 0.0:
            return half_span

        half_span = abs(upper - lower) / 2.0
        if half_span > 0.0:
            return half_span
        return self._default_numeric_step_value()

    def _current_numeric_sync_value(self, sync_values: dict) -> Optional[float]:
        if SyncValue.initial in sync_values:
            return sync_values[SyncValue.initial]
        if SyncValue.centre in sync_values:
            return sync_values[SyncValue.centre]
        return None


class FixedScanOption(NumericScanOption):
    option_tooltip = "Hold this parameter fixed at a single value."

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box = self._make_spin_box()
        self.box.setToolTip("Fixed parameter value")
        layout.addWidget(self.box)

    def write_to_params(self, params: dict) -> None:
        o = {"path": self.path, "value": self.box.value() * self.scale}
        params["overrides"].setdefault(self.schema["fqn"], []).append(o)

    def set_value(self, value) -> None:
        if value is None:
            # Error evaluating defaults, no better guess.
            value = 0.0
        self.box.setValue(float(value) / self.scale)

    def read_sync_values(self, sync_values: dict) -> None:
        value = self._current_numeric_sync_value(sync_values)
        if value is not None:
            self.box.setValue(value)

    def write_sync_values(self, sync_values: dict) -> None:
        sync_values[SyncValue.initial] = self.box.value()
        sync_values[SyncValue.centre] = self.box.value()


class RangeScanOption(NumericScanOption):
    """Base class for different ways of specifying scans across a given numerical
    range.
    """

    def _make_inf_points_box(self):
        box = QtWidgets.QCheckBox()
        box.setToolTip("Infinitely refine scan grid")
        box.setText("∞")
        box.setChecked(True)
        box.stateChanged.connect(self.value_changed)
        return box

    def _build_points_ui(self, layout):
        self.check_infinite = self._make_inf_points_box()
        layout.addWidget(self.check_infinite)
        layout.setStretchFactor(self.check_infinite, 0)

        self.box_points = QtWidgets.QSpinBox()
        self.box_points.setMinimum(2)
        self.box_points.setValue(21)
        self.box_points.setToolTip(
            "Number of points in the finite scan grid when infinite refinement is off"
        )

        # Somewhat gratuitously restrict the number of scan points for sizing, and to
        # avoid the user accidentally pasting in millions of points, etc.
        self.box_points.setMaximum(0xFFFF)

        self.box_points.setSuffix(" pts")
        layout.addWidget(self.box_points)
        layout.setStretchFactor(self.box_points, 0)

        self.check_infinite.setChecked(True)
        self.box_points.setEnabled(False)
        self.check_infinite.stateChanged.connect(
            lambda *_: self.box_points.setEnabled(not self.check_infinite.isChecked())
        )

        self.check_randomise = self.make_randomise_box()
        layout.addWidget(self.check_randomise)
        layout.setStretchFactor(self.check_randomise, 0)

    def write_to_params(self, params: dict) -> None:
        spec = {
            "fqn": self.schema["fqn"],
            "path": self.path,
            "range": {
                "randomise_order": self.check_randomise.isChecked(),
            },
        }
        self.write_type_and_range(spec)
        params["scan"].setdefault("axes", []).append(spec)

    execution_modes = frozenset({"scan"})


class MinMaxScanOption(RangeScanOption):
    option_tooltip = (
        "Scan between lower and upper bounds on either a finite linear grid or an "
        "infinitely refining grid."
    )

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box_start = self._make_spin_box()
        self.box_start.setToolTip("Lower bound or start value")
        layout.addWidget(self.box_start)
        layout.setStretchFactor(self.box_start, 1)

        layout.addWidget(make_divider())

        self._build_points_ui(layout)

        layout.addWidget(make_divider())

        self.box_stop = self._make_spin_box()
        self.box_stop.setToolTip("Upper bound or stop value")
        layout.addWidget(self.box_stop)
        layout.setStretchFactor(self.box_stop, 1)

        lower, upper = self._default_numeric_range_values()
        self.box_start.setValue(lower / self.scale)
        self.box_stop.setValue(upper / self.scale)

    def read_sync_values(self, sync_values: dict) -> None:
        lower, upper = self._default_numeric_range_values()
        if SyncValue.lower in sync_values:
            self.box_start.setValue(sync_values[SyncValue.lower])
        else:
            self.box_start.setValue(lower / self.scale)
        if SyncValue.upper in sync_values:
            self.box_stop.setValue(sync_values[SyncValue.upper])
        else:
            self.box_stop.setValue(upper / self.scale)
        if SyncValue.num_points in sync_values:
            self.box_points.setValue(sync_values[SyncValue.num_points])

    def write_sync_values(self, sync_values: dict) -> None:
        sync_values[SyncValue.num_points] = self.box_points.value()

    def attempt_read_from_axis(self, axis: dict) -> bool:
        lower, upper = self._default_numeric_range_values()
        if axis["type"] == "refining":
            self.check_infinite.setChecked(True)
            self.box_start.setValue(axis["range"].get("lower", lower) / self.scale)
            self.box_stop.setValue(axis["range"].get("upper", upper) / self.scale)
            self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
            return True
        if axis["type"] == "linear":
            self.check_infinite.setChecked(False)
            self.box_start.setValue(axis["range"].get("start", lower) / self.scale)
            self.box_stop.setValue(axis["range"].get("stop", upper) / self.scale)
            self.box_points.setValue(axis["range"].get("num_points", 21))
            self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
            return True
        return False

    def write_type_and_range(self, spec: dict) -> None:
        start = self.box_start.value()
        stop = self.box_stop.value()
        if self.check_infinite.isChecked():
            spec["type"] = "refining"
            spec["range"] |= {
                "lower": start * self.scale,
                "upper": stop * self.scale,
            }
        else:
            spec["type"] = "linear"
            spec["range"] |= {
                "start": start * self.scale,
                "stop": stop * self.scale,
                "num_points": self.box_points.value(),
            }


class CentreSpanScanOption(RangeScanOption):
    option_tooltip = (
        "Scan symmetrically around a centre using a half-span, on either a finite "
        "grid or an infinitely refining grid."
    )

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box_centre = self._make_spin_box()
        self.box_centre.setToolTip("Centre value")
        layout.addWidget(self.box_centre)
        layout.setStretchFactor(self.box_centre, 1)

        self.plusminus = QtWidgets.QLabel("±")
        layout.addWidget(self.plusminus)
        layout.setStretchFactor(self.plusminus, 0)

        self.box_half_span = self._make_spin_box(set_limits_from_spec=False)
        self.box_half_span.setToolTip("Half-span around the centre value")
        layout.addWidget(self.box_half_span)
        layout.setStretchFactor(self.box_half_span, 1)

        layout.addWidget(make_divider())

        self._build_points_ui(layout)

        self.box_centre.setValue(self._default_numeric_centre_value() / self.scale)
        self.box_half_span.setValue(
            self._default_numeric_half_span_value() / self.scale
        )

    def read_sync_values(self, sync_values: dict) -> None:
        value = self._current_numeric_sync_value(sync_values)
        if value is not None:
            self.box_centre.setValue(value)
        else:
            self.box_centre.setValue(
                self._default_numeric_centre_value() / self.scale
            )

        self.box_half_span.setValue(
            self._default_numeric_half_span_value() / self.scale
        )

        if SyncValue.num_points in sync_values:
            self.box_points.setValue(sync_values[SyncValue.num_points])

    def write_sync_values(self, sync_values: dict) -> None:
        sync_values[SyncValue.initial] = self.box_centre.value()
        sync_values[SyncValue.centre] = self.box_centre.value()
        sync_values[SyncValue.num_points] = self.box_points.value()

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] == "centre_span_refining":
            self.check_infinite.setChecked(True)
        elif axis["type"] == "centre_span":
            self.check_infinite.setChecked(False)
        else:
            return False

        # Common to both finite/refining:
        self.box_half_span.setValue(
            axis["range"].get("half_span", self._default_numeric_half_span_value())
            / self.scale
        )
        self.box_centre.setValue(
            axis["range"].get("centre", self._default_numeric_centre_value())
            / self.scale
        )
        self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
        return True

    def write_type_and_range(self, spec: dict) -> None:
        centre = self.box_centre.value()
        half_span = self.box_half_span.value()
        spec["range"] |= {
            "centre": centre * self.scale,
            "half_span": half_span * self.scale,
            "limit_lower": self.min,
            "limit_upper": self.max,
        }
        if self.check_infinite.isChecked():
            spec["type"] = "centre_span_refining"
        else:
            spec["type"] = "centre_span"
            spec["range"]["num_points"] = self.box_points.value()


class ExpandingScanOption(NumericScanOption):
    execution_modes = frozenset({"scan"})
    option_tooltip = "Scan outward from a centre with a fixed spacing between steps."

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box_centre = self._make_spin_box()
        self.box_centre.setToolTip("Centre value")
        layout.addWidget(self.box_centre)
        layout.setStretchFactor(self.box_centre, 1)

        layout.addWidget(make_divider())

        self.check_randomise = self.make_randomise_box()
        layout.addWidget(self.check_randomise)
        layout.setStretchFactor(self.check_randomise, 0)

        layout.addWidget(make_divider())

        self.box_spacing = self._make_spin_box()
        self.box_spacing.setSuffix(self.box_spacing.suffix() + " steps")
        self.box_spacing.setToolTip("Spacing between neighbouring scan points")
        layout.addWidget(self.box_spacing)
        layout.setStretchFactor(self.box_spacing, 1)

        self.box_centre.setValue(self._default_numeric_centre_value() / self.scale)
        self.box_spacing.setValue(self._default_numeric_step_value() / self.scale)

    def write_to_params(self, params: dict) -> None:
        schema = self.schema
        spec = {
            "fqn": schema["fqn"],
            "path": self.path,
            "type": "expanding",
            "range": {
                "centre": self.box_centre.value() * self.scale,
                "spacing": self.box_spacing.value() * self.scale,
                "randomise_order": self.check_randomise.isChecked(),
            },
        }
        spec["range"]["limit_lower"] = self.min
        spec["range"]["limit_upper"] = self.max
        params["scan"].setdefault("axes", []).append(spec)

    def read_sync_values(self, sync_values: dict) -> None:
        value = self._current_numeric_sync_value(sync_values)
        if value is not None:
            self.box_centre.setValue(value)
        else:
            self.box_centre.setValue(self._default_numeric_centre_value() / self.scale)

    def write_sync_values(self, sync_values: dict) -> None:
        sync_values[SyncValue.initial] = self.box_centre.value()
        sync_values[SyncValue.centre] = self.box_centre.value()

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] != "expanding":
            return False
        self.box_centre.setValue(
            axis["range"].get("centre", self._default_numeric_centre_value())
            / self.scale
        )
        self.box_spacing.setValue(
            axis["range"].get("spacing", self._default_numeric_step_value())
            / self.scale
        )
        self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
        return True


class ListScanOption(NumericScanOption):
    execution_modes = frozenset({"scan"})
    option_tooltip = "Scan an explicit list of values."

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        class Validator(QtGui.QValidator):
            def validate(self, input, pos):
                try:
                    [float(f) for f in parse_list_pyon(input)]
                    return QtGui.QValidator.State.Acceptable, input, pos
                except Exception:
                    return QtGui.QValidator.State.Intermediate, input, pos

        self.box_pyon = QtWidgets.QLineEdit()
        self.box_pyon.setValidator(Validator(self))
        self.box_pyon.setToolTip("Comma-separated list of scan values")
        layout.addWidget(self.box_pyon)

        layout.addWidget(make_divider())

        self.check_randomise = self.make_randomise_box()
        layout.addWidget(self.check_randomise)
        layout.setStretchFactor(self.check_randomise, 0)

    def write_to_params(self, params: dict) -> None:
        try:
            values = [v * self.scale for v in parse_list_pyon(self.box_pyon.text())]
        except Exception as e:
            logger.info(e)
            values = []
        spec = {
            "fqn": self.schema["fqn"],
            "path": self.path,
            "type": "list",
            "range": {
                "values": values,
                "randomise_order": self.check_randomise.isChecked(),
            },
        }
        params["scan"].setdefault("axes", []).append(spec)

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] != "list":
            return False
        values = axis["range"].get("values", [])
        list_str = ", ".join([str(v / self.scale) for v in values])
        self.box_pyon.setText(list_str)
        self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
        return True


class BoolScanOption(ScanOption):
    execution_modes = frozenset({"scan"})
    option_tooltip = "Scan both boolean values, false and true."

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        dummy_box = QtWidgets.QCheckBox()
        dummy_box.setTristate()
        dummy_box.setEnabled(False)
        dummy_box.setChecked(True)
        dummy_box.setToolTip("Boolean scan covers both false and true")
        layout.addWidget(dummy_box)
        layout.setStretchFactor(dummy_box, 0)
        layout.addWidget(make_divider())
        self.check_randomise = self.make_randomise_box()
        layout.addWidget(self.check_randomise)
        layout.setStretchFactor(self.check_randomise, 1)

    def write_to_params(self, params: dict) -> None:
        spec = {
            "fqn": self.schema["fqn"],
            "path": self.path,
            "type": "list",
            "range": {
                "values": [False, True],
                "randomise_order": self.check_randomise.isChecked(),
            },
        }
        params["scan"].setdefault("axes", []).append(spec)

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] != "list":
            return False
        self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
        return True


class EnumScanOption(ScanOption):
    execution_modes = frozenset({"scan"})
    option_tooltip = "Scan across all enum members."

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.check_randomise = self.make_randomise_box()
        layout.addWidget(self.check_randomise)
        layout.setStretchFactor(self.check_randomise, 0)

    def write_to_params(self, params: dict) -> None:
        spec = {
            "fqn": self.schema["fqn"],
            "path": self.path,
            "type": "list",
            "range": {
                "values": list(self.schema["spec"]["members"].keys()),
                "randomise_order": self.check_randomise.isChecked(),
            },
        }
        params["scan"].setdefault("axes", []).append(spec)

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] != "list":
            return False
        self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
        return True


class OptimizeAxisOption(NumericScanOption):
    execution_modes = frozenset({"optimise"})
    option_tooltip = (
        "Optimise this parameter between lower and upper bounds, starting from the "
        "initial value."
    )

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        lower_label = make_icon_label(
            QtWidgets.QStyle.StandardPixmap.SP_ArrowDown, "Lower bound"
        )
        layout.addWidget(lower_label)
        layout.setStretchFactor(lower_label, 0)

        self.box_min = self._make_spin_box()
        self.box_min.setToolTip("Lower bound")
        layout.addWidget(self.box_min)
        layout.setStretchFactor(self.box_min, 1)

        layout.addWidget(make_divider())

        initial_label = make_icon_label(
            QtWidgets.QStyle.StandardPixmap.SP_MediaPlay, "Initial value"
        )
        layout.addWidget(initial_label)
        layout.setStretchFactor(initial_label, 0)

        self.box_initial = self._make_spin_box()
        self.box_initial.setToolTip("Initial value")
        layout.addWidget(self.box_initial)
        layout.setStretchFactor(self.box_initial, 1)

        layout.addWidget(make_divider())

        upper_label = make_icon_label(
            QtWidgets.QStyle.StandardPixmap.SP_ArrowUp, "Upper bound"
        )
        layout.addWidget(upper_label)
        layout.setStretchFactor(upper_label, 0)

        self.box_max = self._make_spin_box()
        self.box_max.setToolTip("Upper bound")
        layout.addWidget(self.box_max)
        layout.setStretchFactor(self.box_max, 1)

        self.box_min.setValue(self._default_numeric_spec_value("min") / self.scale)
        default = self._default_numeric_param_value()
        if default is not None:
            self.box_initial.setValue(default / self.scale)
        self.box_max.setValue(self._default_numeric_spec_value("max") / self.scale)

    def write_to_params(self, params: dict) -> None:
        spec = {
            "fqn": self.schema["fqn"],
            "path": self.path,
            "min": self.box_min.value() * self.scale,
            "max": self.box_max.value() * self.scale,
            "initial": self.box_initial.value() * self.scale,
        }
        params.setdefault("optimise", {}).setdefault("parameters", []).append(spec)

    def read_sync_values(self, sync_values: dict) -> None:
        self.box_min.setValue(self._default_numeric_spec_value("min") / self.scale)
        value = self._current_numeric_sync_value(sync_values)
        if value is not None:
            self.box_initial.setValue(value)
        else:
            default = self._default_numeric_param_value()
            if default is not None:
                self.box_initial.setValue(default / self.scale)
        self.box_max.setValue(self._default_numeric_spec_value("max") / self.scale)

    def write_sync_values(self, sync_values: dict) -> None:
        sync_values[SyncValue.initial] = self.box_initial.value()
        sync_values[SyncValue.centre] = self.box_initial.value()

    def attempt_read_from_optimise_parameter(self, parameter: dict) -> bool:
        self.box_min.setValue(
            parameter.get("min", self._default_numeric_spec_value("min")) / self.scale
        )
        self.box_max.setValue(
            parameter.get("max", self._default_numeric_spec_value("max")) / self.scale
        )
        default = self._default_numeric_param_value()
        self.box_initial.setValue(
            parameter.get("initial", 0.0 if default is None else default) / self.scale
        )
        return True


def list_scan_option_types(
    schema_type: str, is_scannable: bool
) -> OrderedDict[str, type[ScanOption]]:
    """Return a list of scan option types appropriate for the given parameter.

    :param schema_type: The "type" field of the parameter schema.
    :param is_scannable: Whether to show non-Fixed options.
    :return: An ordered list of option labels mapping to the ScanOption subclass
        representing them.
    """
    result = OrderedDict([])
    if schema_type == "string":
        result["Fixed"] = StringFixedScanOption
    elif schema_type == "bool":
        result["Fixed"] = BoolFixedScanOption
        if is_scannable:
            result["Scanning"] = BoolScanOption
    elif schema_type == "enum":
        result["Fixed"] = EnumFixedScanOption
        if is_scannable:
            result["Scanning"] = EnumScanOption
    else:
        # TODO: Properly handle int, add errors (or default to PYON value).
        result["Fixed"] = FixedScanOption
        if is_scannable:
            result["Min./Max."] = MinMaxScanOption
            result["Centered"] = CentreSpanScanOption
            result["Expanding"] = ExpandingScanOption
            result["List"] = ListScanOption
            if schema_type == "float":
                result["Optimise"] = OptimizeAxisOption
    return result
