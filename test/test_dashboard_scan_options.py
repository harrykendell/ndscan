import os
import sys
import types
import unittest
from enum import Enum
from itertools import tee

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ndscan._qt import QtWidgets

import itertools

if not hasattr(itertools, "pairwise"):
    def pairwise(iterable):
        first, second = tee(iterable)
        next(second, None)
        return zip(first, second)

    itertools.pairwise = pairwise

pyon = types.ModuleType("sipyco.pyon")
pyon.decode = lambda value: eval(value, {})
sys.modules.setdefault("sipyco", types.ModuleType("sipyco"))
sys.modules["sipyco.pyon"] = pyon
sys.modules["sipyco"].pyon = pyon

scientific_spinbox = types.ModuleType("artiq.gui.scientific_spinbox")


class ScientificSpinBox(QtWidgets.QDoubleSpinBox):
    def setPrecision(self):
        pass

    def setSigFigs(self):
        pass

    def setRelativeStep(self):
        pass


scientific_spinbox.ScientificSpinBox = ScientificSpinBox
gui_tools = types.ModuleType("artiq.gui.tools")
gui_tools.disable_scroll_wheel = lambda box: None


class LayoutWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)

    def addWidget(self, widget, row=0, col=0):
        self.layout.addWidget(widget, row, col)


class WheelFilter:
    pass


gui_tools.LayoutWidget = LayoutWidget
gui_tools.WheelFilter = WheelFilter
units = types.ModuleType("artiq.language.units")
units.__all__ = []
language = types.ModuleType("artiq.language")
language.units = units
fit_base = types.SimpleNamespace(FitBase=object)
fitting = types.ModuleType("oitg.fitting")
fitting.FitBase = fit_base
for name in [
    "cos",
    "decaying_sinusoid",
    "detuned_square_pulse",
    "exponential_decay",
    "gaussian",
    "line",
    "lorentzian",
    "rabi_flop",
    "sinusoid",
    "v_function",
    "shifted_parabola",
]:
    setattr(fitting, name, object())
gui = types.ModuleType("artiq.gui")
gui.scientific_spinbox = scientific_spinbox
gui.tools = gui_tools
artiq = types.ModuleType("artiq")
artiq.gui = gui
artiq.language = language
oitg = types.ModuleType("oitg")
oitg.fitting = fitting
ndscan_utils = types.ModuleType("ndscan.utils")


class ExecutionMode(Enum):
    scan = "Scan"
    optimise = "Optimise"


class NoAxesMode(Enum):
    single = "Single (run once)"
    repeat = "Repeat (save only last)"
    time_series = "Time series (save all, with timestamps)"


ndscan_utils.eval_param_default = lambda value, get_dataset: eval(
    value, {"dataset": get_dataset}
)
ndscan_utils.ExecutionMode = ExecutionMode
ndscan_utils.NoAxesMode = NoAxesMode
ndscan_utils.PARAMS_ARG_KEY = "ndscan_params"
ndscan_utils.shorten_to_unambiguous_suffixes = lambda fqns: fqns
entries = types.ModuleType("artiq.gui.entries")
entries.procdesc_to_entry = lambda procdesc: procdesc
fuzzy_select = types.ModuleType("artiq.gui.fuzzy_select")
fuzzy_select.FuzzySelectWidget = QtWidgets.QWidget
param_tree_dialog = types.ModuleType("ndscan.dashboard.param_tree_dialog")
param_tree_dialog.OverrideProvider = object
param_tree_dialog.OverrideStatus = object
param_tree_dialog.ParamTreeDialog = QtWidgets.QDialog
sys.modules["artiq"] = artiq
sys.modules["artiq.gui"] = gui
sys.modules["artiq.gui.entries"] = entries
sys.modules["artiq.gui.fuzzy_select"] = fuzzy_select
sys.modules["artiq.gui.scientific_spinbox"] = scientific_spinbox
sys.modules["artiq.gui.tools"] = gui_tools
sys.modules["artiq.language"] = language
sys.modules["artiq.language.units"] = units
sys.modules["oitg"] = oitg
sys.modules["oitg.fitting"] = fitting
sys.modules["ndscan.utils"] = ndscan_utils
sys.modules["ndscan.dashboard.param_tree_dialog"] = param_tree_dialog

from ndscan.dashboard.scan_options import (
    CentreSpanScanOption,
    ExpandingScanOption,
    FixedScanOption,
    MinMaxScanOption,
    OptimizeAxisOption,
    SyncValue,
)


class DashboardOptionCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _build_option(self, option):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        option.build_ui(layout)
        container.setLayout(layout)
        self.addCleanup(container.deleteLater)
        return option

    def _make_schema(self, default="1.5", *, min=-2.0, max=3.0, step=0.1):
        return {
            "fqn": "test.param",
            "type": "float",
            "default": default,
            "spec": {
                "min": min,
                "max": max,
                "scale": 1.0,
                "step": step,
                "is_scannable": True,
            },
        }

    def _make_option(self, option_cls, default="1.5", *, min=-2.0, max=3.0, step=0.1):
        schema = self._make_schema(default, min=min, max=max, step=step)
        return self._build_option(option_cls(schema, "*"))


class OptimizeAxisOptionCase(DashboardOptionCase):
    def _make_option(self, default="1.5"):
        return super()._make_option(OptimizeAxisOption, default)

    def test_defaults_come_from_param_schema(self):
        option = self._make_option()
        option.read_sync_values({})

        self.assertEqual(option.box_min.value(), -2.0)
        self.assertEqual(option.box_initial.value(), 1.5)
        self.assertEqual(option.box_max.value(), 3.0)

    def test_sync_only_keeps_current_value_for_optimise(self):
        option = self._make_option()
        option.read_sync_values(
            {
                SyncValue.lower: -1.0,
                SyncValue.initial: 0.25,
                SyncValue.upper: 2.0,
            }
        )

        self.assertEqual(option.box_min.value(), -2.0)
        self.assertEqual(option.box_initial.value(), 0.25)
        self.assertEqual(option.box_max.value(), 3.0)

    def test_explicit_optimise_parameter_values_override_schema_defaults(self):
        option = self._make_option()
        option.attempt_read_from_optimise_parameter(
            {
                "min": -1.0,
                "initial": 0.25,
                "max": 2.0,
            }
        )

        self.assertEqual(option.box_min.value(), -1.0)
        self.assertEqual(option.box_initial.value(), 0.25)
        self.assertEqual(option.box_max.value(), 2.0)

    def test_missing_optimise_fields_fall_back_to_param_schema(self):
        option = self._make_option()
        option.attempt_read_from_optimise_parameter({})

        self.assertEqual(option.box_min.value(), -2.0)
        self.assertEqual(option.box_initial.value(), 1.5)
        self.assertEqual(option.box_max.value(), 3.0)


class ScanAxisOptionCase(DashboardOptionCase):
    def test_min_max_defaults_come_from_param_schema(self):
        option = self._make_option(MinMaxScanOption)
        option.read_sync_values({})

        self.assertEqual(option.box_start.value(), -2.0)
        self.assertEqual(option.box_stop.value(), 3.0)

    def test_centre_span_defaults_come_from_param_schema(self):
        option = self._make_option(CentreSpanScanOption, default="0.0", min=-2.0, max=2.0)
        option.read_sync_values({})

        self.assertEqual(option.box_centre.value(), 0.0)
        self.assertEqual(option.box_half_span.value(), 2.0)

    def test_minmax_alone_does_not_change_current_value(self):
        sync_values = {}

        minmax = self._make_option(MinMaxScanOption)
        minmax.box_start.setValue(-1.0)
        minmax.box_stop.setValue(3.0)
        minmax.write_sync_values(sync_values)

        centred = self._make_option(CentreSpanScanOption)
        centred.read_sync_values(sync_values)

        self.assertEqual(centred.box_centre.value(), 1.5)
        self.assertEqual(centred.box_half_span.value(), 1.5)

    def test_expanding_defaults_use_param_default_and_step(self):
        option = self._make_option(ExpandingScanOption, default="1.5", step=0.25)
        option.read_sync_values({})

        self.assertEqual(option.box_centre.value(), 1.5)
        self.assertEqual(option.box_spacing.value(), 0.25)

    def test_missing_axis_fields_fall_back_to_param_schema(self):
        minmax = self._make_option(MinMaxScanOption)
        self.assertTrue(minmax.attempt_read_from_axis({"type": "linear", "range": {}}))
        self.assertEqual(minmax.box_start.value(), -2.0)
        self.assertEqual(minmax.box_stop.value(), 3.0)

        centred = self._make_option(
            CentreSpanScanOption, default="0.0", min=-2.0, max=2.0
        )
        self.assertTrue(
            centred.attempt_read_from_axis({"type": "centre_span", "range": {}})
        )
        self.assertEqual(centred.box_centre.value(), 0.0)
        self.assertEqual(centred.box_half_span.value(), 2.0)

        expanding = self._make_option(ExpandingScanOption, default="1.5", step=0.25)
        self.assertTrue(
            expanding.attempt_read_from_axis({"type": "expanding", "range": {}})
        )
        self.assertEqual(expanding.box_centre.value(), 1.5)
        self.assertEqual(expanding.box_spacing.value(), 0.25)

    def test_fixed_value_becomes_current_initial_for_optimise(self):
        sync_values = {}

        fixed = self._make_option(FixedScanOption, default="0.0", min=-10.0, max=10.0)
        fixed.box.setValue(2.5)
        fixed.write_sync_values(sync_values)

        optimise = self._make_option(
            OptimizeAxisOption, default="0.0", min=-10.0, max=10.0
        )
        optimise.read_sync_values(sync_values)

        self.assertEqual(optimise.box_initial.value(), 2.5)

    def test_minmax_does_not_override_existing_current_value(self):
        sync_values = {}

        fixed = self._make_option(FixedScanOption, default="0.0", min=-10.0, max=10.0)
        fixed.box.setValue(2.5)
        fixed.write_sync_values(sync_values)

        minmax = self._make_option(MinMaxScanOption, default="0.0", min=-10.0, max=10.0)
        minmax.read_sync_values(sync_values)
        minmax.box_start.setValue(-1.0)
        minmax.box_stop.setValue(3.0)
        minmax.write_sync_values(sync_values)

        optimise = self._make_option(
            OptimizeAxisOption, default="0.0", min=-10.0, max=10.0
        )
        optimise.read_sync_values(sync_values)

        self.assertEqual(optimise.box_initial.value(), 2.5)

    def test_most_recent_centre_overrides_stale_optimise_initial(self):
        sync_values = {}

        optimise = self._make_option(
            OptimizeAxisOption, default="0.0", min=-10.0, max=10.0
        )
        optimise.box_initial.setValue(4.0)
        optimise.write_sync_values(sync_values)

        centred = self._make_option(
            CentreSpanScanOption, default="0.0", min=-10.0, max=10.0
        )
        centred.read_sync_values(sync_values)
        centred.box_centre.setValue(1.0)
        centred.write_sync_values(sync_values)

        optimise_again = self._make_option(
            OptimizeAxisOption, default="0.0", min=-10.0, max=10.0
        )
        optimise_again.read_sync_values(sync_values)

        self.assertEqual(optimise_again.box_initial.value(), 1.0)
