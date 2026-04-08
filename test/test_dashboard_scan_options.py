import os
import sys
import types
import unittest
from enum import Enum
from itertools import tee
from unittest import mock

if os.getenv("NDSCAN_SKIP_GUI"):
    raise unittest.SkipTest("NDSCAN_SKIP_GUI")

from ndscan._qt import QtCore, QtWidgets

import itertools

if not hasattr(itertools, "pairwise"):
    def pairwise(iterable):
        first, second = tee(iterable)
        next(second, None)
        return zip(first, second)

    itertools.pairwise = pairwise

pyon = types.ModuleType("sipyco.pyon")
pyon.decode = lambda value: eval(value, {})
sipyco = types.ModuleType("sipyco")
sipyco.pyon = pyon

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
ndscan_utils.merge_ndscan_params = lambda default_params, state_params: (
    default_params
    if state_params is None
    else {**default_params, **state_params}
)
ndscan_utils.shorten_to_unambiguous_suffixes = (
    lambda fqns, _formatter=None: {fqn: fqn for fqn in fqns}
)
entries = types.ModuleType("artiq.gui.entries")
entries.procdesc_to_entry = lambda procdesc: procdesc
fuzzy_select = types.ModuleType("artiq.gui.fuzzy_select")
fuzzy_select.FuzzySelectWidget = QtWidgets.QWidget
param_tree_dialog = types.ModuleType("ndscan.dashboard.param_tree_dialog")
param_tree_dialog.OverrideProvider = object
param_tree_dialog.OverrideStatus = object
param_tree_dialog.ParamTreeDialog = QtWidgets.QDialog

_stubbed_modules = {
    "artiq": artiq,
    "artiq.gui": gui,
    "artiq.gui.entries": entries,
    "artiq.gui.fuzzy_select": fuzzy_select,
    "artiq.gui.scientific_spinbox": scientific_spinbox,
    "artiq.gui.tools": gui_tools,
    "artiq.language": language,
    "artiq.language.units": units,
    "oitg": oitg,
    "oitg.fitting": fitting,
    "sipyco": sipyco,
    "sipyco.pyon": pyon,
    "ndscan.utils": ndscan_utils,
    "ndscan.dashboard.param_tree_dialog": param_tree_dialog,
}

with mock.patch.dict(sys.modules, _stubbed_modules):
    from ndscan.dashboard.argument_editor import ScanOptions as EditorScanOptions
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


class ScanOptionsCase(DashboardOptionCase):
    def test_algorithm_settings_use_single_container(self):
        options = EditorScanOptions({"optimise": {"algorithm": {}}})

        labels = [label for label, _widget in options.get_rows()]
        self.assertEqual(labels.count("Algorithm settings"), 1)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(options.algorithm_settings_container)
        container.setLayout(layout)
        container.setFixedWidth(700)
        container.show()
        self.addCleanup(container.deleteLater)

        self._app.processEvents()

        self.assertEqual(
            options.algorithm_box.mapTo(container, QtCore.QPoint(0, 0)).y(),
            options.max_evals_box.mapTo(container, QtCore.QPoint(0, 0)).y(),
        )
        self.assertIs(
            options.max_evals_box.parentWidget(), options.algorithm_settings_container
        )
        self.assertIs(
            options.algorithm_box.parentWidget(), options.algorithm_settings_container
        )
        self.assertIs(
            options.xatol_box.parentWidget(), options.algorithm_settings_container
        )

    def test_optimise_skip_wording_mentions_np_inf_cost(self):
        options = EditorScanOptions({"optimise": {"algorithm": {}}, "scan": {}})

        labels = [label for label, _widget in options.get_rows()]
        self.assertIn("Skip point if transitory errors persist", labels)
        self.assertIn(
            "Apply maximally bad objective result if transitory errors persist",
            labels,
        )
        self.assertIn(
            "skip it and attempt the next point",
            options.skip_persistently_failing_box.toolTip(),
        )
        self.assertIn(
            "np.inf",
            options.optimise_skip_persistently_failing_box.toolTip(),
        )

    def test_algorithm_settings_keep_fatol_and_xatol_on_one_row(self):
        options = EditorScanOptions({"optimise": {"algorithm": {}}})

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(options.algorithm_settings_container)
        container.setLayout(layout)
        container.setFixedWidth(480)
        container.show()
        self.addCleanup(container.deleteLater)

        self._app.processEvents()

        self.assertEqual(options.fatol_box.y(), options.xatol_box.y())
        self.assertGreaterEqual(
            options.fatol_box.width(), options.fatol_box.minimumSizeHint().width()
        )
        self.assertGreaterEqual(
            options.xatol_box.width(), options.xatol_box.minimumSizeHint().width()
        )

    def test_optimise_acquisition_defaults_and_serialisation(self):
        options = EditorScanOptions(
            {
                "execution_mode": "optimise",
                "optimise": {
                    "algorithm": {},
                    "num_repeats_per_point": 5,
                    "averaging_method": "median",
                },
            }
        )

        self.assertEqual(options.optimise_num_repeats_per_point_box.value(), 5)
        self.assertEqual(options.optimise_averaging_method_box.currentText(), "Median")

        params = {}
        options.write_to_params(params)

        self.assertEqual(params["execution_mode"], "optimise")
        self.assertEqual(params["optimise"]["num_repeats_per_point"], 5)
        self.assertEqual(params["optimise"]["averaging_method"], "median")


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
