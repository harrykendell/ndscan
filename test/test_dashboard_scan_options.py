import unittest

from ndscan._qt import QtWidgets
from ndscan.dashboard.optimise_options import (
    _OPTIMISE_AXIS_FIELDS,
    OptimiseAxisOption,
)
from ndscan.dashboard.scan_options import (
    CentreSpanScanOption,
    FixedScanOption,
    NumericScanOption,
    SyncValue,
    list_scan_option_types,
)


class _ValueBox:
    def __init__(self, value=0.0):
        self._value = value

    def setValue(self, value):
        self._value = value

    def value(self):
        return self._value


def _schema(default="5.0", **spec):
    return {
        "fqn": "test.parameter",
        "default": default,
        "spec": {"scale": 1.0, **spec},
    }


class NumericDefaultsTest(unittest.TestCase):
    def test_unbounded_range_uses_default_and_finite_step(self):
        option = NumericScanOption(_schema(step=2.0), "*")

        self.assertEqual(option._default_numeric_param_value(), 5.0)
        self.assertEqual(option._default_numeric_range_values(), (3.0, 7.0))
        self.assertEqual(option._default_numeric_half_span_value(), 2.0)

    def test_finite_bounds_determine_range_around_default(self):
        option = NumericScanOption(_schema(min=0.0, max=10.0, step=0.5), "*")

        self.assertEqual(option._default_numeric_range_values(), (0.0, 10.0))
        self.assertEqual(option._default_numeric_centre_value(), 5.0)
        self.assertEqual(option._default_numeric_half_span_value(), 5.0)

    def test_missing_dataset_and_non_finite_step_have_safe_fallbacks(self):
        option = NumericScanOption(
            _schema("dataset('missing')", step=float("nan")), "*"
        )

        self.assertIsNone(option._default_numeric_param_value())
        self.assertEqual(option._default_numeric_step_value(), 1.0)
        self.assertEqual(option._default_numeric_range_values(), (0.0, 1.0))

    def test_fixed_and_centre_options_share_initial_value(self):
        fixed = FixedScanOption(_schema(), "*")
        fixed.box = _ValueBox(4.0)
        sync_values = {}
        fixed.write_sync_values(sync_values)

        self.assertEqual(sync_values[SyncValue.initial], 4.0)

        centred = CentreSpanScanOption(_schema(min=0.0, max=10.0), "*")
        centred.box_centre = _ValueBox()
        centred.box_half_span = _ValueBox()
        centred.box_points = _ValueBox(21)
        centred.read_sync_values(sync_values)

        self.assertEqual(centred.box_centre.value(), 4.0)
        self.assertEqual(centred.box_half_span.value(), 5.0)


class ScanOptionTooltipTest(unittest.TestCase):
    def test_all_current_upstream_option_types_have_explanatory_tooltips(self):
        for schema_type in ("string", "bool", "enum", "float"):
            with self.subTest(schema_type=schema_type):
                options = list_scan_option_types(schema_type, True)
                self.assertTrue(options)
                self.assertTrue(
                    all(option.option_tooltip for option in options.values())
                )


class OptimiseAxisDefaultsTest(unittest.TestCase):
    def test_axis_option_has_explanatory_tooltip(self):
        self.assertIn("initial value", OptimiseAxisOption.option_tooltip)

    def test_axis_fields_use_lower_initial_and_upper_icons(self):
        self.assertEqual(
            _OPTIMISE_AXIS_FIELDS,
            (
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
            ),
        )

    def test_optimise_axis_uses_parameter_default_as_initial_value(self):
        option = OptimiseAxisOption(_schema("6.0", min=0.0, max=10.0), "*")
        option.box_min = _ValueBox()
        option.box_initial = _ValueBox()
        option.box_max = _ValueBox()

        option.read_sync_values({})
        self.assertEqual(option.box_min.value(), 0.0)
        self.assertEqual(option.box_initial.value(), 6.0)
        self.assertEqual(option.box_max.value(), 10.0)

        option.read_sync_values({SyncValue.initial: 7.0})
        self.assertEqual(option.box_initial.value(), 7.0)

    def test_partial_saved_optimise_axis_uses_current_defaults(self):
        option = OptimiseAxisOption(_schema("6.0", min=0.0, max=10.0), "*")
        option.box_min = _ValueBox()
        option.box_initial = _ValueBox()
        option.box_max = _ValueBox()

        option.attempt_read_from_optimise_parameter({"initial": 8.0})
        self.assertEqual(option.box_min.value(), 0.0)
        self.assertEqual(option.box_initial.value(), 8.0)
        self.assertEqual(option.box_max.value(), 10.0)


if __name__ == "__main__":
    unittest.main()
