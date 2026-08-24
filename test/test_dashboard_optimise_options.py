import unittest

from ndscan.dashboard.optimise_options import OptimiseOptions


class _ValueWidget:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class _TextWidget:
    def __init__(self, text):
        self._text = text

    def currentText(self):
        return self._text


class _CheckedWidget:
    def __init__(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


class OptimiseOptionsSchemaTest(unittest.TestCase):
    def test_serialises_current_optimisation_schema(self):
        options = object.__new__(OptimiseOptions)
        options._objective_paths = {"Counts": "counts"}
        options.objective_box = _TextWidget("Counts")
        options.direction_box = _TextWidget("Maximise")
        options._algorithm_kinds = {"Nelder-Mead": "nelder_mead"}
        options.algorithm_box = _TextWidget("Nelder-Mead")
        options.algorithm_parameter_widgets = {
            "nelder_mead": {
                "xatol": _ValueWidget(0.01),
                "fatol": _ValueWidget(2.0),
                "simplex_step_fraction": _ValueWidget(0.5),
                "user_seed": _ValueWidget(42),
            }
        }
        options.repeats_box = _ValueWidget(3)
        options._averaging_methods = {"Median": "median"}
        options.averaging_box = _TextWidget("Median")
        options.max_evals_box = _ValueWidget(80)
        options._reference_methods = {"Divide": "divide"}
        options.reference_box = _TextWidget("Divide")
        options.reference_interval_box = _ValueWidget(5)
        options.skip_errors_box = _CheckedWidget(True)
        params = {"optimise": {"parameters": [{"fqn": "example.x"}]}}

        options.write_to_params(params)

        self.assertEqual(params["optimise"]["parameters"], [{"fqn": "example.x"}])
        self.assertEqual(
            params["optimise"]["objective"],
            {"channel": "counts", "direction": "max"},
        )
        self.assertEqual(
            params["optimise"]["algorithm"],
            {
                "kind": "nelder_mead",
                "xatol": 0.01,
                "fatol": 2.0,
                "simplex_step_fraction": 0.5,
                "user_seed": 42,
            },
        )
        self.assertEqual(params["optimise"]["num_repeats_per_point"], 3)
        self.assertEqual(params["optimise"]["averaging_method"], "median")
        self.assertEqual(params["optimise"]["max_evals"], 80)
        self.assertEqual(params["optimise"]["reference_normalisation"], "divide")
        self.assertEqual(params["optimise"]["reference_resample_interval"], 5)
        self.assertTrue(params["optimise"]["skip_on_persistent_transitory_error"])


if __name__ == "__main__":
    unittest.main()
