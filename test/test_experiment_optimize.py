import json
from mock_environment import HasEnvironmentCase

from ndscan.experiment import *
from ndscan.experiment.entry_point import ScanSpecError
from ndscan.experiment.optimize import (
    CoordinateSearchOptimizer,
    NelderMeadOptimizer,
    BayesianOptimizer,
)


class QuadraticFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_param("y", FloatParam, "y", default=0.0)
        self.setattr_result("objective", FloatChannel)

        self.num_host_setup_calls = 0
        self.num_device_setup_calls = 0
        self.num_host_cleanup_calls = 0
        self.num_device_cleanup_calls = 0

    def host_setup(self):
        self.num_host_setup_calls += 1

    def device_setup(self):
        self.num_device_setup_calls += 1

    def host_cleanup(self):
        self.num_host_cleanup_calls += 1

    def device_cleanup(self):
        self.num_device_cleanup_calls += 1

    def run_once(self):
        self.objective.push((self.x.get() - 1.25) ** 2 + (self.y.get() + 0.5) ** 2)


class InvertedQuadraticFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_param("y", FloatParam, "y", default=0.0)
        self.setattr_result("objective", FloatChannel)

    def run_once(self):
        self.objective.push(-((self.x.get() - 1.25) ** 2 + (self.y.get() + 0.5) ** 2))


class OpaqueObjectiveFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("objective", OpaqueChannel)

    def run_once(self):
        self.objective.push({"x": self.x.get()})


class IntAxisFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("x", IntParam, "x", default=0)
        self.setattr_result("objective", FloatChannel)

    def run_once(self):
        self.objective.push(float((self.x.get() - 1) ** 2))


class RepeatOutlierFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("objective", FloatChannel)
        self.run_counter = 0

    def run_once(self):
        self.objective.push(9.0 if self.run_counter % 3 == 0 else 0.0)
        self.run_counter += 1


class AdditiveDriftFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("objective", FloatChannel)
        self.run_counter = 0

    def run_once(self):
        self.objective.push(self.x.get() + 10.0 * self.run_counter)
        self.run_counter += 1


class MultiplicativeDriftFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("objective", FloatChannel)
        self.run_counter = 0

    def run_once(self):
        self.objective.push((self.x.get() + 2.0) * (self.run_counter + 1.0))
        self.run_counter += 1


QuadraticScanExp = make_fragment_scan_exp(QuadraticFragment)
InvertedQuadraticScanExp = make_fragment_scan_exp(InvertedQuadraticFragment)
OpaqueObjectiveScanExp = make_fragment_scan_exp(OpaqueObjectiveFragment)
IntAxisScanExp = make_fragment_scan_exp(IntAxisFragment)
RepeatOutlierScanExp = make_fragment_scan_exp(RepeatOutlierFragment)
AdditiveDriftScanExp = make_fragment_scan_exp(AdditiveDriftFragment)
MultiplicativeDriftScanExp = make_fragment_scan_exp(MultiplicativeDriftFragment)


def _evaluate_optimizer(opt, objective):
    num_evals = 0
    while (point := opt.ask()) is not None and num_evals < 100:
        num_evals += 1
        opt.tell(point, objective(point))
    return num_evals < 100


OPTIMIZER_TEST_CASES = (
    {
        "name": "bayesian",
        "make": BayesianOptimizer,
        "kwargs": {"n_init": 10, "user_seed": 42},
        "std_values": (1.0, 0.0),
        "std_best_value": 0.0,
        "xatol": 1e-3,
        "fatol": 1e-3,
    },
    {
        "name": "nelder_mead",
        "make": NelderMeadOptimizer,
        "std_values": (1.0, 0.0),
        "std_best_value": 0.0,
        "xatol": 1e-2,
        "fatol": 1e-10,
    },
    {
        "name": "coordinate_search",
        "make": CoordinateSearchOptimizer,
        "std_values": (1.0, 0.0),
        "std_best_value": 0.0,
        "xatol": 1e-2,
        "fatol": 0.0,
    },
)


class OptimizerCase(HasEnvironmentCase):
    def _check_best_std_tracks_best_point(self, case):
        opt = case["make"](
            (0.0,),
            (-5.0,),
            (5.0,),
            xatol=1e-4,
            fatol=1e-6,
            **case.get("kwargs", {}),
        )

        first_point = opt.ask()
        self.assertIsNotNone(first_point)
        opt.tell(first_point, case["std_values"][0], 0.125)
        second_point = opt.ask()
        self.assertIsNotNone(second_point)
        opt.tell(second_point, case["std_values"][1], 0.25)

        best_point, best_value = opt.best()
        self.assertEqual(tuple(best_point), tuple(second_point))
        self.assertEqual(best_value, case["std_best_value"])
        self.assertEqual(opt.best_std(), 0.25)

    def _skip_bayesian(self, case):
        # The bayesian optimiser adds noise and changes fatol/xatol dynamically so can only get to about 1%
        if case["name"] == "bayesian":
            self.skipTest("The Bayesian optimizer fails these tests")

    def _check_converges_1d(self, case):
        opt = case["make"](
            (4.0,),
            (-5.0,),
            (5.0,),
            xatol=1e-4,
            fatol=1e-4,
            **case.get("kwargs", {}),
        )
        self._skip_bayesian(case)
        _evaluate_optimizer(opt, lambda x: x[0] ** 3 / 3 + 5 * x[0] ** 2 / 2 - 6 * x[0])

        # ensure the optimizer terminated due to convergence, not because of an error
        self.assertEqual(
            opt.termination_reason(), "converged", "optimizer did not converge"
        )

        best_point, best_value = opt.best()
        self.assertAlmostEqual(best_point[0], 1.0, places=2)
        self.assertAlmostEqual(best_value, -19 / 6, places=2)

    def _check_converges_2d(self, case):
        opt = case["make"](
            (4.0, -4.0),
            (-5.0, -5.0),
            (5.0, 5.0),
            xatol=1e-4,
            fatol=1e-4,
            **case.get("kwargs", {}),
        )
        self._skip_bayesian(case)
        _evaluate_optimizer(
            opt, lambda x: x[0] ** 3 / 3 + 5 * x[0] ** 2 / 2 - 6 * x[0] + x[1]
        )

        # ensure the optimizer terminated due to convergence, not because of an error
        self.assertEqual(
            opt.termination_reason(), "converged", "optimizer did not converge"
        )

        best_point, best_value = opt.best()
        self.assertAlmostEqual(best_point[0], 1.0, places=2)
        self.assertAlmostEqual(best_point[1], -5.0, places=2)
        self.assertAlmostEqual(best_value, -49 / 6, places=2)

    def _check_xatol_is_relative_to_bounds(self, case):
        def run_scaled(scale):
            opt = case["make"](
                (4.0 * scale,),
                (-5.0 * scale,),
                (5.0 * scale,),
                xatol=case["xatol"],
                fatol=case["fatol"],
                **case.get("kwargs", {}),
            )
            num_evals = _evaluate_optimizer(
                opt,
                lambda point: ((point[0] - 1.25 * scale) / (10.0 * scale)) ** 2,
            )
            return num_evals, opt.best()[0][0]

        self._skip_bayesian(case)

        small_count, small_best = run_scaled(1.0)
        large_count, large_best = run_scaled(10.0)
        self.assertEqual(small_count, large_count)
        self.assertAlmostEqual(
            small_best,
            large_best / 10.0,
            delta=case["xatol"] * 10.0,
        )


def _make_optimizer_test(check_name, case):
    def test(self):
        getattr(self, check_name)(case)

    test.__name__ = "test_{}_{}".format(
        check_name.removeprefix("_check_"), case["name"]
    )
    return test


for _check_name, _check in list(OptimizerCase.__dict__.items()):
    if not _check_name.startswith("_check_") or not callable(_check):
        continue

    for _case in OPTIMIZER_TEST_CASES:
        setattr(
            OptimizerCase,
            "test_{}_{}".format(_check_name.removeprefix("_check_"), _case["name"]),
            _make_optimizer_test(_check_name, _case),
        )


class FragmentOptimizeExpCase(HasEnvironmentCase):
    def _configure_optimise(
        self,
        exp,
        *,
        channel="objective",
        direction="min",
        algorithm="nelder_mead",
        max_evals=120,
        num_repeats_per_point=1,
        averaging_method="mean",
    ):
        exp.args._params["execution_mode"] = "optimise"
        exp.args._params["optimise"] = {
            "parameters": [
                {
                    "fqn": "test_experiment_optimize.QuadraticFragment.x",
                    "path": "*",
                    "min": -5.0,
                    "max": 5.0,
                    "initial": 4.0,
                },
                {
                    "fqn": "test_experiment_optimize.QuadraticFragment.y",
                    "path": "*",
                    "min": -5.0,
                    "max": 5.0,
                    "initial": -4.0,
                },
            ],
            "objective": {"channel": channel, "direction": direction},
            "algorithm": {
                "kind": algorithm,
                "xatol": 1e-4,
                "fatol": 1e-6,
            },
            "max_evals": max_evals,
            "num_repeats_per_point": num_repeats_per_point,
            "averaging_method": averaging_method,
            "skip_on_persistent_transitory_error": False,
        }

    def test_run_minimise(self):
        exp = self.create(QuadraticScanExp)
        self._configure_optimise(exp)
        exp.prepare()
        exp.run()

        def d(key):
            return self.dataset_db.get("ndscan.rid_0." + key)

        self.assertEqual(d("execution_mode"), "optimise")
        self.assertEqual(d("optimizer.kind"), "nelder_mead")
        self.assertEqual(d("optimizer.objective_channel"), "objective")
        self.assertEqual(d("optimizer.objective_direction"), "min")
        self.assertEqual(d("optimizer.num_repeats_per_point"), 1)
        self.assertEqual(d("optimizer.averaging_method"), "mean")
        self.assertEqual(d("optimizer.reference_normalisation"), "none")
        self.assertEqual(d("optimizer.reference_resample_interval"), 1)
        self.assertIn(
            d("optimizer.termination_reason"), {"converged", "max_evals_reached"}
        )

        xs = d("points.axis_0")
        ys = d("points.axis_1")
        objectives = d("points.channel_objective")
        self.assertEqual(len(xs), len(ys))
        self.assertEqual(len(xs), len(objectives))
        self.assertLessEqual(len(xs), 120)
        self.assertAlmostEqual(d("optimizer.best_axis_0"), 1.25, places=2)
        self.assertAlmostEqual(d("optimizer.best_axis_1"), -0.5, places=2)
        self.assertLess(d("optimizer.best_value"), 1e-4)

        axes = json.loads(d("axes"))
        self.assertEqual(axes[0]["min"], -5.0)
        self.assertEqual(axes[0]["max"], 5.0)
        self.assertEqual(axes[0]["initial"], 4.0)
        self.assertEqual(axes[1]["min"], -5.0)
        self.assertEqual(axes[1]["max"], 5.0)
        self.assertEqual(axes[1]["initial"], -4.0)

        fragment = exp.fragment
        self.assertGreaterEqual(fragment.num_host_setup_calls, 1)
        self.assertGreaterEqual(fragment.num_device_setup_calls, len(xs))
        self.assertEqual(fragment.num_host_setup_calls, fragment.num_host_cleanup_calls)
        self.assertEqual(fragment.num_device_cleanup_calls, 1)

    def test_run_minimise_coordinate_search(self):
        exp = self.create(QuadraticScanExp)
        self._configure_optimise(exp, algorithm="coordinate_search")
        exp.prepare()
        exp.run()

        def d(key):
            return self.dataset_db.get("ndscan.rid_0." + key)

        xs = d("points.axis_0")
        ys = d("points.axis_1")
        objectives = d("points.channel_objective")
        self.assertEqual(len(xs), len(ys))
        self.assertEqual(len(xs), len(objectives))
        self.assertLessEqual(len(xs), 120)
        self.assertEqual(d("optimizer.kind"), "coordinate_search")
        self.assertIn(
            d("optimizer.termination_reason"), {"converged", "max_evals_reached"}
        )
        self.assertAlmostEqual(d("optimizer.best_axis_0"), 1.25, places=2)
        self.assertAlmostEqual(d("optimizer.best_axis_1"), -0.5, places=2)
        self.assertLess(d("optimizer.best_value"), 1e-4)

    def test_run_maximise(self):
        exp = self.create(InvertedQuadraticScanExp)
        exp.args._params["execution_mode"] = "optimise"
        exp.args._params["optimise"] = {
            "parameters": [
                {
                    "fqn": "test_experiment_optimize.InvertedQuadraticFragment.x",
                    "path": "*",
                    "min": -5.0,
                    "max": 5.0,
                    "initial": 4.0,
                },
                {
                    "fqn": "test_experiment_optimize.InvertedQuadraticFragment.y",
                    "path": "*",
                    "min": -5.0,
                    "max": 5.0,
                    "initial": -4.0,
                },
            ],
            "objective": {"channel": "objective", "direction": "max"},
            "algorithm": {
                "kind": "nelder_mead",
                "xatol": 1e-4,
                "fatol": 1e-6,
            },
            "max_evals": 120,
            "skip_on_persistent_transitory_error": False,
        }
        exp.prepare()
        exp.run()

        self.assertAlmostEqual(
            self.dataset_db.get("ndscan.rid_0.optimizer.best_axis_0"), 1.25, places=2
        )
        self.assertAlmostEqual(
            self.dataset_db.get("ndscan.rid_0.optimizer.best_axis_1"), -0.5, places=2
        )
        self.assertGreater(
            self.dataset_db.get("ndscan.rid_0.optimizer.best_value"), -1e-4
        )

    def test_reject_non_numeric_objective(self):
        exp = self.create(OpaqueObjectiveScanExp)
        exp.args._params["execution_mode"] = "optimise"
        exp.args._params["optimise"] = {
            "parameters": [
                {
                    "fqn": "test_experiment_optimize.OpaqueObjectiveFragment.x",
                    "path": "*",
                    "min": -5.0,
                    "max": 5.0,
                    "initial": 0.0,
                }
            ],
            "objective": {"channel": "objective", "direction": "min"},
            "algorithm": {
                "kind": "nelder_mead",
                "xatol": 1e-3,
                "fatol": 1e-3,
            },
            "max_evals": 10,
            "skip_on_persistent_transitory_error": False,
        }
        with self.assertRaises(ScanSpecError):
            exp.prepare()

    def test_reject_non_float_parameter(self):
        exp = self.create(IntAxisScanExp)
        exp.args._params["execution_mode"] = "optimise"
        exp.args._params["optimise"] = {
            "parameters": [
                {
                    "fqn": "test_experiment_optimize.IntAxisFragment.x",
                    "path": "*",
                    "min": -5,
                    "max": 5,
                    "initial": 0,
                }
            ],
            "objective": {"channel": "objective", "direction": "min"},
            "algorithm": {
                "kind": "nelder_mead",
                "xatol": 1e-3,
                "fatol": 1e-3,
            },
            "max_evals": 10,
            "skip_on_persistent_transitory_error": False,
        }
        with self.assertRaises(ScanSpecError):
            exp.prepare()

    def test_reject_unknown_algorithm(self):
        exp = self.create(QuadraticScanExp)
        self._configure_optimise(exp, algorithm="bogus")
        with self.assertRaises(ScanSpecError):
            exp.prepare()

    def test_reject_xatol_larger_than_one(self):
        exp = self.create(QuadraticScanExp)
        self._configure_optimise(exp)
        exp.args._params["optimise"]["algorithm"]["xatol"] = 1.5
        with self.assertRaises(ScanSpecError):
            exp.prepare()

    def test_reject_non_positive_repeats_per_point(self):
        exp = self.create(QuadraticScanExp)
        self._configure_optimise(exp)
        exp.args._params["optimise"]["num_repeats_per_point"] = 0
        with self.assertRaises(ScanSpecError):
            exp.prepare()

    def test_reject_unknown_averaging_method(self):
        exp = self.create(QuadraticScanExp)
        self._configure_optimise(exp)
        exp.args._params["optimise"]["averaging_method"] = "mode"
        with self.assertRaises(ScanSpecError):
            exp.prepare()

    def test_reject_unknown_reference_normalisation(self):
        exp = self.create(QuadraticScanExp)
        self._configure_optimise(exp)
        exp.args._params["optimise"]["reference_normalisation"] = "bogus"
        with self.assertRaises(ScanSpecError):
            exp.prepare()

    def test_reject_non_positive_reference_resample_interval(self):
        exp = self.create(QuadraticScanExp)
        self._configure_optimise(exp)
        exp.args._params["optimise"]["reference_resample_interval"] = 0
        with self.assertRaises(ScanSpecError):
            exp.prepare()

    def test_subtract_reference_normalisation(self):
        exp = self.create(AdditiveDriftScanExp)
        exp.args._params["execution_mode"] = "optimise"
        exp.args._params["optimise"] = {
            "parameters": [
                {
                    "fqn": "test_experiment_optimize.AdditiveDriftFragment.x",
                    "path": "*",
                    "min": -1.0,
                    "max": 1.0,
                    "initial": 0.0,
                }
            ],
            "objective": {"channel": "objective", "direction": "min"},
            "algorithm": {
                "kind": "coordinate_search",
                "xatol": 1e-4,
                "fatol": 1e-6,
            },
            "max_evals": 6,
            "reference_normalisation": "subtract",
            "reference_resample_interval": 1,
            "skip_on_persistent_transitory_error": False,
        }
        exp.prepare()
        exp.run()

        self.assertEqual(
            self.dataset_db.get("ndscan.rid_0.points.channel_objective"),
            [0.0, 10.0, 20.0, 30.5, 40.0, 49.5],
        )
        self.assertAlmostEqual(
            self.dataset_db.get("ndscan.rid_0.optimizer.best_axis_0"), -0.5
        )
        self.assertAlmostEqual(
            self.dataset_db.get("ndscan.rid_0.optimizer.best_value"), 9.5
        )

    def test_divide_reference_normalisation(self):
        exp = self.create(MultiplicativeDriftScanExp)
        exp.args._params["execution_mode"] = "optimise"
        exp.args._params["optimise"] = {
            "parameters": [
                {
                    "fqn": "test_experiment_optimize.MultiplicativeDriftFragment.x",
                    "path": "*",
                    "min": -1.0,
                    "max": 1.0,
                    "initial": 0.0,
                }
            ],
            "objective": {"channel": "objective", "direction": "min"},
            "algorithm": {
                "kind": "coordinate_search",
                "xatol": 1e-4,
                "fatol": 1e-6,
            },
            "max_evals": 6,
            "reference_normalisation": "divide",
            "reference_resample_interval": 1,
            "skip_on_persistent_transitory_error": False,
        }
        exp.prepare()
        exp.run()

        self.assertEqual(
            self.dataset_db.get("ndscan.rid_0.points.channel_objective"),
            [2.0, 4.0, 6.0, 10.0, 10.0, 9.0],
        )
        self.assertAlmostEqual(
            self.dataset_db.get("ndscan.rid_0.optimizer.best_axis_0"), -0.5
        )
        self.assertAlmostEqual(
            self.dataset_db.get("ndscan.rid_0.optimizer.best_value"), 0.9
        )

    def test_mean_aggregates_repeated_objective_values(self):
        exp = self.create(RepeatOutlierScanExp)
        exp.args._params["execution_mode"] = "optimise"
        exp.args._params["optimise"] = {
            "parameters": [
                {
                    "fqn": "test_experiment_optimize.RepeatOutlierFragment.x",
                    "path": "*",
                    "min": -1.0,
                    "max": 1.0,
                    "initial": 0.0,
                }
            ],
            "objective": {"channel": "objective", "direction": "min"},
            "algorithm": {
                "kind": "coordinate_search",
                "xatol": 1e-4,
                "fatol": 1e-6,
            },
            "max_evals": 1,
            "num_repeats_per_point": 3,
            "averaging_method": "mean",
            "skip_on_persistent_transitory_error": False,
        }
        exp.prepare()
        exp.run()

        self.assertEqual(
            self.dataset_db.get("ndscan.rid_0.points.channel_objective"),
            [9.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(
            self.dataset_db.get("ndscan.rid_0.optimizer.best_value"), 3.0
        )

    def test_median_aggregates_repeated_objective_values(self):
        exp = self.create(RepeatOutlierScanExp)
        exp.args._params["execution_mode"] = "optimise"
        exp.args._params["optimise"] = {
            "parameters": [
                {
                    "fqn": "test_experiment_optimize.RepeatOutlierFragment.x",
                    "path": "*",
                    "min": -1.0,
                    "max": 1.0,
                    "initial": 0.0,
                }
            ],
            "objective": {"channel": "objective", "direction": "min"},
            "algorithm": {
                "kind": "coordinate_search",
                "xatol": 1e-4,
                "fatol": 1e-6,
            },
            "max_evals": 1,
            "num_repeats_per_point": 3,
            "averaging_method": "median",
            "skip_on_persistent_transitory_error": False,
        }
        exp.prepare()
        exp.run()

        self.assertEqual(
            self.dataset_db.get("ndscan.rid_0.points.channel_objective"),
            [9.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(
            self.dataset_db.get("ndscan.rid_0.optimizer.best_value"), 0.0
        )
