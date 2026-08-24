import unittest

from ndscan.results.arguments import summarise


class ArgumentSummaryTest(unittest.TestCase):
    def test_summarises_optimisation_state(self):
        schema = {
            "execution_mode": "optimise",
            "schemata": {
                "example.detuning": {
                    "description": "Detuning",
                    "spec": {"unit": "MHz", "scale": 1e6},
                }
            },
            "optimise": {
                "parameters": [
                    {
                        "fqn": "example.detuning",
                        "path": "*",
                        "min": -2e6,
                        "initial": 0.0,
                        "max": 3e6,
                    }
                ],
                "objective": {"channel": "counts", "direction": "max"},
                "algorithm": {"kind": "bayesian", "xatol": 0.01, "fatol": 2.0},
                "max_evals": 50,
                "num_repeats_per_point": 3,
                "averaging_method": "median",
                "reference_normalisation": "divide",
                "reference_resample_interval": 4,
                "skip_on_persistent_transitory_error": True,
            },
            "overrides": {},
        }

        result = summarise(schema)

        self.assertIn("Optimise settings\n=================", result)
        self.assertIn("Detuning (example.detuning@*)", result)
        self.assertIn("min=-2.0 MHz, initial=0.0 MHz, max=3.0 MHz", result)
        self.assertIn("Objective channel: counts", result)
        self.assertIn("Algorithm: bayesian", result)
        self.assertIn("Reference normalisation: divide", result)

    def test_scan_summary_remains_the_default(self):
        schema = {
            "scan": {"axes": [], "no_axes_mode": "single"},
            "overrides": {},
        }

        result = summarise(schema)

        self.assertIn("Scan settings\n=============", result)
        self.assertIn("No scan (mode: single)", result)


if __name__ == "__main__":
    unittest.main()
