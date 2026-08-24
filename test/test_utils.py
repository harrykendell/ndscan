import unittest
from itertools import permutations

from ndscan.experiment.optimize import default_optimise_params
from ndscan.utils import (
    merge_ndscan_params,
    shorten_to_unambiguous_suffixes,
    strip_prefix,
    strip_suffix,
)


class StripTest(unittest.TestCase):
    def test_strip_prefix(self):
        self.assertEqual(strip_prefix("foo_bar", "foo_"), "bar")
        self.assertEqual(strip_prefix("foo_bar", "_bar"), "foo_bar")

    def test_strip_suffix(self):
        self.assertEqual(strip_suffix("foo_bar", "foo_"), "foo_bar")
        self.assertEqual(strip_suffix("foo_bar", "_bar"), "foo")


class ShortenTest(unittest.TestCase):
    def test_shorten(self):
        def shorten_at_slash(fqns):
            return shorten_to_unambiguous_suffixes(
                fqns, lambda fqn, n: "/".join(fqn.split("/")[-n:])
            )

        def test(expected):
            # Test all orderings.
            for keys in permutations(expected.keys()):
                self.assertEqual(shorten_at_slash(keys), expected)

        test({})

        test({"foo": "foo"})
        test({"": "", "foo/bar": "foo/bar", "foo/baz": "baz", "baz/bar": "baz/bar"})

        test({"a1/b": "a1/b", "a2/b": "a2/b"})
        test({"a1/b/c": "a1/b/c", "a2/b/c": "a2/b/c"})
        test({"a1/b/c/d": "a1/b/c/d", "a2/b/c/d": "a2/b/c/d"})
        test({"a1/b/c/d/e": "a1/b/c/d/e", "a2/b/c/d/e": "a2/b/c/d/e"})

        test({"bar": "bar", "foo/bar": "foo/bar"})

        # Test repeated fqns.
        with self.assertRaises(ValueError):
            shorten_at_slash(["foo/bar", "foo/bar"])


class MergeNdscanParamsTest(unittest.TestCase):
    def setUp(self):
        self.defaults = {
            "execution_mode": "scan",
            "scan": {"num_repeats": 1, "no_axes_mode": "single"},
            "optimise": default_optimise_params(),
            "overrides": {},
        }

    def test_none_state_uses_defaults(self):
        merged = merge_ndscan_params(self.defaults, None)
        self.assertEqual(merged, self.defaults)
        self.assertIsNot(merged, self.defaults)

    def test_merges_nested_saved_state_with_new_defaults(self):
        merged = merge_ndscan_params(
            self.defaults,
            {
                "execution_mode": "optimise",
                "scan": {"num_repeats": 3},
                "optimise": {
                    "objective": {"channel": "channel_result"},
                    "max_evals": 500,
                },
            },
        )

        self.assertEqual(merged["execution_mode"], "optimise")
        self.assertEqual(merged["scan"], {"num_repeats": 3, "no_axes_mode": "single"})
        self.assertEqual(
            merged["optimise"]["objective"],
            {"channel": "channel_result", "direction": "min"},
        )
        self.assertEqual(
            merged["optimise"]["algorithm"], self.defaults["optimise"]["algorithm"]
        )
        self.assertEqual(merged["optimise"]["max_evals"], 500)
        self.assertEqual(merged["optimise"]["reference_normalisation"], "none")
        self.assertEqual(merged["optimise"]["reference_resample_interval"], 1)
        self.assertEqual(merged["optimise"]["algorithm"]["user_seed"], -1)
