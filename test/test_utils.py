import unittest
from itertools import permutations

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
    def test_none_state_uses_defaults(self):
        default = {
            "execution_mode": "scan",
            "scan": {"num_repeats": 1},
            "optimise": {
                "objective": {"channel": "", "direction": "min"},
                "algorithm": {"kind": "nelder_mead", "max_evals": 100},
            },
            "overrides": {},
        }

        merged = merge_ndscan_params(default, None)
        self.assertEqual(merged, default)
        self.assertIsNot(merged, default)

    def test_state_overrides_and_nested_merge(self):
        default = {
            "execution_mode": "scan",
            "scan": {"num_repeats": 1, "no_axes_mode": "single"},
            "optimise": {
                "parameters": [],
                "objective": {"channel": "", "direction": "min"},
                "algorithm": {
                    "kind": "nelder_mead",
                    "max_evals": 100,
                    "xatol": 1e-3,
                    "fatol": 1e-3,
                },
            },
            "overrides": {},
        }
        state = {
            "execution_mode": "optimise",
            "scan": {"num_repeats": 3},
            "optimise": {
                "objective": {"channel": "channel_result"},
                "algorithm": {"max_evals": 500},
            },
        }

        merged = merge_ndscan_params(default, state)
        self.assertEqual(merged["execution_mode"], "optimise")
        self.assertEqual(
            merged["scan"], {"num_repeats": 3, "no_axes_mode": "single"}
        )
        self.assertEqual(
            merged["optimise"]["objective"],
            {"channel": "channel_result", "direction": "min"},
        )
        self.assertEqual(
            merged["optimise"]["algorithm"],
            {
                "kind": "nelder_mead",
                "max_evals": 500,
                "xatol": 1e-3,
                "fatol": 1e-3,
            },
        )
