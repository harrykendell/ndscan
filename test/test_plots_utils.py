import unittest

from ndscan.plots.utils import *


class FragmentScanExpCase(unittest.TestCase):
    def test_find_neighbour_index(self):
        # This could be a good fit for property-based testing… For now, just check a
        # a few edge cases with a single input order.
        vals = [5, 3, 2, 4, 1]
        self.assertEqual(find_neighbour_index(vals, vals.index(1), -2), vals.index(1))
        self.assertEqual(find_neighbour_index(vals, vals.index(1), -1), vals.index(1))
        self.assertEqual(find_neighbour_index(vals, vals.index(1), 0), vals.index(1))
        self.assertEqual(find_neighbour_index(vals, vals.index(1), 1), vals.index(2))
        self.assertEqual(find_neighbour_index(vals, vals.index(1), 2), vals.index(3))

        self.assertEqual(find_neighbour_index(vals, vals.index(3), -2), vals.index(1))
        self.assertEqual(find_neighbour_index(vals, vals.index(3), -1), vals.index(2))
        self.assertEqual(find_neighbour_index(vals, vals.index(3), 0), vals.index(3))
        self.assertEqual(find_neighbour_index(vals, vals.index(3), 1), vals.index(4))
        self.assertEqual(find_neighbour_index(vals, vals.index(3), 2), vals.index(5))

        self.assertEqual(find_neighbour_index(vals, vals.index(5), -2), vals.index(3))
        self.assertEqual(find_neighbour_index(vals, vals.index(5), -1), vals.index(4))
        self.assertEqual(find_neighbour_index(vals, vals.index(5), 0), vals.index(5))
        self.assertEqual(find_neighbour_index(vals, vals.index(5), 1), vals.index(5))
        self.assertEqual(find_neighbour_index(vals, vals.index(5), 2), vals.index(5))


class FormatLabelValueTest(unittest.TestCase):
    def test_negative_value_uses_finite_precision(self):
        self.assertEqual(format_label_value(-1.234, 1.0, (0.0, 0.0), ""), "-1.234")

    def test_zero_and_non_finite_spans_do_not_raise(self):
        self.assertEqual(format_label_value(2.0, 0.0, (0.0, 0.0), " V"), "0 V")
        self.assertEqual(format_label_value(2.0, float("inf"), (0.0, 1.0), ""), "inf")

    def test_integer_labels_never_gain_fractional_digits(self):
        self.assertEqual(format_label_value(2, 1e-6, (0, 10), "", {"type": "int"}), "0")
