import unittest

import numpy as np

from ndscan.experiment.annotations import (
    AnnotationContext,
    AnnotationValueRef,
    curve,
    curve_1d,
)
from ndscan.experiment.result_channels import OpaqueChannel


class CurveTestCase(unittest.TestCase):
    def setUp(self):
        self.context = AnnotationContext(
            None, lambda channel: channel.path, lambda _: True
        )

    def test_accepts_annotation_value_ref(self):
        reference = AnnotationValueRef("analysis_result", name="fit_time")

        annotation = curve_1d(
            "x",
            reference,
            OpaqueChannel("signal"),
            [1.0, 2.0],
        )

        self.assertEqual(
            annotation.describe(self.context)["coordinates"],
            {
                "x": {"kind": "analysis_result", "name": "fit_time"},
                "channel_signal": {"kind": "fixed", "value": [1.0, 2.0]},
            },
        )

    def test_accepts_result_channel_ref(self):
        reference = OpaqueChannel("fit_time")

        annotation = curve_1d(
            "x",
            reference,
            OpaqueChannel("signal"),
            np.array([1.0, 2.0]),
        )

        self.assertEqual(
            annotation.describe(self.context)["coordinates"],
            {
                "x": {"kind": "analysis_result", "name": "fit_time"},
                "channel_signal": {"kind": "fixed", "value": [1.0, 2.0]},
            },
        )

    def test_rejects_mismatched_literal_lengths(self):
        with self.assertRaisesRegex(ValueError, "previously had 2"):
            curve(
                {
                    "referenced": AnnotationValueRef("fixed", value=[]),
                    "x": [1.0, 2.0],
                    "y": [1.0],
                }
            )


if __name__ == "__main__":
    unittest.main()
