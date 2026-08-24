import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ndscan.plots.container_widgets import make_plot_for_dimensional_model
from ndscan.plots.optimise_Nd import LinearNDInterpolator, _DelaunayInterpolationLayer


class PlotSelectionTest(unittest.TestCase):
    def test_one_dimensional_optimisation_uses_standard_history_plot(self):
        model = SimpleNamespace(axes=[{}], execution_mode="optimise")
        with patch("ndscan.plots.container_widgets.XY1DPlotWidget") as widget:
            make_plot_for_dimensional_model(model)
        widget.assert_called_once_with(model)

    def test_multidimensional_optimisation_uses_irregular_plot(self):
        model = SimpleNamespace(axes=[{}, {}, {}], execution_mode="optimise")
        with patch("ndscan.plots.container_widgets.OptimisePlotWidget") as widget:
            make_plot_for_dimensional_model(model)
        widget.assert_called_once_with(model)

    def test_two_dimensional_scan_uses_upstream_image_plot(self):
        model = SimpleNamespace(axes=[{}, {}], execution_mode="scan")
        with patch("ndscan.plots.container_widgets.Image2DPlotWidget") as widget:
            make_plot_for_dimensional_model(model)
        widget.assert_called_once_with(model)


class IrregularInterpolationTest(unittest.TestCase):
    @unittest.skipIf(LinearNDInterpolator is None, "SciPy interpolation is optional")
    def test_interpolates_inside_sampled_triangle_and_preserves_samples(self):
        layer = _DelaunayInterpolationLayer(None)
        layer.reset((0.0, 1.0, 0.5), (0.0, 1.0, 0.5))
        layer.insert(0.0, 0.0, 0.0)
        layer.insert(1.0, 0.0, 1.0)
        layer.insert(0.0, 1.0, 1.0)

        result = layer.interpolate(np.full((3, 3), np.nan))

        self.assertEqual(result[0, 0], 0.0)
        self.assertEqual(result[2, 0], 1.0)
        self.assertEqual(result[0, 2], 1.0)
        self.assertAlmostEqual(result[1, 1], 1.0)
