import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ndscan.plots.container_widgets import make_plot_for_dimensional_model
from ndscan.plots.optimise_Nd import (
    LinearNDInterpolator,
    OptimisePlotWidget,
    _best_so_far_indices,
    _DelaunayInterpolationLayer,
    _image_data_for_display,
    _image_pixel_centres,
    _ImagePlot,
    _make_optimisation_image_item,
    _resolve_objective_channel_name,
)


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
    def test_image_item_matches_transposed_row_major_upload(self):
        with patch("ndscan.plots.optimise_Nd.ClickableImageItem") as image_item_cls:
            image_item = _make_optimisation_image_item()

        image_item_cls.assert_called_once_with(axisOrder="row-major")
        self.assertIs(image_item, image_item_cls.return_value)

        image_data = np.arange(6).reshape(2, 3)
        displayed = _image_data_for_display(image_data)
        self.assertEqual(displayed.shape, (3, 2))
        self.assertEqual(displayed[2, 1], image_data[1, 2])

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

    @unittest.skipIf(LinearNDInterpolator is None, "SciPy interpolation is optional")
    def test_interpolation_uses_displayed_pixel_centres(self):
        layer = _DelaunayInterpolationLayer(None)
        range_spec = (0.0, 1.0, 0.3)
        layer.reset(range_spec, range_spec)
        for x, y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
            layer.insert(x, y, x + 2 * y)

        result = layer.interpolate(np.full((4, 4), np.nan))
        centres = _image_pixel_centres(range_spec, 4)

        expected = centres[:, None] + 2 * centres[None, :]
        np.testing.assert_allclose(result, expected)

    def test_rewind_rebuilds_interpolation_from_visible_points(self):
        class Surface:
            def __init__(self):
                self.values_by_coord = {}

            def reset(self, _x_range, _y_range):
                self.values_by_coord.clear()

            def insert(self, x, y, z):
                self.values_by_coord[(x, y)] = z
                return x, y

            def interpolate(self, fallback):
                return fallback

            def redraw_markers(self, _updated_cell):
                pass

        class Colorbar:
            def setColorMap(self, _cmap):
                pass

            def setLevels(self, _levels):
                pass

        class ImageItem:
            def setImage(self, _image, autoLevels=False):
                pass

            def setRect(self, _rect):
                pass

        class CrosshairLabel:
            def set_image_data(self, *_args):
                pass

        plot = object.__new__(_ImagePlot)
        plot.image_item = ImageItem()
        plot.colorbar = Colorbar()
        plot.channels = {"objective": {"min": 0.0, "max": 10.0, "display_hints": {}}}
        plot.active_channel_name = "objective"
        plot.x_min, plot.x_max, plot.x_increment = -1.0, 2.0, 1.0
        plot.y_min, plot.y_max, plot.y_increment = -1.0, 2.0, 1.0
        plot.x_range = (-1.0, 2.0, 1.0)
        plot.y_range = (-1.0, 2.0, 1.0)
        plot.sample_data = np.full((4, 4), np.nan)
        plot.image_data = plot.sample_data
        plot.image_rect = None
        plot.num_shown = 4
        plot.current_z_limits = (0.0, 10.0)
        plot.averaging_enabled = False
        plot.averages_by_coords = {
            (0.0, 0.0): (0.0, 1),
            (1.0, 0.0): (1.0, 1),
            (0.0, 1.0): (2.0, 1),
            (1.0, 1.0): (3.0, 1),
        }
        plot.interpolated_surface = Surface()
        plot.interpolated_surface.values_by_coord.update(
            {coord: value for coord, (value, _) in plot.averages_by_coords.items()}
        )
        plot.z_crosshair_label = CrosshairLabel()

        plot.data_changed(
            {
                "axis_0": [0.0, 1.0, 0.0],
                "axis_1": [0.0, 0.0, 1.0],
                "channel_objective": [0.0, 1.0, 2.0],
            }
        )

        self.assertEqual(
            set(plot.interpolated_surface.values_by_coord),
            {(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)},
        )


class OptimisationConvergenceTest(unittest.TestCase):
    def test_legend_text_respects_pyqtgraph_theme_colours(self):
        self.assertEqual(OptimisePlotWidget._html_color("w"), "#ffffff")
        self.assertEqual(OptimisePlotWidget._html_color("d"), "#969696")

    def test_best_so_far_respects_objective_direction(self):
        values = [2.0, 1.0, 3.0, np.nan, 2.5]
        np.testing.assert_array_equal(
            _best_so_far_indices(values, "min"), [0, 1, 1, 1, 1]
        )
        np.testing.assert_array_equal(
            _best_so_far_indices(values, "max"), [0, 0, 2, 2, 2]
        )

    def test_objective_channel_is_resolved_by_path(self):
        model = SimpleNamespace(
            optimisation_objective={"channel": "fragment/objective", "direction": "min"}
        )
        channels = {
            "display_first": {"path": "fragment/display"},
            "objective": {"path": "fragment/objective"},
        }
        self.assertEqual(_resolve_objective_channel_name(model, channels), "objective")
