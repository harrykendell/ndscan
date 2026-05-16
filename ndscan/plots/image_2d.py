"""Pseudocolor 2D plot for equidistant data."""

import logging
from itertools import chain, repeat
from typing import Any

import numpy as np
import pyqtgraph

try:
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import QhullError
except ImportError:
    LinearNDInterpolator = None
    QhullError = ValueError

from .._qt import QtCore, QtGui
from . import colormaps
from .cursor import CrosshairAxisLabel, CrosshairLabel, LabeledCrosshairCursor
from .model import ScanModel
from .model.select_point import SelectPointFromScanModel
from .model.slice import create_slice_roots
from .model.subscan import create_subscan_roots
from .plot_widgets import SliceableMenuPanesWidget, add_source_id_label
from .utils import (
    CONTRASTING_COLOR_TO_HIGHLIGHT,
    HIGHLIGHT_PEN,
    call_later,
    enum_to_numeric,
    extract_linked_datasets,
    extract_scalar_channels,
    find_neighbour_index,
    format_param_identity,
    get_axis_scaling_info,
    setup_axis_item,
    slice_data_along_axis,
)

logger = logging.getLogger(__name__)


def _calc_range_spec(preset_min, preset_max, preset_increment, data):
    sorted_data = np.unique(data).astype(float)

    lower = preset_min if preset_min is not None else sorted_data[0]
    upper = preset_max if preset_max is not None else sorted_data[-1]

    if preset_increment:
        increment = preset_increment
    elif len(sorted_data) > 1:
        increment = np.min(sorted_data[1:] - sorted_data[:-1])
    else:
        # Only one point on this (i.e. all data so far is from one row/column), and no
        # way to infer what the increment is going to be. To be able to still display
        # the data as it comes in, fall back on an arbitrary increment for now.
        #
        # If we have lower/upper limits, we can at least try to guess a reasonable order
        # of magnitude.
        if lower != upper:
            increment = (upper - lower) / 32
        else:
            increment = 1.0

    return lower, upper, increment


def _num_points_in_range(range_spec):
    min, max, increment = range_spec
    return int(np.rint((max - min) / increment + 1))


def _coords_to_indices(coords, range_spec):
    min, max, increment = range_spec
    return np.rint((np.array(coords) - min) / increment).astype(int)


def _axis_min(schema):
    return schema.get("min", schema["param"]["spec"].get("min"))


def _axis_max(schema):
    return schema.get("max", schema["param"]["spec"].get("max"))


def _axis_increment(schema):
    return schema.get("increment", schema["param"]["spec"].get("step"))


def _axis_bounds(schema) -> tuple[float, float] | None:
    lower = _axis_min(schema)
    upper = _axis_max(schema)
    if lower is None or upper is None:
        return None
    return lower, upper


def _axis_span(schema, values) -> float:
    bounds = _axis_bounds(schema)
    if bounds is not None and bounds[1] != bounds[0]:
        return abs(bounds[1] - bounds[0])
    if len(values) > 1:
        span = np.nanmax(values) - np.nanmin(values)
        if span:
            return abs(span)
    return 1.0


class CrosshairZDataLabel(CrosshairLabel):
    """Crosshair label for the z value of a 2D image"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_range = None
        self.y_range = None
        self.image_data = None
        self.z_limits = None

    def set_crosshair_info(
        self, unit_suffix: str, data_to_display_scale: float, _color
    ):
        """Update the unit/scale information of the underlying data.

        :param unit_suffix: The unit (including a leading space).
        :param data_to_display_scale: The scaling factor corresponding to the unit.
        :param _color: This parameter has no effect.
        """
        self.unit_suffix = unit_suffix
        self.data_to_display_scale = data_to_display_scale

    def set_image_data(
        self,
        image_data: np.ndarray,
        x_range: tuple[float, float, float],
        y_range: tuple[float, float, float],
        z_limits: tuple[float, float],
    ):
        """Update the underlying image data object and the data limits.

        :param image_data: 2D numpy array containing the data that is displayed.
        :param z_limits: The current colormap limits.
        """
        self.image_data = image_data
        self.x_range = x_range
        self.y_range = y_range
        self.z_limits = z_limits

    def update_coords(self, data_coords):
        if self.image_data is None:
            return
        z = np.nan

        x_idx = _coords_to_indices([data_coords.x()], self.x_range)[0]
        y_idx = _coords_to_indices([data_coords.y()], self.y_range)[0]
        shape = self.image_data.shape
        if (0 <= x_idx < shape[0]) and (0 <= y_idx < shape[1]):
            z = self.image_data[x_idx, y_idx]
        if np.isnan(z):
            self.set_visible(False)
        else:
            self.set_value(z, self.z_limits)


class ClickableImageItem(pyqtgraph.ImageItem):
    """An ImageItem that emits a signal when clicked."""

    sigClicked = QtCore.pyqtSignal(QtCore.QPointF)

    def __init__(self):
        # ndscan stores image data as [x, y], but pyqtgraph's default col-major path
        # swaps axes internally. With float images containing NaNs, pyqtgraph 0.13.x can
        # then apply a transparency mask with the pre-swap indices to the post-swap
        # image buffer. Use row-major input instead and pass a transposed contiguous
        # array from _ImagePlot.update().
        super().__init__(axisOrder="row-major")

    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.sigClicked.emit(event.pos())
            event.accept()


class _DelaunayInterpolationLayer:
    """Interpolate sampled points onto the displayed image grid.

    Optimiser scans generally visit irregular coordinates, so the equidistant image
    buffer is a display surface rather than a direct representation of acquired data.
    Delaunay triangulation gives a piecewise-linear surface inside the convex hull of
    measured points while keeping the actual evaluation locations explicit as markers.
    """

    def __init__(self, plot_item):
        self.plot_item = plot_item
        self.x_range: tuple[float, float, float] | None = None
        self.y_range: tuple[float, float, float] | None = None
        self.values_by_coord: dict[tuple[float, float], float] = {}
        self.marker_items = []

    def reset(
        self,
        x_range: tuple[float, float, float],
        y_range: tuple[float, float, float],
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.values_by_coord.clear()
        self._clear_items()

    def _clear_items(self):
        for item in self.marker_items:
            if item.scene() is not None:
                self.plot_item.removeItem(item)
        self.marker_items = []

    def insert(self, x: float, y: float, z: float):
        key = (float(x), float(y))
        self.values_by_coord[key] = float(z)
        return key

    @staticmethod
    def _marker_pen(updated: bool):
        pen = pyqtgraph.mkPen(
            CONTRASTING_COLOR_TO_HIGHLIGHT if updated else (30, 30, 30, 190),
            width=1.2 if updated else 0.8,
        )
        pen.setCosmetic(True)
        return pen

    def interpolate(self, fallback_data: np.ndarray) -> np.ndarray:
        if LinearNDInterpolator is None:
            return fallback_data
        if self.x_range is None or self.y_range is None:
            return fallback_data
        if len(self.values_by_coord) < 3:
            return fallback_data

        coords = np.array(list(self.values_by_coord.keys()), dtype=float)
        values = np.array(list(self.values_by_coord.values()), dtype=float)

        x_min, x_max, _ = self.x_range
        y_min, y_max, _ = self.y_range
        x_coords = np.linspace(x_min, x_max, fallback_data.shape[0])
        y_coords = np.linspace(y_min, y_max, fallback_data.shape[1])
        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="ij")

        try:
            interpolator = LinearNDInterpolator(coords, values, fill_value=np.nan)
            interpolated = interpolator(grid_x, grid_y)
        except (QhullError, ValueError):
            return fallback_data

        # Preserve exact evaluated pixels, including early points on the hull where the
        # triangulation can otherwise leave tiny NaN gaps due to floating point edges.
        x_indices = _coords_to_indices(coords[:, 0], self.x_range)
        y_indices = _coords_to_indices(coords[:, 1], self.y_range)
        for x_idx, y_idx, value in zip(x_indices, y_indices, values):
            if 0 <= x_idx < interpolated.shape[0] and 0 <= y_idx < interpolated.shape[1]:
                interpolated[x_idx, y_idx] = value

        return interpolated

    def redraw_markers(self, updated_cell):
        self._clear_items()
        marker_spots = []
        updated_marker_spots = []

        for coord, value in self.values_by_coord.items():
            if np.isnan(value):
                continue

            updated = coord == updated_cell
            spot = {
                "pos": coord,
                "size": 8 if updated else 5,
                "pen": self._marker_pen(updated),
                "brush": pyqtgraph.mkBrush(
                    CONTRASTING_COLOR_TO_HIGHLIGHT
                    if updated else (255, 255, 255, 185)
                ),
            }
            if updated:
                updated_marker_spots.append(spot)
            else:
                marker_spots.append(spot)

        if marker_spots:
            item = pyqtgraph.ScatterPlotItem(pxMode=True)
            item.setData(marker_spots)
            item.setZValue(3)
            self.plot_item.addItem(item, ignoreBounds=True)
            self.marker_items.append(item)

        if updated_marker_spots:
            item = pyqtgraph.ScatterPlotItem(pxMode=True)
            item.setData(updated_marker_spots)
            item.setZValue(4)
            self.plot_item.addItem(item, ignoreBounds=True)
            self.marker_items.append(item)


class _ImagePlot:
    def __init__(
        self,
        image_item: ClickableImageItem,
        colorbar: pyqtgraph.ColorBarItem,
        active_channel_name: str,
        x_min: float | None,
        x_max: float | None,
        x_increment: float | None,
        y_min: float | None,
        y_max: float | None,
        y_increment: float | None,
        channels: dict[str, dict],
    ):
        self.image_item = image_item
        self.colorbar = colorbar
        self.channels = channels

        self.x_min = x_min
        self.x_max = x_max
        self.x_increment = x_increment

        self.y_min = y_min
        self.y_max = y_max
        self.y_increment = y_increment

        self.points: dict[str, Any] | None = None
        self.num_shown = 0
        self.current_z_limits = None
        self.x_range = None
        self.y_range = None
        self.sample_data = None
        self.image_data = None
        self.interpolated_surface = None

        #: Whether to average points with the same coordinates.
        self.averaging_enabled = False
        #: Keeps track of the running average and the number of samples therein.
        self.averages_by_coords = dict[tuple[float, float], tuple[float, int]]()

        self.z_crosshair_label = CrosshairZDataLabel(self.image_item.getViewBox())

        self.activate_channel(active_channel_name)

    def activate_channel(self, channel_name: str):
        self.active_channel_name = channel_name

        channel = self.channels[channel_name]
        label = channel["description"]
        if not label:
            label = channel["path"].split("/")[-1]
        crosshair_info = setup_axis_item(
            self.colorbar.getAxis("right"),
            [(label, channel["path"], channel["type"], None, channel)],
        )
        # Update crosshair label.
        self.z_crosshair_label.set_crosshair_info(*crosshair_info[0])

        self._invalidate_current()
        self.update(self.averaging_enabled)

    def data_changed(self, points, invalidate_previous: bool = False):
        self.points = points
        if invalidate_previous:
            self._invalidate_current()
        self.update(self.averaging_enabled)

    def set_axis_ranges(
        self,
        x_min: float | None,
        x_max: float | None,
        x_increment: float | None,
        y_min: float | None,
        y_max: float | None,
        y_increment: float | None,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.x_increment = x_increment
        self.y_min = y_min
        self.y_max = y_max
        self.y_increment = y_increment
        self.x_range = None
        self.y_range = None
        self.sample_data = None
        self.image_data = None
        self._invalidate_current()

    def _invalidate_current(self):
        self.num_shown = 0
        self.current_z_limits = None
        self.averages_by_coords.clear()

    def _active_fixed_z_limits(self) -> tuple[float, float] | None:
        channel = self.channels[self.active_channel_name]
        if channel.get("min") is None:
            return None
        if channel.get("max") is None:
            return None
        return channel["min"], channel["max"]

    def update(self, averaging_enabled):
        if not self.points:
            return

        x_data = self.points["axis_0"]
        y_data = self.points["axis_1"]
        z_data = self.points["channel_" + self.active_channel_name]

        # Figure out how many complete data points we have, and whether there are any
        # not already shown.

        num_to_show = min(len(x_data), len(y_data), len(z_data))

        if (
            num_to_show == self.num_shown
            and averaging_enabled == self.averaging_enabled
        ):
            return
        num_skip = self.num_shown

        # Update running averages.
        for x, y, z in zip(
            x_data[num_skip:num_to_show],
            y_data[num_skip:num_to_show],
            z_data[num_skip:num_to_show],
        ):
            avg, num = self.averages_by_coords.get((x, y), (0.0, 0))
            num += 1
            avg += (z - avg) / num
            self.averages_by_coords[(x, y)] = (avg, num)

        # Determine range of x/y values to show and prepare image buffer accordingly if
        # it changed.
        x_range = _calc_range_spec(self.x_min, self.x_max, self.x_increment, x_data)
        y_range = _calc_range_spec(self.y_min, self.y_max, self.y_increment, y_data)

        if x_range != self.x_range or y_range != self.y_range:
            self.x_range = x_range
            self.y_range = y_range

            # TODO: Splat old data for progressively less blurry look on refining scans?
            self.sample_data = np.full(
                (_num_points_in_range(x_range), _num_points_in_range(y_range)), np.nan
            )
            self.image_data = self.sample_data

            self.image_rect = QtCore.QRectF(
                QtCore.QPointF(
                    x_range[0] - x_range[2] / 2, y_range[0] - y_range[2] / 2
                ),
                QtCore.QPointF(
                    x_range[1] + x_range[2] / 2, y_range[1] + y_range[2] / 2
                ),
            )

            num_skip = 0
            if self.interpolated_surface is not None:
                self.interpolated_surface.reset(x_range, y_range)

        # Revisit all coordinates in current image if averaging was toggled.
        if averaging_enabled != self.averaging_enabled:
            num_skip = 0

        if num_skip == 0:
            self.sample_data.fill(np.nan)
            if self.interpolated_surface is not None:
                self.interpolated_surface.reset(self.x_range, self.y_range)

        x_inds = _coords_to_indices(x_data[num_skip:num_to_show], self.x_range)
        y_inds = _coords_to_indices(y_data[num_skip:num_to_show], self.y_range)
        for i, (x_idx, y_idx) in enumerate(zip(x_inds, y_inds)):
            data_idx = num_skip + i
            coords, z = (x_data[data_idx], y_data[data_idx]), z_data[data_idx]
            self.sample_data[x_idx, y_idx] = (
                self.averages_by_coords[coords][0] if averaging_enabled else z
            )
        updated_cell = None
        if self.interpolated_surface is not None:
            start = num_skip
            for data_idx in range(start, num_to_show):
                z = self.averages_by_coords[(x_data[data_idx], y_data[data_idx])][0]
                if not averaging_enabled:
                    z = z_data[data_idx]
                updated_cell = self.interpolated_surface.insert(
                    x_data[data_idx], y_data[data_idx], z
                )

        cmap = colormaps.plasma
        channel = self.channels[self.active_channel_name]
        display_hints = channel.get("display_hints", {})
        if display_hints.get("coordinate_type", "") == "cyclic":
            cmap = colormaps.kovesi_c8
        self.colorbar.setColorMap(cmap)

        self.image_data = self.sample_data
        if self.interpolated_surface is not None:
            self.image_data = self.interpolated_surface.interpolate(self.sample_data)

        # Update z autorange if active.
        z_limits = self._active_fixed_z_limits()
        if z_limits is None:  # TODO: Provide manual override.
            z_limits = (np.nanmin(self.image_data), np.nanmax(self.image_data))
        self.current_z_limits = z_limits
        self.colorbar.setLevels(z_limits)

        self.image_item.setImage(
            np.ascontiguousarray(self.image_data.T), autoLevels=False
        )
        self.z_crosshair_label.set_image_data(
            self.image_data, self.x_range, self.y_range, self.current_z_limits
        )
        if self.interpolated_surface is not None:
            self.interpolated_surface.redraw_markers(updated_cell)
        if num_skip == 0:
            # Image size has changed, set plot item size accordingly.
            self.image_item.setRect(self.image_rect)

        self.num_shown = num_to_show
        self.averaging_enabled = averaging_enabled


class Image2DPlotWidget(SliceableMenuPanesWidget):
    def __init__(self, model: ScanModel):
        super().__init__()

        self.model = model

        self.model.channel_schemata_changed.connect(self._initialise_series)
        self.model.points_appended.connect(lambda p: self._update_points(p, False))
        self.model.points_rewritten.connect(lambda p: self._update_points(p, True))

        self.selected_point_model = SelectPointFromScanModel(self.model)

        self.data_names = []

        self.x_axis_idx = 0
        self.y_axis_idx = 1
        self._use_pca_projection = len(self.model.axes) > 2
        self.x_schema = self.model.axes[self.x_axis_idx]
        self.y_schema = self.model.axes[self.y_axis_idx]

        self.plot_item = self.add_pane()
        self.plot_item.showGrid(x=True, y=True)
        self.convergence_plot_item = None
        self.convergence_curves = []
        if self._use_pca_projection:
            self.convergence_plot_item = self.add_pane()
            self.convergence_plot_item.showGrid(x=True, y=True)
            self.convergence_plot_item.setLabel("bottom", "Evaluation")
            self.convergence_plot_item.setLabel(
                "left", "Best-so-far parameter value", units="axis range"
            )
            self.convergence_plot_item.setYRange(0, 1, padding=0)
            self.convergence_plot_item.addLegend(offset=(5, 5))
            self.layout.setRowPreferredHeight(0, 10000)
            self.layout.setRowPreferredHeight(1, 20000)
        self.plot: _ImagePlot | None = None
        self.crosshair = None
        self._highlighted_xy = (None, None)

        self.found_duplicate_coords = False
        self.unique_coords = set[tuple[float, float]]()
        self._pca_components = None
        self._pca_range_spec = None

        if (channels := self.model.get_channel_schemata()) is not None:
            call_later(lambda: self._initialise_series(channels))
            if (points := self.model.get_point_data()) is not None:
                call_later(lambda: self._update_points(points, False))

    def _initialise_series(self, channels):
        if self.plot is not None:
            self.plot_item.removeItem(self.plot.image_item)
            self.plot.image_item.destroyLater()
            self.plot = None

        try:
            self.data_names, _ = extract_scalar_channels(channels)
        except ValueError as e:
            self.error.emit(str(e))

        if not self.data_names:
            self.error.emit("No scalar result channels to display")

        self._setup_display_axes()

        def range_spec(schema):
            return _axis_min(schema), _axis_max(schema), _axis_increment(schema)

        image_item = ClickableImageItem()
        image_item.sigClicked.connect(self._point_clicked)

        self.plot_item.addItem(image_item)
        colorbar = self.plot_item.addColorBar(image_item, width=15.0, interactive=False)
        x_range_spec = (
            (None, None, None)
            if self._use_pca_projection
            else range_spec(self.x_schema)
        )
        if self._use_pca_projection:
            y_range_spec = (None, None, None)
        else:
            y_range_spec = range_spec(self.y_schema)

        self.plot = _ImagePlot(
            image_item,
            colorbar,
            self.data_names[0],
            *x_range_spec,
            *y_range_spec,
            channels,
        )
        self.plot.interpolated_surface = _DelaunayInterpolationLayer(self.plot_item)

        x_bounds = None if self._use_pca_projection else _axis_bounds(self.x_schema)
        y_bounds = None if self._use_pca_projection else _axis_bounds(self.y_schema)
        if x_bounds is not None and y_bounds is not None:
            self.plot_item.setRange(xRange=x_bounds, yRange=y_bounds, padding=0)

        highlight_pen = pyqtgraph.mkPen(**HIGHLIGHT_PEN)
        brush = pyqtgraph.mkBrush(CONTRASTING_COLOR_TO_HIGHLIGHT)
        self.highlight_point_item = pyqtgraph.ScatterPlotItem(
            pen=highlight_pen, brush=brush, size=8, symbol="o"
        )
        self.highlight_point_item.setZValue(2)  # Show above all other points.
        self.plot_item.addItem(self.highlight_point_item, ignoreBounds=True)

        if self._use_pca_projection:
            x_scaling_info = ("", 1)
            y_scaling_info = ("", 1)
        else:
            x_scaling_info = get_axis_scaling_info(self.x_schema["param"]["spec"])
            y_scaling_info = get_axis_scaling_info(self.y_schema["param"]["spec"])

        x_label = CrosshairAxisLabel(
            self.plot_item.getViewBox(), *x_scaling_info, is_x=True
        )
        y_label = CrosshairAxisLabel(
            self.plot_item.getViewBox(), *y_scaling_info, is_x=False
        )

        self.crosshair = LabeledCrosshairCursor(
            self, self.plot_item, [x_label, y_label, self.plot.z_crosshair_label]
        )

        add_source_id_label(self.plot_item.getViewBox(), self.model.context)

        self.subscan_roots = create_subscan_roots(self.selected_point_model)
        self.slice_roots = (
            {} if self._use_pca_projection
            else create_slice_roots(self.model, self.selected_point_model)
        )

        self.ready.emit()

    def _setup_display_axes(self):
        if self._use_pca_projection:
            self.plot_item.getAxis("bottom").setScale(1)
            self.plot_item.getAxis("bottom").setLabel("<b>Principal Component 1</b>")
            self.plot_item.getAxis("left").setScale(1)
            self.plot_item.getAxis("left").setLabel("<b>Principal Component 2</b>")
            return

        def setup_axis(schema, location):
            param = schema["param"]
            setup_axis_item(
                self.plot_item.getAxis(location),
                [
                    (
                        param["description"],
                        format_param_identity(schema),
                        param["type"],
                        None,
                        param["spec"],
                    )
                ],
            )

        setup_axis(self.x_schema, "bottom")
        setup_axis(self.y_schema, "left")

    def _update_points(self, points, invalidate):
        if self.plot:
            if invalidate:
                self.found_duplicate_coords = False
                self.unique_coords.clear()
            plot_points = self._project_points(points)
            if self._use_pca_projection:
                self.found_duplicate_coords = False
                self.unique_coords.clear()
                self._set_pca_ranges(plot_points)
                self._update_parameter_convergence(points)
                invalidate = True
            # If all points were unique so far, check if we have duplicates now.
            if not self.found_duplicate_coords:
                num_skip = len(self.unique_coords)
                for x in zip(
                    plot_points["axis_0"][num_skip:], plot_points["axis_1"][num_skip:]
                ):
                    if x in self.unique_coords:
                        self.found_duplicate_coords = True
                        break
                    else:
                        self.unique_coords.add(x)

            self.plot.data_changed(plot_points, invalidate_previous=invalidate)

    def _project_points(self, points):
        plot_points = {
            name: values
            for name, values in points.items()
            if not name.startswith("axis_")
        }
        if self._use_pca_projection:
            x, y = self._pca_project_points(points)
            plot_points["axis_0"] = x
            plot_points["axis_1"] = y
        else:
            plot_points["axis_0"] = list(points[f"axis_{self.x_axis_idx}"])
            plot_points["axis_1"] = list(points[f"axis_{self.y_axis_idx}"])
        if self._use_pca_projection:
            return plot_points

        if self.x_schema["param"]["type"] == "enum":
            plot_points["axis_0"] = enum_to_numeric(
                self.x_schema["param"]["spec"]["members"].keys(), plot_points["axis_0"]
            )
        if self.y_schema["param"]["type"] == "enum":
            plot_points["axis_1"] = enum_to_numeric(
                self.y_schema["param"]["spec"]["members"].keys(), plot_points["axis_1"]
            )
        return plot_points

    def _pca_project_points(self, points):
        axis_values = []
        num_points = None
        for axis_idx, schema in enumerate(self.model.axes):
            values = self._axis_values_numeric(
                points, axis_idx, len(points.get(f"axis_{axis_idx}", []))
            )
            num_points = (
                len(values) if num_points is None else min(num_points, len(values))
            )
            axis_values.append((schema, values))
        if not num_points:
            return [], []

        columns = []
        for schema, values in axis_values:
            values = np.asarray(values[:num_points], dtype=float)
            span = _axis_span(schema, values)
            columns.append(values / span)
        data = np.column_stack(columns)
        finite_rows = np.all(np.isfinite(data), axis=1)
        if not np.any(finite_rows):
            return [0.0] * num_points, [0.0] * num_points

        center = np.nanmean(data[finite_rows], axis=0)
        centered = np.nan_to_num(data - center)
        if np.count_nonzero(finite_rows) < 2:
            projected = np.zeros((num_points, 2))
        else:
            _, _, vh = np.linalg.svd(centered[finite_rows], full_matrices=False)
            components = np.zeros((2, data.shape[1]))
            components[: min(2, vh.shape[0])] = vh[:2]
            components = self._align_pca_components(components)
            self._pca_components = components
            projected = centered @ components.T
        return projected[:, 0].tolist(), projected[:, 1].tolist()

    def _align_pca_components(self, components):
        if self._pca_components is None:
            return components
        previous = self._pca_components
        aligned = components.copy()
        for idx in range(min(len(previous), len(aligned))):
            if np.dot(previous[idx], aligned[idx]) < 0:
                aligned[idx] *= -1
        return aligned

    def _set_pca_ranges(self, plot_points):
        x = np.asarray(plot_points["axis_0"], dtype=float)
        y = np.asarray(plot_points["axis_1"], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return False

        def axis_range(values):
            lower = float(np.nanmin(values[finite]))
            upper = float(np.nanmax(values[finite]))
            if lower == upper:
                lower -= 0.5
                upper += 0.5
            else:
                padding = 0.05 * (upper - lower)
                lower -= padding
                upper += padding
            increment = max((upper - lower) / 160, 1e-12)
            return lower, upper, increment

        range_spec = (*axis_range(x), *axis_range(y))
        if self._pca_range_spec == range_spec:
            return False
        self._pca_range_spec = range_spec
        self.plot.set_axis_ranges(*range_spec)
        return True

    def _axis_values_numeric(self, points, axis_idx: int, num_points: int):
        schema = self.model.axes[axis_idx]
        if schema["param"]["type"] == "enum":
            return enum_to_numeric(
                schema["param"]["spec"]["members"].keys(),
                points[f"axis_{axis_idx}"][:num_points],
            )
        return np.asarray(points[f"axis_{axis_idx}"][:num_points], dtype=float)

    def _update_parameter_convergence(self, points):
        if self.convergence_plot_item is None:
            return

        z_key = "channel_" + self.plot.active_channel_name
        z_data = np.asarray(points.get(z_key, []), dtype=float)
        num_points = min(
            [len(z_data)]
            + [
                len(points.get(f"axis_{idx}", []))
                for idx in range(len(self.model.axes))
            ]
        )
        if num_points == 0:
            return

        while len(self.convergence_curves) < len(self.model.axes):
            axis_idx = len(self.convergence_curves)
            color = pyqtgraph.intColor(axis_idx, hues=len(self.model.axes))
            curve = pyqtgraph.PlotDataItem(
                pen=pyqtgraph.mkPen(color, width=1.5),
                name=self._axis_description(axis_idx),
            )
            self.convergence_plot_item.addItem(curve)
            self.convergence_curves.append(curve)

        finite_z = np.isfinite(z_data[:num_points])
        if not np.any(finite_z):
            for curve in self.convergence_curves:
                curve.setData([], [])
            return

        best_indices = np.empty(num_points, dtype=int)
        best_idx = None
        best_z = np.inf
        for idx, z in enumerate(z_data[:num_points]):
            if np.isfinite(z) and (best_idx is None or z < best_z):
                best_idx = idx
                best_z = z
            best_indices[idx] = 0 if best_idx is None else best_idx

        evaluations = np.arange(num_points)
        for axis_idx, curve in enumerate(self.convergence_curves):
            values = np.asarray(
                self._axis_values_numeric(points, axis_idx, num_points), dtype=float
            )
            values = values[best_indices]
            curve.setData(evaluations, self._normalise_axis_values(axis_idx, values))

        self.convergence_plot_item.setXRange(0, max(num_points - 1, 1), padding=0)

    def _normalise_axis_values(self, axis_idx: int, values):
        schema = self.model.axes[axis_idx]
        bounds = _axis_bounds(schema)
        if bounds is None or bounds[0] == bounds[1]:
            lower = np.nanmin(values)
            upper = np.nanmax(values)
        else:
            lower, upper = bounds
        if lower == upper:
            return np.full_like(values, 0.5, dtype=float)
        return (values - lower) / (upper - lower)

    def _axis_description(self, axis_idx: int):
        param = self.model.axes[axis_idx]["param"]
        return param.get("description") or f"axis_{axis_idx}"

    def build_context_menu(self, pane_idx: int | None, builder):
        if self.model.context.is_online_master() and not self._use_pca_projection:
            x_datasets = extract_linked_datasets(self.x_schema["param"])
            y_datasets = extract_linked_datasets(self.y_schema["param"])
            for d, axis_idx in chain(
                zip(x_datasets, repeat(0)), zip(y_datasets, repeat(1))
            ):
                action = builder.append_action(f"Set '{d}' from crosshair")
                action.triggered.connect(
                    lambda *a, axis_idx=axis_idx, d=d: self._set_dataset_from_crosshair(
                        d, axis_idx
                    )
                )
            if len(x_datasets) == 1 and len(y_datasets) == 1:
                action = builder.append_action("Set both from crosshair")

                def set_both():
                    self._set_dataset_from_crosshair(x_datasets[0], 0)
                    self._set_dataset_from_crosshair(y_datasets[0], 1)

                action.triggered.connect(set_both)
        builder.ensure_separator()

        if self.found_duplicate_coords:
            action = builder.append_action("Average points with same coordinates")
            action.setCheckable(True)
            action.setChecked(self.plot.averaging_enabled)
            action.triggered.connect(
                lambda *a: self.plot.update(not self.plot.averaging_enabled)
            )
            builder.ensure_separator()

        self.channel_menu_group = QtGui.QActionGroup(self)
        for name in self.data_names:
            action = builder.append_action(name)
            action.setCheckable(True)
            action.setActionGroup(self.channel_menu_group)
            action.setChecked(name == self.plot.active_channel_name)
            action.triggered.connect(
                lambda *a, name=name: self._activate_channel(name)
            )

        builder.ensure_separator()

        super().build_context_menu(pane_idx, builder)
        builder.ensure_separator()

    def _activate_channel(self, name):
        self.plot.activate_channel(name)
        if self._use_pca_projection:
            points = self.model.get_point_data()
            if points is not None:
                self._update_points(points, True)

    def _set_dataset_from_crosshair(self, dataset, axis_idx):
        if not self.plot:
            logger.warning("Plot not initialised yet, ignoring set dataset request")
            return
        self.model.context.set_dataset(
            dataset, self.crosshair.labels[axis_idx].last_value
        )

    def _point_clicked(self, pos: QtCore.QPointF):
        """Callback for when `self.plot` is clicked.

        :param pos: Position of the click in `plot`'s coordinates.
            Here, these are in units of the point indices
        """
        if self._use_pca_projection:
            x = pos.x()
            y = pos.y()
        else:
            x_idx = np.floor(pos.x())
            y_idx = np.floor(pos.y())
            x = self.plot.x_range[0] + x_idx * self.plot.x_range[2]
            y = self.plot.y_range[0] + y_idx * self.plot.y_range[2]

        source_idx = self._xy_to_source_index(x, y)
        if source_idx is not None:
            self._highlight_point_at_index(source_idx)

    def keyPressEvent(self, event):
        """Handle arrow key presses to move the highlighted point."""
        key = event.key()
        is_left = key == QtCore.Qt.Key.Key_Left
        is_right = key == QtCore.Qt.Key.Key_Right
        is_up = key == QtCore.Qt.Key.Key_Up
        is_down = key == QtCore.Qt.Key.Key_Down

        if is_left or is_right:
            axis = 0
        elif is_up or is_down:
            axis = 1
        else:
            return super().keyPressEvent(event)

        step = -1 if is_left or is_down else 1
        neighbour_idx = self._get_highlighted_neighbour_index(axis, step)
        if neighbour_idx is not None:
            self._highlight_point_at_index(neighbour_idx)
        event.accept()

    def _highlight_point_at_index(self, source_idx: int | None):
        """Highlight the point at the given index of the source data."""
        self.selected_point_model.set_source_index(source_idx)

        if source_idx is None:
            self._highlighted_xy = (None, None)
            if self.highlight_point_item.parentItem():
                self.plot_item.removeItem(self.highlight_point_item)
            return

        x = self.plot.points["axis_0"][source_idx]
        y = self.plot.points["axis_1"][source_idx]

        if source_idx is None:
            return

        self.highlight_point_item.setData([x], [y], data=source_idx)
        self._highlighted_xy = (x, y)
        if not self.highlight_point_item.parentItem():
            self.plot_item.addItem(self.highlight_point_item, ignoreBounds=True)

    def _xy_to_source_index(self, x, y) -> int | None:
        """Get the source index of the point at the given coordinates."""
        x_source = self.plot.points["axis_0"]
        y_source = self.plot.points["axis_1"]

        if self._use_pca_projection:
            distances = (
                (np.asarray(x_source) - x) ** 2
                + (np.asarray(y_source) - y) ** 2
            )
            finite = np.flatnonzero(np.isfinite(distances))
            if not finite.size:
                return None
            return int(finite[np.nanargmin(distances[finite])])

        # KLUDGE: For some reason, the range spec/… calculation introduces more than the
        # possibly expected few ulp roundoff error; use a relatively loose tolerance of
        # 1e-14 (about 50 epsilon). A slightly cleaner solution would be to use an
        # absolute tolerance based on the minimal spacing of source data points.
        source_idx = np.flatnonzero(
            np.isclose(x_source, x, atol=0.0, rtol=1e-14)
            & np.isclose(y_source, y, atol=0.0, rtol=1e-14)
        )

        if source_idx.size == 0:
            return None

        return source_idx[0]

    def _get_highlighted_neighbour_index(self, axis: int, step: int) -> int | None:
        """Get the source index of the neighbouring point along the given axis."""
        if not self.plot or self._highlighted_xy == (None, None):
            return None

        source = self.plot.points

        sliced_idxs = slice_data_along_axis(
            source, self.selected_point_model.get_source_index(), axis
        )

        sliced_axis_name = f"axis_{axis}"
        fixed_axis_name = f"axis_{1 - axis}"

        slicing_axis_source = np.asarray(source[sliced_axis_name])
        fixed_axis_source = np.asarray(source[fixed_axis_name])

        sliced_axis_source = slicing_axis_source[sliced_idxs]

        # Coordinates of the point along and orthogonal to the slice axis.
        sliced_axis_coord = self._highlighted_xy[axis]
        fixed_axis_coord = self._highlighted_xy[1 - axis]

        # Find index of the highlighted point along the slice.
        try:
            idx_along_slice = np.flatnonzero(sliced_axis_source == sliced_axis_coord)[0]
        except IndexError:  # no matches found
            return None

        # Find coordinate of the neighbour along the slice.
        neighbour_coord = sliced_axis_source[
            find_neighbour_index(sliced_axis_source, idx_along_slice, step)
        ]
        # Map back to source index.
        return np.argmax(
            np.logical_and(
                slicing_axis_source == neighbour_coord,
                fixed_axis_source == fixed_axis_coord,
            )
        )
