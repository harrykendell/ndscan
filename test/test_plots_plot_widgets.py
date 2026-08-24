import unittest

from ndscan.plots.plot_widgets import MultiYAxisPlotItem


class _Rect:
    def contains(self, _position):
        return False

    def center(self):
        return "centre"


class _ViewBox:
    def mapFromScene(self, position):
        return "mapped-" + position

    def boundingRect(self):
        return _Rect()

    def wheelEvent(self, event):
        self.forwarded_event = event


class _Event:
    def __init__(self):
        self.accepted = None

    def scenePos(self):
        return "position"

    def delta(self):
        return 120

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


class WheelForwardingTest(unittest.TestCase):
    def test_plot_item_forwards_wheel_event_to_view_box(self):
        view_box = _ViewBox()
        item = type("Item", (), {"getViewBox": lambda _self: view_box})()
        source_event = _Event()

        MultiYAxisPlotItem.wheelEvent(item, source_event)

        forwarded = view_box.forwarded_event
        self.assertEqual(forwarded.delta(), 120)
        self.assertEqual(forwarded.pos(), "centre")
        forwarded.accept()
        self.assertTrue(source_event.accepted)
        forwarded.ignore()
        self.assertFalse(source_event.accepted)


if __name__ == "__main__":
    unittest.main()
