import json
import unittest

from sipyco.sync_struct import Notifier

from ndscan.plots.model import Context
from ndscan.plots.model.subscriber import SubscriberRoot
from ndscan.utils import SCHEMA_REVISION, SCHEMA_REVISION_KEY


class SinglePointTest(unittest.TestCase):
    def setUp(self):
        self.context = Context()
        self.root = SubscriberRoot("ndscan.", self.context)
        self.datasets = Notifier(
            {
                "ndscan.axes": (False, "[]", {}),
                "ndscan.channels": (
                    False,
                    json.dumps(
                        {
                            "foo": {
                                "description": "Foo",
                                "path": "foo",
                                "type": "int",
                                "unit": "",
                            },
                            "bar": {
                                "description": "Bar",
                                "path": "foo",
                                "type": "int",
                                "unit": "",
                            },
                        }
                    ),
                    {},
                ),
                ("ndscan." + SCHEMA_REVISION_KEY): (False, SCHEMA_REVISION, {}),
            }
        )
        self.pending_mods = []
        self.datasets.publish = lambda a: self.pending_mods.append(a)

    def init(self):
        self.pending_mods = [
            {"action": "init", "struct": self.datasets.raw_view.copy()}
        ]
        self.sync()

    def sync(self):
        values = {k: v[1] for k, v in self.datasets.raw_view.items()}
        self.root.data_changed(values, self.pending_mods)
        self.pending_mods.clear()

    def test_new_point(self):
        self.init()
        self.datasets["ndscan.point.foo"] = (False, 42, {})
        self.datasets["ndscan.point.bar"] = (False, 23, {})
        self.datasets["ndscan.point_phase"] = (False, True, {})
        self.sync()
        self.assertEqual(self.root.get_model().get_point(), {"foo": 42, "bar": 23})

    def test_halfway(self):
        self.datasets["ndscan.point.foo"] = (False, 42, {})
        self.init()

        # No complete point yet.
        self.assertIsNone(self.root.get_model().get_point())

        self.datasets["ndscan.point.bar"] = (False, 23, {})
        self.datasets["ndscan.point_phase"] = (False, True, {})
        self.sync()
        self.assertEqual(self.root.get_model().get_point(), {"foo": 42, "bar": 23})

    def test_one_and_a_half(self):
        self.datasets["ndscan.point.foo"] = (False, 42, {})
        self.init()

        # No complete point yet.
        self.assertIsNone(self.root.get_model().get_point())

        self.datasets["ndscan.point.bar"] = (False, 23, {})
        self.datasets["ndscan.point_phase"] = (False, True, {})

        # Already write foo value of next point.
        self.datasets["ndscan.point.foo"] = (False, 0, {})
        self.sync()

        # Foo should still be the old value.
        self.assertEqual(self.root.get_model().get_point(), {"foo": 42, "bar": 23})

    def test_preexisting(self):
        self.datasets["ndscan.point.foo"] = (False, 42, {})
        self.datasets["ndscan.point.bar"] = (False, 42, {})
        self.datasets["ndscan.point_phase"] = (False, True, {})
        self.datasets["ndscan.point.foo"] = (False, 0, {})
        self.init()

        # Can't know whether point is complete (it indeed isn't).
        self.assertIsNone(self.root.get_model().get_point())

        self.datasets["ndscan.point.bar"] = (False, 1, {})
        self.datasets["ndscan.point_phase"] = (False, False, {})
        self.sync()

        self.assertEqual(self.root.get_model().get_point(), {"foo": 0, "bar": 1})

    def test_already_completed(self):
        self.datasets["ndscan.point.foo"] = (False, 42, {})
        self.datasets["ndscan.point.bar"] = (False, 23, {})
        self.datasets["ndscan.point_phase"] = (False, True, {})
        self.datasets["ndscan.completed"] = (False, True, {})
        self.init()
        self.assertEqual(self.root.get_model().get_point(), {"foo": 42, "bar": 23})


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.context = Context()
        self.root = SubscriberRoot("ndscan.", self.context)
        self.datasets = Notifier(
            {
                "ndscan.axes": (
                    False,
                    json.dumps(
                        [
                            {
                                "param": {
                                    "description": "Foo",
                                    "spec": {"unit": ""},
                                },
                                "path": "foo",
                            },
                            {
                                "param": {
                                    "description": "Bar",
                                    "spec": {"unit": ""},
                                },
                                "path": "bar",
                            },
                        ]
                    ),
                    {},
                ),
                "ndscan.channels": (
                    False,
                    json.dumps(
                        {
                            "foo": {
                                "description": "Foo",
                                "path": "foo",
                                "type": "float",
                                "unit": "",
                            },
                        }
                    ),
                    {},
                ),
                "ndscan.online_analyses": (False, "{}", {}),
                ("ndscan." + SCHEMA_REVISION_KEY): (False, SCHEMA_REVISION, {}),
            }
        )
        self.pending_mods = []
        self.datasets.publish = lambda a: self.pending_mods.append(a)

    def init(self):
        self.pending_mods = [
            {"action": "init", "struct": self.datasets.raw_view.copy()}
        ]
        self.sync()

    def sync(self):
        values = {k: v[1] for k, v in self.datasets.raw_view.items()}
        self.root.data_changed(values, self.pending_mods)
        self.pending_mods.clear()

    def test_is_optimising_known_when_model_emitted(self):
        self.datasets["ndscan.execution_mode"] = (False, "optimise", {})
        emitted_modes = []
        self.root.model_changed.connect(
            lambda model: emitted_modes.append(model.is_optimising())
        )

        self.init()

        self.assertEqual(emitted_modes, [True])
        self.assertTrue(self.root.get_model().is_optimising())

    def test_scan_mode_is_not_optimising(self):
        self.datasets["ndscan.execution_mode"] = (False, "scan", {})

        self.init()

        self.assertFalse(self.root.get_model().is_optimising())
