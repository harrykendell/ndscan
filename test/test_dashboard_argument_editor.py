import unittest

from ndscan.dashboard.argument_editor import _set_execution_option_rows_visible
from ndscan.utils import ExecutionMode


class _TreeItem:
    def __init__(self):
        self.hidden = False

    def setHidden(self, hidden):
        self.hidden = hidden


class ExecutionOptionVisibilityTest(unittest.TestCase):
    def test_only_rows_for_initial_and_selected_mode_are_visible(self):
        scan_items = [_TreeItem(), _TreeItem()]
        optimise_items = [_TreeItem(), _TreeItem()]

        _set_execution_option_rows_visible(
            scan_items, optimise_items, ExecutionMode.scan.name
        )
        self.assertTrue(all(not item.hidden for item in scan_items))
        self.assertTrue(all(item.hidden for item in optimise_items))

        _set_execution_option_rows_visible(
            scan_items, optimise_items, ExecutionMode.optimise.name
        )
        self.assertTrue(all(item.hidden for item in scan_items))
        self.assertTrue(all(not item.hidden for item in optimise_items))

        _set_execution_option_rows_visible(
            scan_items, optimise_items, ExecutionMode.scan.name
        )
        self.assertTrue(all(not item.hidden for item in scan_items))
        self.assertTrue(all(item.hidden for item in optimise_items))


if __name__ == "__main__":
    unittest.main()
