import os
import unittest

from app import should_show_history


class HistoryVisibilityTests(unittest.TestCase):
    def test_history_hidden_on_render_by_default(self):
        os.environ["RENDER"] = "true"
        os.environ.pop("SHOW_HISTORY", None)
        self.assertFalse(should_show_history())

    def test_history_shown_locally_by_default(self):
        os.environ.pop("RENDER", None)
        os.environ.pop("SHOW_HISTORY", None)
        self.assertTrue(should_show_history())

    def test_history_can_be_forced_on_or_off(self):
        os.environ.pop("RENDER", None)
        os.environ["SHOW_HISTORY"] = "false"
        self.assertFalse(should_show_history())

        os.environ["SHOW_HISTORY"] = "true"
        self.assertTrue(should_show_history())


if __name__ == "__main__":
    unittest.main()
