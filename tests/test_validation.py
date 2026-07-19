import tempfile
import unittest
from pathlib import Path

from PIL import Image

from utils.file_utils import validate_image, validate_prompt


class ValidationTests(unittest.TestCase):
    def test_validate_prompt_rejects_blank_input(self):
        self.assertRaises(ValueError, validate_prompt, "   ")
        self.assertRaises(ValueError, validate_prompt, None)

    def test_validate_prompt_returns_trimmed_prompt(self):
        self.assertEqual(validate_prompt("  hello world  "), "hello world")

    def test_validate_image_rejects_missing_file(self):
        self.assertFalse(validate_image(None))
        self.assertFalse(validate_image("missing.png"))

    def test_validate_image_accepts_valid_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (10, 10), color="red").save(path)
            self.assertTrue(validate_image(str(path)))


if __name__ == "__main__":
    unittest.main()
