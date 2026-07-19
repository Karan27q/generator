import unittest

from services.config import get_text_to_image_model_candidates


class TextToImageFallbackTests(unittest.TestCase):
    def test_fallback_models_include_free_options(self):
        candidates = get_text_to_image_model_candidates()

        self.assertIn("black-forest-labs/FLUX.1-schnell", candidates)
        self.assertIn("stabilityai/stable-diffusion-2-1-base", candidates)
        self.assertIn("runwayml/stable-diffusion-v1-5", candidates)


if __name__ == "__main__":
    unittest.main()
