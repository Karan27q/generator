from datetime import datetime

from PIL import Image

from services.config import DEFAULT_IMAGE_MODEL, get_inference_client, get_text_to_image_model_candidates
from services.prompts import build_prompt
from utils.file_utils import save_output, validate_prompt
from utils.history import add_entry
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_image(prompt: str, style: str | None = None) -> tuple[Image.Image, str]:
    """Generate an image from a text prompt using HF Inference API."""
    prompt = validate_prompt(prompt)

    style = style or "None"
    full_prompt = build_prompt(prompt, style)
    logger.info("Generating image with prompt: %s", full_prompt[:100])

    client = get_inference_client()
    model_candidates = get_text_to_image_model_candidates()
    last_error = None

    for model_name in model_candidates:
        try:
            logger.info("Calling HF Inference API (model=%s)...", model_name)
            image = client.text_to_image(full_prompt, model=model_name)
            break
        except Exception as exc:  # pragma: no cover - runtime fallback path
            last_error = exc
            logger.warning("Model %s failed: %s", model_name, exc)
    else:
        raise RuntimeError(f"All text-to-image fallback models failed. Last error: {last_error}")

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    output_path = save_output(image, "image")
    logger.info("Saved output to %s", output_path)

    add_entry({
        "time": datetime.now().isoformat(timespec="seconds"),
        "prompt": prompt,
        "type": "text-image",
        "output": output_path,
        "style": style,
    })

    logger.info("Completed text-to-image generation.")
    return image, output_path
