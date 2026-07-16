from datetime import datetime

from PIL import Image

from services.config import DEFAULT_IMAGE_MODEL, get_inference_client
from services.prompts import build_prompt
from utils.file_utils import save_output
from utils.history import add_entry
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_image(prompt: str, style: str | None = None) -> tuple[Image.Image, str]:
    """Generate an image from a text prompt using HF Inference API."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is required.")

    style = style or "None"
    full_prompt = build_prompt(prompt.strip(), style)
    logger.info("Generating image with prompt: %s", full_prompt[:100])

    client = get_inference_client()
    logger.info("Calling HF Inference API (model=%s)...", DEFAULT_IMAGE_MODEL)

    image = client.text_to_image(full_prompt, model=DEFAULT_IMAGE_MODEL)
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    output_path = save_output(image, "image")
    logger.info("Saved output to %s", output_path)

    add_entry({
        "time": datetime.now().isoformat(timespec="seconds"),
        "prompt": prompt.strip(),
        "type": "text-image",
        "output": output_path,
        "style": style,
    })

    logger.info("Completed text-to-image generation.")
    return image, output_path
