import io
import tempfile
from datetime import datetime
from pathlib import Path

from PIL import Image
from gradio_client import handle_file

from services.config import (
    DEFAULT_IMG2IMG_MODEL,
    DEFAULT_IMG2IMG_PROVIDER,
    HF_TOKEN,
    IMG2IMG_SPACE,
    get_space_client,
)
from utils.file_utils import save_output, to_pil_image, validate_image
from utils.history import add_entry
from utils.logger import get_logger

logger = get_logger(__name__)


def _provider_hf_inference(image, prompt: str) -> Image.Image:
    """Primary provider: HF Inference Providers, routed to fal-ai, via FLUX.2-dev."""
    from huggingface_hub import InferenceClient
    from services.config import DEFAULT_IMG2IMG_MODEL, DEFAULT_IMG2IMG_PROVIDER, HF_TOKEN

    client = InferenceClient(provider=DEFAULT_IMG2IMG_PROVIDER, api_key=HF_TOKEN)

    pil = to_pil_image(image)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)

    logger.info(
        "Calling HF Inference img2img (provider=%s, model=%s)...",
        DEFAULT_IMG2IMG_PROVIDER, DEFAULT_IMG2IMG_MODEL,
    )
    result = client.image_to_image(
        buf.read(),
        prompt=prompt,
        model=DEFAULT_IMG2IMG_MODEL,
    )
    return result.convert("RGB")


def _provider_hf_space(image, prompt: str) -> Image.Image:
    """Try a public HF Space via gradio_client."""
    client = get_space_client(IMG2IMG_SPACE)
    pil = to_pil_image(image)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil.save(tmp.name, format="PNG")
        tmp_path = tmp.name

    try:
        logger.info("Calling HF Space %s...", IMG2IMG_SPACE)
        result = client.predict(
            handle_file(tmp_path),
            prompt,
            50,
            "Randomize Seed",
            1371,
            "Fix CFG",
            7.5,
            1.5,
            api_name="/generate",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if isinstance(result, str) and Path(result).exists():
        return Image.open(result).convert("RGB")
    if isinstance(result, tuple) and len(result) >= 4:
        img_val = result[3]
        if isinstance(img_val, str):
            return Image.open(img_val).convert("RGB")
        if isinstance(img_val, dict) and "path" in img_val:
            return Image.open(img_val["path"]).convert("RGB")
        return to_pil_image(img_val)
    if isinstance(result, tuple) and result:
        first = result[0]
        if isinstance(first, str):
            return Image.open(first).convert("RGB")
        return to_pil_image(first)
    if isinstance(result, Image.Image):
        return result.convert("RGB")
    raise RuntimeError(f"Unexpected Space response type: {type(result)}")



PROVIDERS = [
    _provider_hf_space,       # primary — free HF Space (instruct-pix2pix)
    _provider_hf_inference,   # fallback — paid HF Inference via fal-ai
]


def transform_image(image, prompt: str) -> tuple[Image.Image, str]:
    """Transform an image using a provider priority chain."""
    if not validate_image(image):
        raise ValueError("Invalid or missing image upload.")
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is required.")

    prompt = prompt.strip()
    logger.info("Starting image-to-image: %s", prompt[:80])

    last_error: Exception | None = None
    for provider in PROVIDERS:
        try:
            logger.info("Trying provider: %s", provider.__name__)
            result = provider(image, prompt)
            output_path = save_output(result, "image")
            logger.info("Saved output to %s", output_path)
            add_entry({
                "time": datetime.now().isoformat(timespec="seconds"),
                "prompt": prompt,
                "type": "image-image",
                "output": output_path,
            })
            logger.info("Provider %s succeeded.", provider.__name__)
            return result, output_path
        except Exception as e:
            logger.warning("%s failed: %s", provider.__name__, e)
            last_error = e

    raise RuntimeError(f"All providers failed. Last error: {last_error}")
