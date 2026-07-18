import tempfile
from datetime import datetime
from pathlib import Path

from gradio_client import handle_file

from services.config import HF_TOKEN, LTX_SPACE, get_space_client
from services.text_to_video import _download_video
from utils.file_utils import save_output, to_pil_image
from utils.history import add_entry
from utils.logger import get_logger

logger = get_logger(__name__)


def _provider_ltx(image, prompt: str) -> str:
    """Animate an uploaded image using LTX Video Space."""
    client = get_space_client(LTX_SPACE)
    pil = to_pil_image(image)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil.save(tmp.name, format="PNG")
        tmp_path = tmp.name

    try:
        result = client.predict(
            prompt,
            "",                              # negative_prompt
            handle_file(tmp_path),           # input_image_filepath — the uploaded image
            None,                            # input_video_filepath
            512,                             # height_ui
            704,                             # width_ui
            "image-to-video",                # mode
            3.0,                             # duration_ui
            9,                               # ui_frames_to_use
            0,                               # seed_ui
            True,                            # randomize_seed
            3.0,                             # ui_guidance_scale
            True,                            # improve_texture_flag
            api_name="/image_to_video",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    generated_video = result[0] if isinstance(result, tuple) else result
    video_bytes = _download_video(generated_video)
    return save_output(video_bytes, "video")


PROVIDERS = [
    _provider_ltx,
]


def animate_image(image, prompt: str) -> str:
    """Animate an uploaded image using the provider priority chain."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is required.")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is not set. Add your Hugging Face token to .env")

    prompt = prompt.strip()
    logger.info("Starting image-to-video: %s", prompt[:80])

    last_error: Exception | None = None
    for provider in PROVIDERS:
        try:
            logger.info("Trying provider: %s", provider.__name__)
            output_path = provider(image, prompt)
            add_entry({
                "time": datetime.now().isoformat(timespec="seconds"),
                "prompt": prompt,
                "type": "image-video",
                "output": output_path,
            })
            logger.info("Provider %s succeeded. Saved to %s", provider.__name__, output_path)
            return output_path
        except Exception as e:
            logger.warning("%s failed: %s", provider.__name__, e)
            last_error = e

    raise RuntimeError(f"All providers failed. Last error: {last_error}")


