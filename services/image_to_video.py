import tempfile
from datetime import datetime
from pathlib import Path

from services.config import HF_TOKEN, LTX_SPACE, get_space_client
from services.text_to_video import _download_video
from utils.file_utils import save_output, to_pil_image
from utils.history import add_entry
from utils.logger import get_logger

logger = get_logger(__name__)


def animate_image(image, prompt: str) -> str:
    """Animate an uploaded image using LTX (image-conditioned mode)."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is required.")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is not set. Add your Hugging Face token to .env")

    prompt = prompt.strip()
    logger.info("Starting image-to-video: %s", prompt[:80])

    client = get_space_client(LTX_SPACE)
    pil = to_pil_image(image)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil.save(tmp.name, format="PNG")
        tmp_path = tmp.name

    try:
        result = client.predict(
            prompt,
            "",                  # negative_prompt
            tmp_path,            # input_image_filepath — the uploaded image
            None,                # input_video_filepath
            512,                 # height_ui        — GUESS, verify
            768,                 # width_ui          — GUESS, verify
            "image-to-video",    # mode              — GUESS, verify exact literal string
            5,                   # duration_ui       — GUESS, verify
            None,                # ui_frames_to_use
            0,                   # seed_ui
            True,                # randomize_seed
            3.0,                 # ui_guidance_scale — GUESS, verify
            True,                # improve_texture_flag
            api_name="/text_to_video",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    generated_video = result[0] if isinstance(result, tuple) else result
    video_bytes = _download_video(generated_video)
    output_path = save_output(video_bytes, "video")

    add_entry({
        "time": datetime.now().isoformat(timespec="seconds"),
        "prompt": prompt,
        "type": "image-video",
        "output": output_path,
    })
    logger.info("Saved image-to-video output to %s", output_path)
    return output_path
