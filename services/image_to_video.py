import tempfile
from datetime import datetime
from pathlib import Path

from gradio_client import handle_file

from services.config import HF_TOKEN, LTX_SPACE, WAN_SPACE, get_space_client
from services.text_to_video import _download_video, _is_valid_video
from utils.file_utils import save_output, to_pil_image
from utils.history import add_entry
from utils.logger import get_logger

logger = get_logger(__name__)


def _provider_ltx(image, prompt: str) -> str:
    """Animate an uploaded image using LTX Video Space."""
    try:
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
    except Exception as exc:
        from services.config import clear_space_client_cache
        clear_space_client_cache(LTX_SPACE)
        raise RuntimeError(f"LTX Space call failed: {exc}") from exc


def _provider_ltx_2(image, prompt: str) -> str:
    """Animate an image using the newer Lightricks/ltx-2-distilled Space."""
    try:
        client = get_space_client("Lightricks/ltx-2-distilled")
        pil = to_pil_image(image)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        try:
            result = client.predict(
                handle_file(tmp_path),
                prompt,
                3.0,     # duration
                True,    # enhance_prompt
                42,      # seed
                True,    # randomize_seed
                512,     # height
                768,     # width
                api_name="/generate_video",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        generated_video = result[0] if isinstance(result, tuple) else result
        video_bytes = _download_video(generated_video)
        return save_output(video_bytes, "video")
    except Exception as exc:
        from services.config import clear_space_client_cache
        clear_space_client_cache("Lightricks/ltx-2-distilled")
        raise RuntimeError(f"LTX-2 Space call failed: {exc}") from exc


def _provider_wan_i2v(image, prompt: str) -> str:
    """Animate an uploaded image using Wan Video Space."""
    try:
        import time
        client = get_space_client(WAN_SPACE)
        logger.info("Calling Wan Space Image-to-Video (%s)...", WAN_SPACE)

        pil = to_pil_image(image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        try:
            client.predict(
                prompt,
                handle_file(tmp_path),
                False,   # watermark_wan
                -1,      # seed
                api_name="/i2v_generation_async",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        max_wait_seconds = 300
        poll_interval = 5
        waited = 0

        while waited < max_wait_seconds:
            time.sleep(poll_interval)
            waited += poll_interval
            result = client.predict(api_name="/status_refresh")
            generated_video = result[0] if isinstance(result, tuple) else result

            if _is_valid_video(generated_video):
                logger.info("Wan generation finished after %ss", waited)
                actual_video = generated_video
                if isinstance(generated_video, dict) and generated_video.get("__type__") == "update":
                    actual_video = generated_video.get("value", generated_video)
                video_bytes = _download_video(actual_video)
                return save_output(video_bytes, "video")

            logger.info("Wan still generating... (%ss elapsed)", waited)

        raise RuntimeError(f"Wan generation timed out after {max_wait_seconds}s")
    except Exception as exc:
        from services.config import clear_space_client_cache
        clear_space_client_cache(WAN_SPACE)
        raise RuntimeError(f"Wan Space Image-to-Video call failed: {exc}") from exc



PROVIDERS = [
    _provider_ltx,
    _provider_ltx_2,
    _provider_wan_i2v,
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


