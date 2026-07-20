import tempfile
import time
from datetime import datetime
from pathlib import Path

from services.config import (
    HF_TOKEN,
    LTX_SPACE,
    WAN_SPACE,
    get_space_client,
)

from utils.file_utils import save_output
from utils.history import add_entry
from utils.logger import get_logger

logger = get_logger(__name__)


def _download_video(result) -> bytes:
    """Extract video bytes from various gradio_client return types."""
    if isinstance(result, (bytes, bytearray)):
        return bytes(result)

    if isinstance(result, dict):
        if "video" in result:
            return _download_video(result["video"])
        if "path" in result:
            return _download_video(result["path"])

    if isinstance(result, str):
        if result.startswith(("http://", "https://")):
            import requests
            resp = requests.get(result)
            resp.raise_for_status()
            return resp.content
        path = Path(result)
        if path.exists():
            return path.read_bytes()
        raise RuntimeError(f"Video path does not exist: {result}")

    if isinstance(result, tuple) and result:
        return _download_video(result[0])

    raise RuntimeError(f"Unexpected video response type: {type(result)}")


def _is_valid_video(val) -> bool:
    if not val:
        return False
    if isinstance(val, str):
        return True
    if isinstance(val, dict):
        if "__type__" in val:
            if val["__type__"] == "update":
                nested_val = val.get("value")
                if nested_val:
                    return _is_valid_video(nested_val)
                return False
        if "video" in val and val["video"]:
            return _is_valid_video(val["video"])
        if "path" in val and val["path"]:
            return True
    return False


def _provider_wan(prompt: str) -> str:
    try:
        client = get_space_client(WAN_SPACE)
        logger.info("Calling Wan Space (%s)...", WAN_SPACE)

        client.predict(
            prompt,
            "1280*720",
            False,
            -1,
            api_name="/t2v_generation_async",
        )

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
        raise RuntimeError(f"Wan Space call failed: {exc}") from exc




def _provider_ltx(prompt: str) -> str:
    try:
        client = get_space_client(LTX_SPACE)
        logger.info("Calling LTX Video Space (%s)...", LTX_SPACE)
        result = client.predict(
            prompt,
            "",
            None,
            None,
            512,
            704,
            "text-to-video",
            5.0,
            9,
            0,
            True,
            3.0,
            True,
            api_name="/text_to_video",
        )
        generated_video = result[0] if isinstance(result, tuple) else result
        video_bytes = _download_video(generated_video)
        return save_output(video_bytes, "video")
    except Exception as exc:
        from services.config import clear_space_client_cache
        clear_space_client_cache(LTX_SPACE)
        raise RuntimeError(f"LTX Space call failed: {exc}") from exc


PROVIDERS = [
    _provider_ltx,
    _provider_wan,
]




def generate_video(prompt: str) -> str:
    """Generate a video from text using a provider priority chain. Returns local path."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is required.")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is not set. Add your Hugging Face token to .env")

    prompt = prompt.strip()
    logger.info("Starting text-to-video: %s", prompt[:80])

    last_error: Exception | None = None
    for provider in PROVIDERS:
        try:
            logger.info("Trying provider: %s", provider.__name__)
            output_path = provider(prompt)
            add_entry({
                "time": datetime.now().isoformat(timespec="seconds"),
                "prompt": prompt,
                "type": "text-video",
                "output": output_path,
            })
            logger.info("Provider %s succeeded. Saved to %s", provider.__name__, output_path)
            return output_path
        except Exception as e:
            logger.warning("%s failed: %s", provider.__name__, e)
            last_error = e

    raise RuntimeError(f"All providers failed. Last error: {last_error}")
