import io
from datetime import datetime
from pathlib import Path
from typing import Literal, Union

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
VIDEOS_DIR = OUTPUTS_DIR / "videos"

MAX_IMAGE_SIZE_MB = 10
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def ensure_output_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def save_output(
    data: Union[Image.Image, bytes, str, Path],
    kind: Literal["image", "video"],
) -> str:
    """Write to outputs/images or outputs/videos with a timestamped filename."""
    ensure_output_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if kind == "image":
        path = IMAGES_DIR / f"image_{timestamp}.png"
        if isinstance(data, Image.Image):
            data.save(path, format="PNG")
        elif isinstance(data, (bytes, bytearray)):
            Image.open(io.BytesIO(data)).save(path, format="PNG")
        else:
            raise TypeError(f"Unsupported image data type: {type(data)}")
    else:
        path = VIDEOS_DIR / f"video_{timestamp}.mp4"
        if isinstance(data, (bytes, bytearray)):
            path.write_bytes(data)
        elif isinstance(data, (str, Path)):
            src = Path(data)
            if src.resolve() != path.resolve():
                path.write_bytes(src.read_bytes())
        else:
            raise TypeError(f"Unsupported video data type: {type(data)}")

    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def validate_prompt(prompt) -> str:
    """Validate prompt input and return a trimmed prompt."""
    if prompt is None:
        raise ValueError("Prompt is required.")

    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError("Prompt is required.")
    return prompt


def validate_image(file) -> bool:
    """Check file type, size, and basic corruption before sending to a service."""
    if file is None:
        return False

    if isinstance(file, Image.Image):
        try:
            file.verify()
            return True
        except Exception:
            return False

    if isinstance(file, str):
        path = Path(file)
        if not path.exists():
            return False
        if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            return False
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            return False
        try:
            with Image.open(path) as img:
                img.verify()
            arr = cv2.imread(str(path))
            return arr is not None
        except Exception:
            return False

    if isinstance(file, np.ndarray):
        return file.size > 0

    return False


def to_pil_image(file) -> Image.Image:
    """Convert Gradio upload (path, PIL, or ndarray) to PIL.Image."""
    if isinstance(file, Image.Image):
        return file.convert("RGB")
    if isinstance(file, np.ndarray):
        return Image.fromarray(file).convert("RGB")
    if isinstance(file, str):
        return Image.open(file).convert("RGB")
    raise ValueError("Unsupported image input type")
