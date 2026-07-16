import os
from functools import lru_cache

from dotenv import load_dotenv
from gradio_client import Client
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
FAL_KEY = os.getenv("FAL_KEY", "")

DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
DEFAULT_IMG2IMG_MODEL = "black-forest-labs/FLUX.2-dev"
DEFAULT_IMG2IMG_PROVIDER = "fal-ai"   # also available: "replicate", "wavespeed"
DEFAULT_STYLE = "None"

# Hugging Face Spaces — verified with Client(<id>).view_api()
COGVIDEOX_SPACE = "THUDM/CogVideoX-5B-Space"   # confirmed working
WAN_SPACE = "Wan-AI/Wan2.1"                     # was wrong: "Wan-AI/Wan2.1-T2V-1.3B" is a model repo, not a Space
LTX_SPACE = "Lightricks/ltx-video-distilled"  # confirmed working
IMG2IMG_SPACE = "timbrooks/instruct-pix2pix"


@lru_cache(maxsize=1)
def get_inference_client() -> InferenceClient:
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is not set. Add your Hugging Face token to .env"
        )
    return InferenceClient(token=HF_TOKEN)


@lru_cache(maxsize=8)
def get_space_client(space_id: str) -> Client:
    try:
        return Client(space_id, token=HF_TOKEN or None)
    except TypeError:
        return Client(space_id, hf_token=HF_TOKEN or None)
