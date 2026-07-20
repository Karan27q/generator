import os
import time
from functools import lru_cache

import requests
from dotenv import load_dotenv
from gradio_client import Client
from huggingface_hub import HfApi, InferenceClient
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


HF_TOKEN = os.getenv("HF_TOKEN", "")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
FAL_KEY = os.getenv("FAL_KEY", "")

DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
DEFAULT_IMG2IMG_MODEL = "black-forest-labs/FLUX.2-dev"
DEFAULT_IMG2IMG_PROVIDER = "fal-ai"   # also available: "replicate", "wavespeed"
DEFAULT_STYLE = "None"

FREE_IMAGE_MODELS = [
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-2-1-base",
    "runwayml/stable-diffusion-v1-5",
]

# Hugging Face Spaces — verified with Client(<id>).view_api()
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


def get_text_to_image_model_candidates() -> list[str]:
    return list(FREE_IMAGE_MODELS)


_space_clients = {}


def clear_space_client_cache(space_id: str) -> None:
    """Evict a space client from the cache so it will be recreated next time."""
    if space_id in _space_clients:
        logger.info("Evicting space client %s from cache", space_id)
        client = _space_clients.pop(space_id, None)
        if client is not None:
            try:
                client.close()
            except Exception as e:
                logger.warning("Failed to close client for space %s: %s", space_id, e)


def get_space_client(space_id: str) -> Client:
    # Check if we have a cached client and it's still alive/valid
    client = _space_clients.get(space_id)
    if client is not None:
        executor = getattr(client, "executor", None)
        if executor is not None and getattr(executor, "_shutdown", False):
            logger.info("Cached client for space %s has been shut down. Evicting and recreating...", space_id)
            old_client = _space_clients.pop(space_id, None)
            if old_client is not None:
                try:
                    old_client.close()
                except Exception as e:
                    logger.warning("Failed to close old client for space %s: %s", space_id, e)
            client = None

    if client is None:
        client = _create_space_client(space_id)
        _space_clients[space_id] = client
    return client


import atexit

def close_all_space_clients() -> None:
    """Close all cached space clients to release resources and stop heartbeat threads."""
    try:
        clients = list(_space_clients.items())
        if clients:
            logger.info("Closing all cached space clients...")
            for space_id, client in clients:
                try:
                    client.close()
                except Exception as e:
                    logger.warning("Failed to close client for space %s: %s", space_id, e)
            _space_clients.clear()
    except Exception:
        pass

atexit.register(close_all_space_clients)



def _create_space_client(space_id: str) -> Client:
    # 1. Try to wake up the space if it's sleeping or paused
    try:
        api = HfApi()
        runtime = api.get_space_runtime(space_id)
        stage = runtime.stage
        if stage in ("SLEEPING", "PAUSED"):
            logger.info("Hugging Face Space %s is %s. Waking it up...", space_id, stage)
            requests.get(f"https://huggingface.co/spaces/{space_id}", timeout=10)
            
            waited = 0
            while waited < 300:
                time.sleep(5)
                waited += 5
                runtime = api.get_space_runtime(space_id)
                if runtime.stage == "RUNNING":
                    logger.info("Space %s is now RUNNING.", space_id)
                    break
                logger.info("Space %s is %s, waiting... (%ss)", space_id, runtime.stage, waited)
    except Exception as e:
        logger.warning("Could not check status of space %s: %s", space_id, e)

    # 2. Try to connect unauthenticated with browser headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    last_err = None
    try:
        logger.info("Initializing gradio Client for %s (no token)...", space_id)
        return Client(space_id, token=None, headers=headers)
    except Exception as e:
        last_err = e
        logger.warning("Failed unauthenticated connect to space %s: %s", space_id, e)
        
    # 3. Fallback to connecting with HF_TOKEN
    if HF_TOKEN:
        try:
            logger.info("Retrying connection to %s with token...", space_id)
            return Client(space_id, token=HF_TOKEN, headers=headers)
        except TypeError:
            try:
                return Client(space_id, hf_token=HF_TOKEN, headers=headers)
            except Exception as re:
                last_err = re
        except Exception as re:
            last_err = re
            
    raise RuntimeError(f"Could not connect to Hugging Face space '{space_id}'. Last error: {last_err}")


