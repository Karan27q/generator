# AI Creative Studio — Build Guide (v2)

Confirmed stack and structure for the internship assessment. Use this as the working spec.

---

## 1. Confirmed Stack

| Layer | Choice |
|---|---|
| UI | Gradio |
| Language | Python 3.11+ |
| Image Generation (Text→Image) | HF `InferenceClient`, default model `FLUX.1-schnell`, swappable via config |
| Image-to-Image | Provider-priority chain (see Section 5) |
| Text-to-Video | Provider-priority chain (see Section 5) |
| Image-to-Video | Provider-priority chain (see Section 5) |
| Config | `.env` via `python-dotenv` |
| Logging | Python `logging`, centralized in `utils/logger.py` |
| History | `outputs/history.json`, shown in a 5th Gradio tab |
| Deployment | Hugging Face Spaces |
| Structure | `services/`, `utils/`, `assets/`, `outputs/`, single `app.py` entry point |

**Core principle:** `app.py` only builds the Gradio UI and calls functions from `services/`.
All model-calling logic (which provider, which model, how to parse the response) lives inside
`services/`, so the UI never needs to change if a backend model or provider is swapped.

---

## 2. Project Structure

```
ai-creative-studio/
│
├── app.py                     # Gradio UI, tabs, wires buttons to services/
├── requirements.txt
├── README.md
├── .env                        # HF_TOKEN=... (not committed)
├── .env.example
│
├── services/
│   ├── __init__.py
│   ├── config.py               # env vars, DEFAULT_*_MODEL constants, client instances
│   ├── text_to_image.py        # generate_image(prompt, style=None) -> PIL.Image
│   ├── image_to_image.py       # transform_image(image, prompt) -> PIL.Image (provider chain)
│   ├── text_to_video.py        # generate_video(prompt) -> str (provider chain)
│   ├── image_to_video.py       # animate_image(image, prompt) -> str (provider chain)
│   └── prompts.py              # STYLE_PRESETS dict + prompt-building helper
│
├── utils/
│   ├── __init__.py
│   ├── file_utils.py           # save to outputs/images or outputs/videos, validate uploads
│   ├── error_utils.py          # format exceptions into user-facing Gradio messages
│   ├── logger.py               # centralized logging config
│   └── history.py              # read/write outputs/history.json
│
├── assets/
│   └── style.css               # optional custom CSS, kept out of app.py
│
├── outputs/
│   ├── images/
│   ├── videos/
│   └── history.json
│
└── examples/                   # sample prompts/images for demoing the app
```

---

## 3. Prerequisites

- [ ] Python 3.11+
- [ ] Free Hugging Face account → generate an Access Token (Settings → Access Tokens)
- [ ] Dependencies:
  ```bash
  pip install gradio huggingface_hub gradio_client python-dotenv requests pillow numpy opencv-python
  ```

---

## 4. Environment Variables

`.env` (not committed):
```
HF_TOKEN=your_huggingface_token_here
# Optional, only needed if you add fallback providers later:
REPLICATE_API_TOKEN=
FAL_KEY=
```

`.env.example` (committed):
```
HF_TOKEN=
REPLICATE_API_TOKEN=
FAL_KEY=
```

`services/config.py` loads these once via `python-dotenv` and exposes:
- A shared `InferenceClient(token=HF_TOKEN)`.
- Default model constants (see below) so any single model can be swapped in one place.
- `gradio_client.Client(...)` instances for chosen Spaces, created once, not per-request.

```python
# services/config.py (structure, not final code)
DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
DEFAULT_STYLE = "None"
```

Usage pattern in `text_to_image.py`:
```python
client.text_to_image(prompt, model=DEFAULT_IMAGE_MODEL)
```
Never hardcode the model string inside the generation call itself — always reference the
constant, so switching models (e.g. if FLUX is rate-limited) is a one-line change in `config.py`.

---

## 5. Provider Strategy (Image-to-Image & Video)

Not every HF model supports every mode, and public Spaces sleep, disappear, or change their
API signature. So each of these services is written as a **priority chain**, not a single
hardcoded call — try the first provider, catch failure, fall back to the next.

### Image-to-Image — priority order
1. **HF Inference Endpoint** (if the chosen model supports an `image` + `prompt` call directly)
2. **HF Space** via `gradio_client` (search Spaces tagged "img2img", verify it's running)
3. **Replicate API** (free-tier friendly, stable img2img models available)
4. **Fal AI** (fast inference, has an img2img free tier)

### Text-to-Video & Image-to-Video — preferred providers, in order
1. **CogVideoX** (public HF Space)
2. **Wan** (public HF Space)
3. **LTX Video** (public HF Space)
4. **Stable Video Diffusion** (public HF Space, image-to-video specifically)

Implementation pattern for each service module:
```python
PROVIDERS = [provider_1_fn, provider_2_fn, provider_3_fn]

def generate(...):
    for provider in PROVIDERS:
        try:
            logger.info(f"Trying provider: {provider.__name__}")
            return provider(...)
        except Exception as e:
            logger.warning(f"{provider.__name__} failed: {e}")
    raise RuntimeError("All providers failed")
```
Only implement Replicate/Fal AI as fallbacks if time allows — for the MVP, one working
provider per mode is enough, but structure the code so adding a fallback is just appending
one function to the `PROVIDERS` list, not a rewrite.

Before wiring in any Space, open its **API tab** on Hugging Face (or run
`Client(space_id).view_api()`) to confirm the real `api_name` and parameter order — these vary
per Space and change over time, so never guess them.

---

## 6. Service Modules — Responsibilities

### `services/config.py`
- Load env vars, define `DEFAULT_IMAGE_MODEL` and any other default-model constants.
- Instantiate shared `InferenceClient` and `gradio_client.Client` instances once.

### `services/text_to_image.py`
- `generate_image(prompt: str, style: str | None = None) -> PIL.Image`
- Applies style preset from `prompts.py`.
- Calls `client.text_to_image(prompt, model=DEFAULT_IMAGE_MODEL)`.
- Logs progress via `utils/logger.py`, saves to `outputs/images/`, appends a `history.json` entry.

### `services/image_to_image.py`
- `transform_image(image, prompt: str) -> PIL.Image`
- Runs the provider priority chain from Section 5.
- Logs which provider succeeded/failed, saves output, appends history entry.

### `services/text_to_video.py`
- `generate_video(prompt: str) -> str` (local file path)
- Runs the provider priority chain from Section 5 (video providers).
- Logs, saves to `outputs/videos/`, appends history entry.

### `services/image_to_video.py`
- `animate_image(image, prompt: str) -> str`
- Same pattern, preferring Stable Video Diffusion first since it's image-native.

### `services/prompts.py`
```python
STYLE_PRESETS = {
    "None": "",
    "Realistic": "photorealistic, highly detailed, cinematic lighting",
    "Anime": "anime style, vibrant colors, studio ghibli inspired",
    "Oil Painting": "oil painting, textured brushstrokes, fine art",
    "Fantasy": "fantasy art, magical, dramatic lighting",
    "Cyberpunk": "cyberpunk, neon lighting, futuristic, highly detailed",
    "Pixar": "3d render, pixar style, vibrant, expressive",
    "Sketch": "pencil sketch, hand-drawn, monochrome",
    "Watercolor": "watercolor painting, soft edges, pastel tones",
}

def build_prompt(prompt: str, style: str) -> str:
    suffix = STYLE_PRESETS.get(style, "")
    return f"{prompt}, {suffix}" if suffix else prompt
```

---

## 7. Utils

### `utils/logger.py`
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
```
Every service module logs key lifecycle events: `"Generating image..."`, `"Calling HF..."`,
`"Completed."`, `"Saved output."`, `"Generation failed: <reason>"`. This is a small addition
that makes the app noticeably more professional to review.

### `utils/file_utils.py`
- `save_output(data, kind: Literal["image", "video"]) -> str` — writes to `outputs/images/`
  or `outputs/videos/` with a timestamped filename, returns the path.
- `validate_image(file) -> bool` — checks file type/size before sending to a service (can use
  `opencv-python`/`PIL` here for basic format/corruption checks).

### `utils/error_utils.py`
- `friendly_error(exc: Exception) -> str` — maps common failure modes (empty prompt, missing
  image, API timeout, rate limit, all providers failed, Space cold-start/offline) to short
  user-facing messages for Gradio.

### `utils/history.py`
- `add_entry(entry: dict)` — appends to `outputs/history.json`:
  ```json
  {
    "time": "2026-07-15T10:22:00",
    "prompt": "a futuristic city",
    "type": "text-image",
    "output": "outputs/images/image_001.png"
  }
  ```
- `get_recent(limit: int = 20) -> list[dict]` — reads the file, returns most recent entries
  for display in the History tab.

---

## 8. `app.py` — UI Structure

Single Gradio `Blocks` app (optionally loading `assets/style.css`), title
**"AI Creative Studio"**, with **5** `gr.Tab()`s:

1. **Text → Image** — prompt textbox, style dropdown, Generate button, image output, download button.
2. **Image → Image** — image upload, prompt textbox, Generate button, side-by-side original/result.
3. **Text → Video** — prompt textbox, Generate button, video player, download button.
4. **Image → Video** — image upload, prompt textbox, Generate button, video player.
5. **History** — table or gallery of recent generations pulled from `utils/history.py`,
   showing timestamp, type, prompt, and thumbnail/link to output.

Each Generate button:
- Disables itself and shows a loading state while running.
- Calls the matching `services/` function inside a try/except, using `error_utils.friendly_error`
  on failure.
- On success, calls `utils/history.py` to log the generation, and refreshes the History tab data.

`app.py` should contain **no direct API/model/provider logic** — only UI layout, event wiring,
and calls into `services/` and `utils/`.

---

## 9. Build Order (Phases)

1. **Setup** — scaffold folders, install deps, configure `.env` + `config.py` + `logger.py`.
2. **Text → Image** — implement + test in isolation.
3. **Image → Image** — implement primary provider first; add fallback(s) only if time allows.
4. **Text → Video** — implement primary provider, verify it's live before coding against it.
5. **Image → Video** — same.
6. **History** — wire up `history.json` writes from each service, build the History tab.
7. **UI polish** — loading states, disabled buttons, download buttons, style dropdown,
   optional `assets/style.css`.
8. **Error handling pass** — empty prompts, missing images, invalid file types, API failures,
   timeouts, rate limits, all-providers-failed case.
9. **Testing** — run every workflow end-to-end, including invalid inputs and provider fallback.
10. **Documentation** — write README (see below).
11. **Deploy** — push to Hugging Face Spaces, add secrets, verify the public demo works.

---

## 10. README.md Should Include

- Project overview
- Features (5 tabs, including History)
- **Architecture** section:
  ```
  Gradio UI
      ↓
  Service Layer (services/)
      ↓
  Model Provider Layer (priority chains)
      ↓
  Hugging Face Inference API / Spaces / (optional) Replicate / Fal AI
  ```
- Technologies used
- Installation steps
- Environment variables (`.env.example` reference)
- How to run locally
- Example screenshots/GIFs of each tab
- Known limitations (free-tier rate limits, Space cold-starts, provider fallback behavior)
- Future improvements

---

## 11. Deliverables Checklist

- [ ] Working Gradio app with all 5 tabs functional (including History)
- [ ] `services/` and `utils/` cleanly separated from `app.py`
- [ ] Provider priority chains implemented for image-to-image and video (at least primary provider working, fallback structure in place)
- [ ] Centralized logging across all services
- [ ] `.env.example` provided, `.env` not committed
- [ ] `requirements.txt` accurate
- [ ] README complete with Architecture section and screenshots
- [ ] Deployed to Hugging Face Spaces with a working public link
- [ ] Graceful handling of empty prompts, bad uploads, API/timeout/rate-limit/all-providers-failed errors

---

## 12. requirements.txt

```
gradio
huggingface_hub
gradio_client
python-dotenv
requests
pillow
numpy
opencv-python
```

---

## 13. Notes on Finding Video/Image Spaces

Spaces on Hugging Face change frequently (some go offline, others get added). Before building
any service that depends on one:

1. Search Hugging Face Spaces for the provider name (e.g. "CogVideoX", "Wan", "LTX Video",
   "Stable Video Diffusion", "img2img").
2. Open a candidate Space, confirm it's actively running (not sleeping/errored).
3. Check its **API** tab, or run `Client(space_id).view_api()`, to get the exact `api_name`
   and parameter order.
4. Hardcode that verified signature into the corresponding service module.
5. If it stops working later, that's exactly what the provider priority chain (Section 5)
   is for — add a new provider function and move on, without touching `app.py` or other services.