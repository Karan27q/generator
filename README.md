# AI Creative Studio

A Gradio-based creative AI app for generating images and videos from text and images. Built as an internship assessment project using Hugging Face models and Spaces.

## Features

- **Text → Image** — Generate images from prompts with optional style presets (FLUX.1-schnell)
- **Image → Image** — Transform uploaded images with instruction prompts (provider fallback chain)
- **Text → Video** — Generate short videos from text prompts
- **Image → Video** — Animate still images into video clips
- **History** — View recent generations with timestamps, types, prompts, and output paths

## Architecture

```
Gradio UI (app.py)
    ↓
Service Layer (services/)
    ↓
Model Provider Layer (priority chains)
    ↓
Hugging Face Inference API / Spaces / (optional) Replicate / Fal AI
```

- `app.py` only handles UI layout and event wiring — no direct API calls.
- Each service module owns provider selection, model calls, logging, saving, and history writes.
- Image-to-image and video modes use a **priority chain**: try provider 1, catch failure, fall back to the next.

## Technologies

| Layer | Choice |
|---|---|
| UI | Gradio |
| Language | Python 3.11+ |
| Text → Image | Hugging Face `InferenceClient` + FLUX.1-schnell |
| Image → Image | HF Inference → HF Space → Replicate → Fal (chain) |
| Video | CogVideoX → Wan → LTX → SVD Spaces (chain) |
| Config | `.env` via `python-dotenv` |
| Logging | Centralized in `utils/logger.py` |
| History | `outputs/history.json` |

## Installation

1. **Clone or download** this project.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   copy .env.example .env          # Windows
   cp .env.example .env            # macOS/Linux
   ```
   Edit `.env` and set your Hugging Face token:
   ```
   HF_TOKEN=your_huggingface_token_here
   ```
   Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

   Optional fallbacks (not required for MVP):
   ```
   REPLICATE_API_TOKEN=
   FAL_KEY=
   ```

## Run Locally

```bash
python app.py
```

Open the URL shown in the terminal (typically `http://127.0.0.1:7860`).

## Project Structure

```
├── app.py                     # Gradio UI entry point
├── requirements.txt
├── .env.example
├── services/
│   ├── config.py              # env vars, model constants, clients
│   ├── text_to_image.py
│   ├── image_to_image.py
│   ├── text_to_video.py
│   ├── image_to_video.py
│   └── prompts.py             # style presets
├── utils/
│   ├── logger.py
│   ├── file_utils.py
│   ├── error_utils.py
│   └── history.py
├── assets/style.css
├── outputs/
│   ├── images/
│   ├── videos/
│   └── history.json
└── examples/                  # sample prompts
```

## Example Prompts

See [`examples/README.md`](examples/README.md) for demo prompts for each tab.

## Deploy to Hugging Face Spaces

1. Create a new **Gradio** Space on [huggingface.co/new-space](https://huggingface.co/new-space).
2. Push this repository to the Space (or upload files).
3. Add `HF_TOKEN` as a **Repository Secret** in Space Settings.
4. Ensure `app.py` is the entry point (default for Gradio Spaces).

## Known Limitations

- **Free-tier rate limits** — HF Inference API has usage caps; heavy use may hit 429 errors.
- **Space cold-starts** — Public HF Spaces sleep when idle; first request may take 1–3 minutes.
- **Space API changes** — Space endpoints can change; verify with `Client(space_id).view_api()` if a provider fails.
- **Video generation** — Requires live Spaces; not all providers may be available at all times.
- **Replicate / Fal** — Optional fallbacks; only used if tokens are configured.

## Future Improvements

- Add thumbnail previews in the History tab
- Implement Fal AI as a fully wired img2img fallback
- Add progress bars for long-running Space jobs
- Support batch generation and seed control
- Cache Space clients with health checks before calling

## License

MIT — for educational and assessment purposes.
