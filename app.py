"""AI Creative Studio — Gradio UI entry point."""

import asyncio
import os
from http import HTTPStatus
from pathlib import Path

import gradio as gr
from fastapi import FastAPI

from services.image_to_image import transform_image
from services.image_to_video import animate_image
from services.prompts import STYLE_PRESETS
from services.text_to_image import generate_image
from services.text_to_video import generate_video
from utils.error_utils import friendly_error
from utils.file_utils import ensure_output_dirs
from utils.history import format_history_table

PROJECT_ROOT = Path(__file__).resolve().parent
CSS_PATH = PROJECT_ROOT / "assets" / "style.css"

ensure_output_dirs()

css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


def should_show_history() -> bool:
    show_history = os.getenv("SHOW_HISTORY", "").strip().lower()
    if show_history in {"1", "true", "yes", "on"}:
        return True
    if show_history in {"0", "false", "no", "off"}:
        return False
    return os.getenv("RENDER", "").lower() != "true"


def _refresh_history():
    return format_history_table()


def _disable_btn():
    return gr.update(interactive=False)


def _enable_btn():
    return gr.update(interactive=True)


def _run_text_to_image(prompt, style):
    try:
        image, path = generate_image(prompt, style)
        return (
            image,
            gr.update(value=str(PROJECT_ROOT / path), visible=True),
            gr.update(value="", visible=False),
            _refresh_history(),
        )
    except Exception as exc:
        return (
            None,
            gr.update(visible=False),
            gr.update(value=friendly_error(exc), visible=True),
            _refresh_history(),
        )


def _run_image_to_image(image, prompt):
    try:
        result, path = transform_image(image, prompt)
        return (
            result,
            gr.update(value=str(PROJECT_ROOT / path), visible=True),
            gr.update(value="", visible=False),
            _refresh_history(),
        )
    except Exception as exc:
        return (
            None,
            gr.update(visible=False),
            gr.update(value=friendly_error(exc), visible=True),
            _refresh_history(),
        )


def _run_text_to_video(prompt):
    try:
        path = generate_video(prompt)
        full_path = str(PROJECT_ROOT / path)
        return (
            full_path,
            gr.update(value=full_path, visible=True),
            gr.update(value="", visible=False),
            _refresh_history(),
        )
    except Exception as exc:
        return (
            None,
            gr.update(visible=False),
            gr.update(value=friendly_error(exc), visible=True),
            _refresh_history(),
        )


def _run_image_to_video(image, prompt):
    try:
        path = animate_image(image, prompt)
        full_path = str(PROJECT_ROOT / path)
        return (
            full_path,
            gr.update(value=full_path, visible=True),
            gr.update(value="", visible=False),
            _refresh_history(),
        )
    except Exception as exc:
        return (
            None,
            gr.update(visible=False),
            gr.update(value=friendly_error(exc), visible=True),
            _refresh_history(),
        )


def _wire_generate(btn, fn, inputs, outputs):
    """Wire a generate button with loading state and history refresh."""
    btn.click(_disable_btn, outputs=[btn], queue=False).then(
        fn,
        inputs=inputs,
        outputs=outputs,
    ).then(_enable_btn, outputs=[btn], queue=False)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Creative Studio") as app:
        gr.Markdown(
            """
            # AI Creative Studio
            Generate images and videos from text and images using Hugging Face models.
            """
        )

        with gr.Tab("Text → Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2i_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="A futuristic city at sunset with flying cars...",
                        lines=3,
                    )
                    t2i_style = gr.Dropdown(
                        label="Style",
                        choices=list(STYLE_PRESETS.keys()),
                        value="None",
                    )
                    t2i_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=1):
                    t2i_output = gr.Image(label="Generated Image", type="pil")
                    t2i_download = gr.DownloadButton("Download Image", visible=False)
                    t2i_error = gr.Textbox(label="Error", visible=False, interactive=False)

        with gr.Tab("Image → Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2i_input = gr.Image(label="Upload Image", type="filepath")
                    i2i_prompt = gr.Textbox(
                        label="Transformation Prompt",
                        placeholder="Make it look like a watercolor painting...",
                        lines=3,
                    )
                    i2i_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=1):
                    with gr.Row():
                        i2i_original = gr.Image(label="Original", type="pil", interactive=False)
                        i2i_output = gr.Image(label="Result", type="pil")
                    i2i_download = gr.DownloadButton("Download Image", visible=False)
                    i2i_error = gr.Textbox(label="Error", visible=False, interactive=False)

            i2i_input.change(lambda img: img, inputs=[i2i_input], outputs=[i2i_original])

        with gr.Tab("Text → Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2v_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="A cat walking through a garden in slow motion...",
                        lines=3,
                    )
                    t2v_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=1):
                    t2v_output = gr.Video(label="Generated Video")
                    t2v_download = gr.DownloadButton("Download Video", visible=False)
                    t2v_error = gr.Textbox(label="Error", visible=False, interactive=False)

        with gr.Tab("Image → Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2v_input = gr.Image(label="Upload Image", type="filepath")
                    i2v_prompt = gr.Textbox(
                        label="Motion Prompt",
                        placeholder="Gentle camera zoom with leaves blowing in the wind...",
                        lines=3,
                    )
                    i2v_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=1):
                    i2v_output = gr.Video(label="Generated Video")
                    i2v_download = gr.DownloadButton("Download Video", visible=False)
                    i2v_error = gr.Textbox(label="Error", visible=False, interactive=False)

        if should_show_history():
            with gr.Tab("History"):
                gr.Markdown("Recent generations from this session and past runs.")
                history_table = gr.Dataframe(
                    headers=["Time", "Type", "Prompt", "Output"],
                    value=_refresh_history(),
                    interactive=False,
                    wrap=True,
                )
                refresh_btn = gr.Button("Refresh History")
                refresh_btn.click(fn=_refresh_history, outputs=[history_table])
        else:
            history_table = None

        t2i_outputs = [t2i_output, t2i_download, t2i_error, history_table]
        _wire_generate(t2i_btn, _run_text_to_image, [t2i_prompt, t2i_style], t2i_outputs)

        i2i_outputs = [i2i_output, i2i_download, i2i_error, history_table]
        _wire_generate(i2i_btn, _run_image_to_image, [i2i_input, i2i_prompt], i2i_outputs)

        t2v_outputs = [t2v_output, t2v_download, t2v_error, history_table]
        _wire_generate(t2v_btn, _run_text_to_video, [t2v_prompt], t2v_outputs)

        i2v_outputs = [i2v_output, i2v_download, i2v_error, history_table]
        _wire_generate(i2v_btn, _run_image_to_video, [i2v_input, i2v_prompt], i2v_outputs)

    return app


def create_fastapi_app() -> FastAPI:
    demo = build_app()
    return gr.mount_gradio_app(
        FastAPI(title="AI Creative Studio"),
        demo,
        path="/",
        css=css,
        theme=gr.themes.Soft(),
    )


class AsgiToWsgi:
    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    def __call__(self, environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")
        query_string = environ.get("QUERY_STRING", "")
        content_length = environ.get("CONTENT_LENGTH", "")
        body = b""

        if content_length:
            try:
                body = environ["wsgi.input"].read(int(content_length))
            except (TypeError, ValueError):
                body = b""

        async def receive():
            if not hasattr(receive, "called"):
                receive.called = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        response_status = 500
        response_headers = []
        response_body_chunks = []

        async def send(message):
            nonlocal response_status, response_headers, response_body_chunks
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = [
                    (
                        key.decode("latin-1") if isinstance(key, bytes) else key,
                        value.decode("latin-1") if isinstance(value, bytes) else value,
                    )
                    for key, value in message.get("headers", [])
                ]
            elif message["type"] == "http.response.body":
                if message.get("body"):
                    response_body_chunks.append(message["body"])

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": environ.get("SERVER_PROTOCOL", "HTTP/1.1"),
            "method": method,
            "scheme": environ.get("wsgi.url_scheme", "http"),
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": query_string.encode("utf-8"),
            "headers": [
                (key.lower().encode("latin-1"), value.encode("latin-1"))
                for key, value in environ.get("HTTP_HEADERS", {}).items()
            ],
            "client": (environ.get("REMOTE_ADDR", "127.0.0.1"), 0),
            "server": (environ.get("SERVER_NAME", "localhost"), int(environ.get("SERVER_PORT", "80"))),
            "root_path": "",
        }

        scope["headers"] = [
            (key.encode("latin-1") if isinstance(key, str) else key, value.encode("latin-1") if isinstance(value, str) else value)
            for key, value in [
                (key, value)
                for key, value in environ.items()
                if key.startswith("HTTP_")
            ]
        ]
        scope["headers"].extend(
            [
                (b"host", environ.get("HTTP_HOST", "localhost").encode("latin-1")),
                (b"content-length", str(len(body)).encode("latin-1")),
                (b"content-type", environ.get("CONTENT_TYPE", "").encode("latin-1")),
            ]
        )

        asyncio.run(self.asgi_app(scope, receive, send))

        status_text = f"{response_status} {HTTPStatus(response_status).phrase}" if response_status in HTTPStatus._value2member_map_ else f"{response_status} OK"
        if method == "HEAD":
            response_body = b""
        else:
            response_body = b"".join(response_body_chunks)

        start_response(status_text, response_headers)
        return [response_body]


asgi_app = create_fastapi_app()
app = AsgiToWsgi(asgi_app)
application = app


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        css=css,
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        server_port=7860,
    )
