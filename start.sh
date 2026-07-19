#!/usr/bin/env bash
set -euo pipefail
exec uvicorn app:asgi_app --host 0.0.0.0 --port "${PORT:-8000}"
