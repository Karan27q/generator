import os

# Gunicorn configuration file
# This file is automatically loaded by Gunicorn when started in this directory.

bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
