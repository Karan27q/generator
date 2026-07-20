def friendly_error(exc: Exception) -> str:
    """Map exceptions to short user-facing Gradio messages."""
    msg = str(exc).lower()
    exc_type = type(exc).__name__

    if "empty" in msg or "prompt" in msg and "required" in msg:
        return "Please enter a prompt before generating."

    if "image" in msg and ("missing" in msg or "required" in msg or "none" in msg):
        return "Please upload an image before generating."

    if "invalid" in msg and "image" in msg:
        return "The uploaded file is not a valid image. Use PNG, JPG, or WEBP under 10 MB."

    if "rate limit" in msg or "429" in msg or "too many requests" in msg:
        return "Rate limit reached. Wait a moment and try again."

    if "timeout" in msg or "timed out" in msg:
        return "The request timed out. The model may be starting up — try again in a minute."

    if "401" in msg or "unauthorized" in msg or "invalid token" in msg:
        return "Authentication failed. Check that HF_TOKEN is set correctly in your .env file."

    if "403" in msg or "forbidden" in msg:
        return "Access denied. Your token may lack permission for this model."

    if "all providers failed" in msg:
        return "All generation providers failed. Try again later or check your API token."

    if "zerogpu" in msg or "gpu time" in msg or "hugging face pro" in msg:
        return "You have exceeded your Hugging Face ZeroGPU quota limit. Please wait a while for it to reset or check your account."

    if "shutdown" in msg and "futures" in msg:
        return "A connection error occurred. Please try generating again."

    if "space" in msg and ("sleep" in msg or "offline" in msg or "building" in msg):
        return "The Hugging Face Space is waking up or offline. Please try again shortly."

    if "connection" in msg or "network" in msg:
        return "Network error. Check your internet connection and try again."

    if exc_type in ("ValueError", "TypeError") and str(exc):
        return str(exc)

    return f"Generation failed: {exc}"
