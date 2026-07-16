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
