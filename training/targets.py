"""Target patterns for training.

A target is an (X, Y, 4) float array in [0, 1]: RGB + alpha, centered in the grid.
Alpha is used as the target matter density, RGB as the target color (first 3 hidden channels).

Accepted names:
    gecko, butterfly, fish, octopus, mushroom, smiley, ...   (see EMOJIS)
    emoji:<char>       any emoji (rendered with the Apple Color Emoji font)
    circle, ring, square, star, cross, two_dots             (procedural, white)
    text:<chars>       rendered with the default font
    image:<path>       any image file (PNG with alpha recommended; otherwise luminance is used as alpha)
"""
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
EMOJIS = {
    "gecko": "🦎", "butterfly": "🦋", "fish": "🐠", "octopus": "🐙", "mushroom": "🍄",
    "smiley": "😀", "spider": "🕷️", "snowflake": "❄️", "lobster": "🦞", "microbe": "🦠",
}
SHAPES = ["circle", "ring", "square", "star", "cross", "two_dots"]


def available() -> list[str]:
    return list(EMOJIS) + SHAPES + ["emoji:<char>", "text:<chars>", "image:<path>"]


def _render_emoji(ch: str) -> Image.Image:
    if not pathlib.Path(EMOJI_FONT).exists():
        raise FileNotFoundError(f"{EMOJI_FONT} not found: use image:<path> targets on this system")
    font = ImageFont.truetype(EMOJI_FONT, 160)  # bitmap font: 160 is one of its native sizes
    im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, 0), ch, font=font, embedded_color=True)
    return im


def _render_text(txt: str) -> Image.Image:
    font = ImageFont.load_default(size=160)
    im = Image.new("RGBA", (160 * max(len(txt), 1), 200), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, 0), txt, font=font, fill=(255, 255, 255, 255))
    return im


def _render_shape(name: str, n: int = 256) -> Image.Image:
    yy, xx = (np.mgrid[:n, :n] + 0.5) / n * 2 - 1  # [-1, 1]
    rr, th = np.hypot(xx, yy), np.arctan2(yy, xx)
    if name == "circle":
        a = rr < 0.9
    elif name == "ring":
        a = (rr < 0.9) & (rr > 0.55)
    elif name == "square":
        a = (np.abs(xx) < 0.8) & (np.abs(yy) < 0.8)
    elif name == "star":
        a = rr < 0.45 + 0.45 * np.abs(np.cos(2.5 * th)) ** 3
    elif name == "cross":
        a = ((np.abs(xx) < 0.25) | (np.abs(yy) < 0.25)) & (rr < 0.95)
    elif name == "two_dots":
        a = (np.hypot(xx - 0.45, yy) < 0.4) | (np.hypot(xx + 0.45, yy) < 0.4)
    else:
        raise ValueError(name)
    rgba = np.ones((n, n, 4)) * a[..., None]
    return Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")


def _render(name: str) -> Image.Image:
    if name in EMOJIS:
        return _render_emoji(EMOJIS[name])
    if name.startswith("emoji:"):
        return _render_emoji(name[len("emoji:"):])
    if name.startswith("text:"):
        return _render_text(name[len("text:"):])
    if name.startswith("image:"):
        im = Image.open(name[len("image:"):])
        if im.mode != "RGBA":  # no alpha: use luminance as alpha
            L = im.convert("L")
            im = im.convert("RGB")
            im.putalpha(L)
        return im
    if name in SHAPES:
        return _render_shape(name)
    raise ValueError(f"unknown target {name!r}; available: {', '.join(available())}")


def load_target(name: str, size: int, X: int, Y: int) -> np.ndarray:
    """Render target `name`, crop to its content, fit it in a size x size box, center it in an X x Y grid."""
    im = _render(name).convert("RGBA")
    bbox = im.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError(f"target {name!r} is empty")
    im = im.crop(bbox)
    scale = size / max(im.size)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 255.  # (h, w, 4), rows first
    arr[..., :3] *= arr[..., 3:] > 0  # no color where there is no matter
    out = np.zeros((X, Y, 4), np.float32)
    h, w = arr.shape[:2]
    x0, y0 = (X - h) // 2, (Y - w) // 2
    out[x0:x0 + h, y0:y0 + w] = arr
    return out
