import base64
import json
import logging
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image

logger = logging.getLogger(__name__)


def _maybe_parse_json_content(content: Any) -> Any:
    """Handle cases where OpenAI-style `content` list is JSON-dumped into a string."""
    if not isinstance(content, str):
        return content
    s = content.strip()
    if not s:
        return content
    # Fast path: only attempt JSON parse when it looks like a JSON container.
    if not (s.startswith("[") or s.startswith("{")):
        return content
    try:
        return json.loads(s)
    except Exception:
        return content


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a shallow-normalized messages list with parsed JSON-string content if applicable."""
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        new_msg = dict(msg)
        if "content" in new_msg:
            new_msg["content"] = _maybe_parse_json_content(new_msg["content"])
        normalized.append(new_msg)
    return normalized


def iter_content_parts(content: Any) -> Iterable[Dict[str, Any]]:
    content = _maybe_parse_json_content(content)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                yield part


def _extract_image_url_from_part(part: Dict[str, Any]) -> Optional[str]:
    # OpenAI-style: {"type":"image_url","image_url":{"url":"data:image/..."}}
    if "image_url" in part:
        val = part.get("image_url")
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            url = val.get("url")
            return url if isinstance(url, str) else None
        return None

    # Some datasets/tools: {"type":"image","image":"file:///..."} or {"image":"..."}
    if "image" in part:
        val = part.get("image")
        return val if isinstance(val, str) else None

    # Fallback by `type`
    t = part.get("type")
    if t == "image_url":
        val = part.get("url")
        return val if isinstance(val, str) else None
    if t == "image":
        val = part.get("image")
        return val if isinstance(val, str) else None

    return None


def extract_image_urls_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract image url/path/data-uri strings in encounter order."""
    urls: List[str] = []
    for msg in normalize_messages(messages):
        for part in iter_content_parts(msg.get("content")):
            url = _extract_image_url_from_part(part)
            if url:
                urls.append(url)
    return urls


def has_image_in_messages(messages: List[Dict[str, Any]]) -> bool:
    return len(extract_image_urls_from_messages(messages)) > 0


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        white = Image.new("RGB", img.size, (255, 255, 255))
        white.paste(img, mask=img.split()[3])
        return white
    return img.convert("RGB")


def load_pil_image(image_ref: str, timeout_s: int = 30) -> Image.Image:
    """Load an image reference (data URI / http(s) URL / file:// / local path) into RGB PIL.Image."""
    if image_ref.startswith("data:image") and "base64," in image_ref:
        _, b64 = image_ref.split("base64,", 1)
        raw = base64.b64decode(b64)
        with BytesIO(raw) as bio:
            img = Image.open(bio).copy()
        return _to_rgb(img)

    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        resp = requests.get(image_ref, stream=True, timeout=timeout_s)
        resp.raise_for_status()
        with BytesIO(resp.content) as bio:
            img = Image.open(bio).copy()
        return _to_rgb(img)

    if image_ref.startswith("file://"):
        image_ref = image_ref[7:]

    img = Image.open(image_ref).copy()
    return _to_rgb(img)


def load_pil_images(image_refs: List[str], timeout_s: int = 30) -> List[Image.Image]:
    images: List[Image.Image] = []
    for ref in image_refs:
        try:
            images.append(load_pil_image(ref, timeout_s=timeout_s))
        except Exception as e:
            logger.warning(f"Failed to load image ref={ref!r}: {e}")
            raise
    return images


def encode_image_for_rollout_engine(image: Image.Image) -> str:
    """Encode a PIL image as PNG bytes, then base64 string (no data-uri prefix)."""
    buffer = BytesIO()
    image = _to_rgb(image)
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def split_messages_prompt_and_assistant(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """If the last message is an assistant, split it out; else return (messages, None)."""
    if not messages:
        return [], None
    last = messages[-1]
    if isinstance(last, dict) and last.get("role") == "assistant":
        return messages[:-1], last
    return messages, None
