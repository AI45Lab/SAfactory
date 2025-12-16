import base64
import json
import pathlib
import requests

RENDER_URL = "http://100.101.28.243:8000/render"
OUTPUT_PATH = pathlib.Path("render_frame.png")

def main():
    resp = requests.post(RENDER_URL, timeout=60)
    resp.raise_for_status()

    payload = resp.json()
    img_b64 = payload.get("image_base64")
    if not img_b64:
        raise RuntimeError("render response missing image_base64")

    image_bytes = base64.b64decode(img_b64)
    OUTPUT_PATH.write_bytes(image_bytes)
    print(f"Saved render frame to {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()