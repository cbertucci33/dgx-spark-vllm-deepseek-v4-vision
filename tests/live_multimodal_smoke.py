#!/usr/bin/env python3
import base64
import json
import pathlib
import sys
import urllib.request

ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "DeepSeek-V4-Flash-0731-Vision"
OUT = pathlib.Path(sys.argv[1])
IMAGES = [(pathlib.Path(sys.argv[2]), "ORBIT-7391"), (pathlib.Path(sys.argv[3]), "NOVA-2846")]
OUT.mkdir(parents=True, exist_ok=True)


def request(payload):
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.loads(response.read())


def content(response):
    return response["choices"][0]["message"].get("content") or ""


common = {
    "model": MODEL,
    "temperature": 0,
    "max_tokens": 64,
}

text_payload = dict(common)
text_payload["chat_template_kwargs"] = {"thinking": False}
text_payload["messages"] = [{"role": "user", "content": "Reply with exactly: VISION TEXT PATH OK"}]
text_response = request(text_payload)
(OUT / "text-response.json").write_text(json.dumps(text_response, indent=2) + "\n")
text_answer = content(text_response).strip()
if text_answer != "VISION TEXT PATH OK":
    raise AssertionError(f"text answer mismatch: {text_answer!r}")
print(f"text_smoke=PASS answer={text_answer!r}")

for image_path, expected in IMAGES:
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    payload = dict(common)
    payload["chat_template_kwargs"] = {"thinking": False}
    payload["messages"] = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            {"type": "text", "text": "Read the large alphanumeric code in this image. Return only the code, with no explanation."},
        ],
    }]
    response = request(payload)
    (OUT / f"{image_path.stem}-response.json").write_text(json.dumps(response, indent=2) + "\n")
    answer = content(response).strip()
    if answer != expected:
        raise AssertionError(f"{image_path.name}: expected {expected!r}, got {answer!r}")
    print(f"image_smoke=PASS image={image_path.name} answer={answer!r}")

print("live_multimodal_contract=PASS")