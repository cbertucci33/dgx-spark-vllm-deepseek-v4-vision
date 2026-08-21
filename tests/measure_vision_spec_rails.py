#!/usr/bin/env python3
"""Measure DSpark acceptance and inter-node rail traffic for one image request."""

import base64
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.request

if len(sys.argv) != 3:
    raise SystemExit(f"usage: {sys.argv[0]} OUTPUT_DIRECTORY IMAGE.png")

OUT = pathlib.Path(sys.argv[1]).resolve()
IMAGE = pathlib.Path(sys.argv[2]).resolve()
BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
WORKER_SSH = os.environ.get("WORKER_SSH", "worker")
HCAS = tuple(x for x in os.environ.get("NCCL_IB_HCA", "").split(",") if x)
if not HCAS:
    raise SystemExit("NCCL_IB_HCA must explicitly list the target RoCE devices")
MODEL = os.environ.get("SERVED_MODEL_NAME", "DeepSeek-V4-Flash-0731-Vision")
METRICS = BASE_URL.rstrip("/") + "/metrics"
CHAT = BASE_URL.rstrip("/") + "/v1/chat/completions"
OUT.mkdir(parents=True, exist_ok=False)
if not IMAGE.is_file():
    raise ValueError(f"image does not exist: {IMAGE}")


def local_counters():
    result = {}
    for hca in HCAS:
        for counter in ("port_xmit_data", "port_rcv_data"):
            path = pathlib.Path(f"/sys/class/infiniband/{hca}/ports/1/counters/{counter}")
            result[f"{hca}.{counter}"] = int(path.read_text())
    return result


def remote_counters():
    code = (
        "import json,pathlib; hs=" + repr(HCAS) + "; "
        "print(json.dumps({f'{h}.{c}':int(pathlib.Path(f'/sys/class/infiniband/{h}/ports/1/counters/{c}').read_text()) "
        "for h in hs for c in ('port_xmit_data','port_rcv_data')}))"
    )
    return json.loads(
        subprocess.check_output(["ssh", WORKER_SSH, "python3", "-c", code], text=True)
    )


def spec_metrics():
    with urllib.request.urlopen(METRICS, timeout=30) as response:
        text = response.read().decode()
    result = {}
    for name in (
        "spec_decode_num_draft_tokens_total",
        "spec_decode_num_accepted_tokens_total",
    ):
        match = re.search(
            rf'^vllm:{name}\{{[^\n]*\}} ([0-9.e+-]+)$', text, re.MULTILINE
        )
        if not match:
            raise RuntimeError(f"missing metric {name}")
        result[name] = float(match.group(1))
    return result


before = {"head": local_counters(), "worker": remote_counters(), "spec": spec_metrics()}
payload = {
    "model": MODEL,
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 64,
    "min_tokens": 48,
    "chat_template_kwargs": {"thinking": False},
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(IMAGE.read_bytes()).decode()
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this image in detail, including exact text, colors, and layout.",
                },
            ],
        }
    ],
}
request = urllib.request.Request(
    CHAT,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    completion = json.loads(response.read())
after = {"head": local_counters(), "worker": remote_counters(), "spec": spec_metrics()}

deltas = {
    host: {key: after[host][key] - before[host][key] for key in before[host]}
    for host in ("head", "worker")
}
deltas["spec"] = {
    key: after["spec"][key] - before["spec"][key] for key in before["spec"]
}
evidence = {"before": before, "after": after, "deltas": deltas, "completion": completion}
(OUT / "vision-spec-rail-evidence.json").write_text(
    json.dumps(evidence, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "deltas": deltas,
            "message": completion["choices"][0]["message"],
            "usage": completion.get("usage"),
        },
        indent=2,
    )
)