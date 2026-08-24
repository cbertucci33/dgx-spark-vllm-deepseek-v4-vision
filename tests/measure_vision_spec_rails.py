#!/usr/bin/env python3
"""Measure DSpark acceptance and inter-node rail traffic for one image request."""

import base64
import json
import os
import pathlib
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from dspark_acceptance import (
    calculate_acceptance_stats,
    delta_spec_metrics,
    parse_spec_metrics,
    validate_minimum_drafts,
    validate_minimum_rate,
    validate_qualification,
)

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
MIN_ACCEPTANCE_RATE = validate_minimum_rate(
    float(os.environ.get("MIN_DSPARK_ACCEPTANCE_RATE", "0.20"))
)
MIN_DRAFTS = validate_minimum_drafts(os.environ.get("MIN_DSPARK_DRAFTS", "8"))
MAX_REQUESTS = validate_minimum_drafts(
    os.environ.get("MAX_DSPARK_QUALIFICATION_REQUESTS", "8")
)
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
    return parse_spec_metrics(text)


before_spec = spec_metrics()
before = {"head": local_counters(), "worker": remote_counters()}
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
completions = []
completion_tokens = 0
after_spec = before_spec
for _ in range(MAX_REQUESTS):
    request = urllib.request.Request(
        CHAT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        completion = json.loads(response.read())
    usage = completion.get("usage")
    if not isinstance(usage, dict) or "completion_tokens" not in usage:
        raise RuntimeError("completion response is missing usage.completion_tokens")
    completion_tokens += usage["completion_tokens"]
    completions.append(completion)
    after_spec = spec_metrics()
    if delta_spec_metrics(before_spec, after_spec).num_drafts >= MIN_DRAFTS:
        break
after = {"head": local_counters(), "worker": remote_counters()}

deltas = {
    host: {key: after[host][key] - before[host][key] for key in before[host]}
    for host in ("head", "worker")
}
evidence = {
    "before": {**before, "spec": before_spec.as_evidence()},
    "after": {**after, "spec": after_spec.as_evidence()},
    "deltas": deltas,
    "minimum_acceptance_rate": MIN_ACCEPTANCE_RATE,
    "minimum_drafts": MIN_DRAFTS,
    "completions": completions,
    "maximum_requests": MAX_REQUESTS,
}
evidence_path = OUT / "vision-spec-rail-evidence.json"
try:
    spec_delta = delta_spec_metrics(before_spec, after_spec)
    deltas["spec"] = spec_delta._asdict()
    stats = calculate_acceptance_stats(spec_delta)
    evidence["acceptance"] = stats._asdict()
    validate_qualification(
        spec_delta,
        minimum_rate=MIN_ACCEPTANCE_RATE,
        minimum_drafts=MIN_DRAFTS,
        completion_tokens=completion_tokens,
    )
except Exception as exc:
    evidence["qualification_error"] = str(exc)
    evidence_path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    raise
evidence_path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
print(
    json.dumps(
        {
            "deltas": deltas,
            "acceptance": stats._asdict(),
            "minimum_acceptance_rate": MIN_ACCEPTANCE_RATE,
            "minimum_drafts": MIN_DRAFTS,
            "messages": [item["choices"][0]["message"] for item in completions],
            "completion_tokens": completion_tokens,
            "request_count": len(completions),
        },
        indent=2,
        allow_nan=False,
    )
)
