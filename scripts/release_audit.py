#!/usr/bin/env python3
"""Fail closed on privacy- and release-sensitive content."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from urllib.parse import urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT_PATH = "scripts/release_audit.py"
MAX_FILE_BYTES = 5 * 1024 * 1024
GENERIC_NAME = "Repository Maintainers"
GENERIC_EMAIL = "noreply@github.com"

APPROVED_URL_NAMESPACES = {
    "github.com": {
        "Anemll",
        "FlyCockpit",
        "MiaAI-Lab",
        "cbertucci33",
        "facebookresearch",
    },
    "huggingface.co": {
        "$repo",
        "FlyCockpit",
        "cbert33",
        "cebeuq",
        "deepseek-ai",
    },
}

# Reviewed non-person phrases currently used by documentation and licenses.
APPROVED_NAME_LIKE_PHRASES = {
    "Abliterated Vision",
    "Accepting Warranty",
    "Additional Liability",
    "Anemll Torch",
    "Apache License",
    "Container Toolkit",
    "Copyright License",
    "Derivative Works",
    "Docker Compose",
    "Efficient Million",
    "Hugging Face",
    "Legal Entity",
    "Patch Embedding",
    "Patent License",
    "Programming Language",
    "Qualified Anemll",
    "Relative Positional",
    "Repository Maintainers",
    "Stock Transformers",
    "The Anemll",
    "The Compose",
    "This License",
    "Token Context",
    "Towards Highly",
    "Unless You",
    "Vision Assets",
    "Vision Runtime",
    "When True",
}

PRIVATE_KEY_MARKERS = (
    "BEGIN OPENSSH PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
    "BEGIN EC PRIVATE KEY",
    "BEGIN DSA PRIVATE KEY",
)
PRIVATE_IPV4 = re.compile(
    r"(?<![0-9])(?:10(?:\.[0-9]{1,3}){3}|192\.168(?:\.[0-9]{1,3}){2}|"
    r"172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2})(?![0-9])"
)
USER_HOME_PATH = re.compile(r"(?:^|[\s='\"])/(?:Users|home)/[^/\s'\"]+/")
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
CONCRETE_NETWORK_DEVICE = re.compile(
    r"\b(?:enp[0-9][A-Za-z0-9]*|ens[0-9][A-Za-z0-9]*|mlx[0-9][A-Za-z0-9_]*|"
    r"roce[A-Za-z0-9][A-Za-z0-9]*)\b",
    re.IGNORECASE,
)
LOCAL_HOSTNAME = re.compile(
    r"(?i)(?<![A-Z0-9_])(?:[A-Z0-9-]+\.)+(?:local|lan|home|internal)"
    r"(?=[:/\s'\"]|$)"
)
NAME_LIKE = re.compile(
    r"\b[A-Z][a-z]{2,}(?:[-'][A-Z][a-z]+)?[ \t]+[A-Z][a-z]{2,}\b"
)
URL = re.compile(r"https://(?:github\.com|huggingface\.co)/[^\s)`\"']+")
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(api[_-]?key|password|passwd|secret|access[_-]?token|auth[_-]?token)"
    r"\s*[:=]\s*['\"]?([A-Za-z0-9_./+:-]{16,})"
)
SECRET_SIGNATURES = {
    "GitHub token": re.compile(r"\b(?:github_pat_|gh[oprsu]_)[A-Za-z0-9_]{20,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "Bearer credential": re.compile(r"(?i)\bBearer[ \t]+[A-Za-z0-9._~+/-]{20,}\b"),
}
FORBIDDEN_LEGACY_PLATFORM_TERM = re.compile("gx" + "10", re.IGNORECASE)
ALLOWED_EMAILS = {GENERIC_EMAIL}
HOST_ASSIGNMENT = re.compile(
    r"^[ \t]*(MASTER_ADDR|VLLM_HOST_IP|WORKER_HOST|WORKER_SSH|SSH_HOST|HOSTNAME)"
    r"[ \t]*[:=][ \t]*['\"]?([^'\"#\s]+)",
    re.MULTILINE,
)

errors: set[str] = set()


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def add_error(category: str, relative: str, source: str) -> None:
    errors.add(f"{category}: {relative} ({source})")


def scan_text(relative: str, text: str, source: str) -> None:
    is_audit = relative == AUDIT_PATH

    if FORBIDDEN_LEGACY_PLATFORM_TERM.search(text):
        add_error("forbidden legacy platform term", relative, source)

    if not is_audit:
        for marker in PRIVATE_KEY_MARKERS:
            if marker in text:
                add_error("private-key marker", relative, source)

    if PRIVATE_IPV4.search(text):
        add_error("private-network address", relative, source)
    if USER_HOME_PATH.search(text):
        add_error("user-specific home path", relative, source)
    if LOCAL_HOSTNAME.search(text):
        add_error("local hostname", relative, source)

    for address in EMAIL.findall(text):
        if address.lower() not in ALLOWED_EMAILS:
            add_error("unapproved email address", relative, source)

    privacy_text = re.sub(r"CHANGE_ME_[A-Z0-9_]+", "", text)
    if CONCRETE_NETWORK_DEVICE.search(privacy_text):
        add_error("concrete network device", relative, source)

    if relative.endswith((".env", ".example", ".example.env", ".yml", ".yaml", ".sh")):
        for match in HOST_ASSIGNMENT.finditer(text):
            value = match.group(2)
            allowed = (
                "CHANGE_ME" in value
                or value.startswith("$")
                or value in {"127.0.0.1", "0.0.0.0", "worker"}
                or bool(re.fullmatch(r"[A-Z][A-Z0-9_]+", value))
            )
            if not allowed:
                add_error("concrete host assignment", relative, source)

    for raw_url in URL.findall(text):
        parsed = urlparse(raw_url)
        parts = parsed.path.strip("/").split("/")
        namespace = parts[0] if parts else ""
        if namespace not in APPROVED_URL_NAMESPACES.get(parsed.netloc, set()):
            add_error("unapproved public URL namespace", relative, source)

    for phrase in NAME_LIKE.findall(text):
        if phrase not in APPROVED_NAME_LIKE_PHRASES:
            add_error("unreviewed name-like phrase", relative, source)

    if not is_audit:
        for match in SECRET_ASSIGNMENT.finditer(text):
            value = match.group(2)
            if not any(
                word in value.upper()
                for word in ("REDACTED", "EXAMPLE", "PLACEHOLDER", "CHANGE_ME")
            ):
                add_error("possible hard-coded secret assignment", relative, source)
        for label, pattern in SECRET_SIGNATURES.items():
            if pattern.search(text):
                add_error(f"possible {label}", relative, source)


def scan_bytes(relative: str, data: bytes, source: str) -> None:
    if len(data) > MAX_FILE_BYTES:
        add_error(f"file exceeds {MAX_FILE_BYTES}-byte limit", relative, source)
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        add_error("binary/non-UTF-8 file is forbidden", relative, source)
        return
    scan_text(relative, text, source)


# Scan the exact candidate working tree, including non-ignored untracked files.
candidates = subprocess.check_output(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
    cwd=ROOT,
    text=True,
).splitlines()
for relative in candidates:
    path = ROOT / relative
    if path.is_symlink():
        add_error("tracked symlink", relative, "worktree")
    elif path.is_file():
        if relative.endswith(".env") and not relative.endswith(".example.env"):
            add_error("private env file", relative, "worktree")
        else:
            scan_bytes(relative, path.read_bytes(), "worktree")

# Scan every unique blob reachable from HEAD so deletion in a later commit cannot
# hide content that remains publicly addressable through history.
seen_blobs: set[str] = set()
for commit in git("rev-list", "HEAD").decode().splitlines():
    for entry in git("ls-tree", "-rz", commit).split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, sha = metadata.decode().split()
        relative = raw_path.decode()
        if kind != "blob" or sha in seen_blobs:
            continue
        seen_blobs.add(sha)
        if mode == "120000":
            add_error("historical symlink", relative, commit[:12])
            continue
        scan_bytes(relative, git("cat-file", "blob", sha), commit[:12])

# Every reachable author and committer must use the generic public identity.
for line in git(
    "log", "HEAD", "--format=%H%x00%an%x00%ae%x00%cn%x00%ce"
).decode().splitlines():
    commit, author, author_email, committer, committer_email = line.split("\x00")
    if [author, author_email, committer, committer_email] != [
        GENERIC_NAME,
        GENERIC_EMAIL,
        GENERIC_NAME,
        GENERIC_EMAIL,
    ]:
        errors.add(f"non-generic commit identity: {commit[:12]}")

# Annotated public tags must use the same generic identity.
for tag in git("tag").decode().splitlines():
    kind = git("cat-file", "-t", tag).decode().strip()
    if kind == "tag":
        body = git("cat-file", "-p", tag).decode(errors="replace")
        match = re.search(r"^tagger (.+) <([^>]+)>", body, re.MULTILINE)
        if not match or [match.group(1), match.group(2)] != [GENERIC_NAME, GENERIC_EMAIL]:
            errors.add(f"non-generic annotated tag identity: {tag}")

if errors:
    print("RELEASE_AUDIT_FAILED", file=sys.stderr)
    for error in sorted(errors):
        print(f"- {error}", file=sys.stderr)
    raise SystemExit(1)

print(
    "RELEASE_AUDIT_PASS "
    f"files={len(candidates)} commits={len(git('rev-list', 'HEAD').decode().splitlines())} "
    f"history_blobs={len(seen_blobs)}"
)
