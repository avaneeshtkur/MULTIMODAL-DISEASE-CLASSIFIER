"""
image_hash.py — exact-content hashing.

SHA-256 of the uploaded file bytes identifies *byte-identical* files only.
It says nothing about patient identity, and two visually similar images will
have completely different hashes.
"""
import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def short_hash(h: str, n: int = 12) -> str:
    return h[:n] if h else ""
