#!/usr/bin/env python3
"""Download the exact NAIF kernels used by the published evaluator."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data" / "ephemerides"
CHUNK_SIZE = 1024 * 1024
FILES = {
    "de430.bsp": {
        "url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp",
        "sha256": "6e1b277c5f07135a84950604b83e56b736be696a7f3560bcddb1d4aeb944fca1",
        "size": 119741440,
    },
    "naif0012.tls": {
        "url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
        "sha256": "678e32bdb5a744117a467cd9601cd6b373f0e9bc9bbde1371d5eee39600a039b",
        "size": 5257,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, spec: dict[str, object]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(spec["size"])
        and sha256(path) == str(spec["sha256"])
    )


def download(name: str, spec: dict[str, object]) -> None:
    destination = DATA_DIR / name
    if verified(destination, spec):
        print(f"[OK] {name}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        str(spec["url"]), headers={"User-Agent": "SimpleTES-data-setup/1.0"}
    )
    print(f"[download] {spec['url']}")
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        while chunk := response.read(CHUNK_SIZE):
            output.write(chunk)

    if not verified(temporary, spec):
        raise RuntimeError(
            f"Downloaded {name} failed size or SHA-256 verification; "
            f"temporary file retained at {temporary}"
        )
    os.replace(temporary, destination)
    print(f"[verified] {name}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in FILES.items():
        download(name, spec)
    marker = DATA_DIR / ".verified"
    marker.write_text(
        "\n".join(f"{spec['sha256']}  {name}" for name, spec in FILES.items())
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
