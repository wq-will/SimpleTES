"""Code normalization helpers for GPUMode tasks."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any


_INVALID_ESCAPE_RE = re.compile(r"\\([A-Za-z])")
_VALID_ESCAPE_START = set(list("abfnrtv\\'\"0xuUN"))


def _sanitize_module_docstring_for_python312(src: str) -> str:
    """
    Best-effort sanitizer for Python 3.12+:
    - A module-level docstring like \"\"\"...\\s...\"\"\" triggers `SyntaxWarning: invalid escape sequence '\\s'`.
    - This is harmless but noisy, and in some environments warnings can be treated as errors.

    We avoid touching executable code by only operating on a *leading module docstring*.
    Strategy:
    - Detect an initial triple-quoted string literal (after leading comments/blank lines).
    - If it contains invalid escape sequences, rewrite it to a raw string literal by adding an `r` prefix.
      (If the docstring ends with a backslash, raw strings are invalid; in that rare case, we drop the docstring.)
    """
    s = str(src or "")
    if not s.strip():
        return s

    # Match leading comments/blank lines, then an optional string prefix + triple-quoted literal.
    m = re.match(
        r"^(?P<header>(?:[ \t]*#.*\n|[ \t]*\n)*)"
        r"(?P<prefix>[rRuUfFbB]{0,4})"
        r"(?P<quote>\"\"\"|''')"
        r"(?P<body>[\s\S]*?)"
        r"(?P=quote)"
        r"(?P<tail>[ \t]*\n)?",
        s,
    )
    if not m:
        return s

    header = m.group("header") or ""
    prefix = m.group("prefix") or ""
    quote = m.group("quote") or '"""'
    body = m.group("body") or ""
    tail = m.group("tail") or ""

    # Detect *invalid* escape sequences like \s, \d, etc. (we keep this heuristic simple).
    invalid = False
    for em in _INVALID_ESCAPE_RE.finditer(body):
        ch = em.group(1)
        if ch not in _VALID_ESCAPE_START:
            invalid = True
            break
    if not invalid:
        return s

    # Raw strings cannot end with an odd backslash.
    if body.rstrip().endswith("\\"):
        rest = s[m.end() :]
        return header + rest

    if "r" in prefix.lower():
        return s  # already raw; nothing to do

    new_prefix = prefix + "r"
    rest = s[m.end() :]
    return header + new_prefix + quote + body + quote + tail + rest


def normalize_triton_submission_py(submission_code: Any) -> str:
    """
    Normalize Triton/Python submission content into a full `submission.py` string.

    Accepts:
    - a raw string (submission.py content)
    - a dict with key "submission.py"

    Strict mode: the submission MUST define `custom_kernel(data)` (GPUMode entrypoint).
    We do NOT auto-wrap `kernel(...)` into `custom_kernel(data)`; generation must follow the
    required format.
    """
    if isinstance(submission_code, dict):
        text = str(submission_code.get("submission.py", "") or "")
    else:
        text = str(submission_code or "")
    s = text.strip()
    if not s:
        return ""
    s = _sanitize_module_docstring_for_python312(s)
    if re.search(r"def\s+custom_kernel\s*\(", s):
        return s
    raise ValueError("GPUMode submission.py must define custom_kernel(data)")


def parse_cuda_xml_sources(code: str) -> dict[str, str]:
    patterns = {
        "kernel.h": r'<header_file name="kernel\.h">(.*?)</header_file>',
        "kernel.cu": r'<cuda_file name="kernel\.cu">(.*?)</cuda_file>',
        "main.cpp": r'<cpp_file name="main\.cpp">(.*?)</cpp_file>',
    }
    files: dict[str, str] = {}
    for filename, pattern in patterns.items():
        match = re.search(pattern, str(code or ""), re.DOTALL)
        if not match:
            raise ValueError(f"Missing {filename} block in CUDA XML output")
        content = (match.group(1) or "").strip()
        if not content:
            raise ValueError(f"Empty {filename} block in CUDA XML output")
        files[filename] = content
    return files


def normalize_cuda_sources(submission_code: Any) -> dict[str, str]:
    if isinstance(submission_code, dict):
        files = {str(k): str(v) for k, v in submission_code.items()}
    else:
        files = parse_cuda_xml_sources(str(submission_code or ""))
    required = ("kernel.h", "kernel.cu", "main.cpp")
    missing = [k for k in required if not str(files.get(k, "")).strip()]
    if missing:
        raise ValueError(f"Missing required CUDA sources: {missing}")
    return {k: str(files[k]) for k in required}


def cuda_sources_to_submission_py(sources: dict[str, str]) -> str:
    """Convert CUDA sources into a Python `submission.py` that builds a torch extension."""
    sources = normalize_cuda_sources(sources)
    sources_json = json.dumps(sources)
    digest = hashlib.sha1(sources_json.encode("utf-8")).hexdigest()[:12]
    return f"""import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_SOURCES = json.loads(r'''{sources_json}''')
_EXT = None

def _build_ext():
    build_root = Path(os.environ.get("FIB_CACHE_PATH", str(Path.home() / ".cache" / "k_search" / "cache")))
    build_dir = build_root / "gpumode_cuda" / "{digest}"
    build_dir.mkdir(parents=True, exist_ok=True)
    for name, content in _SOURCES.items():
        (build_dir / name).write_text(content)
    sources = [str(build_dir / "main.cpp"), str(build_dir / "kernel.cu")]
    ext = load(
        name="gpumode_cuda_{digest}",
        sources=sources,
        extra_include_paths=[str(build_dir)],
        with_cuda=True,
        build_directory=str(build_dir),
        verbose=False,
    )
    return ext

def custom_kernel(data):
    global _EXT
    if _EXT is None:
        _EXT = _build_ext()
    input_tensor, mask, weights, config = data
    return _EXT.run(input_tensor, mask, weights, config)
"""


