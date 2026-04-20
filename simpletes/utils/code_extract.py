"""EVOLVE-BLOCK and fenced-code parsing for LLM output.

Extracted from ``simpletes.node`` so the data model (Node,
NodeDatabase) stays focused on archive management while this module
owns the string-level parsing of generator output.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_CODE_FENCE_RE = re.compile(
    r"```([^\n`]*)\s*\r?\n(.*?)(?:\r?\n)?```",
    re.DOTALL,
)
_LANG_TAG_RE = re.compile(r"^[A-Za-z0-9_.+\-]{1,32}$")
_CODE_HINT_RE = re.compile(
    r"^\s*(def|class|import|from|if __name__|for|while|try:|with|@)\b",
    re.MULTILINE,
)

# EVOLVE-BLOCK markers for extracting only the evolving part
_EVOLVE_BLOCK_START = "# EVOLVE-BLOCK-START"
_EVOLVE_BLOCK_END = "# EVOLVE-BLOCK-END"


def _is_python_lang(tag: str) -> bool:
    tag = tag.strip().lower()
    if not tag:
        return False
    tag = tag.split()[0]
    if tag in {"py", "python"}:
        return True
    if tag.startswith("python"):
        return True
    return tag.startswith("py")


def _strip_possible_lang_header(text: str) -> str:
    head, sep, rest = text.partition("\n")
    if sep and _LANG_TAG_RE.match(head.strip()):
        return rest
    return text


def _looks_like_code(text: str) -> bool:
    if _CODE_HINT_RE.search(text):
        return True
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


@dataclass
class EvolveBlockContext:
    """Context for EVOLVE-BLOCK extraction.

    Stores the fixed parts (prefix before START and suffix after END) from the
    original init_program. When extracting code from LLM output, only the content
    between the markers is taken, then merged with these fixed parts.
    """
    prefix: str = ""  # Everything before # EVOLVE-BLOCK-START (including the marker)
    suffix: str = ""  # Everything after # EVOLVE-BLOCK-END (including the marker)
    has_markers: bool = False  # Whether the original program has EVOLVE-BLOCK markers

    @classmethod
    def from_program(cls, program_code: str) -> EvolveBlockContext:
        """Extract fixed parts from the original program code."""
        if not program_code:
            return cls()

        lines = program_code.splitlines(keepends=True)
        start_idx = -1
        end_idx = -1

        for i, line in enumerate(lines):
            if _EVOLVE_BLOCK_START in line:
                start_idx = i
            elif _EVOLVE_BLOCK_END in line and start_idx >= 0:
                end_idx = i
                break

        if start_idx < 0 or end_idx < 0:
            return cls(has_markers=False)

        # Include markers in prefix/suffix so they remain in the reconstructed program.
        prefix = "".join(lines[: start_idx + 1]).rstrip("\n")
        suffix = "".join(lines[end_idx:]).lstrip("\n")
        return cls(prefix=prefix, suffix=suffix, has_markers=True)

    def merge_with_evolved_block(self, evolved_block: str) -> str | None:
        """Merge an extracted evolved block back with the surrounding fixed parts."""
        evolved_clean = evolved_block.strip("\n")
        if not evolved_clean:
            return None

        # Combine: prefix + evolved_block + suffix
        # prefix ends with START marker, suffix starts with END marker
        return f"{self.prefix}\n{evolved_clean}\n{self.suffix}"


def _extract_evolve_block(text: str) -> str | None:
    """Extract content between EVOLVE-BLOCK markers if present.

    Returns the content between markers (without markers), or None if markers not found.
    """
    start_idx = text.find(_EVOLVE_BLOCK_START)
    if start_idx == -1:
        return None

    end_idx = text.find(_EVOLVE_BLOCK_END, start_idx)
    if end_idx == -1:
        return None

    # Extract content between the markers
    start_content = start_idx + len(_EVOLVE_BLOCK_START)
    return text[start_content:end_idx].strip("\n")


def extract_code_detailed(
    llm_output: str,
    evolve_context: EvolveBlockContext | None = None,
) -> tuple[str | None, str]:
    """Extract Python code with a reason for success/failure.

    When evolve_context is provided, EVOLVE-BLOCK markers are required in LLM output.
    The extracted block is merged with fixed parts from evolve_context.

    Returns (code, reason), where code is None on failure and reason indicates why.
    """
    text = llm_output or ""
    if not text.strip():
        return None, "empty_output"

    if evolve_context is not None:
        if not evolve_context.has_markers:
            return None, "init_program_missing_evolve_block_markers"
        evolved_block = _extract_evolve_block(text)
        if evolved_block is None:
            return None, "missing_evolve_block_markers"
        if not evolved_block.strip():
            return None, "empty_evolve_block"
        merged_code = evolve_context.merge_with_evolved_block(evolved_block)
        if merged_code:
            return merged_code, "evolve_block_merged"
        return None, "empty_evolve_block"

    # Optional EVOLVE-BLOCK extraction when no evolve_context is provided.
    evolved_block = _extract_evolve_block(text)
    if evolved_block is not None:
        if evolved_block.strip():
            return evolved_block, "evolve_block"
        return None, "empty_evolve_block"

    matches = list(_CODE_FENCE_RE.finditer(text))
    if matches:
        chosen = None
        for match in matches:
            lang = (match.group(1) or "").strip()
            if _is_python_lang(lang):
                chosen = match
        if chosen is None:
            chosen = matches[-1]
        code = (chosen.group(2) or "").strip()
        if code:
            return code, "code_block"
        return None, "empty_code_block"

    fence_idx = text.rfind("```")
    if fence_idx != -1:
        tail = text[fence_idx + 3:]
        tail = tail.lstrip("\r\n")
        tail = _strip_possible_lang_header(tail)
        code = tail.strip()
        if code:
            return code, "code_block_unclosed"

    candidate = text.strip()
    if _looks_like_code(candidate):
        return candidate, "raw_output"

    return None, "no_code_block"


def extract_code(
    llm_output: str,
    evolve_context: EvolveBlockContext | None = None,
) -> str | None:
    """Extract Python code from LLM output.

    Prefers EVOLVE-BLOCK marked regions if present, then fenced Python code blocks,
    and falls back to raw output if it parses as valid Python.
    """
    code, _reason = extract_code_detailed(llm_output, evolve_context)
    return code
