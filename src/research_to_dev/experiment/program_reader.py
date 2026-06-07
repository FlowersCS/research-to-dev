"""Parse program.md frontmatter and body using manual regex split + yaml.safe_load (D1).

Produces a ``ProgramSpec`` dataclass with all mandatory fields validated
at parse time.  Duration strings are converted to seconds immediately
(D5), direction is inferred from success_criteria operators if not
explicitly overridden (D6).

Usage::

    from research_to_dev.experiment.program_reader import parse_program_md
    spec = parse_program_md(Path("program.md"))
    print(spec.time_budget_seconds)  # 1800
    print(spec.direction)            # "minimize"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Duration parsing (D5)
# ---------------------------------------------------------------------------


def parse_duration(s: str) -> int:
    """Convert ``"30m"``, ``"2h"``, ``"1h30m"`` to integer seconds.

    Args:
        s: Duration string in ``[Nh][Nm]`` format.

    Returns:
        Integer representing total seconds.

    Raises:
        ValueError: If the format is invalid or the string is empty.
    """
    s = s.strip()
    if not s:
        raise ValueError(
            "Duration string is empty. Use format like '30m', '2h', or '1h30m'."
        )

    total = 0
    idx = 0
    n = len(s)

    # Optional hours: e.g. "2h"
    h_match = re.match(r"(\d+)h", s[idx:])
    if h_match:
        total += int(h_match.group(1)) * 3600
        idx += h_match.end()

    # Required minutes: e.g. "30m"
    m_match = re.match(r"(\d+)m", s[idx:])
    if m_match:
        total += int(m_match.group(1)) * 60
        idx += m_match.end()

    # Must consume the entire string and have parsed something
    if idx != n:
        raise ValueError(
            f"Invalid duration format '{s}'. "
            f"Use format like '30m', '2h', or '1h30m'."
        )
    if total == 0:
        raise ValueError(
            f"Duration '{s}' resolves to 0 seconds. "
            f"Specify a positive duration like '30m', '2h', or '1h30m'."
        )

    return total


# ---------------------------------------------------------------------------
# Direction inference (D6)
# ---------------------------------------------------------------------------

_OPERATOR_DIRECTION = [
    (r">\s*=", "maximize"),
    (r">(?!=)", "maximize"),
    (r"<\s*=", "minimize"),
    (r"<(?!=)", "minimize"),
]


def _infer_direction(success_criteria: str) -> str:
    """Infer direction from the operator in ``success_criteria``.

    ``>=``, ``>`` → ``"maximize"``
    ``<=``, ``<`` → ``"minimize"``

    Raises ``ValueError`` if no clear operator is found.
    """
    for pattern, direction in _OPERATOR_DIRECTION:
        if re.search(pattern, success_criteria):
            return direction

    raise ValueError(
        f"Cannot infer direction from success_criteria "
        f"'{success_criteria}'. "
        f"Use an operator like '>=', '>', '<=', or '<', "
        f"or set 'direction' explicitly in program.md frontmatter."
    )


# ---------------------------------------------------------------------------
# ProgramSpec dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProgramSpec:
    """Parsed representation of a ``program.md`` file.

    All mandatory fields are validated at parse time — missing fields
    raise ``ValueError`` immediately (fail-loud, T3).
    """

    hypothesis_id: str
    target_metric: str
    success_criteria: str
    time_budget: str  # Original string e.g. "30m"
    max_iterations: int
    baseline: dict[str, float]  # e.g. {"val_accuracy": 0.72}
    run_command: str
    coding_agent_model: str
    time_budget_seconds: int  # Parsed from time_budget
    direction: str  # "maximize" or "minimize"
    body: str  # Markdown body (everything after frontmatter)


# ---------------------------------------------------------------------------
# Parser entry point
# ---------------------------------------------------------------------------

# Mandatory frontmatter fields (D3, D4, D12)
_MANDATORY_FIELDS = [
    "hypothesis_id",
    "target_metric",
    "success_criteria",
    "time_budget",
    "max_iterations",
    "baseline",
    "run_command",
    "coding_agent_model",
]


_FRONTMATTER_RE = re.compile(r"^---\s*$", re.MULTILINE)


def parse_program_md(path: Path) -> ProgramSpec:
    """Parse a ``program.md`` file into a ``ProgramSpec``.

    Splits on ``---`` delimiters, parses YAML frontmatter, validates
    mandatory fields, converts durations, and infers direction.

    Args:
        path: Path to the ``program.md`` file.

    Returns:
        ``ProgramSpec`` with all fields populated.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed or mandatory fields
            are missing.

    Spec refs: PR-01 through PR-06.
    """
    raw = path.read_text(encoding="utf-8")

    # Split on --- lines (D1)
    parts = _FRONTMATTER_RE.split(raw, maxsplit=2)

    if len(parts) < 3:
        raise ValueError(
            f"Invalid program.md format in '{path}': "
            f"expected YAML frontmatter delimited by '---' on its own line."
        )

    yaml_text = parts[1].strip()
    body = parts[2].lstrip("\n")

    # Parse YAML (D1)
    try:
        frontmatter = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML in program.md frontmatter: {exc}"
        ) from exc

    if not isinstance(frontmatter, dict):
        raise ValueError(
            f"program.md frontmatter must be a YAML mapping, "
            f"got {type(frontmatter).__name__}."
        )

    # Validate mandatory fields (D3, D4, D12)
    _validate_mandatory_fields(frontmatter, path)

    # Parse duration (D5)
    time_budget_str = str(frontmatter["time_budget"])
    time_budget_seconds = parse_duration(time_budget_str)

    # Validate max_iterations
    max_iterations = frontmatter["max_iterations"]
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError(
            f"max_iterations must be a positive integer, got {max_iterations!r}. "
            f"Fix it in {path}."
        )

    # Validate baseline (D4) — must be a dict
    baseline = frontmatter["baseline"]
    if not isinstance(baseline, dict) or len(baseline) == 0:
        raise ValueError(
            f"baseline must be a non-empty dict (e.g. {{val_accuracy: 0.72}}), "
            f"got {baseline!r}. Fix it in {path}."
        )
    # Convert values to float
    for key, val in baseline.items():
        baseline[key] = float(val)

    # Infer direction from success_criteria unless explicitly overridden (D6)
    direction = frontmatter.get("direction")
    if direction is not None:
        direction = str(direction)
        if direction not in ("maximize", "minimize"):
            raise ValueError(
                f"direction must be 'maximize' or 'minimize', "
                f"got '{direction}'. Fix it in {path}."
            )
    else:
        success_criteria = str(frontmatter["success_criteria"])
        direction = _infer_direction(success_criteria)

    return ProgramSpec(
        hypothesis_id=str(frontmatter["hypothesis_id"]),
        target_metric=str(frontmatter["target_metric"]),
        success_criteria=str(frontmatter["success_criteria"]),
        time_budget=time_budget_str,
        max_iterations=max_iterations,
        baseline=baseline,
        run_command=str(frontmatter["run_command"]),
        coding_agent_model=str(frontmatter["coding_agent_model"]),
        time_budget_seconds=time_budget_seconds,
        direction=direction,
        body=body,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_mandatory_fields(frontmatter: dict, path: Path) -> None:
    """Ensure all mandatory frontmatter fields are present.

    Raises ``ValueError`` listing any missing fields.
    """
    missing = [f for f in _MANDATORY_FIELDS if f not in frontmatter]
    if missing:
        fields_str = ", ".join(missing)
        raise ValueError(
            f"Missing mandatory field(s) in program.md frontmatter: "
            f"{fields_str}. Add {'them' if len(missing) > 1 else 'it'} "
            f"to {path}."
        )
