from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../portfolio_hjb
VENV_PY = ROOT / ".venv" / "bin" / "python"


@dataclass
class ActionResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


def run_daily_now(
    *,
    window: int,
    annual_days: int,
    nx: int,
    x_min_factor: float,
    x_max_factor: float,
    xmin_floor: float,
    bc: str,
    timeout_s: int = 180,
    outputsize: str = "compact",
    max_steps: int | None = None,
    dry_run: bool = False,
) -> ActionResult:
    cmd = [
        str(VENV_PY),
        "src/main.py",
        "run_daily",
        "--outputsize", outputsize,
        "--window", str(window),
        "--annual_days", str(annual_days),
        "--nx", str(nx),
        "--x_min_factor", str(x_min_factor),
        "--x_max_factor", str(x_max_factor),
        "--xmin_floor", str(xmin_floor),
        "--bc", str(bc),
    ]
    if max_steps is not None:
        cmd += ["--max_steps", str(max_steps)]
    if dry_run:
        cmd += ["--dry_run"]

    try:
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return ActionResult(
            ok=(p.returncode == 0),
            stdout=(p.stdout or "").strip(),
            stderr=(p.stderr or "").strip(),
            returncode=p.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as e:
        return ActionResult(
            ok=False,
            stdout=(getattr(e, "stdout", "") or "").strip(),
            stderr=(getattr(e, "stderr", "") or "").strip(),
            returncode=124,
            timed_out=True,
        )
