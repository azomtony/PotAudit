from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class StatusReport:
    updated: int
    ok: int
    bad: int
    running: int
    pending: int
    lines: List[str]   # per-job human-readable lines


def _utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _read_state(p: Path) -> Dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _write_state(p: Path, state: Dict) -> None:
    p.write_text(json.dumps(state, indent=2) + "\n")


def _state_path(jobdir: Path) -> Path:
    return jobdir / "state.json"


def _run(cmd: List[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
    return r.stdout.strip()


def _slurm_squeue_state(jobid: str) -> Optional[str]:
    # returns e.g. RUNNING, PENDING or None if not in squeue
    try:
        out = _run(["squeue", "-j", jobid, "-h", "-o", "%T"])
    except Exception:
        return None
    out = out.strip()
    if not out:
        return None
    return out.splitlines()[0].strip()


def _slurm_sacct_state(jobid: str) -> Tuple[Optional[str], Optional[str]]:
    # returns (State, ExitCode) like ("COMPLETED", "0:0")
    out = _run(["sacct", "-j", jobid, "--format=State,ExitCode", "-n", "-P"])
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split("|")
        if len(parts) >= 2:
            return parts[0].strip(), parts[1].strip()
    return None, None


def _vasp_validate(jobdir: Path) -> Tuple[bool, str, Optional[float], Optional[int]]:
    """
    Returns: (ok, reason, energy_toten_ev, nions)
    """
    outcar = jobdir / "OUTCAR"
    if not outcar.exists() or outcar.stat().st_size < 1000:
        return False, "missing_or_small_OUTCAR", None, None

    txt = outcar.read_text(errors="ignore")

    if "General timing and accounting" not in txt:
        return False, "no_timing_footer", None, None

    nions = None
    m = re.search(r"NIONS\s*=\s*(\d+)", txt)
    if m:
        nions = int(m.group(1))

    toten = None
    for m in re.finditer(r"free\s+energy\s+TOTEN\s*=\s*([-\d\.Ee+]+)\s+eV", txt):
        toten = float(m.group(1))
    if toten is None:
        return False, "no_TOTEN_found", None, nions

    if "TOTAL-FORCE (eV/Angst)" not in txt:
        return False, "no_force_block", toten, nions

    return True, "ok", toten, nions


def _compact_live_state(sq: str) -> str:
    s = sq.strip().upper()
    # normalize common slurm strings to your labels
    if s.startswith("RUN"):
        return "running"
    if s.startswith("PEND"):
        return "pending"
    # other states (COMPLETING, CONFIGURING, SUSPENDED, etc.)
    return s.lower()


def status_update(*, out_root: str) -> StatusReport:
    out_root_p = Path(out_root).resolve()
    updated = 0
    ok = bad = running = pending = 0
    lines: List[str] = []

    for jobdir in sorted(out_root_p.iterdir()):
        if not jobdir.is_dir():
            continue

        st_path = _state_path(jobdir)
        st = _read_state(st_path)

        # Only report jobs we have submitted (your original behavior)
        if not st or not st.get("submitted"):
            continue

        jid = st.get("slurm_job_id")
        if not jid:
            lines.append(f"{jobdir.name} [missing_jobid]")
            continue

        sq = _slurm_squeue_state(str(jid))
        if sq is not None:
            st["slurm_state"] = sq
            st["checked_at"] = _utcnow()
            updated += 1

            label = _compact_live_state(sq)
            if label == "running":
                running += 1
            elif label == "pending":
                pending += 1

            lines.append(f"{jobdir.name} [{label}]")
            _write_state(st_path, st)
            continue

        # Not in squeue => terminal state from sacct
        state, exitcode = _slurm_sacct_state(str(jid))
        st["slurm_state"] = state
        st["slurm_exit_code"] = exitcode
        st["checked_at"] = _utcnow()

        terminal_ok = (state == "COMPLETED") and (exitcode == "0:0")

        if terminal_ok:
            v_ok, reason, toten, nions = _vasp_validate(jobdir)

            st["vasp_ok"] = bool(v_ok)
            st["vasp_reason"] = reason
            st["vasp_energy_toten_ev"] = toten
            st["vasp_nions"] = nions

            st["completed"] = True
            st["completed_at"] = _utcnow()
            st["failed"] = (not v_ok)

            if v_ok:
                ok += 1
                lines.append(f"{jobdir.name} [completed/ok]")
            else:
                bad += 1
                lines.append(f"{jobdir.name} [completed/bad:{reason}]")
        else:
            st["completed"] = True
            st["failed"] = True
            st["completed_at"] = _utcnow()
            st["vasp_ok"] = False
            st["vasp_reason"] = f"slurm_{state}_{exitcode}"

            bad += 1
            lines.append(f"{jobdir.name} [completed/bad:slurm_{state}_{exitcode}]")

        updated += 1
        _write_state(st_path, st)

    return StatusReport(
        updated=updated,
        ok=ok,
        bad=bad,
        running=running,
        pending=pending,
        lines=lines,
    )