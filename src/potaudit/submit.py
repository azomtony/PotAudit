from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SubmitReport:
    submitted: List[str]   # job folder names (e.g., "000123")
    skipped: List[str]     # already submitted/completed/etc
    inflight: int


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


def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    return r.stdout.strip()


def _parse_sbatch_jobid(s: str) -> str:
    # "Submitted batch job 29496284"
    m = re.search(r"Submitted batch job\s+(\d+)", s)
    if not m:
        raise RuntimeError(f"Could not parse sbatch job id from: {s}")
    return m.group(1)


def _get_inflight_jobids(out_root: Path) -> List[str]:
    """
    Fast path: count jobs we *know* are inflight based on our state.json.
    (We can cross-check with squeue in status(), not here.)
    """
    jobids: List[str] = []
    for d in sorted(out_root.iterdir()):
        if not d.is_dir():
            continue
        st = _read_state(_state_path(d))
        if not st:
            continue
        if st.get("submitted") and not st.get("completed"):
            jid = st.get("slurm_job_id")
            if jid:
                jobids.append(str(jid))
    return jobids


def submit_jobs(
    *,
    out_root: str,
    max_inflight: int = 100,
    watch: bool = False,
    poll_sec: int = 60,
    limit: Optional[int] = None,
) -> SubmitReport:
    out_root_p = Path(out_root).resolve()
    if not out_root_p.exists():
        raise FileNotFoundError(f"out_root not found: {out_root_p}")

    def one_pass() -> SubmitReport:
        inflight = len(_get_inflight_jobids(out_root_p))
        capacity = max(0, max_inflight - inflight)
        if limit is not None:
            capacity = min(capacity, max(0, int(limit)))

        submitted: List[str] = []
        skipped: List[str] = []

        if capacity == 0:
            return SubmitReport(submitted=submitted, skipped=skipped, inflight=inflight)

        for jobdir in sorted(out_root_p.iterdir()):
            if capacity == 0:
                break
            if not jobdir.is_dir():
                continue

            st_path = _state_path(jobdir)
            st = _read_state(st_path)

            # must be prepared
            if not st.get("prepared", False):
                skipped.append(jobdir.name)
                continue

            # already done or already submitted
            if st.get("completed", False) or st.get("submitted", False):
                skipped.append(jobdir.name)
                continue

            sub_sh = jobdir / "sub_vasp.sh"
            if not sub_sh.exists():
                raise FileNotFoundError(f"Missing sub_vasp.sh in {jobdir}")

            # submit
            out = _run(["sbatch", "sub_vasp.sh"], cwd=jobdir)
            jid = _parse_sbatch_jobid(out)

            st["submitted"] = True
            st["slurm_job_id"] = jid
            st["slurm_state"] = "SUBMITTED"
            st["submitted_at"] = _utcnow()
            st["checked_at"] = _utcnow()
            _write_state(st_path, st)

            submitted.append(jobdir.name)
            capacity -= 1
            inflight += 1

        return SubmitReport(submitted=submitted, skipped=skipped, inflight=inflight)

    if not watch:
        return one_pass()

    # Watch mode: keep topping up until no more to submit
    while True:
        rep = one_pass()
        # stop if we didn't submit anything
        if len(rep.submitted) == 0:
            return rep
        import time
        time.sleep(max(5, int(poll_sec)))