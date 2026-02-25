from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# If you have status_update() in potaudit.status, we can call it in --watch mode.
# If you prefer not to import here, you can pass a callback instead.
try:
    from .status import status_update  # type: ignore
except Exception:  # pragma: no cover
    status_update = None  # optional


@dataclass(frozen=True)
class SubmitReport:
    submitted: List[str]   # job folder names (e.g., "000123")
    skipped: List[str]     # already submitted/completed/etc
    inflight: int
    capacity: int
    remaining_ready: int   # ready-to-submit jobs still left after this pass


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
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
    return r.stdout.strip()


def _parse_sbatch_jobid(s: str) -> str:
    # "Submitted batch job 29496284"
    m = re.search(r"Submitted batch job\s+(\d+)", s)
    if not m:
        raise RuntimeError(f"Could not parse sbatch job id from: {s}")
    return m.group(1)


def _squeue_jobids() -> Set[str]:
    """
    Returns set of job IDs currently in Slurm queue for *this user*.
    This is safer than trusting state.json alone, especially after restarts.
    """
    try:
        out = _run(["squeue", "-h", "-o", "%A"])
    except Exception:
        return set()
    ids: Set[str] = set()
    for ln in out.splitlines():
        s = ln.strip()
        if s:
            ids.add(s)
    return ids


def _inflight_count(out_root: Path, crosscheck_squeue: bool = True) -> int:
    """
    Inflight = jobs that are submitted but not completed.
    If crosscheck_squeue=True, only count those whose jobid is still in squeue.
    This avoids "inflight stuck at 50" due to stale state.json.
    """
    q = _squeue_jobids() if crosscheck_squeue else set()

    n = 0
    for d in sorted(out_root.iterdir()):
        if not d.is_dir():
            continue
        st = _read_state(_state_path(d))
        if not st:
            continue
        if st.get("submitted") and not st.get("completed"):
            jid = st.get("slurm_job_id")
            # If no jid, treat as inflight only if you *want* to be conservative.
            # Better: treat as NOT inflight so we can resubmit or flag it.
            if not jid:
                continue
            if (not crosscheck_squeue) or (str(jid) in q):
                n += 1
    return n


def _ready_to_submit_count(out_root: Path) -> int:
    """
    Count jobs that look eligible:
      prepared=True, submitted=False, completed=False
    """
    n = 0
    for d in sorted(out_root.iterdir()):
        if not d.is_dir():
            continue
        st = _read_state(_state_path(d))
        if not st:
            continue
        if st.get("prepared", False) and (not st.get("submitted", False)) and (not st.get("completed", False)):
            n += 1
    return n


def submit_jobs(
    *,
    out_root: str,
    max_inflight: int = 100,
    watch: bool = False,
    poll_sec: int = 60,
    limit: Optional[int] = None,
    verbose: bool = True,
    crosscheck_squeue: bool = True,
    refresh_status_in_watch: bool = True,
) -> SubmitReport:
    """
    Submit per-frame VASP jobs under out_root/<jobdir>/sub_vasp.sh

    Key improvements vs your version:
      - inflight is computed with optional squeue cross-check (avoids stale state.json inflight).
      - watch mode DOES NOT exit just because "submitted=0" once (it may be at capacity).
      - prints which job folders were submitted in that pass.
      - reports remaining ready-to-submit jobs so you can see if it’s done.
      - (optional) calls status_update() each watch loop to mark completed jobs and free capacity.

    Notes:
      - If you ran submit previously and got partial/stale state, crosscheck_squeue=True is safer.
      - If sbatch succeeds but you want more metadata, you can also store submit stdout in state.json.
    """
    out_root_p = Path(out_root).resolve()
    if not out_root_p.exists():
        raise FileNotFoundError(f"out_root not found: {out_root_p}")

    def one_pass() -> SubmitReport:
        inflight = _inflight_count(out_root_p, crosscheck_squeue=crosscheck_squeue)
        capacity = max(0, int(max_inflight) - inflight)
        if limit is not None:
            capacity = min(capacity, max(0, int(limit)))

        submitted: List[str] = []
        skipped: List[str] = []

        # quick ready count before submitting
        remaining_ready_before = _ready_to_submit_count(out_root_p)

        if capacity == 0:
            rep = SubmitReport(
                submitted=submitted,
                skipped=skipped,
                inflight=inflight,
                capacity=capacity,
                remaining_ready=remaining_ready_before,
            )
            return rep

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
            st["sbatch_stdout"] = out
            _write_state(st_path, st)

            submitted.append(jobdir.name)
            capacity -= 1
            inflight += 1

        remaining_ready_after = _ready_to_submit_count(out_root_p)

        return SubmitReport(
            submitted=submitted,
            skipped=skipped,
            inflight=inflight,
            capacity=capacity,
            remaining_ready=remaining_ready_after,
        )

    if not watch:
        rep = one_pass()
        if verbose:
            if rep.submitted:
                print(f"[PotAudit] submitted {len(rep.submitted)}: {', '.join(rep.submitted[:10])}"
                      + (" ..." if len(rep.submitted) > 10 else ""))
            print(f"[PotAudit] inflight={rep.inflight} remaining_ready={rep.remaining_ready}")
        return rep

    # Watch mode: keep topping up until no more work remains.
    # We do NOT exit just because one pass submits 0; could be at capacity.
    while True:
        if refresh_status_in_watch and status_update is not None:
            try:
                srep = status_update(out_root=str(out_root_p))
                if verbose and srep.updated:
                    print(f"[PotAudit] status: updated={srep.updated} ok={srep.ok} bad={srep.bad} "
                          f"running={srep.running} pending={srep.pending}")
            except Exception as e:
                if verbose:
                    print(f"[PotAudit] status_update failed (continuing): {e}")

        rep = one_pass()

        if verbose:
            if rep.submitted:
                print(f"[PotAudit] submitted {len(rep.submitted)}: {', '.join(rep.submitted)}")
            print(f"[PotAudit] inflight={rep.inflight} remaining_ready={rep.remaining_ready}")

        # Done condition: nothing left ready to submit AND no inflight jobs.
        if rep.remaining_ready == 0 and rep.inflight == 0:
            return rep

        # If no capacity (or no eligible jobs right now), just wait and re-check.
        time.sleep(max(5, int(poll_sec)))