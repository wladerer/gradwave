"""Machine and process provenance for run outputs (Layer C).

Every ``run()`` output carries a ``provenance`` block answering, months
later: which machine and code produced this file, whether the box was
contested by other work, how hot it ran, and what the process actually
consumed. A recorded timing without that context is unusable — the same
SCF drifts 2-3x on a thermally-throttled or shared host.

Everything here is best-effort and dependency-free (``/proc``, ``/sys``,
``nvidia-smi``): a field that cannot be read on this platform is absent
or None, never an exception. All values are plain JSON-serializable
Python types.
"""

from __future__ import annotations

import datetime
import os
import platform
import resource
import subprocess
import time
from pathlib import Path


def _read(path) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def _run(cmd, timeout=3) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout)
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_info() -> dict:
    import torch

    model = None
    cpuinfo = _read("/proc/cpuinfo")
    if cpuinfo:
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {"model": model, "logical_cores": os.cpu_count(),
            "torch_threads": torch.get_num_threads()}


def memory_info() -> dict:
    fields = {}
    meminfo = _read("/proc/meminfo")
    if meminfo:
        for line in meminfo.splitlines():
            key = line.split(":")[0]
            if key in ("MemTotal", "MemAvailable"):
                fields[key] = round(int(line.split()[1]) / 1024**2, 2)  # GB
    return {"total_gb": fields.get("MemTotal"),
            "available_gb": fields.get("MemAvailable")}


def load_info() -> dict:
    """Load averages plus the busiest OTHER processes — the contested-
    machine indicator. Sampled at run start (before this process spins
    its threads) the 1-minute load is other people's work; sampled at
    the end it should be ≈ our own thread count if the box is ours."""
    try:
        l1, l5, l15 = os.getloadavg()
    except OSError:
        l1 = l5 = l15 = None
    others = []
    ps = _run(["ps", "-eo", "pid,pcpu,comm", "--sort=-pcpu"])
    if ps:
        me = os.getpid()
        for line in ps.splitlines()[1:6]:
            parts = line.split(None, 2)
            if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) != me:
                pcpu = float(parts[1])
                if pcpu >= 10.0:
                    others.append({"comm": parts[2], "pcpu": pcpu})
    return {"load_1m": l1, "load_5m": l5, "load_15m": l15,
            "busy_other_processes": others}


def thermal_info() -> dict:
    """Best-effort temperatures from /sys/class/thermal (laptops expose
    CPU package zones there) — absent on hosts without the sysfs zones."""
    zones = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        ztype = (_read(zone / "type") or "").strip()
        raw = (_read(zone / "temp") or "").strip()
        if ztype and raw.lstrip("-").isdigit():
            temp_c = int(raw) / 1000.0
            if -50.0 < temp_c < 150.0:
                zones[ztype] = round(temp_c, 1)
    return {"zones_c": zones,
            "max_c": max(zones.values()) if zones else None}


def gpu_info() -> dict | None:
    import torch

    if not torch.cuda.is_available():
        return None
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    info = {"name": props.name,
            "vram_total_gb": round(props.total_memory / 1024**3, 2),
            "capability": f"{props.major}.{props.minor}"}
    smi = _run(["nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits"])
    if smi:
        parts = [p.strip() for p in smi.splitlines()[dev].split(",")]
        if len(parts) == 3:
            info.update(utilization_pct=float(parts[0]),
                        vram_used_gb=round(float(parts[1]) / 1024, 2),
                        temperature_c=float(parts[2]))
    return info


def _git_commit() -> str | None:
    src = Path(__file__).resolve().parent
    out = _run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"])
    return out.strip() if out else None


def machine_snapshot() -> dict:
    """Full static + dynamic machine state; take one at run start."""
    import torch

    from gradwave import __version__

    uname = platform.uname()
    return {
        "timestamp": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "host": {"hostname": uname.node,
                 "os": f"{uname.system} {uname.release}",
                 "arch": uname.machine},
        "code": {"gradwave": __version__, "git": _git_commit(),
                 "python": platform.python_version(),
                 "torch": torch.__version__},
        "cpu": cpu_info(),
        "memory": memory_info(),
        "gpu": gpu_info(),
        "load": load_info(),
        "thermal": thermal_info(),
    }


class ProcessMeter:
    """Start/stop accounting of what THIS process consumed: wall and CPU
    time (their ratio = effective threads — a contested-box fingerprint
    when it lands far below torch_threads), peak RSS, CUDA peak memory."""

    def __init__(self):
        import torch

        self._t0 = time.perf_counter()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        self._cpu0 = ru.ru_utime + ru.ru_stime
        self._cuda = torch.cuda.is_available()
        if self._cuda:
            torch.cuda.reset_peak_memory_stats()

    def stop(self) -> dict:
        import torch

        wall = time.perf_counter() - self._t0
        ru = resource.getrusage(resource.RUSAGE_SELF)
        cpu = ru.ru_utime + ru.ru_stime - self._cpu0
        out = {
            "wall_s": round(wall, 2),
            "cpu_s": round(cpu, 2),
            "effective_threads": round(cpu / wall, 2) if wall > 0 else None,
            "peak_rss_gb": round(ru.ru_maxrss / 1024**2, 2),  # Linux: KB
        }
        if self._cuda:
            out["cuda_peak_alloc_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024**3, 3)
            out["cuda_peak_reserved_gb"] = round(
                torch.cuda.max_memory_reserved() / 1024**3, 3)
        return out


def provenance_block(start_snapshot: dict, meter: ProcessMeter) -> dict:
    """The block written into every <task>.json: the start-of-run machine
    snapshot, an end-of-run resample of the volatile parts (load and
    temperature drift over a run; the delta is the throttling/contention
    story), and the process accounting."""
    return {
        **start_snapshot,
        "load_end": load_info(),
        "thermal_end": thermal_info(),
        "process": meter.stop(),
    }
