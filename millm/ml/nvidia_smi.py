"""
GPU inventory from nvidia-smi: what each card has, without a CUDA context.

`torch.cuda.mem_get_info(i)` initialises a CUDA context on card i — a few
hundred MB that stays for the life of the process. Asking every card how much
it has free, to decide where a model goes, therefore took memory from every
card, including the ones the model never touched. miStudio's placement reads
live free memory on the same node, so a context miLLM left on the other card
was memory miStudio could not have. nvidia-smi reads NVML, which needs no
context in this process.

Indices here are nvidia-smi's. Placement maps a card to its torch index by UUID
(see `millm.ml.gpu_placement.list_gpus`).
"""

import subprocess
from typing import Any, Optional

import structlog

logger = structlog.get_logger()

#: `name` last, so a name containing a comma cannot shift the numeric fields.
NVIDIA_SMI_QUERY = (
    "index,uuid,utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu,name"
)
_FIELD_COUNT = len(NVIDIA_SMI_QUERY.split(","))


def _smi_int(value: str) -> int:
    """nvidia-smi reports "[N/A]" for fields a card does not expose; count it as 0."""
    try:
        return int(float(value))
    except ValueError:
        return 0


def parse_nvidia_smi_gpus(stdout: str) -> list[dict[str, Any]]:
    """
    Parse `nvidia-smi --query-gpu=<NVIDIA_SMI_QUERY>` CSV output, one GPU per line.

    An earlier parser split the whole output on ", " as if it were one line.
    With two GPUs a field became "36\\n33", int() raised, and every card was
    reported as zero — the UI showed "No GPU" on a two-GPU node.
    """
    gpus: list[dict[str, Any]] = []
    for line in stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < _FIELD_COUNT or not parts[0].isdigit():
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "utilization": _smi_int(parts[2]),
                "memory_used_mb": _smi_int(parts[3]),
                "memory_total_mb": _smi_int(parts[4]),
                "memory_free_mb": _smi_int(parts[5]),
                "temperature": _smi_int(parts[6]),
                "name": ", ".join(parts[_FIELD_COUNT - 1:]),
            }
        )
    return gpus


def _run_query() -> Optional[str]:
    """nvidia-smi's CSV output, or None when it is absent, fails or hangs."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={NVIDIA_SMI_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("nvidia_smi_failed", error=str(e))
        return None
    if result.returncode != 0:
        logger.debug("nvidia_smi_nonzero_exit", returncode=result.returncode)
        return None
    return result.stdout


def query_gpus() -> list[dict[str, Any]]:
    """Every card nvidia-smi reports. Empty when nvidia-smi is unavailable."""
    stdout = _run_query()
    return parse_nvidia_smi_gpus(stdout) if stdout else []
