import os
import atexit
import subprocess
import tempfile
import socket
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCK_DIR  = Path(tempfile.gettempdir())
_NODE      = socket.gethostname()
_LOCK_TMPL = "gpu_{node}_{idx}.lock"


def _all_gpu_indices() -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
        env=os.environ
    )
    return [int(i) for i in out.strip().splitlines()]


def _try_lock(idx: int) -> bool:
    """Create a lock file for the (node, GPU) pair. Return True on success."""
    path =_LOCK_DIR / _LOCK_TMPL.format(node=_NODE, idx=idx)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        return False

    atexit.register(lambda: path.unlink(missing_ok=True))
    return True


def reserve_one_gpu() -> int:
    """Reserve one GPU and hide all others from this process."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    indices = _all_gpu_indices()
    logger.info("Found GPU indices %s", indices)
    for idx in indices:
        if _try_lock(idx):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            logger.info("Process %s bound to GPU %s on node %s", os.getpid(), idx, _NODE)
            return idx
    raise RuntimeError(f"[{_NODE}] No free GPU available!")
