"""
One bot per environment per machine.

Two engines trading the same account would each apply their own limits and
double the exposure, so the engine holds an exclusive operating-system lock on
a small file for as long as it runs. The lock disappears with the process -
a crash never leaves a stale one behind.
"""
from __future__ import annotations

import os
from pathlib import Path


class InstanceLock:
    def __init__(self, path: Path):
        self.path = path
        self._f = None

    def acquire(self) -> bool:
        if self._f is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self._f = f
        return True

    def release(self) -> None:
        f, self._f = self._f, None
        if f is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            f.close()
