"""Advisory file lock with crash-safe release.

We layer three mechanisms:

1. ``os.open(path, O_EXCL | O_CREAT)`` for cross-process atomicity.  Creating a
   lock file with O_EXCL is atomic across processes on POSIX and Windows; the
   loser gets EEXIST, which we map to a timeout.

2. ``fcntl.flock`` (POSIX) or ``msvcrt.locking`` (Windows) on the lock *file*
   for an OS-level hold.  When the process dies — even SIGKILL — the kernel
   closes the fd and the flock is released.  This is the property the brief is
   asking for that a plain lock file does not give us.

3. A stale-owner record *inside* the lock file so a *new* process can decide
   whether the lock file itself is orphaned.  The record is written *after*
   the flock is acquired so it is never written by a process that does not
   hold the OS lock.

Interpretation choice: the lock is exclusive (write) only.  No read/shared
mode is supported.
"""

from __future__ import annotations

import errno
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional


class LockTimeoutError(TimeoutError):
    """Raised when a lock cannot be acquired within the requested budget."""


class LockState:
    """Why a contender decided a lock file is stale.

    Used by ``FileLock.acquire`` to report, via the ``stale_reason`` attribute,
    exactly which path triggered a takeover.  Keeping this as a small enum-like
    class rather than a bare string makes log assertions type-stable.
    """

    __slots__ = ()
    OWNED_BY_OTHER = "owned_by_other"
    STALE_OWNER_DEAD = "stale_owner_dead"


def _monotonic() -> float:
    """Default clock.  Kept as a module-level so tests can monkeypatch it."""
    return time.monotonic()


_IS_WINDOWS = sys.platform.startswith("win") or sys.platform == "cygwin"


class FileLock:
    """An exclusive advisory lock backed by a file.

    Parameters
    ----------
    path:
        Filesystem path to the lock file.  Parent directories must exist.
    poll_interval:
        Seconds between attempts when contending.  Must be positive.
    clock:
        Monotonic callable returning float seconds.  Injectable for tests so
        the test suite never sleeps or reads wall-clock time.
    """

    def __init__(
        self,
        path: "str | os.PathLike[str]",
        *,
        poll_interval: float = 0.1,
        clock: "Any" = _monotonic,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._path = Path(path)
        self._poll_interval = float(poll_interval)
        self._clock = clock
        self._fd: Optional[int] = None
        # pid of the process whose record is currently inside the lock file.
        # Only set while *this* object holds the OS lock.
        self._owner_pid: Optional[int] = os.getpid()
        # Human-readable diagnostic set on the most recent acquire.
        self.stale_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_locked(self) -> bool:
        """True if this object currently holds the OS lock."""
        return self._fd is not None

    def acquire(self, timeout: Optional[float] = None) -> None:
        """Block until the lock is acquired or ``timeout`` elapses.

        ``timeout=None`` means wait forever.  ``timeout=0`` means try once.
        Raises :class:`LockTimeoutError` on timeout.
        """
        if self._fd is not None:
            return
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")

        deadline = None
        if timeout is not None:
            deadline = self._clock() + float(timeout)

        attempt = 0
        while True:
            attempt += 1
            if self._try_acquire_once():
                return
            if deadline is not None:
                # Use the injected clock so tests can drive time.
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise LockTimeoutError(
                        f"could not acquire {self._path} within {timeout}s "
                        f"after {attempt} attempts"
                    )
            # Never sleep in the zero-timeout case; the injected clock in
            # tests returns whatever it returns, and a sleep would make the
            # suite non-deterministic on slow machines.
            if timeout is None or timeout > 0:
                time.sleep(self._poll_interval)

    def release(self) -> None:
        """Release the lock.  Safe to call when not held."""
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        self._owner_pid = None
        self.stale_reason = None
        # Release the OS lock *before* unlinking: another waiter may then grab
        # the flock on a new inode we create.  Order matters.
        self._unlock_fd(fd)
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self) -> None:
        # Best-effort.  If the interpreter is tearing down module globals,
        # ``os`` may already be gone; guard with getattr.
        try:
            self.release()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _try_acquire_once(self) -> bool:
        """One attempt.  Returns True on success."""
        self.stale_reason = None
        fd = self._create_lock_file()
        if fd is not None:
            # Fresh file: we own it, now take the OS lock.
            self._flock(fd)
            self._fd = fd
            self._owner_pid = os.getpid()
            self._write_owner(fd)
            return True

        # File existed.  Open it and decide: locked by someone else, or stale?
        fd = self._open_existing()
        if fd is None:
            # Someone deleted it between our stat and open; retry.
            return False

        try:
            self._flock(fd)  # raises TimeoutError if contended
        except LockTimeoutError:
            self.stale_reason = LockState.OWNED_BY_OTHER
            try:
                os.close(fd)
            except OSError:
                pass
            return False

        # We got the flock.  If the file's content names a dead owner, the
        # lock is stale and we can take it over.  Otherwise we must assume
        # the file is legitimately held by a process that crashed *before*
        # writing its record — treat that as contended and retry.
        record = self._read_owner(fd)
        if not self._is_stale(record):
            self.stale_reason = LockState.OWNED_BY_OTHER
            self._unlock_fd(fd)
            try:
                os.close(fd)
            except OSError:
                pass
            return False

        self.stale_reason = LockState.STALE_OWNER_DEAD
        self._fd = fd
        self._owner_pid = os.getpid()
        self._write_owner(fd)
        return True

    def _create_lock_file(self) -> Optional[int]:
        """O_EXCL create.  Returns fd or None if the file already exists."""
        flags = os.O_CREAT | os.O_EXCL | getattr(os, "O_RDWR", os.O_RDWR)
        try:
            return os.open(self._path, flags, 0o600)
        except FileExistsError:
            return None
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                return None
            raise

    def _open_existing(self) -> Optional[int]:
        flags = getattr(os, "O_RDWR", os.O_RDWR)
        try:
            return os.open(self._path, flags)
        except FileNotFoundError:
            return None

    def _flock(self, fd: int) -> None:
        """Acquire an exclusive OS lock on ``fd`` without blocking.

        Raises :class:`LockTimeoutError` if held by another process.
        """
        if _IS_WINDOWS:
            self._lock_msvcrt(fd)
        else:
            self._lock_fcntl(fd)

    def _unlock_fd(self, fd: int) -> None:
        try:
            if _IS_WINDOWS:
                import msvcrt

                # Unlock must cover the range we locked.  We lock byte 0..0,
                # so unlock the same.
                try:
                    msvcrt.locking(fd, 0, 0)
                except OSError:
                    pass
            else:
                import fcntl

                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        except Exception:
            # Never let an unlock failure mask the real flow.
            pass

    @staticmethod
    def _lock_fcntl(fd: int) -> None:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockTimeoutError(str(exc)) from exc

    @staticmethod
    def _lock_msvcrt(fd: int) -> None:
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            # On Windows the errno is not always EWOULDBLOCK, but any failure
            # to acquire here means "busy".
            raise LockTimeoutError(str(exc)) from exc

    def _write_owner(self, fd: int) -> None:
        record = {
            "pid": os.getpid(),
            # birthtime is a monotonic marker the new contender checks; we do
            # *not* compare it to wall clock, only test if the PID is live.
            "birth": self._clock(),
        }
        data = (json.dumps(record) + "\n").encode("utf-8")
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, data)
            os.fsync(fd)
        except OSError:
            pass

    def _read_owner(self, fd: int) -> Optional[dict]:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            chunk = os.read(fd, 4096)
        except OSError:
            return None
        if not chunk:
            return None
        try:
            return json.loads(chunk.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def _is_stale(self, record: Optional[dict]) -> bool:
        """A lock file is stale if its recorded owner is not alive."""
        if not isinstance(record, dict):
            # Unreadable or empty: assume contended (conservative).
            return False
        pid = record.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False
        if pid == os.getpid():
            # We left a record from a previous run in this same process
            # object.  Not stale in the "dead process" sense, but we *are*
            # the owner, so take it.
            return True
        return not _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` exists.

    POSIX: ``os.kill(pid, 0)``.  Windows: ``OpenProcess`` via ctypes (avoids
    any third-party dependency).  We deliberately treat "permission denied"
    as "alive" because the current process may not have the rights to signal
    another user's process, but its existence still implies the lock is held.
    """
    if pid <= 0:
        return False
    if _IS_WINDOWS:
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_alive_windows(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    OpenProcess = kernel32.OpenProcess
    OpenProcess.restype = wintypes.HANDLE
    OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    GetExitCodeProcess = kernel32.GetExitCodeProcess
    GetExitCodeProcess.restype = wintypes.BOOL
    GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    CloseHandle = kernel32.CloseHandle
    CloseHandle.restype = wintypes.BOOL
    CloseHandle.argtypes = (wintypes.HANDLE,)

    handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # ERROR_INVALID_PARAMETER (87) is returned for non-existent PIDs.
        return ctypes.get_last_error() != 87
    try:
        code = wintypes.DWORD()
        if not GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == STILL_ACTIVE
    finally:
        CloseHandle(handle)
