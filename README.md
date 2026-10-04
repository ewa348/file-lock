# file_lock

An advisory file lock for Python that releases even when the holding process dies badly (SIGKILL, power loss, segfault). Standard library only.

```python
from file_lock import FileLock

with FileLock("/var/run/myjob.lock") as lock:
    # critical section — if this process is killed, the OS drops the flock
    # and the next contender can detect the stale owner record and take over.
    pass
```

## Why

A plain lock file created with `O_EXCL` is atomic across processes, but it never goes away when the holder crashes — the next process sees the file, can't create it, and waits forever. Locking the same file with `flock` (POSIX) or `msvcrt.locking` (Windows) fixes that: the kernel releases the lock when the holding process dies, even under SIGKILL.

This library layers both. `O_EXCL` gives us cross-process atomic creation of the lock file. The OS-level lock gives us crash-safe release. A small JSON record inside the file names the current owner by PID, so a *new* contender that finds the file already present can decide whether to take it over (dead owner) or keep waiting (live owner).

Exclusive (write) lock only. No read/shared mode. The `poll_interval` controls contention backoff; pass a `clock` callable in tests so the suite never sleeps or reads wall-clock time.

## Edge cases

- A corrupt or empty lock file is treated as *contended*, not stale. The conservative choice is to wait rather than clobber a file we can't reason about. Delete it by hand if you know it's garbage.
- `pid == os.getpid()` in the record is treated as takeable: a fresh `FileLock` object reclaiming its own previous file is safe.
- On Windows, `OpenProcess` with `PROCESS_QUERY_LIMITED_INFORMATION` is used to check liveness; access-denied is interpreted as "alive" (the process exists, we just can't query it).

## Exports

- `FileLock(path, *, poll_interval=0.1, clock=time.monotonic)` — the lock. Methods: `acquire(timeout=None)`, `release()`, `is_locked`. Context-manager capable.
- `LockTimeoutError` — subclass of `TimeoutError`, raised on timeout.
- `LockState` — constants `OWNED_BY_OTHER`, `STALE_OWNER_DEAD` used in `FileLock.stale_reason`.
