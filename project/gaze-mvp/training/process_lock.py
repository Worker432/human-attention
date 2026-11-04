"""Nonblocking process lock on Windows and POSIX; released when handle closes."""
import os
from contextlib import contextmanager


def acquire_lock(path):
    handle = open(path, 'a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f'Cannot acquire training lock: {path}. Check for another running process.') from exc
    return handle


@contextmanager
def process_lock(path):
    handle = acquire_lock(path)
    try:
        yield handle
    finally:
        handle.close()
