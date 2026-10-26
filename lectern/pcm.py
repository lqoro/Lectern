"""Tiny PCM helpers (16-bit little-endian). Replaces the stdlib `audioop`,
which was removed in Python 3.13 and would break the app on new installs."""
from __future__ import annotations

import math
import sys
from array import array


def _samples(raw: bytes) -> array:
    a = array("h")
    a.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if sys.byteorder == "big":
        a.byteswap()
    return a


def rms(raw: bytes, width: int = 2) -> int:
    assert width == 2
    a = _samples(raw)
    if not a:
        return 0
    return int(math.sqrt(sum(x * x for x in a) / len(a)))


def peak(raw: bytes, width: int = 2) -> int:
    assert width == 2
    a = _samples(raw)
    return max((abs(x) for x in a), default=0)
