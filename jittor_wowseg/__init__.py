"""Jittor inference implementation for WOW-Seg.

The model import is lazy so utility modules such as ``preprocess`` and the
weight converter remain usable in environments where Jittor is not installed.
"""

__all__ = ["WowSegJittor"]


def __getattr__(name):
    if name == "WowSegJittor":
        from .model import WowSegJittor

        return WowSegJittor
    raise AttributeError(name)
