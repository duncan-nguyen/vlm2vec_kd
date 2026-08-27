"""Lightweight step profiler for the distillation training loop.

Disabled by default: every hook degrades to a shared no-op context manager, so
leaving the `profiler.section(...)` calls in the hot path costs nothing.

Enable with::

    VLM2VEC_PROFILE=1 bash train_scripts/rebuttal_hierd_grounding.sh

Environment variables
---------------------
VLM2VEC_PROFILE        1 to enable (default 0).
VLM2VEC_PROFILE_SYNC   1 to torch.cuda.synchronize() around every section
                       (default 1). CUDA is async, so without the sync the
                       timings get attributed to whichever section happens to
                       block first. Turn it off only to measure the wall clock
                       of the loop without the sync overhead.
VLM2VEC_PROFILE_EVERY  print a report every N steps (default 50).
VLM2VEC_PROFILE_STEPS  stop the process after N steps (default 0 = never).
                       Handy for a quick profiling run without a full epoch.
"""

import functools
import os
import time
from collections import OrderedDict
from contextlib import contextmanager

import torch


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "no")


ENABLED = _env_flag("VLM2VEC_PROFILE", "0")
_SYNC = _env_flag("VLM2VEC_PROFILE_SYNC", "1")
_EVERY = int(os.environ.get("VLM2VEC_PROFILE_EVERY", "50"))
_MAX_STEPS = int(os.environ.get("VLM2VEC_PROFILE_STEPS", "0"))


@contextmanager
def _null_section():
    yield


_NULL = _null_section


class StepProfiler:
    """Accumulates wall-clock time per named section, reported per step."""

    def __init__(self):
        self.enabled = ENABLED
        self.sync = _SYNC
        self.totals = OrderedDict()
        self.counts = OrderedDict()
        self.n_steps = 0
        self._stack = []
        self._t_epoch = None

    # -- measurement ------------------------------------------------------
    def _sync_cuda(self):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    def section(self, name):
        if not self.enabled:
            return _NULL()
        return self._section(name)

    @contextmanager
    def _section(self, name):
        self._stack.append(name)
        key = "/".join(self._stack)
        self._sync_cuda()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync_cuda()
            dt = time.perf_counter() - t0
            self.totals[key] = self.totals.get(key, 0.0) + dt
            self.counts[key] = self.counts.get(key, 0) + 1
            self._stack.pop()

    def step(self):
        """Mark the end of one training step."""
        if not self.enabled:
            return
        self.n_steps += 1
        if self._t_epoch is None:
            self._t_epoch = time.perf_counter()

    @property
    def should_report(self):
        return self.enabled and _EVERY > 0 and self.n_steps > 0 and self.n_steps % _EVERY == 0

    @property
    def should_stop(self):
        return self.enabled and _MAX_STEPS > 0 and self.n_steps >= _MAX_STEPS

    # -- reporting --------------------------------------------------------
    def report(self, header="profile"):
        if not self.enabled or self.n_steps == 0:
            return ""
        # Top-level sections only, for the "share of a step" column: a nested
        # section is already counted inside its parent.
        top_total = sum(v for k, v in self.totals.items() if "/" not in k)
        lines = [
            "",
            "=" * 78,
            f"[{header}] steps={self.n_steps}  sync={'on' if self.sync else 'off'}",
            f"{'section':<38}{'ms/step':>10}{'% top':>9}{'calls/step':>12}",
            "-" * 78,
        ]
        for key, total in self.totals.items():
            depth = key.count("/")
            label = "  " * depth + key.split("/")[-1]
            ms = total / self.n_steps * 1000.0
            share = (total / top_total * 100.0) if (depth == 0 and top_total > 0) else float("nan")
            share_s = f"{share:8.1f}" if share == share else " " * 8
            lines.append(
                f"{label:<38}{ms:>10.1f}{share_s:>9}{self.counts[key] / self.n_steps:>12.2f}"
            )
        lines.append("-" * 78)
        lines.append(f"{'TOTAL (top-level)':<38}{top_total / self.n_steps * 1000.0:>10.1f}")
        if self._t_epoch is not None:
            wall = time.perf_counter() - self._t_epoch
            lines.append(f"{'wall clock':<38}{wall / self.n_steps * 1000.0:>10.1f} ms/step")
        lines.append("=" * 78)
        return "\n".join(lines)

    def reset(self):
        self.totals.clear()
        self.counts.clear()
        self.n_steps = 0
        self._t_epoch = None


profiler = StepProfiler()


def section(name):
    """Module-level shortcut so call sites read `with profiling.section('x'):`."""
    return profiler.section(name)


def timed(name):
    """Decorator form of `section`, for wrapping a whole function.

    Returns the function untouched when profiling is off, so there is no wrapper
    frame in the hot path.
    """
    def decorator(fn):
        if not profiler.enabled:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with profiler.section(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def timed_iter(iterable, name="data"):
    """Wrap a DataLoader so the time spent waiting on the next batch is measured.

    With num_workers>0 and enough prefetch this should be close to zero; a large
    value here means the input pipeline is the bottleneck.
    """
    if not profiler.enabled:
        for item in iterable:
            yield item
        return
    it = iter(iterable)
    while True:
        with profiler.section(name):
            try:
                item = next(it)
            except StopIteration:
                return
        yield item
