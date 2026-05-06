"""
Lightweight per-section profiler for the BOP3000 main loop.

Usage:
    from utilities.profiler import Profiler

    profiler = Profiler(log_every_sec=2.0)

    # Context manager style (recommended):
    with profiler.section("yolo_wide"):
        result = model(...)

    # Manual start/stop:
    profiler.start("close_inference")
    close_out = run_close_inference(...)
    profiler.stop("close_inference")

    # Periodic logging:
    if profiler.should_log():
        logger.info(profiler.format())

    # Final summary at shutdown:
    print(profiler.summary())
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict


@dataclass
class SectionStats:
    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0


class Profiler:
    """
    Tracks elapsed time per named section across many iterations.
    Thread-safe for single-threaded main loop usage.
    """

    def __init__(self, log_every_sec: float = 2.0):
        self.log_every_sec = float(log_every_sec)
        self._last_log_time = time.perf_counter()
        self._sections: Dict[str, SectionStats] = {}
        self._active: Dict[str, float] = {}  # section_name -> start_time

    def _get_stats(self, name: str) -> SectionStats:
        if name not in self._sections:
            self._sections[name] = SectionStats(name=name)
        return self._sections[name]

    def start(self, name: str) -> None:
        """Mark the beginning of a named section."""
        self._active[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """
        Mark the end of a named section. Returns elapsed milliseconds.
        Raises KeyError if start() was never called for this section.
        """
        t0 = self._active.pop(name)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        stats = self._get_stats(name)
        stats.count += 1
        stats.total_ms += elapsed_ms
        if elapsed_ms < stats.min_ms:
            stats.min_ms = elapsed_ms
        if elapsed_ms > stats.max_ms:
            stats.max_ms = elapsed_ms

        return elapsed_ms

    @contextmanager
    def section(self, name: str):
        """Context manager for a named section."""
        self.start(name)
        try:
            yield self
        finally:
            self.stop(name)

    def should_log(self) -> bool:
        now = time.perf_counter()
        if now - self._last_log_time < self.log_every_sec:
            return False
        self._last_log_time = now
        return True

    def format(self, top_n: int = 10) -> str:
        """
        Format current stats as a readable string.
        Shows top-N sections by average time (descending).
        """
        if not self._sections:
            return ""

        lines = []
        sorted_sections = sorted(
            self._sections.values(),
            key=lambda s: s.avg_ms,
            reverse=True,
        )[:top_n]

        for stats in sorted_sections:
            if stats.count == 0:
                continue
            lines.append(
                f"  {stats.name:<25s} "
                f"avg={stats.avg_ms:>7.1f}ms "
                f"min={stats.min_ms:>6.1f}ms "
                f"max={stats.max_ms:>7.1f}ms "
                f"(n={stats.count})"
            )

        return "\n".join(lines)

    def summary(self) -> str:
        """Final summary with totals and percentage breakdown."""
        lines = ["\n=== PROFILER SUMMARY ==="]

        total_avg_per_frame = 0.0
        if self._sections:
            for stats in self._sections.values():
                if stats.count > 0:
                    total_avg_per_frame += stats.avg_ms

        lines.append(f"Total estimated frame time: {total_avg_per_frame:.1f}ms")
        if total_avg_per_frame > 0:
            lines.append(f"Estimated max FPS: {1000.0 / total_avg_per_frame:.1f}")
        else:
            lines.append("Estimated max FPS: N/A")
        lines.append("")

        sorted_sections = sorted(
            self._sections.values(),
            key=lambda s: s.avg_ms,
            reverse=True,
        )

        for stats in sorted_sections:
            if stats.count == 0:
                continue
            pct = (stats.avg_ms / total_avg_per_frame * 100) if total_avg_per_frame > 0 else 0.0
            lines.append(
                f"  {stats.name:<25s} "
                f"avg={stats.avg_ms:>7.1f}ms "
                f"min={stats.min_ms:>6.1f}ms "
                f"max={stats.max_ms:>7.1f}ms "
                f"{pct:>5.1f}%  (n={stats.count})"
            )

        lines.append("======================\n")
        return "\n".join(lines)
