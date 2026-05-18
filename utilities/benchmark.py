import time


class OCRThroughputStats:
    def __init__(self, log_every_sec=2.0):
        self.log_every_sec = float(log_every_sec)
        self._last_log_time = time.perf_counter()

    def should_log(self, now=None):
        ts = time.perf_counter() if now is None else float(now)
        if ts - self._last_log_time < self.log_every_sec:
            return False
        self._last_log_time = ts
        return True

    def format_line(self, stats):
        line = (
            "OCR stats | "
            f"in_q: +{stats['in_enqueued']} / -{stats['in_dequeued']} / drop_old={stats['in_dropped_oldest']} | "
            f"out_q: +{stats['out_enqueued']} / -{stats['out_dequeued']} / drop_old={stats['out_dropped_oldest']} | "
            f"processed={stats['ocr_processed']} errors={stats['ocr_errors']} empty={stats['ocr_empty_return']} | "
            f"shm_ok={stats.get('shm_write_ok', 0)} shm_drop={stats.get('shm_oversize_drop', 0)}"
        )

        # Append inference timing if available
        inf_count = stats.get("ocr_inference_count", 0)
        if inf_count and inf_count > 0:
            total_ms = stats.get("ocr_total_time_ms", 0.0)
            min_ms = stats.get("ocr_min_ms", 0.0)
            max_ms = stats.get("ocr_max_ms", 0.0)
            avg_ms = total_ms / inf_count
            throughput = inf_count / (total_ms / 1000.0) if total_ms > 0 else 0
            line += (
                f" | GLM inference: {inf_count} calls, "
                f"avg={avg_ms:.0f}ms min={min_ms:.0f}ms max={max_ms:.0f}ms, "
                f"{throughput:.1f} img/s"
            )

        return line