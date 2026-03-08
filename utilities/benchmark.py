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
        return (
            "OCR stats | "
            f"in_q: +{stats['in_enqueued']} / -{stats['in_dequeued']} / drop_old={stats['in_dropped_oldest']} | "
            f"out_q: +{stats['out_enqueued']} / -{stats['out_dequeued']} / drop_old={stats['out_dropped_oldest']} | "
            f"processed={stats['ocr_processed']} errors={stats['ocr_errors']} empty={stats['ocr_empty_return']} | "
            f"shm_ok={stats.get('shm_write_ok', 0)} shm_drop={stats.get('shm_oversize_drop', 0)}"
        )
