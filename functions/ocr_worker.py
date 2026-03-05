import multiprocessing as mp
from queue import Empty, Full
from typing import Any, Dict, List, Optional


_STOP = "__STOP__"

def _inc(counter, delta: int = 1) -> None:
    if counter is None:
        return
    with counter.get_lock():
        counter.value += delta


def _drop_oldest_and_put(q: mp.Queue, item: Any) -> tuple[bool, bool]:
    """Best-effort queue policy: drop one oldest item when full, then enqueue."""
    try:
        q.put_nowait(item)
        return True, False
    except Full:
        try:
            _ = q.get_nowait()
        except Empty:
            return False, False
        try:
            q.put_nowait(item)
            return True, True
        except Full:
            return False, True


def _ocr_process_main(
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    stop_event: mp.Event,
    stats: Dict[str, Any],
) -> None:
    """
    OCR worker process entrypoint.

    PaddleOCR is initialized in this process context by importing register_helmet
    here (not in parent process), which avoids serialization/fork issues.
    """
    from functions.register_helmet import register_helmet

    while not stop_event.is_set():
        try:
            item = in_queue.get(timeout=0.1)
        except Empty:
            continue

        if item == _STOP:
            break

        _inc(stats["in_dequeued"])
        _inc(stats["ocr_processed"])

        tid = int(item.get("track_id", -1))
        bbox = item.get("bbox", (0, 0, 0, 0))
        image = item.get("image")

        if image is None:
            result = {
                "track_id": tid,
                "bbox": bbox,
                "helmet_number": "",
                "ocr_conf": 0.0,
            }
            ok, dropped = _drop_oldest_and_put(out_queue, result)
            if ok:
                _inc(stats["out_enqueued"])
            if dropped:
                _inc(stats["out_dropped_oldest"])
            continue

        helmet = {
            "image": image,
            "bbox": bbox,
            "conf": float(item.get("conf", 0.0)),
            "track_id": tid,
        }

        try:
            out = register_helmet([helmet], debug=False) # To Do: Make register helmet return a single helmet
            result = out[0] if out else {
                "track_id": tid,
                "bbox": bbox,
                "helmet_number": "",
                "ocr_conf": 0.0,
            }
        except Exception:
            _inc(stats["ocr_errors"])
            result = {
                "track_id": tid,
                "bbox": bbox,
                "helmet_number": "",
                "ocr_conf": 0.0,
            }

        ok, dropped = _drop_oldest_and_put(out_queue, result)
        if ok:
            _inc(stats["out_enqueued"])
        if dropped:
            _inc(stats["out_dropped_oldest"])


class OCRWorker:
    """Realtime-friendly OCR process wrapper with non-blocking queues."""

    def __init__(
        self,
        max_in_size: int = 256,
        max_out_size: int = 256,
        start_method: str = "spawn",
        thresh: float = 0.3,
    ):
        self._ctx = mp.get_context(start_method)
        self.ocr_in_queue: mp.Queue = self._ctx.Queue(maxsize=max_in_size)
        self.ocr_out_queue: mp.Queue = self._ctx.Queue(maxsize=max_out_size)
        self._stop_event: mp.Event = self._ctx.Event()
        self._process: Optional[mp.Process] = None
        self._thresh = thresh # Not used yet

        # Simply counters for stats
        self._stats: Dict[str, Any] = {
            "in_enqueued": self._ctx.Value("i", 0),
            "in_dequeued": self._ctx.Value("i", 0),
            "in_dropped_oldest": self._ctx.Value("i", 0),
            "out_enqueued": self._ctx.Value("i", 0),
            "out_dequeued": self._ctx.Value("i", 0),
            "out_dropped_oldest": self._ctx.Value("i", 0),
            "ocr_processed": self._ctx.Value("i", 0),
            "ocr_errors": self._ctx.Value("i", 0),
        }

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return

        self._stop_event.clear()
        self._process = self._ctx.Process(
            target=_ocr_process_main,
            args=(self.ocr_in_queue, self.ocr_out_queue, self._stop_event, self._stats),
            daemon=False,
        )
        self._process.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        _drop_oldest_and_put(self.ocr_in_queue, _STOP)

        if self._process is not None:
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process = None

        # Explicit queue cleanup avoids leaked semaphore warnings at shutdown.
        for q in (self.ocr_in_queue, self.ocr_out_queue):
            try:
                q.close()
            except Exception:
                pass
            try:
                q.join_thread()
            except Exception:
                pass

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def submit(self, item: Dict[str, Any]) -> bool:
        """
        Non-blocking submit.

        Returns True if accepted. When input queue is full, drops oldest to keep
        latency low and still tries to enqueue the newest task.
        """
        accepted, dropped = _drop_oldest_and_put(self.ocr_in_queue, item)
        if accepted:
            _inc(self._stats["in_enqueued"])
        if dropped:
            _inc(self._stats["in_dropped_oldest"])
        return accepted

    # When should this be used. Can be removed?
    def get_result_nowait(self) -> Optional[Dict[str, Any]]:
        try:
            return self.ocr_out_queue.get_nowait()
        except Empty:
            return None

    def drain_results(self, max_items: int = 64) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for _ in range(max_items):
            try:
                items.append(self.ocr_out_queue.get_nowait())
            except Empty:
                break
        if items:
            _inc(self._stats["out_dequeued"], len(items))
        return items

    def get_stats(self) -> Dict[str, int]:
        return {key: int(counter.value) for key, counter in self._stats.items()}
