import multiprocessing as mp
import os
from queue import Empty, Full
from typing import Any, Dict, List, Optional

import numpy as np

from functions.shm_ring import SharedMemoryRing


_STOP = "__STOP__"

# Benchmark logging
def _inc(counter, delta: int = 1) -> None:
    if counter is None:
        return
    with counter.get_lock():
        counter.value += delta


def _drop_oldest_and_put(q: mp.Queue, item: Any) -> tuple[bool, bool]:
    """
    Best-effort queue policy: drop one oldest item when full, then enqueue.
    This is best for low latency
    """
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
    shm_name: str,
    shm_slots: int,
    shm_max_h: int,
    shm_max_w: int,
    shm_channels: int,
    shm_dtype: str,
) -> None:
    """
    OCR worker main loop.

    PaddleOCR is initialized in this process context by importing register_helmet
    here (not in parent process), which avoids serialization/pickling issues.

    Currently, it has a lot of room for silent errors and returning empty images.
    """
    from functions.register_helmet import register_helmet

    # We already make a shared memory in start()
    # We then connect to that same memory space by having create=False and correct name=
    # This is not a new memory space just new object referring to same place of memory.
    ring = SharedMemoryRing(
        name=shm_name,
        slots=shm_slots,
        max_h=shm_max_h,
        max_w=shm_max_w,
        channels=shm_channels,
        dtype=shm_dtype,
        create=False,
    )

    try:
        while not stop_event.is_set():
            try:
                item = in_queue.get(timeout=0.1)
            except Empty:
                continue

            if item == _STOP:
                break

            # Increment stats for benchmarks logs
            _inc(stats["in_dequeued"])
            _inc(stats["ocr_processed"])

            tid = int(item.get("track_id", -1))
            bbox = item.get("bbox", (0, 0, 0, 0))

            image = item.get("image")

            if "shm_slot" in item:
                image = ring.read(item["shm_slot"], item["shm_h"], item["shm_w"]).copy()

            # Default result
            if image is None:
                result = {
                    "track_id": tid,
                    "bbox": bbox,
                    "helmet_number": "",
                    "ocr_conf": 0.0,
                }
                # Bool returns for stat logs
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
                out = register_helmet([helmet], debug=False)
                if not out:
                    _inc(stats["ocr_empty_return"])

                # Give empty default if the register_helmet returns nothing
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

            # Logging for stats
            ok, dropped = _drop_oldest_and_put(out_queue, result)
            if ok:
                _inc(stats["out_enqueued"])
            if dropped:
                _inc(stats["out_dropped_oldest"])
    finally:
        ring.close()


class OCRWorker:
    """
    OCR process wrapper with non-blocking queues.

    The worker contains logic for queue handling and writing/reading from shared memory ring.
    The queue handles metadata for the images
    While the shared memory ring contains the actual image
    So the queue just contaisn the information it needs to find the image from the memory ring.
    """

    def __init__(
        self,
        max_in_size: int = 256,
        max_out_size: int = 256,
        start_method: str = "spawn",
        shm_slots: int = 1024,
        shm_max_h: int = 256,
        shm_max_w: int = 256,
        shm_channels: int = 3,
        shm_dtype: str = "uint8",
    ):
        self._ctx = mp.get_context(start_method)
        self.max_in_size = max_in_size
        self.max_out_size = max_out_size
        self.ocr_in_queue = self._ctx.Queue(maxsize=max_in_size)
        self.ocr_out_queue = self._ctx.Queue(maxsize=max_out_size)
        self._stop_event: mp.Event = self._ctx.Event()
        self._process: Optional[mp.Process] = None
        self._ring: Optional[SharedMemoryRing] = None
        self._shm_name = f"ocr_ring_{os.getpid()}"
        self._shm_slots = int(shm_slots)
        self._shm_max_h = int(shm_max_h)
        self._shm_max_w = int(shm_max_w)
        self._shm_channels = int(shm_channels)
        self._shm_dtype = str(shm_dtype)

        # Simply counters for benchmarking
        self._stats: Dict[str, Any] = {
            "in_enqueued": self._ctx.Value("i", 0),
            "in_dequeued": self._ctx.Value("i", 0),
            "in_dropped_oldest": self._ctx.Value("i", 0),
            "out_enqueued": self._ctx.Value("i", 0),
            "out_dequeued": self._ctx.Value("i", 0),
            "out_dropped_oldest": self._ctx.Value("i", 0),
            "ocr_processed": self._ctx.Value("i", 0),
            "ocr_errors": self._ctx.Value("i", 0),
            "ocr_empty_return": self._ctx.Value("i", 0),
            "shm_oversize_drop": self._ctx.Value("i", 0),
            "shm_write_ok": self._ctx.Value("i", 0),
        }


    def start(self) -> None:
        """
        Starts single process for handling OCR workload
        It uses _ocrprocess_main as the target function as the main loop.
        """
        if self._process is not None and self._process.is_alive():
            return

        if self.ocr_in_queue is None:
            self.ocr_in_queue = self._ctx.Queue(maxsize=self.max_in_size)
        if self.ocr_out_queue is None:
            self.ocr_out_queue = self._ctx.Queue(maxsize=self.max_out_size)

        # Actually create the shared memory space
        self._ring = SharedMemoryRing(
            name=self._shm_name,
            slots=self._shm_slots,
            max_h=self._shm_max_h,
            max_w=self._shm_max_w,
            channels=self._shm_channels,
            dtype=self._shm_dtype,
            create=True,
        )
        self._stop_event.clear()

        # Create the process and start
        self._process = self._ctx.Process(
            target=_ocr_process_main,
            args=(
                self.ocr_in_queue,
                self.ocr_out_queue,
                self._stop_event,
                self._stats,
                self._shm_name,
                self._shm_slots,
                self._shm_max_h,
                self._shm_max_w,
                self._shm_channels,
                self._shm_dtype,
            ),
            daemon=False,
        )
        self._process.start()


    def stop(self, timeout: float = 2.0) -> None:
        """
        Stops the current running process
        """
        self._stop_event.set()
        if self.ocr_in_queue is not None:
            _drop_oldest_and_put(self.ocr_in_queue, _STOP) # Enqueue a stop Event

        if self._process is not None:
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process = None
        if self._ring is not None:
            try:
                self._ring.close()
            except Exception:
                pass
            try:
                self._ring.unlink()
            except Exception:
                pass
            self._ring = None

        # Explicit queue cleanup avoids leaked semaphore warnings at shutdown.
        for q in (self.ocr_in_queue, self.ocr_out_queue):
            if q is None:
                continue
            try:
                q.close()
            except Exception:
                pass
            try:
                q.join_thread()
            except Exception:
                pass

        # Set them to none so no one can reuse old queues.
        self.ocr_in_queue = None
        self.ocr_out_queue = None

    def submit(self, item: Dict[str, Any]) -> bool:
        """
        Non-blocking submit.

        Returns True if accepted. When input queue is full, drops oldest to keep
        latency low and still tries to enqueue the newest task.
        """
        payload = dict(item)
        image = payload.get("image")
        if isinstance(image, np.ndarray) and self._ring is not None:
            img = image
            if img.ndim == 2:
                # Make grayscale images into BGR (2 channels -> 3 channels)
                img = np.repeat(img[:, :, None], 3, axis=2)
            if img.ndim != 3 or img.shape[2] != self._shm_channels:
                _inc(self._stats["shm_oversize_drop"])
                payload["image"] = None
                print("Image wrong size was dropped!!")
            elif img.shape[0] > self._shm_max_h or img.shape[1] > self._shm_max_w:
                _inc(self._stats["shm_oversize_drop"])
                payload["image"] = None
                print("Image crop is too big and was dropped!!")
            else:
                slot, h, w = self._ring.write(img)
                payload.pop("image", None)
                payload["shm_slot"] = slot
                payload["shm_h"] = h
                payload["shm_w"] = w
                _inc(self._stats["shm_write_ok"])

        accepted, dropped = _drop_oldest_and_put(self.ocr_in_queue, payload)
        if accepted:
            _inc(self._stats["in_enqueued"])
        if dropped:
            _inc(self._stats["in_dropped_oldest"])
        return accepted

    def drain_results(self, max_items: int = 24) -> List[Dict[str, Any]]:
        """
        Drains the queue in batches of max_items
        Returns a List of Dicts of tracks
        """
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
        """
        Return stats for the ocr_benchmark
        It can be found in utilities/benchmark.py
        """
        return {key: int(counter.value) for key, counter in self._stats.items()}
