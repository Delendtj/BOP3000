import threading
from queue import Empty, Full, Queue

from functions.register_helmet import initialize_ocr, register_helmet


class OCRWorker:
    def __init__(
        self,
        max_in_size=256,
        max_out_size=256,
    ):
        self.ocr_in_queue = Queue(maxsize=max_in_size)
        self.ocr_out_queue = Queue(maxsize=max_out_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        try:
            self.ocr_in_queue.put_nowait(None)
        except Full:
            pass
        self._thread.join(timeout=2.0)

    def submit(self, item):
        try:
            self.ocr_in_queue.put_nowait(item)
            return True
        except Full:
            return False

    def _run(self):
        # Initialize PaddleOCR in worker context before first task.
        initialize_ocr(debug=False)

        while not self._stop_event.is_set():
            try:
                item = self.ocr_in_queue.get(timeout=0.1)
            except Empty:
                continue

            if item is None:
                self.ocr_in_queue.task_done()
                continue

            result = self._process_item(item)
            try:
                self.ocr_out_queue.put_nowait(result)
            except Full:
                # Drop oldest result to keep queue fresh and non-blocking.
                try:
                    _ = self.ocr_out_queue.get_nowait()
                    self.ocr_out_queue.task_done()
                except Empty:
                    pass
                try:
                    self.ocr_out_queue.put_nowait(result)
                except Full:
                    pass

            self.ocr_in_queue.task_done()

    def _process_item(self, item):
        tid = int(item.get("track_id", -1))
        bbox = item.get("bbox", (0, 0, 0, 0))
        image = item.get("image")
        if image is None:
            return {
                "track_id": tid,
                "bbox": bbox,
                "helmet_number": "",
                "ocr_conf": 0.0,
            }

        helmet = {
            "image": image,
            "bbox": bbox,
            "conf": float(item.get("conf", 0.0)),
            "track_id": tid,
        }

        try:
            out = register_helmet([helmet], debug=False)
        except Exception:
            return {
                "track_id": tid,
                "bbox": bbox,
                "helmet_number": "",
                "ocr_conf": 0.0,
            }

        if not out:
            return {
                "track_id": tid,
                "bbox": bbox,
                "helmet_number": "",
                "ocr_conf": 0.0,
            }
        return out[0]
