import cv2
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from functions.Inference_roi import crop_frame_to_roi
from utilities.downscale_to_1080p import downscale_to_1080p


#
# Datastruktur for en frame
#
@dataclass
class FrameItem:
    """
        En "pakke" vi sender mellom tråder.
     ts: tidspunkt (perf_counter) når frame ble tatt ut av VideoCapture
     frame: selve bildet (numpy array fra OpenCV)
        """
    # `frame` is the full frame used for drawing/tracking.
    # `inference_frame` is an optional ROI-cropped frame for model inference.
    ts: float
    frame: Any
    inference_frame: Any = None
    inference_offset: tuple[int, int] = (0, 0)


class LatestQueue(queue.Queue):
    """
        vaanlig queue kan bygge kø og skape latency (forsinkelse).
        I realtime vil vi heller ha "nyeste frame" enn å prosessere gamle frames.

        put_latest():
        om køen er full: dropp én gammel frame og legg inn den nye.
        resultat: vi ligger nærmest mulig live, selv om prosesseringen blir litt treg.
        """
    # Keep only the newest item to avoid latency buildup.
    def put_latest(self, item):
        try:
            self.put_nowait(item)
        except queue.Full:
            try:
                _ = self.get_nowait()  # dropp én gammel
            except queue.Empty:
                pass
            self.put_nowait(item)



# 3) Tråd 1: Capture (leser frames fra video/kamera)
def _capture_loop(cap: cv2.VideoCapture,
                  out_q: LatestQueue,
                  stop_event: threading.Event,
                  frame_skip: int = 1):
    """
    leser frames fra OpenCV VideoCapture i en egen tråd.
    gjør at hovedtråden slipper å "vente" på disk/kamera.
    frame_skip kan brukes for å hoppe over frames (redusere load).

    out_q: kø som får FrameItem (ts + frame).
    """
    i = 0
    # Throttle fps if the source is a video (has fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        frame_interval = 1.0 / fps
    else:
        frame_interval = None


    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # video ferdig / kamera feilet
            print(f"[capture_loop] ret=False after {i} frames")
            stop_event.set()
            break

        frame = downscale_to_1080p(frame)

        i += 1
        if frame_skip > 1 and (i % frame_skip != 0):
            continue

        out_q.put_latest(FrameItem(time.perf_counter(), frame))

        # Throttle fps
        if frame_interval is not None:
            time.sleep(frame_interval)


# 4) Tråd 2: Preprocessing (valgfritt steg egt kan sees på)
def _preprocess_loop(in_q: LatestQueue,
                     out_q: LatestQueue,
                     stop_event: threading.Event,
                     roi_getter=None, resize=None, crop_padding=None):
    """
    Henter frames fra capture-kø og gjør lett preprocessing i egen tråd.
    Eksempel: resize / crop / fargekonvertering etc.

    resize: f.eks. (1280, 720) eller None for ingen resize
    """
    while not stop_event.is_set():
        try:
            item: FrameItem = in_q.get(timeout=0.2)
        except queue.Empty:
            continue

        frame = item.frame

        # Eksempel preprocessing: resize
        if resize is not None:
            frame = cv2.resize(frame, resize)

        roi = roi_getter() if roi_getter is not None else None
        inference_frame, inference_offset = crop_frame_to_roi(
            frame,
            roi,
            padding=crop_padding,
        )
        if inference_frame is None:
            inference_frame = frame
            inference_offset = (0, 0)

        out_q.put_latest(
            FrameItem(
                ts=item.ts,
                frame=frame,
                inference_frame=inference_frame,
                inference_offset=inference_offset,
            )
        )

# 5) Pipeline-klassen (det vi bruker i main.py)
class AsyncFramePipeline:
    def __init__(
        self,
        source,
        frame_skip: int = 1,


        resize=None,
        queue_size: int = 3,
        inference_roi=None,
        crop_padding: int = 0,
    ):
        self.stop_event = threading.Event()
        self.cap = cv2.VideoCapture(source)
        self._roi_lock = threading.Lock()
        self._inference_roi = inference_roi

        if not self.cap.isOpened():
            raise RuntimeError(f"Kunne ikke åpne video/kamera: {source}")

        # Køene
        self.raw_q = LatestQueue(maxsize=queue_size)
        self.proc_q = LatestQueue(maxsize=queue_size)

        # Thread 1: capture
        self.t_cap = threading.Thread(
            target=_capture_loop,
            args=(self.cap, self.raw_q, self.stop_event, frame_skip),
            daemon=False,
        )

        # Thread 2: preprocessing
        self.t_pre = threading.Thread(
            target=_preprocess_loop,
            args=(
                # In queue
                self.raw_q,
                # Out queue
                self.proc_q,
                self.stop_event,
                self._get_inference_roi,
                resize,
                crop_padding
            ),
            daemon=False,
        )

    def start(self):
        """Start trådene."""
        self.t_cap.start()
        self.t_pre.start()

    def read(self, timeout: float = 0.5) -> Optional[FrameItem]:
        """
        Henter en ferdig frame fra pipeline.
        Returnerer:
        - FrameItem hvis det finnes frame
        - None hvis ingen frame innen timeout

        main.py kan da gjøre:
        item = pipeline.read()
        if item is None: continue
        frame = item.frame
        """
        try:
            return self.proc_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stopper pipeline og frigjør VideoCapture."""
        self.stop_event.set()

        # Join worker threads first so cap.release() doesn't race with cap.read().
        if self.t_pre.is_alive():
            self.t_pre.join(timeout=1.0)
        if self.t_cap.is_alive():
            self.t_cap.join(timeout=2.0)

        self.cap.release()

        # Final short joins in case release() unblocks a lingering read.
        if self.t_cap.is_alive():
            self.t_cap.join(timeout=0.5)
        if self.t_pre.is_alive():
            self.t_pre.join(timeout=0.5)

    def set_inference_roi(self, roi):
        with self._roi_lock:
            self._inference_roi = roi

    def _get_inference_roi(self):
        with self._roi_lock:
            if self._inference_roi is None:
                return None
            return [tuple(pt) for pt in self._inference_roi]
