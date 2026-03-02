import cv2
import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Any


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
    ts: float
    frame: Any


# 2) Queue som alltid holder "siste frame"
#
class LatestQueue(queue.Queue):
    """
    vaanlig queue kan bygge kø og skape latency (forsinkelse).
    I realtime vil vi heller ha "nyeste frame" enn å prosessere gamle frames.

    put_latest():
    om køen er full: dropp én gammel frame og legg inn den nye.
    resultat: vi ligger nærmest mulig live, selv om prosesseringen blir litt treg.
    """
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
#
def capture_loop(cap: cv2.VideoCapture,
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
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # video ferdig / kamera feilet
            stop_event.set()
            break

        i += 1
        if frame_skip > 1 and (i % frame_skip != 0):
            continue

        out_q.put_latest(FrameItem(time.perf_counter(), frame))


# 4) Tråd 2: Preprocessing (valgfritt steg egt kan sees på)
def preprocess_loop(in_q: LatestQueue,
                    out_q: LatestQueue,
                    stop_event: threading.Event,
                    resize=None):
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

        out_q.put_latest(FrameItem(item.ts, frame))


# 5) Pipeline-klassen (det vi bruker i main.py)
# 
class AsyncFramePipeline:
    """
Dette er "wrapperen" main.py bruker.


    fordi main.py skal bare gjøre: item = pipeline.read()
    pipeline sørger for at capture + preprocessing skjer i bakgrunnen.

    Konsept:
    Thread 1 capture_loop() legger frames i raw_q
    Thread 2 preprocess_loop() tar fra raw_q og legger i proc_q
     main.py - read() tar siste frame fra proc_q og gjør detection/tracking/display
    """

    def __init__(self,
                 source,
                 frame_skip: int = 1,
                 resize=None,
                 queue_size: int = 3):
        """
        source: video path eller 0/1 for webcam
        frame_skip: hopp over frames ved capture hvis ønskelig
        resize: (w,h) eller None
        queue_size: små køer gir lav latency (3 er ofte fint)
        """
        self.stop_event = threading.Event()
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Kunne ikke åpne video/kamera: {source}")

        # Køene
        self.raw_q = LatestQueue(maxsize=queue_size)
        self.proc_q = LatestQueue(maxsize=queue_size)

        # Thread 1: capture
        self.t_cap = threading.Thread(
            target=capture_loop,
            args=(self.cap, self.raw_q, self.stop_event, frame_skip),
            daemon=True
        )

        # Thread 2: preprocessing
        self.t_pre = threading.Thread(
            target=preprocess_loop,
            args=(self.raw_q, self.proc_q, self.stop_event, resize),
            daemon=True
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
        self.cap.release()
