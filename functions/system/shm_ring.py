from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory

import numpy as np


class SharedMemoryRing:
    """
    We make a Shared Memory Ring buffer so that we can write images (big objects) into shared memory.
    So it can be used by both main and the ocr worker.
    This makes it so that we don't need to serialize big objects across processes.

    This buffer is a fixed size calculated based on expected image size and data type of the object.

    A limitation is that it overwrites slots in memory because it doesn't track if workers are currently
    working with current slot. This should however be fine, especially if we cap frames. e.g 30/60fps
    """
    def __init__(
        self,
        name: str,
        slots: int,
        max_h: int,
        max_w: int,
        channels: int = 3,
        dtype: str = "uint8",
        create: bool = True,
    ) -> None:
        self.name = str(name)
        self.slots = int(slots)
        self.max_h = int(max_h)
        self.max_w = int(max_w)
        self.channels = int(channels)
        self.dtype = np.dtype(dtype)

        # Size per memory slot (1 slot = 1 image)
        # image size expected max_h and max_w
        # color channels (3)
        # bytes per channel. dtype("uint8") = 1
        self._slot_bytes = self.max_h * self.max_w * self.channels * self.dtype.itemsize

        # size per slot * number of slots
        size = self._slot_bytes * self.slots

        # Allocate new memory space
        if create:
            self._shm = SharedMemory(name=self.name, create=True, size=size)
        # Connect to existing memory space
        else:
            self._shm = SharedMemory(name=self.name, create=False)

        self._write_idx = 0
        self._lock = Lock()

    def write(self, image: np.ndarray) -> tuple[int, int, int]:
        h, w = image.shape[:2]

        # Places lock so that two writes don't pick same idx at the same time.
        # Doesn't prevent writer from writing on data currently being read.
        with self._lock:
            idx = self._write_idx
            # Increment by one, but wrap around back to zero if it goes over self.slots
            self._write_idx = (self._write_idx + 1) % self.slots

        start = idx * self._slot_bytes
        end = start + self._slot_bytes
        slot = np.ndarray(
            (self.max_h, self.max_w, self.channels),
            dtype=self.dtype,
            buffer=self._shm.buf[start:end],
        )
        # Actually write into memory
        slot[:h, :w, :] = image

        # Return slot number (idx) and h/w
        return idx, int(h), int(w)

    def read(self, idx: int, h: int, w: int) -> np.ndarray:
        start = int(idx) * self._slot_bytes
        end = start + self._slot_bytes
        slot = np.ndarray(
            (self.max_h, self.max_w, self.channels),
            dtype=self.dtype,
            buffer=self._shm.buf[start:end],
        )

        # Retrieve from ememory
        return slot[: int(h), : int(w), :]

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()

