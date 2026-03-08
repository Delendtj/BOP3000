from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory

import numpy as np


class SharedMemoryRing:
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
        self._slot_bytes = self.max_h * self.max_w * self.channels * self.dtype.itemsize
        size = self._slot_bytes * self.slots

        # ????
        if create:
            self._shm = SharedMemory(name=self.name, create=True, size=size)
        else:
            self._shm = SharedMemory(name=self.name, create=False)

        self._write_idx = 0
        self._lock = Lock()

    def write(self, image: np.ndarray) -> tuple[int, int, int]:
        h, w = image.shape[:2]
        with self._lock:
            idx = self._write_idx
            self._write_idx = (self._write_idx + 1) % self.slots

        start = idx * self._slot_bytes
        end = start + self._slot_bytes
        slot = np.ndarray(
            (self.max_h, self.max_w, self.channels),
            dtype=self.dtype,
            buffer=self._shm.buf[start:end],
        )
        slot[:h, :w, :] = image
        return idx, int(h), int(w)

    def read(self, idx: int, h: int, w: int) -> np.ndarray:
        start = int(idx) * self._slot_bytes
        end = start + self._slot_bytes
        slot = np.ndarray(
            (self.max_h, self.max_w, self.channels),
            dtype=self.dtype,
            buffer=self._shm.buf[start:end],
        )
        return slot[: int(h), : int(w), :]

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()

