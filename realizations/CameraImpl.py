from time import sleep, time

import numpy as np
from numpy.typing import NDArray
import cv2 as cv

from interface.Camera import Camera
from threading import Thread, Lock, Event


class CameraImpl (Camera):
    """Читает кадры с камеры в отдельном потоке и хранит последний полученный кадр."""
    def __init__(self,
                 camera_id: int,
                 timeout: int = 2,
                 frame_size: tuple[int, int] = (None, None),
                 fps: int = None,) -> None:
        """Инициализирует камеру, применяет настройки и запускает поток захвата.

               Args:
                   camera_id: Индекс камеры для OpenCV, например 0 для первой доступной камеры.
                   frame_size: Желаемое разрешение в пикселях в формате (ширина, высота).
                       Если один из элементов равен None, соответствующий параметр не изменяется.
                   fps: Желаемая частота кадров. Если None, частота не изменяется.

               Raises:
                   RuntimeError: Если камеру не удалось открыть.
               """
        self._timeout = timeout
        self._stop_event_ = Event()
        self.current_frame: NDArray = None
        self._camera_id = camera_id
        self._lock = Lock()
        self._frame_size = frame_size
        self._fps = fps
        self._cap = cv.VideoCapture(self._camera_id)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Не удалось открыть камеру с id={camera_id}")
        self._config_cap_()
        self._video_thread: Thread = Thread(target=self.__capture_loop__)
        self._video_thread.start()

    def __enter__(self) -> "CameraImpl":
        """Возвращает объект камеры для использования в конструкции with."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Останавливает поток захвата и освобождает камеру при выходе из with."""
        self.close()

    def close(self) -> None:
        """Останавливает поток захвата и освобождает ресурс камеры.

        Метод безопасно вызывать повторно. После закрытия чтение кадров
        становится невозможным.
        """
        self._stop_event_.set()

        if self._video_thread.is_alive():
            self._video_thread.join(timeout=2.0)

        if self._cap.isOpened():
            self._cap.release()

    def _config_cap_(self) -> None:
        if None not in self._frame_size:
            self._cap.set(cv.CAP_PROP_FRAME_WIDTH, self._frame_size[0])
            self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, self._frame_size[1])
        if self._fps is not None:
            self._cap.set(cv.CAP_PROP_FPS, self._fps)

    def __capture_loop__(self):
        while not self._stop_event_.is_set():
            ret, frame = self._cap.read()
            if not ret:
                print("Lost video")
                sleep(1)
                continue
            with self._lock:
                self.current_frame = frame

    def get_frame(self) -> np.ndarray:
        frame: NDArray = None

        t1 = time()
        while self.current_frame is None:
            t2 = time()
            if t2 - t1 > self._timeout:
                raise RuntimeError("No frame available. Capture is lost")

        with self._lock:
            frame = self.current_frame.copy()
        return frame