from typing import Protocol

import numpy as np


class Camera(Protocol):
    """Интерфейс взаимодействия с камерой"""

    def get_frame(self) -> np.ndarray:
        pass