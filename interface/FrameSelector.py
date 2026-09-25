from typing import Protocol

import numpy as np

from dto.DroneLocationDto import DroneLocationDto


class FrameSelector(Protocol):
    """Интерфейс класса поиска снимка со спутника"""

    def get_frame(self, position: DroneLocationDto) -> np.ndarray:
        """"Метод получения снимка, соответсвующего переданной позицией дрона.
        Класс, реализующий этот метод, должен осуществлять поиск снимка, который соответствует
        координатам дрона, переданными в метод"""
        pass