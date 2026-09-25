from asyncio import Protocol

import numpy as np

from dto.DroneLocationDto import DroneLocationDto


class AILocationCorrector(Protocol):
    """Интерфейс класса коррекции координат при помощи ИИ"""

    def correct(self,
                drone_location: DroneLocationDto,
                frame_from_storage: np.ndarray,
                frame_from_drone: np.ndarray) -> DroneLocationDto:
        """Класс, реализующий этот метод, должен сопостовлять снимок с дрона и снимок со спутника,
        используя ИИ модель
        (скорее всего параметр drone_location лишний)"""
        pass