from typing import Protocol

import numpy as np

from dto.MatchResult import MatchResult


class AILocationCorrector(Protocol):
    """Интерфейс класса коррекции координат при помощи ИИ"""

    def correct(self,
                frame_from_storage: np.ndarray,
                frame_from_drone: np.ndarray) -> MatchResult:
        """Класс, реализующий этот метод, должен сопостовлять снимок с дрона и снимок со спутника,
        используя ИИ модель
        (скорее всего параметр drone_location лишний)"""
        pass