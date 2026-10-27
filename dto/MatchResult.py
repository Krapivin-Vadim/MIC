from dataclasses import dataclass

import numpy as np


@dataclass
class MatchResult:
    # Результат работы модели
    points_frame: np.ndarray  # Точки на кадре дрона (N x 2 массив [x, y])
    points_map: np.ndarray  # Соответствующие точки на карте (N x 2 массив [x, y])
    scores: np.ndarray  # Оценки совпадений от LightGlue (N массив, от 0 до 1)
    processing_time_ms: float  # Время обработки кадра в миллисекундах
    num_matches: int  # Количество найденных совпадений
