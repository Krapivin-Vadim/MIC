from typing import Protocol

from dto.GeoFrameDto import  GeoFrameDto
from dto.DroneLocationDto import DroneLocationDto


class FrameSelector(Protocol):
    """Интерфейс класса поиска снимка со спутника"""

    def get_frame(self, position: DroneLocationDto) -> GeoFrameDto:
        """"Метод получения снимка, соответсвующего переданной позицией дрона.
        Класс, реализующий этот метод, должен осуществлять поиск снимка, который соответствует
        координатам дрона, переданными в метод"""
        pass