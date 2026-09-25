from typing import Protocol

from dto.DroneLocationDto import DroneLocationDto


class MavrosManager(Protocol):
    """"Интерфейс класса взаиможействия с полётным контроллером посредством Mavros"""

    def get_drone_location(self) -> DroneLocationDto:
        pass