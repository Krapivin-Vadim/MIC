from dataclasses import dataclass


@dataclass(frozen=True)
class DroneLocationDto:
    latitude: float
    longitude: float
    altitude: float
