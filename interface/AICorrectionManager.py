
import numpy as np

from dto.DroneLocationDto import DroneLocationDto
from interface.AILocationCorrector import AILocationCorrector
from interface.Camera import Camera
from interface.FrameSelector import FrameSelector
from interface.MavrosManager import MavrosManager


class AICorrectionManager:
    def __init__(self,
                 cam: Camera,
                 selector: FrameSelector,
                 corrector: AILocationCorrector,
                 mavros: MavrosManager) -> None:
        self.cam: Camera = cam
        self.selector: FrameSelector = selector
        self.corrector: AILocationCorrector = corrector
        self.mavros: MavrosManager = mavros


    def valid(self, drone: DroneLocationDto, corrected: DroneLocationDto) -> bool:
        """Проверка поправок на адекватность"""
        pass

    def correct_position(self) -> DroneLocationDto:
        """Коррекция координат дрона"""
        frame_from_drone: np.ndarray = self.cam.get_frame()
        drone_location =  self.mavros.get_drone_location()
        frame_from_storage: np.ndarray = self.selector.get_frame(drone_location)
        corrections: DroneLocationDto = self.corrector.correct(drone_location, frame_from_storage, frame_from_drone)
        if self.valid(drone_location, corrections):
            return corrections
        return drone_location
