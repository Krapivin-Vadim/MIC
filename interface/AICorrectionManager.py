
import numpy as np

from dto.GeoFrameDto import GeoFrameDto
from dto.DroneLocationDto import DroneLocationDto
from interface.AILocationCorrector import AILocationCorrector
from interface.Camera import Camera
from interface.FrameSelector import FrameSelector
from interface.MavrosManager import MavrosManager
from dto.MatchResult import MatchResult
import cv2 as cv


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

    @staticmethod
    def extract_point(geo_frame_dto: GeoFrameDto,
                      frame_from_drone: cv.Mat,
                      matches: MatchResult,
                      drone_location: DroneLocationDto) -> DroneLocationDto:
        H, _ = cv.findHomography(matches.points_frame, matches.points_map)
        if H is None:
            raise ValueError("Unable to extract homography from frame")

        height, width = frame_from_drone.shape[:2]
        source_center = np.array(
            [[[width / 2.0, height / 2.0]]],
            dtype=np.float64,
        )

        target_center = cv.perspectiveTransform(source_center, H)
        pixel_x, pixel_y = map(float, target_center[0, 0])
        longitude, latitude = geo_frame_dto.transformer * (
            pixel_x,
            pixel_y,
        )

        return DroneLocationDto(
            latitude=float(latitude),
            longitude=float(longitude),
            altitude=drone_location.altitude
        )


    def correct_position(self) -> DroneLocationDto:
        """Коррекция координат дрона"""
        frame_from_drone: np.ndarray = self.cam.get_frame()
        drone_location: DroneLocationDto =  self.mavros.get_drone_location()
        geo_frame_dto: GeoFrameDto = self.selector.get_frame(drone_location)
        frame_from_storage: np.ndarray = geo_frame_dto.image
        corrections: DroneLocationDto = self.corrector.correct(drone_location, frame_from_storage, frame_from_drone)
        if self.valid(drone_location, corrections):
            return corrections
        self.mavros.set_drone_location(drone_location)
        return drone_location
