
import numpy as np

from dto.GeoFrameDto import GeoFrameDto
from dto.DroneLocationDto import DroneLocationDto
from interface.AILocationCorrector import AILocationCorrector
from interface.Camera import Camera
from interface.FrameSelector import FrameSelector
from interface.MavrosManager import MavrosManager
from dto.MatchResult import MatchResult
import cv2 as cv
from pyproj import CRS, Transformer



class AICorrectionManager:
    def __init__(self,
                 cam: Camera,
                 selector: FrameSelector,
                 corrector: AILocationCorrector,
                 mavros: MavrosManager,
                 drone_crs: str,
                 metric_crs: str) -> None:
        self.cam: Camera = cam
        self.selector: FrameSelector = selector
        self.corrector: AILocationCorrector = corrector
        self.mavros: MavrosManager = mavros

        self._drone_crs = CRS.from_user_input(drone_crs)
        self._to_metric = Transformer.from_crs(
            self._drone_crs, CRS.from_user_input(metric_crs), always_xy=True
        )
        self._frame_to_drone: dict[str, Transformer] = {}

    def _get_frame_to_drone(self, frame_crs) -> Transformer:
        """frame_crs - rasterio.crs.CRS из GeoFrameDto."""
        key = frame_crs.to_wkt()
        if key not in self._frame_to_drone:
            self._frame_to_drone[key] = Transformer.from_crs(
                CRS.from_wkt(key), self._drone_crs, always_xy=True
            )
        return self._frame_to_drone[key]


    def valid(self, drone: DroneLocationDto, corrected: DroneLocationDto) -> bool:
        """Проверка поправок на адекватность"""
        pass

    def extract_point(self,
                      geo_frame_dto: GeoFrameDto,
                      frame_from_drone: cv.Mat,
                      matches: MatchResult,
                      drone_location: DroneLocationDto) -> DroneLocationDto:
        H, _ = cv.findHomography(matches.points_frame, matches.points_map)
        if H is None:
            raise ValueError("Unable to extract homography from frame")

        height, width = frame_from_drone.shape[:2]
        center = np.array([[[width / 2.0, height / 2.0]]], dtype=np.float64)
        map_px, map_py = map(float, cv.perspectiveTransform(center, H)[0, 0])

        # Пиксели кропа -> координаты в CRS снимка (Affine из GeoFrameDto).
        x, y = geo_frame_dto.transformer * (map_px, map_py)

        # CRS снимка -> CRS дрона (из настроек).
        lon, lat = self._get_frame_to_drone(geo_frame_dto.crs).transform(x, y)

        return DroneLocationDto(
            latitude=float(lat),
            longitude=float(lon),
            altitude=drone_location.altitude,
        )


    def correct_position(self) -> DroneLocationDto:
        """Коррекция координат дрона"""
        frame_from_drone: np.ndarray = self.cam.get_frame()
        drone_location: DroneLocationDto =  self.mavros.get_drone_location()
        geo_frame_dto: GeoFrameDto = self.selector.get_frame(drone_location)
        frame_from_storage: np.ndarray = geo_frame_dto.image
        matches: MatchResult = self.corrector.correct(frame_from_storage, frame_from_drone)
        corrected: DroneLocationDto = self.extract_point(
            geo_frame_dto,
            frame_from_drone,
            matches,
            drone_location
        )
        self.mavros.set_drone_location(corrected)
        return corrected
