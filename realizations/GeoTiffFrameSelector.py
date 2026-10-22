import math

import numpy as np
import logging

from shapely.ops import transform

from interface.FrameSelector import FrameSelector
from dto.DroneLocationDto import DroneLocationDto
from dto.GeoFrameDto import GeoFrameDto
from pathlib import Path
import rasterio
from rasterio.windows import Window, bounds
from rasterio.crs import CRS
from shapely import STRtree, box, Polygon, Point
from pyproj import Transformer
from dataclasses import dataclass
import cv2 as cv

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FrameMeta:
    path: Path
    geom: Polygon
    crs: CRS
    pixel_size: float

@dataclass(frozen=True)
class CropSize:
    width_m: float
    height_m: float

class GeoTiffFrameSelector(FrameSelector):
    def __init__(self,
                 frame_base_dir: Path,
                 source_crs: str,
                 target_crs: str,
                 fov_v_degree: float,
                 fov_h_degree: float,
                 crop_margin: float= 1.5):
        """Инициализирует селектор фреймов и строит пространственный индекс.

        Args:
            frame_base_dir (Path): Директория с файлами снимков (.tif, .tiff).
            source_crs (str): Исходная CRS координат БПЛА (например, 'EPSG:4326').
            target_crs (str): Целевая CRS для пространственного индекса и расчетов (например, 'EPSG:3857').
            fov_v_degree (float): Вертикальный угол обзора камеры (FOV) в градусах.
            fov_h_degree (float): Горизонтальный угол обзора камеры (FOV) в градусах.
            crop_margin (float, optional): Коэффициент запаса для размера вырезаемого кропа. По умолчанию 1.5.
        """
        self.FRAME_BASE_DIR: Path = frame_base_dir.absolute()
        self.SOURCE_CRS: str = source_crs
        self.TARGET_CRS: str = target_crs
        self.CROP_MARGIN: float = crop_margin
        self.FOV_H_DEGREE: float = fov_h_degree
        self.FOV_V_DEGREE: float = fov_v_degree
        self._build_index()
        self.drone_to_index_transformer = Transformer.from_crs(self.SOURCE_CRS, self.TARGET_CRS, always_xy=True)

    def _build_index(self):
        """Метод для построения индекса для быстрого поиска файла по координатам дрона"""
        self.file_meta_order: list[FrameMeta] = list()
        tiff_files = list(self.FRAME_BASE_DIR.rglob("*.tif")) + list(self.FRAME_BASE_DIR.rglob("*.tiff"))
        for file in tiff_files:
            try:
                with rasterio.open(file) as src:
                    if src.crs is None:
                        logger.warning(f"CRS not found for file {file}")
                        continue
                    transformer = Transformer.from_crs(src.crs, self.TARGET_CRS, always_xy=True)

                    left, bottom, right, top = src.bounds
                    geom = box(left, bottom, right, top)
                    index_geom = transform(transformer.transform, geom)
                    pixel_size = min(abs(src.transform.a), abs(src.transform.e))
                    file_meta = FrameMeta(
                        path=file,
                        geom=index_geom,
                        crs=src.crs,
                        pixel_size=pixel_size
                    )
                    self.file_meta_order.append(file_meta)
            except Exception as e:
                logger.error(f"Error while building index for file {file}: {e}")
                continue
        if len(self.file_meta_order) == 0:
            raise RuntimeError("No valid frames found")
        self.index = STRtree([meta.geom for meta in self.file_meta_order])

    def _read_frame(self, path: Path, pos: Point, crop_size: CropSize) -> GeoFrameDto:
        """Читает фрагмент снимка с предположительным местонахождением дрона"""
        with rasterio.open(path) as src:
            frame_crs = src.crs
            transformer = Transformer.from_crs(self.TARGET_CRS, frame_crs, always_xy=True)
            x_frame, y_frame = transformer.transform(pos.x, pos.y)

            half_w = crop_size.width_m / 2.0
            half_h = crop_size.height_m / 2.0

            left = x_frame - half_w
            right = x_frame + half_w
            bottom = y_frame - half_h
            top = y_frame + half_h

            window = rasterio.windows.from_bounds(
                left=left,
                bottom=bottom,
                right=right,
                top=top,
                transform=src.transform
            )

            full_window = Window(0, 0, src.width, src.height)
            window = window.intersection(full_window)

            window = window.round_offsets().round_lengths()

            num_bands = min(src.count, 3)
            band_indexes = list(range(1, num_bands + 1))

            img = src.read(band_indexes, window=window)

            img = np.moveaxis(img, 0, -1)

            if img.shape[-1] == 3:
                img = img[..., :: -1].copy()

            window_transform = src.window_transform(window)
            window_bounds = bounds(window, src.transform)

            return GeoFrameDto(
                image=img,
                crs=frame_crs,
                transformer=window_transform,
                bounding_box=window_bounds,
                path=path,
            )

    @staticmethod
    def _get_most_efficient_frame(position: Point, frames: list[FrameMeta]) -> FrameMeta:
        """Метод определения наиболее эффективного снимка. Наиболее эффективным снимком считается тот,
        1) расстояние от центра которого до предолагаемой позицией дрона будет наименьшим
        2) у которого наименьший размер пикселя"""

        candidates: list[tuple[float, float, FrameMeta]] = list()
        for meta in frames:
            dist = position.distance(meta.geom.centroid)
            candidates.append((dist, meta.pixel_size, meta))
        candidates.sort(key=lambda tup: (tup[0], tup[1]))
        best_meta = candidates[0][2]
        return best_meta

    def _compute_crop(self,
                      drone_location: DroneLocationDto) -> CropSize:
        """Метод предназначен для рассчёта размеров вырезаемой области из найденного снимка"""

        H = drone_location.altitude
        fov_h_rad = math.radians(self.FOV_H_DEGREE)
        fov_v_rad = math.radians(self.FOV_V_DEGREE)
        return CropSize(
            2 * H * math.tan(fov_h_rad / 2) * self.CROP_MARGIN,
            2 * H * math.tan(fov_v_rad / 2) * self.CROP_MARGIN
        )

    def get_frame(self, position: DroneLocationDto) -> GeoFrameDto:
        row_pos = Point(position.longitude, position.latitude)
        drone_x, drone_y = self.drone_to_index_transformer.transform(row_pos.x, row_pos.y)
        clean_pos = Point(drone_x, drone_y)
        candidates: list[FrameMeta] = [self.file_meta_order[i] for i in self.index.query(clean_pos)]
        matches = [candidate for candidate in candidates if candidate.geom.intersects(clean_pos)]
        if len(matches) == 0:
            raise RuntimeError("No matches for current position")

        best_meta = matches[0] if len(matches) == 1 \
            else self._get_most_efficient_frame(clean_pos, matches)

        crop_size: CropSize = self._compute_crop(position)
        return self._read_frame(best_meta.path, clean_pos, crop_size)