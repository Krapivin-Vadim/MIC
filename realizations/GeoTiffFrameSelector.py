from logging import log

import numpy as np
import logging

import shapely

from interface.FrameSelector import FrameSelector
from dto.DroneLocationDto import DroneLocationDto
from pathlib import Path
import rasterio
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



class GeoTiffFrameSelector(FrameSelector):
    def __init__(self,
                 frame_base_dir: Path,
                 source_crs: str,
                 target_crs: str):
        self.FRAME_BASE_DIR: Path = frame_base_dir.absolute()
        self.SOURCE_CRS: str = source_crs
        self.TARGET_CRS: str = target_crs
        self.build_index()
        self.drone_to_index_transformer = Transformer.from_crs(self.SOURCE_CRS, self.TARGET_CRS, always_xy=True)

    def build_index(self):
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
                    index_geom = shapely.transform(transformer.transform, geom)
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

    @staticmethod
    def _read_frame(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            img = src.read([1, 2, 3])
            img = np.moveaxis(img, 0, -1)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            return img

    def get_most_efficient_frame(self, position: Point, frames: list[FrameMeta]) -> np.ndarray:
        candidates: list[tuple[float, float, FrameMeta]] = list()
        for meta in frames:
            dist = position.distance(meta.geom.centroid)
            candidates.append((dist, meta.pixel_size, meta))
        candidates.sort(key=lambda tup: (tup[0], tup[1]))
        best_meta = candidates[0][2]
        return self._read_frame(best_meta.path)



    def get_frame(self, position: DroneLocationDto) -> np.ndarray:
        row_pos = Point(position.longitude, position.latitude)
        drone_x, drone_y = self.drone_to_index_transformer.transform(row_pos.x, row_pos.y)
        clean_pos = Point(drone_x, drone_y)
        candidates: list[FrameMeta] = [self.file_meta_order[i] for i in self.index.query(clean_pos)]
        matches = [candidate for candidate in candidates if candidate.geom.contains(clean_pos)]
        if len(matches) == 0:
            raise RuntimeError("No matches for current position")

        if len(matches) == 1:
            return self._read_frame(matches[0].path)

        return self.get_most_efficient_frame(clean_pos, matches)





