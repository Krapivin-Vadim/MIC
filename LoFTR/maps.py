"""Провайдеры спутниковой карты: JPG, GeoTIFF, MBTiles."""
from __future__ import annotations

import io
import math
import sqlite3
from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image

from core import MapConfig


class MapProvider(ABC):
    @property
    @abstractmethod
    def width(self) -> int: ...

    @property
    @abstractmethod
    def height(self) -> int: ...

    @abstractmethod
    def crop_pixels(self, cx: int, cy: int, size_px: int) -> np.ndarray: ...


class JPGMapProvider(MapProvider):
    def __init__(self, path: str):
        self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(f"Не найден файл карты: {path}")

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    def crop_pixels(self, cx: int, cy: int, size_px: int) -> np.ndarray:
        h, w = self.image.shape[:2]
        half = size_px // 2
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, x0 + size_px)
        y1 = min(h, y0 + size_px)
        return self.image[y0:y1, x0:x1]


class GeoTIFFMapProvider(MapProvider):
    def __init__(self, path: str):
        import rasterio
        self.src = rasterio.open(path)

    def get_patch(self, lat, lon, size_px, zoom) -> np.ndarray:
        from rasterio.windows import Window
        row, col = self.src.index(lon, lat)
        half = size_px // 2
        window = Window(col - half, row - half, size_px, size_px)
        return self.src.read(1, window=window)

    def crop_pixels(self, cx: int, cy: int, size_px: int) -> np.ndarray:
        from rasterio.windows import Window
        half = size_px // 2
        window = Window(cx - half, cy - half, size_px, size_px)
        return self.src.read(1, window=window)

    def pixels_to_latlon(self, cx, cy):
        lon, lat = self.src.xy(cy, cx)
        return lat, lon

    @property
    def width(self) -> int:
        return self.src.width

    @property
    def height(self) -> int:
        return self.src.height


def build_map_provider(cfg: MapConfig) -> MapProvider:
    t = cfg.type.lower()
    if t == "jpg":
        return JPGMapProvider(cfg.path)
    if t == "geotiff":
        return GeoTIFFMapProvider(cfg.path)
    raise ValueError(f"Неизвестный тип карты: {t}")