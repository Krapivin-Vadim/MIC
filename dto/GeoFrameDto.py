from pathlib import Path

from affine import Affine
from rasterio.crs import CRS
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class GeoFrameDto:
    image: np.ndarray
    crs: CRS
    transformer: Affine
    bounding_box: tuple[float, float, float, float]
    path: Path