import argparse
import csv
import random
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS


def create_monotone_geotiff_with_drone_marker(
    path: Path,
    crs: CRS,
    left: float,
    bottom: float,
    right: float,
    top: float,
    gsd: float,
    drone_x: float,
    drone_y: float,
    marker_radius_m: float = 10.0,
) -> None:
    """
    Создаёт GeoTIFF:
      - фон: тёмно-серый (50, 50, 50);
      - рисуется белый круг в позиции (drone_x, drone_y).
    """
    width_px = int(round((right - left) / gsd))
    height_px = int(round((top - bottom) / gsd))

    if width_px <= 0 or height_px <= 0:
        raise ValueError("Invalid dimensions for GeoTIFF")

    # Фон: тёмно-серый
    img = np.full((height_px, width_px, 3), 50, dtype=np.uint8)

    transform = from_bounds(left, bottom, right, top, width_px, height_px)
    col_drone, row_drone = ~transform * (drone_x, drone_y)
    col_drone = int(round(col_drone))
    row_drone = int(round(row_drone))

    radius_px = max(int(round(marker_radius_m / gsd)), 1)

    # Ограничиваем bounding box маркера границами изображения
    y_min = max(0, row_drone - radius_px)
    y_max = min(height_px, row_drone + radius_px + 1)
    x_min = max(0, col_drone - radius_px)
    x_max = min(width_px, col_drone + radius_px + 1)

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            dx = x - col_drone
            dy = y - row_drone
            if dx * dx + dy * dy <= radius_px * radius_px:
                img[y, x] = [255, 255, 255]

    transform = from_bounds(left, bottom, right, top, width_px, height_px)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height_px,
        width=width_px,
        count=3,
        dtype=img.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(np.moveaxis(img, -1, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=3, help="N x N сетка файлов")
    parser.add_argument("--tile-size-m", type=float, default=1000.0, help="Размер одного тайла в метрах")
    parser.add_argument("--gsd", type=float, default=1.0, help="GSD (м/пиксель)")
    parser.add_argument("--seed", type=int, default=42, help="Seed для генератора случайных чисел")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    crs = CRS.from_epsg(32632)  # UTM 32N

    n = args.grid_size
    tile_size = args.tile_size_m
    gsd = args.gsd

    random.seed(args.seed)

    rows = []

    # Генерируем сетку тайлов
    for i in range(n):
        for j in range(n):
            left = i * tile_size
            right = (i + 1) * tile_size
            bottom = j * tile_size
            top = (j + 1) * tile_size

            # Случайная позиция дрона внутри этого тайла
            drone_x = random.uniform(left, right)
            drone_y = random.uniform(bottom, top)

            filename = f"tile_{i}_{j}.tif"
            path = args.output_dir / filename

            create_monotone_geotiff_with_drone_marker(
                path,
                crs=crs,
                left=left,
                bottom=bottom,
                right=right,
                top=top,
                gsd=gsd,
                drone_x=drone_x,
                drone_y=drone_y,
                marker_radius_m=10.0,
            )

            rows.append(
                {
                    "filename": filename,
                    "drone_x": drone_x,
                    "drone_y": drone_y,
                }
            )

    # Записываем CSV
    csv_path = args.output_dir / "drone_positions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "drone_x", "drone_y"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {n * n} GeoTIFF files in {args.output_dir}")
    print(f"Drone positions saved to {csv_path}")


if __name__ == "__main__":
    main()