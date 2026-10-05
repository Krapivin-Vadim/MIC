import argparse
import csv
from pathlib import Path
import numpy as np
import cv2 as cv
from pyproj import Transformer

from dto.DroneLocationDto import DroneLocationDto
from realizations.GeoTiffFrameSelector import GeoTiffFrameSelector  # замени на реальный импорт

def found_drone(img: np.ndarray):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = cv.inRange(img, 250, 255)
    if np.count_nonzero(mask) > 0:
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Директория с GeoTIFF и drone_positions.csv")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Директория для сохранения кропов")

    parser.add_argument("--source-crs", type=str, default="EPSG:4326")
    parser.add_argument("--target-crs", type=str, default="EPSG:32632")

    parser.add_argument("--fov-h", type=float, default=80.0)
    parser.add_argument("--fov-v", type=float, default=60.0)
    parser.add_argument("--crop-margin", type=float, default=1.0)

    parser.add_argument("--altitude", type=float, default=100.0,
                        help="Высота дрона (м) для всех тестовых позиций")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.data_dir / "drone_positions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    # Создаём селектор
    selector = GeoTiffFrameSelector(
        frame_base_dir=args.data_dir,
        source_crs=args.source_crs,
        target_crs=args.target_crs,
        fov_v_degree=args.fov_v,
        fov_h_degree=args.fov_h,
        crop_margin=args.crop_margin,
    )

    # Трансформер UTM -> WGS84
    utm_to_wgs = Transformer.from_crs(args.target_crs, args.source_crs, always_xy=True)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            drone_x = float(row["drone_x"])
            drone_y = float(row["drone_y"])

            # Перевод в WGS84
            lon, lat = utm_to_wgs.transform(drone_x, drone_y)

            telemetry = DroneLocationDto(
                latitude=lat,
                longitude=lon,
                altitude=args.altitude,
            )

            frame = selector.get_frame(telemetry)

            # Сохраняем кроп
            # frame.image в BGR, можно сохранить как PNG
            out_path = args.output_dir / (Path(filename).stem + ".png")
            cv.imwrite(str(out_path), frame.image)

            found = found_drone(img=frame.image)

            print(f"Saved crop for {filename} -> {out_path.name} "
                  f"(file: {frame.path.name}, shape: {frame.image.shape})")
            print(f"Drone found: {found}")

    print(f"All crops saved to {args.output_dir}")


if __name__ == "__main__":
    main()