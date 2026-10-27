#!/usr/bin/env python3
"""Облёт области в симуляторе Clover и сохранение снимков как GeoTIFF.

Что делает скрипт:
1. Взлетает на заданную высоту.
2. Летит "змейкой" по прямоугольной области (с перекрытием между снимками).
3. В каждой точке делает снимок с нижней камеры и сохраняет его как
   GeoTIFF: к картинке добавляется привязка к земле (границы снимка
   в выбранной системе координат).
4. Пишет metadata.csv с истинными координатами каждого снимка.
5. Возвращается в начальную точку и садится.

Готовую папку можно указать в FRAME_BASE_DIR для GeoTiffFrameSelector.

Запуск (в окружении с ROS и симулятором):
    python3 survey_geotiff.py --out-dir ./map_tiles --crs EPSG:3857
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import rasterio
import rospy
from clover import srv
from cv_bridge import CvBridge
from pyproj import Geod, Transformer
from rasterio.transform import from_bounds
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger

# В системе координат ENU yaw = pi/2 означает "смотреть на север".
# При таком курсе верх снимка направлен на север, а значит снимок
# выровнен по сторонам света, как обычная карта.
NORTH_UP_YAW = math.pi / 2

# Считает расстояния и смещения на поверхности Земли.
GEOD = Geod(ellps="WGS84")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Облёт области и сохранение GeoTIFF")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Куда сохранять снимки")
    p.add_argument("--crs", default="EPSG:3857",
                   help="CRS, в которой записываются GeoTIFF (по умолчанию EPSG:3857)")
    p.add_argument("--altitude", type=float, default=15.0,
                   help="Высота съёмки над точкой взлёта, м")
    p.add_argument("--area-width", type=float, default=40.0,
                   help="Размер области на восток от точки взлёта, м")
    p.add_argument("--area-height", type=float, default=40.0,
                   help="Размер области на север от точки взлёта, м")
    p.add_argument("--overlap", type=float, default=0.6,
                   help="Перекрытие соседних снимков, 0..0.9")
    p.add_argument("--speed", type=float, default=1.0, help="Скорость, м/с")
    p.add_argument("--settle", type=float, default=1.5,
                   help="Пауза перед снимком, чтобы дрон успокоился, с")
    p.add_argument("--camera-topic", default="main_camera/image_raw")
    p.add_argument("--camera-info-topic", default="main_camera/camera_info")
    p.add_argument("--fov-h", type=float, default=None,
                   help="Горизонтальный FOV, градусы (иначе берётся из camera_info)")
    p.add_argument("--fov-v", type=float, default=None,
                   help="Вертикальный FOV, градусы (иначе берётся из camera_info)")
    # rospy.myargv() убирает из аргументов служебные параметры ROS.
    return p.parse_args(rospy.myargv()[1:])


def get_fov(args) -> tuple[float, float]:
    """Возвращает углы обзора камеры (горизонтальный, вертикальный) в радианах."""
    info = None
    if args.fov_h is None or args.fov_v is None:
        info = rospy.wait_for_message(args.camera_info_topic, CameraInfo, timeout=10)

    # Угол обзора считается из размера кадра и фокусного расстояния в пикселях:
    # K[0] = fx, K[4] = fy.
    fov_h = math.radians(args.fov_h) if args.fov_h is not None \
        else 2 * math.atan(info.width / (2 * info.K[0]))
    fov_v = math.radians(args.fov_v) if args.fov_v is not None \
        else 2 * math.atan(info.height / (2 * info.K[4]))
    return fov_h, fov_v


def lawnmower(x0, y0, area_w, area_h, step_x, step_y):
    """Точки обхода "змейкой": ряд слева направо, следующий справа налево и т.д."""
    xs = np.arange(0.0, area_w + 1e-6, step_x)
    ys = np.arange(0.0, area_h + 1e-6, step_y)
    points = []
    for i, dy in enumerate(ys):
        row = xs if i % 2 == 0 else xs[::-1]
        points.extend((x0 + dx, y0 + dy) for dx in row)
    return points


def frame_bounds(lat, lon, width_m, height_m, to_target: Transformer):
    """Границы снимка (west, south, east, north) в целевой CRS.

    Центр снимка - позиция дрона. От неё откладываем половину размера
    снимка на восток/запад/север/юг по поверхности Земли и переводим
    полученные точки в целевую CRS. Так результат верен для любой CRS,
    в том числе для EPSG:3857, где единицы растянуты относительно метров.
    """
    lon_e, lat_e, _ = GEOD.fwd(lon, lat, 90, width_m / 2)
    lon_w, lat_w, _ = GEOD.fwd(lon, lat, 270, width_m / 2)
    lon_n, lat_n, _ = GEOD.fwd(lon, lat, 0, height_m / 2)
    lon_s, lat_s, _ = GEOD.fwd(lon, lat, 180, height_m / 2)

    x_e, _ = to_target.transform(lon_e, lat_e)
    x_w, _ = to_target.transform(lon_w, lat_w)
    _, y_n = to_target.transform(lon_n, lat_n)
    _, y_s = to_target.transform(lon_s, lat_s)
    return min(x_w, x_e), min(y_s, y_n), max(x_w, x_e), max(y_s, y_n)


def save_geotiff(path: Path, img_bgr: np.ndarray, bounds, crs: str) -> None:
    """Сохраняет изображение как GeoTIFF с привязкой к земле."""
    h, w = img_bgr.shape[:2]
    # OpenCV хранит каналы как BGR, а в GeoTIFF принято RGB.
    # GeoTiffFrameSelector при чтении сам разворачивает каналы обратно в BGR.
    rgb = np.ascontiguousarray(np.moveaxis(img_bgr[..., ::-1], -1, 0))
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=3, dtype="uint8",
        crs=crs,
        transform=from_bounds(*bounds, w, h),  # пиксели -> координаты в CRS
        compress="deflate",
    ) as dst:
        dst.write(rgb)


def main() -> None:
    rospy.init_node("geotiff_survey")
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Сервисы Clover: телеметрия, полёт в точку, посадка.
    rospy.wait_for_service("get_telemetry")
    get_telemetry = rospy.ServiceProxy("get_telemetry", srv.GetTelemetry)
    navigate = rospy.ServiceProxy("navigate", srv.Navigate)
    land = rospy.ServiceProxy("land", Trigger)
    bridge = CvBridge()  # переводит сообщения ROS Image в массивы OpenCV

    def navigate_wait(x=0.0, y=0.0, z=0.0, yaw=float("nan"), speed=0.5,
                      frame_id="", auto_arm=False, tolerance=0.2):
        """Летит в точку и ждёт, пока дрон туда не доберётся."""
        navigate(x=x, y=y, z=z, yaw=yaw, speed=speed,
                 frame_id=frame_id, auto_arm=auto_arm)
        while not rospy.is_shutdown():
            t = get_telemetry(frame_id="navigate_target")
            if math.sqrt(t.x ** 2 + t.y ** 2 + t.z ** 2) < tolerance:
                return
            rospy.sleep(0.2)

    # Преобразователь "широта/долгота -> выбранная CRS".
    to_target = Transformer.from_crs("EPSG:4326", args.crs, always_xy=True)

    fov_h, fov_v = get_fov(args)
    print(f"FOV: горизонтальный {math.degrees(fov_h):.1f} град, "
          f"вертикальный {math.degrees(fov_v):.1f} град "
          f"(их же укажите в FRAME_FOV_H / FRAME_FOV_V)")

    # Размер участка земли, попадающего в кадр, на высоте съёмки.
    width_m = 2 * args.altitude * math.tan(fov_h / 2)
    height_m = 2 * args.altitude * math.tan(fov_v / 2)
    step_x = width_m * (1 - args.overlap)
    step_y = height_m * (1 - args.overlap)
    print(f"Снимок покрывает {width_m:.1f} x {height_m:.1f} м, "
          f"шаг {step_x:.1f} x {step_y:.1f} м")

    print("Взлёт...")
    navigate_wait(z=args.altitude, frame_id="body", auto_arm=True)

    start = get_telemetry(frame_id="map")
    points = lawnmower(start.x, start.y, args.area_width, args.area_height,
                       step_x, step_y)
    print(f"Точек съёмки: {len(points)}")

    rows = []
    try:
        for i, (x, y) in enumerate(points):
            navigate_wait(x=x, y=y, z=args.altitude, yaw=NORTH_UP_YAW,
                          speed=args.speed, frame_id="map")
            rospy.sleep(args.settle)

            # Берём свежий кадр и сразу после него телеметрию.
            msg = rospy.wait_for_message(args.camera_topic, Image, timeout=5)
            telem = get_telemetry(frame_id="map")
            if math.isnan(telem.lat) or math.isnan(telem.lon):
                rospy.logwarn(f"Точка {i}: нет GPS, снимок пропущен")
                continue

            img = bridge.imgmsg_to_cv2(msg, "bgr8")
            bounds = frame_bounds(telem.lat, telem.lon, width_m, height_m, to_target)
            name = f"frame_{i:03d}.tif"
            save_geotiff(args.out_dir / name, img, bounds, args.crs)

            rows.append([name, telem.lat, telem.lon, telem.z, telem.x, telem.y, *bounds])
            print(f"[{i + 1}/{len(points)}] {name}  lat={telem.lat:.7f} lon={telem.lon:.7f}")
    finally:
        # Метаданные сохраняем, даже если полёт прервался.
        with open(args.out_dir / "metadata.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "lat", "lon", "alt_agl", "x_map", "y_map",
                        "west", "south", "east", "north"])
            w.writerows(rows)

    print("Возврат и посадка...")
    navigate_wait(x=start.x, y=start.y, z=args.altitude, yaw=NORTH_UP_YAW,
                  speed=args.speed, frame_id="map")
    land()
    print(f"Готово. Снимков: {len(rows)}, папка: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()