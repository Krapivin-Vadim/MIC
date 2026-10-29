"""Склейка блоков + CLI-запуск на одной паре или последовательности."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from core import (
    AppConfig, FilteredPosition, GeometryEstimator, MatchResult,
    PositionEstimate, build_matcher, load_config, FalsePositiveFilter,
)
from maps import build_map_provider


# ============================================================
# Пайплайн
# ============================================================

class DroneLocalizationPipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.matcher = build_matcher(cfg)
        self.geometry = GeometryEstimator(cfg)
        self.filter = FalsePositiveFilter(cfg)

    def process(self, frame: np.ndarray, sat: np.ndarray) -> dict:
        """Старый режим: кадр + уже вырезанный фрагмент карты."""
        t0 = time.time()
        matches = self.matcher.match(frame, sat)
        est = self.geometry.estimate(matches, frame.shape)
        filt = self.filter.update(est)
        return {"matches": matches, "estimate": est,
                "filtered": filt, "total_ms": (time.time() - t0) * 1000}

    def localize(
        self,
        frame: np.ndarray,
        map_provider,
        prior_xy: tuple[float, float] | None = None,
        search_radius_px: int = 300,
        tile_size: int = 640,
        stride: int = 320,
    ) -> FilteredPosition:
        """Ищет позицию дрона на карте.

        Если есть prior_xy — вырезает область вокруг него и матчит один раз.
        Если нет — скользит по всей карте тайлами и выбирает лучший.
        """
        if prior_xy is not None:
            return self._localize_around(frame, map_provider, prior_xy, search_radius_px)

        # Слепой поиск: скользим по карте
        best = None
        map_h, map_w = map_provider.height, map_provider.width
        for cy in range(tile_size // 2, map_h - tile_size // 2, stride):
            for cx in range(tile_size // 2, map_w - tile_size // 2, stride):
                tile = map_provider.crop_pixels(cx, cy, tile_size)
                matches = self.matcher.match(frame, tile)
                if not matches.success or len(matches) < self.cfg.model.min_matches:
                    continue
                est = self.geometry.estimate(matches, frame.shape)
                if not est.valid:
                    continue
                # Центр кадра в координатах карты + смещение тайла
                global_x = est.x + (cx - tile_size // 2)
                global_y = est.y + (cy - tile_size // 2)
                score = est.num_inliers
                if best is None or score > best["score"]:
                    best = {
                        "x": global_x, "y": global_y,
                        "score": score, "inliers": est.num_inliers,
                    }

        if best is None:
            return FilteredPosition(0, 0, time.time(), "no_estimate", 0,
                                    reason="no tile matched")

        est = PositionEstimate(
            x=best["x"], y=best["y"], timestamp=time.time(),
            confidence=0.0, num_inliers=best["inliers"], valid=True,
        )
        return self.filter.update(est)

    def _localize_around(self, frame, map_provider, prior_xy, radius_px):
        """Локализация в окрестности известного приора."""
        cx, cy = int(prior_xy[0]), int(prior_xy[1])
        tile = map_provider.crop_pixels(cx, cy, 2 * radius_px)
        matches = self.matcher.match(frame, tile)
        if not matches.success:
            return FilteredPosition(0, 0, time.time(), "no_estimate", 0)

        est = self.geometry.estimate(matches, frame.shape)
        if est.valid:
            # Переводим локальные координаты тайла в глобальные
            est.x += (cx - radius_px)
            est.y += (cy - radius_px)
        return self.filter.update(est)

    def localize_sliding(
            self,
            frame: np.ndarray,
            map_provider,
            frame_m_per_px: float,
            map_m_per_px: float,
            stride_ratio: float = 0.5,  # шаг = stride_ratio × размер окна
            min_inlier_ratio: float = 0.5,
            max_windows: int | None = None,
    ):
        """Скользящее окно по карте с учётом масштаба.

        Args:
            frame: кадр дрона (grayscale).
            map_provider: провайдер карты.
            frame_m_per_px: разрешение кадра (м/px).
            map_m_per_px: разрешение карты (м/px).
            stride_ratio: шаг скольжения относительно размера окна.
            min_inlier_ratio: минимальная доля инлайеров для валидного окна.
            max_windows: ограничение на число окон (для отладки).
        """
        h_f, w_f = frame.shape[:2]

        # Размер окна на карте, соответствующий покрытию кадра
        win_w = int(round(w_f * frame_m_per_px / map_m_per_px))
        win_h = int(round(h_f * frame_m_per_px / map_m_per_px))
        win = max(win_w, win_h)

        stride = max(32, int(win * stride_ratio))
        map_h, map_w = map_provider.height, map_provider.width

        candidates = []
        n_windows = 0

        for cy in range(win // 2, map_h - win // 2, stride):
            for cx in range(win // 2, map_w - win // 2, stride):
                if max_windows and n_windows >= max_windows:
                    break
                n_windows += 1

                tile = map_provider.crop_pixels(cx, cy, win)
                if tile.shape[0] < 32 or tile.shape[1] < 32:
                    continue

                matches = self.matcher.match(frame, tile)
                if not matches.success or len(matches) < self.cfg.model.min_matches:
                    continue

                est = self.geometry.estimate(matches, frame.shape)
                if not est.valid:
                    continue

                inlier_ratio = est.num_inliers / len(matches)
                if inlier_ratio < min_inlier_ratio:
                    continue

                # Переводим локальные координаты тайла в глобальные
                global_x = est.x + (cx - win // 2)
                global_y = est.y + (cy - win // 2)

                score = est.num_inliers * inlier_ratio
                candidates.append({
                    "x": global_x, "y": global_y,
                    "cx": cx, "cy": cy, "win": win,
                    "num_matches": len(matches),
                    "num_inliers": est.num_inliers,
                    "inlier_ratio": inlier_ratio,
                    "score": score,
                    "tile_shape": tile.shape,
                })

        if not candidates:
            return None, []

        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]

        est = PositionEstimate(
            x=best["x"], y=best["y"], timestamp=time.time(),
            confidence=best["inlier_ratio"], num_inliers=best["num_inliers"],
            valid=True,
        )
        filt = self.filter.update(est)
        return filt, candidates


# ============================================================
# Утилиты
# ============================================================

def load_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {path}")
    return img


def draw_matches(img0, img1, pts0, pts1, max_draw=50, target_h=800):
    a0 = img0 if img0.ndim == 3 else cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    a1 = img1 if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    h0, w0 = a0.shape[:2]
    h1, w1 = a1.shape[:2]
    s0 = target_h / h0
    s1 = target_h / h1

    a0 = cv2.resize(a0, (int(w0 * s0), target_h), interpolation=cv2.INTER_AREA)
    a1 = cv2.resize(a1, (int(w1 * s1), target_h), interpolation=cv2.INTER_AREA)
    pts0 = pts0 * s0
    pts1 = pts1 * s1

    h0, w0 = a0.shape[:2]
    h1, w1 = a1.shape[:2]

    canvas = np.zeros((target_h, w0 + w1, 3), np.uint8)
    canvas[:h0, :w0] = a0
    canvas[:h1, w0:w0 + w1] = a1

    for i in range(min(len(pts0), max_draw)):
        p0 = tuple(np.round(pts0[i]).astype(int))
        p1 = tuple(np.round(pts1[i]).astype(int) + np.array([w0, 0]))
        cv2.circle(canvas, p0, 3, (0, 255, 0), -1)
        cv2.circle(canvas, p1, 3, (0, 255, 0), -1)
        cv2.line(canvas, p0, p1, (0, 200, 255), 1)
    return canvas

def draw_localization_result(
    frame, map_ctx,
    tile_origin=(0, 0),
    best_pos=(0, 0),
    view_radius_px=None,
    target_h=800,
    draw_frame_rect=True,
):
    """Рисует кадр и карту рядом.
    Слева — кадр дрона, справа — карта с крестиком в найденной точке.
    Линии соответствий НЕ рисуются — только позиция.

    Args:
        frame: кадр дрона (grayscale или BGR).
        map_ctx: фрагмент карты для отображения.
        tile_origin: (ox, oy) — координаты левого верхнего угла map_ctx
                     в пикселях полной карты.
        best_pos: (x, y) — найденная позиция в пикселях полной карты.
        view_radius_px: радиус рамки вокруг best_pos (в пикселях карты).
                        Если None — рамка не рисуется.
        target_h: высота обоих изображений на итоговом canvas.
        draw_frame_rect: рисовать ли рамку предполагаемой области кадра.
    """
    a0 = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    a1 = map_ctx if map_ctx.ndim == 3 else cv2.cvtColor(map_ctx, cv2.COLOR_GRAY2BGR)

    # Приводим к одной высоте
    s0 = target_h / a0.shape[0]
    s1 = target_h / a1.shape[0]
    a0 = cv2.resize(a0, (int(a0.shape[1] * s0), target_h), interpolation=cv2.INTER_AREA)
    a1 = cv2.resize(a1, (int(a1.shape[1] * s1), target_h), interpolation=cv2.INTER_AREA)

    ox, oy = tile_origin

    # Крестик в найденной позиции
    bx = int(np.clip((best_pos[0] - ox) * s1, 0, a1.shape[1] - 1))
    by = int(np.clip((best_pos[1] - oy) * s1, 0, a1.shape[0] - 1))
    cv2.drawMarker(a1, (bx, by), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)

    # Рамка предполагаемой области кадра
    if draw_frame_rect and view_radius_px is not None:
        r = int(view_radius_px * s1)
        cv2.rectangle(
            a1,
            (max(0, bx - r), max(0, by - r)),
            (min(a1.shape[1] - 1, bx + r), min(a1.shape[0] - 1, by + r)),
            (0, 255, 0), 2,
        )

    # Центр кадра — для наглядности
    cx0 = a0.shape[1] // 2
    cy0 = a0.shape[0] // 2
    cv2.drawMarker(a0, (cx0, cy0), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)

    canvas = np.zeros((target_h, a0.shape[1] + a1.shape[1], 3), np.uint8)
    canvas[:, :a0.shape[1]] = a0
    canvas[:, a0.shape[1]:] = a1
    return canvas
# ============================================================
# CLI
# ============================================================
def _run_single(cfg, frame_path, map_path):
    out_dir = Path(cfg.io.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_gray(frame_path)
    sat = load_gray(map_path)  # ← cv2.imread, теряет геопривязку
    pipeline = DroneLocalizationPipeline(cfg)
    out = pipeline.process(frame, sat)

    m, e, f = out["matches"], out["estimate"], out["filtered"]
    print(f"Модель:        {m.model_name}")
    print(f"Инференс:      {m.inference_ms:.1f} мс")
    print(f"Всего:         {out['total_ms']:.1f} мс")
    print(f"Совпадений:    {len(m)}")
    print(f"Inlier'ов:     {e.num_inliers}")
    print(f"Статус:        {f.status}")
    if f.status == "accepted":
        print(f"Позиция:       ({f.x:.1f}, {f.y:.1f}), радиус {f.radius_m:.1f} м")

    vis = draw_matches(frame, sat, m.points_frame, m.points_map)
    cv2.imwrite(str(out_dir / "matches.jpg"), vis)
    print(f"Визуализация:  {out_dir / 'matches.jpg'}")

def _run_single_geotiff(cfg, frame_path, prior_xy=None, radius=300):   # ← НОВАЯ
    """НОВЫЙ режим: кадр + GeoTIFF + приор позиции."""
    frame = load_gray(frame_path)
    provider = build_map_provider(cfg.map)
    pipeline = DroneLocalizationPipeline(cfg)

    if prior_xy is None:
        prior_xy = (provider.width // 2, provider.height // 2)

    filt = pipeline.localize(frame, provider, prior_xy=prior_xy,
                             search_radius_px=radius)
    print(f"Статус:   {filt.status}")
    print(f"Позиция:  ({filt.x:.1f}, {filt.y:.1f})")
    if hasattr(provider, "pixels_to_latlon"):
        lat, lon = provider.pixels_to_latlon(int(filt.x), int(filt.y))
        print(f"lat/lon:  ({lat:.6f}, {lon:.6f})")

def _run_sliding(cfg, frame_path, frame_m_per_px, map_m_per_px,
                 stride_ratio=0.5, min_inlier_ratio=0.5, max_windows=None):
    from maps import build_map_provider

    out_dir = Path(cfg.io.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_gray(frame_path)
    provider = build_map_provider(cfg.map)
    pipeline = DroneLocalizationPipeline(cfg)

    t0 = time.time()
    filt, candidates = pipeline.localize_sliding(
        frame, provider,
        frame_m_per_px=frame_m_per_px,
        map_m_per_px=map_m_per_px,
        stride_ratio=stride_ratio,
        min_inlier_ratio=min_inlier_ratio,
        max_windows=max_windows,
    )
    elapsed = time.time() - t0

    print(f"Время:              {elapsed:.1f} с")
    print(f"Кандидатов:         {len(candidates)}")
    if not candidates:
        print("Ни одно окно не прошло пороги")
        return

    for i, c in enumerate(candidates[:5]):
        print(f"  #{i+1}: score={c['score']:.1f} "
              f"inliers={c['num_inliers']}/{c['num_matches']} "
              f"({c['inlier_ratio']*100:.0f}%) "
              f"pos=({c['x']:.0f},{c['y']:.0f})")

    # Визуализация лучшего окна
    best = candidates[0]
    vis_radius = best["win"]

    # Вырезаем окрестность на карте
    cx, cy = int(best["x"]), int(best["y"])
    ctx = provider.crop_pixels(cx, cy, 2 * vis_radius)

    vis = draw_localization_result(
        frame, ctx,
        tile_origin=(cx - vis_radius, cy - vis_radius),
        best_pos=(best["x"], best["y"]),
        view_radius_px=best["win"] // 2,  # половина кадра в пикселях карты
        draw_frame_rect=True,
    )
    cv2.imwrite(str(out_dir / "localization.jpg"), vis)
    print(f"Визуализация: {out_dir / 'localization.jpg'}")

def _run_sequence(cfg, frames_dir, map_path, fps):
    sat = load_gray(map_path)
    pipeline = DroneLocalizationPipeline(cfg)

    frames = sorted(Path(frames_dir).glob("*.jpg")) + sorted(Path(frames_dir).glob("*.png"))
    if not frames:
        print("Кадры не найдены")
        return
    for i, fpath in enumerate(frames):
        frame = load_gray(fpath)
        out = pipeline.process(frame, sat)
        f = out["filtered"]
        print(
            f"[{i:04d}] {fpath.name} | "
            f"m={len(out['matches']):3d} | "
            f"in={out['estimate'].num_inliers:3d} | "
            f"{f.status:18s} | "
            f"pos=({f.x:7.1f},{f.y:7.1f}) | "
            f"r={f.radius_m:5.1f} m | "
            f"{out['total_ms']:6.1f} ms"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Drone localization pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frame", help="Кадр дрона")
    parser.add_argument("--frames", help="Директория с кадрами")
    parser.add_argument("--map", required=True, help="Карта (файл или GeoTIFF)")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--geotiff", action="store_true",
                        help="Использовать режим GeoTIFF с приором")
    parser.add_argument("--prior", type=float, nargs=2, default=None,
                        metavar=("X", "Y"), help="Приор позиции в пикселях карты")
    parser.add_argument("--radius", type=int, default=300,
                        help="Радиус поиска вокруг приора, px")
    parser.add_argument("--sliding", action="store_true",
                        help="Режим скользящего окна")
    parser.add_argument("--frame-m-per-px", type=float, default=None,
                        help="Разрешение кадра в м/px")
    parser.add_argument("--map-m-per-px", type=float, default=None,
                        help="Разрешение карты в м/px")
    parser.add_argument("--stride-ratio", type=float, default=0.5)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.5)
    parser.add_argument("--max-windows", type=int, default=None)

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.sliding:
        if not args.frame:
            parser.error("--sliding требует --frame")
        if args.frame_m_per_px is None or args.map_m_per_px is None:
            parser.error("--sliding требует --frame-m-per-px и --map-m-per-px")
        _run_sliding(
            cfg,
            args.frame,
            frame_m_per_px=args.frame_m_per_px,
            map_m_per_px=args.map_m_per_px,
            stride_ratio=args.stride_ratio,
            min_inlier_ratio=args.min_inlier_ratio,
            max_windows=args.max_windows,
        )
    elif args.geotiff:
        if not args.frame:
            parser.error("--geotiff требует --frame")
        prior = tuple(args.prior) if args.prior else None
        _run_single_geotiff(cfg, args.frame, prior_xy=prior, radius=args.radius)
    elif args.frames:
        _run_sequence(cfg, args.frames, args.map, args.fps)
    elif args.frame:
        _run_single(cfg, args.frame, args.map)
    else:
        parser.error("Укажите --sliding, --geotiff, --frame или --frames")

if __name__ == "__main__":
    raise SystemExit(main())