"""Типы, конфиг, модели-матчеры, геометрия и фильтр ложных срабатываний."""
from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


# ============================================================
# Типы данных
# ============================================================

@dataclass
class MatchResult:
    points_frame: np.ndarray
    points_map: np.ndarray
    scores: np.ndarray
    model_name: str
    inference_ms: float

    @property
    def success(self) -> bool:
        return len(self.scores) > 0

    def __len__(self) -> int:
        return len(self.scores)


@dataclass
class PositionEstimate:
    x: float
    y: float
    timestamp: float
    confidence: float
    num_inliers: int
    valid: bool
    reason: str = ""


@dataclass
class FilteredPosition:
    x: float
    y: float
    timestamp: float
    status: str          # "accepted" | "rejected_vote" | "rejected_distance" | "no_estimate"
    votes: int
    radius_m: float = 0.0
    reason: str = ""


# ============================================================
# Конфиг
# ============================================================

@dataclass
class ModelConfig:
    name: str = "coarse_loftr"
    conf_threshold: float = 0.5
    min_matches: int = 8
    max_size: int = 640


@dataclass
class GeometryConfig:
    min_inliers: int = 8
    ransac_thresh_px: float = 3.0


@dataclass
class FilterConfig:
    map_scale_m_per_px: float = 0.5
    max_drone_speed_mps: float = 20.0
    max_time_gap_s: float = 10.0
    vote_window: int = 7
    vote_min_agree: int = 4
    vote_radius_m: float = 30.0


@dataclass
class MapConfig:
    type: str = "jpg"
    path: str = ""
    zoom: int = 18


@dataclass
class IOConfig:
    output_dir: str = "output"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    map: MapConfig = field(default_factory=MapConfig)
    io: IOConfig = field(default_factory=IOConfig)


def _dc(cls, data):
    if not isinstance(data, dict):
        return cls()
    names = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in names})


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(
        model=_dc(ModelConfig, raw.get("model", {})),
        geometry=_dc(GeometryConfig, raw.get("geometry", {})),
        filter=_dc(FilterConfig, raw.get("filter", {})),
        map=_dc(MapConfig, raw.get("map", {})),
        io=_dc(IOConfig, raw.get("io", {})),
    )


# ============================================================
# Модели-матчеры
# ============================================================

class BaseMatcher(ABC):
    name: str = "base"

    @abstractmethod
    def match(self, frame: np.ndarray, sat: np.ndarray) -> MatchResult: ...


MATCHER_REGISTRY: dict[str, type[BaseMatcher]] = {}


def register(name: str):
    def wrapper(cls):
        MATCHER_REGISTRY[name] = cls
        return cls
    return wrapper


@register("loftr")
class LoFTRMatcher(BaseMatcher):
    name = "loftr"

    def __init__(self, cfg: AppConfig):
        import torch
        from kornia.feature import LoFTR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LoFTR(pretrained="outdoor").to(self.device).eval()
        self.threshold = cfg.model.conf_threshold
        self.max_size = cfg.model.max_size

    def _prep(self, img):
        """Возвращает (tensor, scale), где scale — во сколько раз уменьшили."""
        import torch
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        s = self.max_size / max(h, w)
        if s < 1.0:
            new_w, new_h = int(round(w * s)), int(round(h * s))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            s = 1.0
        t = torch.from_numpy(img).float()[None, None] / 255.0
        return t.to(self.device), s

    def match(self, frame, sat) -> MatchResult:
        import torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            f0, s0 = self._prep(frame)
            f1, s1 = self._prep(sat)
            out = self.model({"image0": f0, "image1": f1})
        if self.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000

        kp0 = out["keypoints0"].cpu().numpy()
        kp1 = out["keypoints1"].cpu().numpy()
        conf = out["confidence"].cpu().numpy()

        # Возвращаем точки в координатах ОРИГИНАЛЬНЫХ изображений
        kp0 = kp0 / s0
        kp1 = kp1 / s1

        mask = conf >= self.threshold
        return MatchResult(kp0[mask], kp1[mask], conf[mask], self.name, ms)


@register("coarse_loftr")
class CoarseLoFTRMatcher(BaseMatcher):
    name = "coarse_loftr"
    REPO_DIR = "coarse_loftr_trt"
    WEIGHTS = "coarse_loftr_trt/weights/LoFTR_teacher.pt"

    def __init__(self, cfg: AppConfig):
        import torch
        repo = Path(self.REPO_DIR).resolve()
        if str(repo) not in sys.path:
            sys.path.append(str(repo))

        from loftr import LoFTR
        from loftr.utils.cvpr_ds_config import default_cfg
        from utils import make_student_config, make_query_image

        self._make_query_image = make_query_image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cfg = make_student_config(default_cfg)
        model = LoFTR(config=model_cfg)
        ckpt = torch.load(self.WEIGHTS, map_location=self.device)
        sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(sd, strict=False)
        self.model = model.eval().to(self.device)
        self.img_size = (model_cfg["input_width"], model_cfg["input_height"])
        self.coarse_res = 16
        self.threshold = cfg.model.conf_threshold

    def _bgr(self, img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    def _scale_for(self, img):
            h, w = img.shape[:2]
            W_t, H_t = self.img_size  # (640, 480)
            return min(W_t / w, H_t / h)

    def match(self, frame, sat) -> MatchResult:
        import torch
        h0, w0 = frame.shape[:2]
        h1, w1 = sat.shape[:2]

        f0 = self._make_query_image(self._bgr(frame), self.img_size)
        f1 = self._make_query_image(self._bgr(sat), self.img_size)

        # Во сколько раз уменьшили (по большей стороне)

        # в match:
        s0 = self._scale_for(frame)
        s1 = self._scale_for(sat)

        t0_in = torch.from_numpy(f0).float()[None, None] / 255.0
        t1_in = torch.from_numpy(f1).float()[None, None] / 255.0
        t0_in, t1_in = t0_in.to(self.device), t1_in.to(self.device)

        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(t0_in, t1_in)
        if self.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000

        cm = out[0] if isinstance(out, tuple) else out
        cm = cm.detach().float()
        if cm.dim() == 4:
            cm = cm.squeeze(1)
        elif cm.dim() == 2:
            cm = cm[None]

        kp0, kp1, conf = self._extract(cm)

        # Возвращаем точки в координатах оригинала
        kp0 = kp0 / s0
        kp1 = kp1 / s1

        mask = conf >= self.threshold
        return MatchResult(kp0[mask], kp1[mask], conf[mask], self.name, ms)

    def _extract(self, cm):
        cm_np = cm.cpu().numpy()
        _, N_q, N_r = cm_np.shape
        H_in, W_in = self.img_size[1], self.img_size[0]
        H_c, W_c = H_in // self.coarse_res, W_in // self.coarse_res
        assert H_c * W_c == N_q == N_r, f"{H_c}x{W_c} vs {N_q}x{N_r}"
        b, i, j = np.nonzero(cm_np > 0.01)
        conf = cm_np[b, i, j]
        iy, ix = np.divmod(i, W_c)
        jy, jx = np.divmod(j, W_c)
        h = self.coarse_res / 2.0
        kp0 = np.stack([ix * self.coarse_res + h, iy * self.coarse_res + h], 1).astype(np.float32)
        kp1 = np.stack([jx * self.coarse_res + h, jy * self.coarse_res + h], 1).astype(np.float32)
        return kp0, kp1, conf


@register("efficient_loftr")
class EfficientLoFTRMatcher(BaseMatcher):
    name = "efficient_loftr"
    HF_ID = "zju-community/efficientloftr"

    def __init__(self, cfg: AppConfig):
        import torch
        from transformers import AutoImageProcessor, AutoModelForKeypointMatching

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(self.HF_ID)
        self.model = (
            AutoModelForKeypointMatching
            .from_pretrained(self.HF_ID)
            .eval()
            .to(self.device)
        )
        self.threshold = cfg.model.conf_threshold

    def _pil(self, img):
        from PIL import Image
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img.astype(np.uint8))

    def match(self, frame, sat) -> MatchResult:
        import torch

        im0, im1 = self._pil(frame), self._pil(sat)
        inputs = self.processor([im0, im1], return_tensors="pt").to(self.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model(**inputs)
        ms = (time.perf_counter() - t0) * 1000

        sizes = [[(frame.shape[0], frame.shape[1]),
                  (sat.shape[0], sat.shape[1])]]
        res = self.processor.post_process_keypoint_matching(
            out, sizes, threshold=self.threshold
        )[0]

        return MatchResult(
            points_frame=res["keypoints0"].cpu().numpy(),
            points_map=res["keypoints1"].cpu().numpy(),
            scores=res["matching_scores"].cpu().numpy(),
            model_name=self.name,
            inference_ms=ms,
        )

@register("efficient_loftr_2")
class EfficientLoFTRMatcher(BaseMatcher):
    """
    EfficientLoFTR через HuggingFace Transformers.
    Обеспечивает высокую скорость и точность для полу-плотного сопоставления.
    """
    name = "efficient_loftr"
    HF_ID = "zju-community/efficientloftr"

    def __init__(self, cfg: AppConfig):
        import torch
        from transformers import AutoImageProcessor, AutoModelForKeypointMatching

        self.threshold = cfg.model.conf_threshold

        # Загрузка процессора и модели с HuggingFace Hub
        self.processor = AutoImageProcessor.from_pretrained(self.HF_ID)
        self.model = AutoModelForKeypointMatching.from_pretrained(self.HF_ID).eval()

        # Перенос на GPU при наличии
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _pil(self, img: np.ndarray):
        """Конвертирует numpy-массив (BGR или Gray) в RGB PIL Image."""
        from PIL import Image
        # Если изображение grayscale, делаем его трехканальным
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        # Если изображение цветное, конвертируем BGR (OpenCV) в RGB (PIL)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img.astype(np.uint8))

    def match(self, frame: np.ndarray, sat: np.ndarray) -> MatchResult:
        import torch
        im0, im1 = self._pil(frame), self._pil(sat)

        # Предобработка через процессор HuggingFace
        inputs = self.processor([im0, im1], return_tensors="pt").to(self.device)

        # Замер времени инференса
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        ms = (time.perf_counter() - t0) * 1000

        # Постобработка для получения читаемого формата
        # Передаем реальные размеры изображений, чтобы координаты были в оригинальном масштабе
        image_sizes = [[(frame.shape[0], frame.shape[1]),
                        (sat.shape[0], sat.shape[1])]]

        # Получаем keypoints и scores
        res = self.processor.post_process_keypoint_matching(
            outputs, image_sizes, threshold=self.threshold
        )[0]

        # Возвращаем результат в вашем стандартном формате
        return MatchResult(
            points_frame=res["keypoints0"].cpu().numpy(),
            points_map=res["keypoints1"].cpu().numpy(),
            scores=res["matching_scores"].cpu().numpy(),
            model_name=self.name,
            inference_ms=ms,
        )
def build_matcher(cfg: AppConfig) -> BaseMatcher:
    name = cfg.model.name
    if name not in MATCHER_REGISTRY:
        raise ValueError(f"Неизвестная модель: {name}. Доступно: {sorted(MATCHER_REGISTRY)}")
    return MATCHER_REGISTRY[name](cfg)


# ============================================================
# Геометрия
# ============================================================

class GeometryEstimator:
    def __init__(self, cfg: AppConfig):
        self.min_inliers = cfg.geometry.min_inliers
        self.ransac_thresh = cfg.geometry.ransac_thresh_px
        self.min_matches = cfg.model.min_matches

    def estimate(self, matches: MatchResult, frame_shape) -> PositionEstimate:
        now = time.time()
        if not matches.success or len(matches) < self.min_matches:
            return PositionEstimate(0, 0, now, 0, 0, False, "not_enough_matches")

        pts0 = matches.points_frame.astype(np.float32)
        pts1 = matches.points_map.astype(np.float32)
        H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, self.ransac_thresh)
        if H is None:
            return PositionEstimate(0, 0, now, 0, 0, False, "homography_failed")

        inliers = int(mask.sum())
        if inliers < self.min_inliers:
            return PositionEstimate(0, 0, now, 0, inliers, False, "not_enough_inliers")

        h, w = frame_shape[:2]
        center = np.array([[[w / 2.0, h / 2.0]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(center, H)[0, 0]
        return PositionEstimate(
            x=float(mapped[0]), y=float(mapped[1]), timestamp=now,
            confidence=inliers / len(pts0), num_inliers=inliers, valid=True,
        )


# ============================================================
# Фильтр ложных срабатываний
# ============================================================

class FalsePositiveFilter:
    """Два механизма: физическая проверка расстояния + голосование большинством."""

    def __init__(self, cfg: AppConfig):
        f = cfg.filter
        self.scale = f.map_scale_m_per_px
        self.max_speed = f.max_drone_speed_mps
        self.max_gap = f.max_time_gap_s
        self.window = f.vote_window
        self.min_agree = f.vote_min_agree
        self.radius_m = f.vote_radius_m
        self.history: deque = deque(maxlen=self.window)
        self.last_accepted: Optional[PositionEstimate] = None

    def _dist_m(self, x1, y1, x2, y2) -> float:
        return float(np.hypot(x1 - x2, y1 - y2) * self.scale)

    def update(self, est: PositionEstimate) -> FilteredPosition:
        if not est.valid:
            return FilteredPosition(0, 0, est.timestamp, "no_estimate", 0, reason=est.reason)

        if self.last_accepted is not None and \
                est.timestamp - self.last_accepted.timestamp > self.max_gap:
            self.history.clear()
            self.last_accepted = None

        if self.last_accepted is not None:
            dt = max(est.timestamp - self.last_accepted.timestamp, 1e-3)
            dist = self._dist_m(est.x, est.y, self.last_accepted.x, self.last_accepted.y)
            max_dist = self.max_speed * dt
            if dist > max_dist:
                return FilteredPosition(
                    est.x, est.y, est.timestamp, "rejected_distance", 0,
                    reason=f"moved {dist:.1f} m in {dt:.2f} s (limit {max_dist:.1f} m)",
                )

        self.history.append(est)
        agreed = [h for h in self.history
                  if self._dist_m(h.x, h.y, est.x, est.y) <= self.radius_m]
        votes = len(agreed)

        if votes < self.min_agree:
            return FilteredPosition(
                est.x, est.y, est.timestamp, "rejected_vote", votes,
                reason=f"only {votes}/{self.min_agree} agree in {self.radius_m} m",
            )

        x = float(np.mean([h.x for h in agreed]))
        y = float(np.mean([h.y for h in agreed]))
        spread_px = float(np.std([(h.x, h.y) for h in agreed], axis=0).mean())
        radius_m = max(spread_px * self.scale, self.radius_m / 2.0)
        self.last_accepted = est
        return FilteredPosition(x, y, est.timestamp, "accepted", votes, radius_m=radius_m)