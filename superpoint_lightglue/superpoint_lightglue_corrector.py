import torch
import numpy as np
import time
from dto.MatchResult import MatchResult
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image


class SuperPointLightGlueCorrector:

    def __init__(self, max_keypoints: int = 1024, device: str = None):

        # Инициализация модели
        # Выбор устройства
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Загрузка модели
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

    def correct(self, frame_from_storage: np.ndarray, frame_from_drone: np.ndarray) -> MatchResult:

        # Сопоставление снимка с дрона и снимка со спутника

        # Время
        start_time = time.perf_counter()

        # Подготовка изображений
        if len(frame_from_drone.shape) == 3:
            frame_from_drone = np.mean(frame_from_drone, axis=2)
        if len(frame_from_storage.shape) == 3:
            frame_from_storage = np.mean(frame_from_storage, axis=2)

        # Нормализация к диапазону [0, 1]
        img0 = frame_from_drone.astype(np.float32) / 255.0
        img1 = frame_from_storage.astype(np.float32) / 255.0

        # Конвертация в тензоры PyTorch
        tensor0 = torch.from_numpy(img0)[None, None].to(self.device)
        tensor1 = torch.from_numpy(img1)[None, None].to(self.device)

        # Извлечение признаков и сопоставление
        with torch.no_grad():  # Отключаем градиенты для ускорения
            feats0 = self.extractor.extract(tensor0)
            feats1 = self.extractor.extract(tensor1)
            matches = self.matcher({'image0': feats0, 'image1': feats1})

        # Извлечение результатов
        matches01 = matches['matches']

        if len(matches01) > 0 and len(matches01[0]) > 0:
            # Индексы совпавших точек
            indices = matches01[0].cpu().numpy()

            # Координаты точек из признаков
            keypoints0 = feats0['keypoints'][0].cpu().numpy()  # Все точки на кадре дрона
            keypoints1 = feats1['keypoints'][0].cpu().numpy()  # Все точки на карте


            points_frame = keypoints0[indices[:, 0]]  # Точки на кадре дрона
            points_map = keypoints1[indices[:, 1]]  # Соответствующие точки на карте

            # Оценки совпадений
            if 'scores' in matches:
                scores = matches['scores'][0].cpu().numpy()
            else:
                # Если scores нет, создаются единичные оценки
                scores = np.ones(len(indices))
        else:
            # Совпадений не найдено
            points_frame = np.empty((0, 2))
            points_map = np.empty((0, 2))
            scores = np.empty(0)

        # Время после
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        # Результат
        result = MatchResult(
            points_frame=points_frame,
            points_map=points_map,
            scores=scores,
            processing_time_ms=processing_time_ms,
            num_matches=len(points_frame)
        )

        return result

