
Данный репозиторий содержит программную реализацию модуля
визуального сопоставления кадров БПЛА со спутниковыми снимками
на базе связки нейросетей SuperPoint + LightGlue

## Структура проекта

superpoint_lightglue_corrector.py
Содержит класс 'SuperPointLightGlueCorrector', готовый к встраиванию в систему дрона

drone_simulation.py
Скрипт для тестирования модели на наборе данных (читает 'metadata.json')

jetson_benchmark.py
Скрипт для замера производительности: FPS, время инференса, потребление RAM на Jetson Orin NX

benchmark_stage2.py
Скрипт для исследования влияния параметров
(разрешение, поворот, кол-во точек) на качество работы

visualize_matches.py
Вспомогательный скрипт для визуализации найденных соответствий

my_data/ - Папка с тестовыми данными ('metadata.json')



## Интеграция
Для использования в общей системе дрона достаточно импортировать класс из главного модуля:
import cv2
from superpoint_lightglue_corrector import SuperPointLightGlueCorrector

## 1. Инициализация корректора (выполняется один раз при старте системы)
corrector = SuperPointLightGlueCorrector(max_keypoints=1024, device='cuda')

## 2. Загрузка изображений (np.ndarray)
frame_drone = cv2.imread('drone_frame.png')
frame_satellite = cv2.imread('satellite_map.tif')

## 3. Получение результатов сопоставления
result = corrector.correct(frame_satellite, frame_drone)

print(f"Найдено совпадений: {result.num_matches}")
print(f"Время обработки: {result.processing_time_ms:.2f} мс")

 result.points_frame, result.points_map, result.scores содержат координаты и оценки

## Запуск тестов
1. Исследование параметров:
   benchmark_stage2.py
   Результат сохраняется в benchmark_results.csv

2. Симуляция полета на наборе данных:
   drone_simulation.py
   Результат сохраняется в simulation_results.json

3. Замер производительности (FPS, память):
   jetson_benchmark.py
   Результат сохраняется в jetson_benchmark_results.csv