import json
import cv2
import time
import torch
import csv
from superpoint_lightglue_corrector import SuperPointLightGlueCorrector

# Загружаем данные
with open('my_data/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Инициализация модель
corrector = SuperPointLightGlueCorrector(max_keypoints=1024)

print(f"Устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Пар кадров: {len(metadata['pairs'])}\n")

results = []

# Прогон всех пар кадров
for i, pair in enumerate(metadata['pairs']):
    print(f"Пара #{i + 1}")

    # Загрузка изображения
    frame_drone = cv2.imread(pair['drone_frame'])
    frame_satellite = cv2.imread(pair['satellite_frame'])

    if frame_drone is None or frame_satellite is None:
        print("Ошибка загрузки изображений")
        continue

    # Замер времени
    times = []
    matches_list = []
    scores_list = []

    for _ in range(5):
        result = corrector.correct(frame_satellite, frame_drone)
        times.append(result.processing_time_ms)
        matches_list.append(result.num_matches)
        scores_list.append(float(result.scores.mean()) if len(result.scores) > 0 else 0.0)

    # Средние значения
    avg_time = sum(times) / len(times)
    avg_matches = sum(matches_list) / len(matches_list)
    avg_score = sum(scores_list) / len(scores_list)
    fps = 1000 / avg_time if avg_time > 0 else 0

    print(f"  Время: {avg_time:.2f} мс")
    print(f"  Совпадений: {avg_matches:.1f}")
    print(f"  Score: {avg_score:.3f}")
    print(f"  FPS: {fps:.1f}\n")

    # Результаты
    results.append({
        'pair_index': i,
        'avg_time_ms': round(avg_time, 2),
        'avg_matches': round(avg_matches, 1),
        'avg_score': round(avg_score, 3),
        'fps': round(fps, 1)
    })

# Статистика

print("Статистика")


if results:
    overall_avg_time = sum(r['avg_time_ms'] for r in results) / len(results)
    overall_fps = 1000 / overall_avg_time if overall_avg_time > 0 else 0
    overall_matches = sum(r['avg_matches'] for r in results) / len(results)

    print(f"Среднее время: {overall_avg_time:.2f} мс")
    print(f"Средний FPS: {overall_fps:.1f}")
    print(f"Среднее совпадений: {overall_matches:.1f}")

# Сохранение в CSV
with open('jetson_benchmark_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['pair_index', 'avg_time_ms', 'avg_matches', 'avg_score', 'fps'])
    writer.writeheader()
    writer.writerows(results)