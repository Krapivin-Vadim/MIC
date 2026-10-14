import torch
import cv2
import csv
import time
import numpy as np
from lightglue import LightGlue, SuperPoint

# Настройка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка моделей
extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
matcher = LightGlue(features='superpoint').eval().to(device)

# Загрузка тестовых изображений
img0_path = 'assets/DSC_0410.JPG'
img1_path = 'assets/DSC_0411.JPG'


# Функция для подготовки картинки
def prepare_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)[None, None].to(device)
    return tensor, image.shape[:2]


tensor0, shape0 = prepare_image(img0_path)
tensor1, shape1 = prepare_image(img1_path)

results = []


# ТЕСТ 1: Влияние разрешения
print("\nТЕСТ 1: Влияние разрешения")
print(f"{'Размер (ширина)':<20} | {'Время (мс)':<15} | {'Совпадений':<15}")

for target_width in [512, 800, 1024]:
    h0, w0 = shape0
    new_h0 = int(h0 * (target_width / w0))

    h1, w1 = shape1
    new_h1 = int(h1 * (target_width / w1))

    img0_resized = cv2.resize(tensor0[0, 0].cpu().numpy(), (target_width, new_h0))
    img1_resized = cv2.resize(tensor1[0, 0].cpu().numpy(), (target_width, new_h1))

    t0 = torch.from_numpy(img0_resized)[None, None].to(device)
    t1 = torch.from_numpy(img1_resized)[None, None].to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        feats0 = extractor.extract(t0)
        feats1 = extractor.extract(t1)
        matches = matcher({'image0': feats0, 'image1': feats1})
    time_ms = (time.perf_counter() - start_time) * 1000

    num_matches = len(matches['matches'][0]) if len(matches['matches']) > 0 else 0

    print(f"{target_width}x{new_h0:<14} | {time_ms:<15.2f} | {num_matches:<15}")
    results.append({
        'Тест': 'Разрешение',
        'Параметр': target_width,
        'Время (мс)': time_ms,
        'Совпадений': num_matches
    })


# ТЕСТ 2: Устойчивость к повороту
print("\nТЕСТ 2: Устойчивость к повороту")
print(f"{'Поворот картинки 2':<20} | {'Время (мс)':<15} | {'Совпадений':<15}")

for angle in [0, 45, 90]:
    img1_np = tensor1[0, 0].cpu().numpy()
    h, w = img1_np.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img1_rotated = cv2.warpAffine(img1_np, matrix, (w, h))
    t1_rot = torch.from_numpy(img1_rotated)[None, None].to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        feats0 = extractor.extract(tensor0)
        feats1 = extractor.extract(t1_rot)
        matches = matcher({'image0': feats0, 'image1': feats1})
    time_ms = (time.perf_counter() - start_time) * 1000

    num_matches = len(matches['matches'][0]) if len(matches['matches']) > 0 else 0

    print(f"{angle} градусов{' ':<9} | {time_ms:<15.2f} | {num_matches:<15}")
    results.append({
        'Тест': 'Поворот',
        'Параметр': f"{angle} deg",
        'Время (мс)': time_ms,
        'Совпадений': num_matches
    })


# ТЕСТ 3: Влияние количества искомых точек
print("\nТЕСТ 3: Влияние количества искомых точек")
print(f"{'Макс. точек':<20} | {'Время (мс)':<15} | {'Совпадений':<15}")

# Экстрактор
extractor_test = SuperPoint().eval().to(device)

for max_kp in [512, 1024, 2048]:
    extractor_test.max_num_keypoints = max_kp

    start_time = time.perf_counter()
    with torch.no_grad():
        feats0 = extractor_test.extract(tensor0)
        feats1 = extractor_test.extract(tensor1)
        matches = matcher({'image0': feats0, 'image1': feats1})
    time_ms = (time.perf_counter() - start_time) * 1000

    num_matches = len(matches['matches'][0]) if len(matches['matches']) > 0 else 0

    print(f"{max_kp}{' ':<15} | {time_ms:<15.2f} | {num_matches:<15}")
    results.append({
        'Тест': 'Кол-во точек',
        'Параметр': max_kp,
        'Время (мс)': time_ms,
        'Совпадений': num_matches
    })

# Сохранение в CSV
csv_filename = 'benchmark_results.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['Тест', 'Параметр', 'Время (мс)', 'Совпадений'])
    writer.writeheader()
    writer.writerows(results)
