# Drone Localizer — Локализация БПЛА по спутниковой карте

## О проекте
Drone Localizer — это исследовательский проект, представляющий собой пайплайн для определения координат беспилотного летательного аппарата (БПЛА) по его кадру и спутниковой карте местности. Система сопоставляет снимок с борта дрона с фрагментом аэрофотоснимка (GeoTIFF, JPG или MBTiles) и вычисляет положение центра кадра на карте.

Цель проекта — решение задачи визуальной локализации в условиях отсутствия или нестабильности GPS-сигнала. В основе лежит полу-плотное сопоставление признаков (dense matching) с помощью моделей семейства LoFTR, робастное вычисление гомографии и многоуровневая фильтрация ложных срабатываний. Проект реализован в виде набора Python-модулей с CLI-интерфейсом.

## Возможности
Сопоставление снимков — поддержка трёх архитектур матчеров: LoFTR (Kornia), Coarse LoFTR (TensorRT-совместимая реализация) и EfficientLoFTR (HuggingFace Transformers).
Оценка геометрии — вычисление матрицы гомографии методом RANSAC с контролем числа инлайеров.
Фильтрация ложных срабатываний — проверка физической правдоподобности перемещения (ограничение скорости) и голосование большинством в скользящем окне.
Три картографических провайдера — растровые JPG-карты, GeoTIFF с геопривязкой (перевод пикселей в lat/lon) и заготовка для MBTiles.
Режимы поиска — точная локализация в окрестности известного приора и «слепой» поиск скользящим окном по всей карте с учётом разницы масштабов.
Визуализация — отрисовка найденных соответствий и итоговой позиции с рамкой предполагаемой области кадра.
Пакетная обработка — последовательная обработка директории кадров.

## Архитектура
Пайплайн построен по модульному принципу и разделён на независимые слои:
- core.py — ядро системы: dataclass-типы (MatchResult, PositionEstimate, FilteredPosition), загрузка YAML-конфигурации, реестр матчеров, оценщик геометрии и фильтр ложных срабатываний.
- maps.py — абстрактный MapProvider и реализации для разных форматов карт.
- pipeline.py — оркестратор DroneLocalizationPipeline, связывающий матчер, геометрию и фильтр; CLI-интерфейс и утилиты визуализации.

Матчеры регистрируются через декоратор register(name) в реестре MATCHER_REGISTRY, что позволяет добавлять новые модели без изменения остального кода. Выбор модели задаётся в config.yaml.

## Логика локализации:
Кадр дрона и фрагмент карты попадают в Matcher, который выдаёт MatchResult. Затем GeometryEstimator вычисляет PositionEstimate, а FalsePositiveFilter выдаёт итоговый FilteredPosition.
Для «слепого» поиска кадр сопоставляется с каждым тайлом карты, а локальные координаты пересчитываются в глобальные; выбирается кандидат с максимальным числом инлайеров.

## Технологии
1) Матчинг признаков: LoFTR (Kornia), Coarse LoFTR, EfficientLoFTR (Transformers).
2) Глубокое обучение: PyTorch, torchvision.
3) Геометрия и обработка изображений: OpenCV.
4) Работа с картами: rasterio (GeoTIFF), Pillow, SQLite (MBTiles).
5) Численные вычисления: NumPy.
6) Конфигурация: PyYAML.
7) Язык: Python 3.11+.

## Установка

Клонируйте репозиторий:
git clone <URL_репозитория>
cd <директория_проекта>

Создайте виртуальное окружение и активируйте его:
python -m venv venv
Для Windows: venv\Scripts\activate
Для Linux/macOS: source venv/bin/activate

Установите зависимости:
pip install -r req.txt

Примечание: для работы coarse_loftr требуется локальный репозиторий coarse_loftr_trt с весами LoFTR_teacher.pt. Модель loftr автоматически скачивает предобученные веса через Kornia, а efficient_loftr — через HuggingFace Hub (требуется доступ в интернет при первом запуске). Для GeoTIFF-режима необходим пакет rasterio.

## Конфигурация
Все параметры задаются в config.yaml.

Секция model:
- name — Архитектура матчера: loftr, coarse_loftr, efficient_loftr.
- conf_threshold — Порог уверенности для отбора соответствий.
- min_matches — Минимальное число совпадений для запуска оценки.
- max_size — Максимальный размер стороны изображения при препроцессинге.

Секция geometry:
min_inliers — Минимальное число инлайеров гомографии.
ransac_thresh_px — Порог RANSAC в пикселях.

Секция filter:
- map_scale_m_per_px — Масштаб карты, м/px.
- max_drone_speed_mps — Максимальная допустимая скорость дрона.
- max_time_gap_s — Максимальный разрыв между кадрами до сброса истории.
- vote_window — Размер окна голосования.
- vote_min_agree — Минимум согласных оценок для принятия позиции.
- vote_radius_m — Радиус согласия при голосовании, м.

Секция map:
type — Тип карты: jpg, geotiff, mbtiles.
path — Путь к файлу карты.
zoom — Уровень зума (для тайловых карт).

Секция io:
output_dir — Директория для результатов.

## Использование
Одиночная пара «кадр + фрагмент карты»:
python pipeline.py --map data/map/fragment.jpg --frame data/frames/frame_001.jpg
Выводит статистику по совпадениям, инлайерам и статусу фильтрации, сохраняет визуализацию matches.jpg.

Локализация на GeoTIFF с приором позиции:
python pipeline.py --config config.yaml --map data/map/alferyevo_bing_700m_utm36n.tif --frame data/frames/frame_001.jpg --geotiff --prior 5000 5000 --radius 400
Выполняет поиск в окне radius вокруг пиксельной координаты приора и дополнительно выводит широту/долготу найденной точки.

Слепой поиск скользящим окном:
python pipeline.py --map data/map/map.tif --frame data/frames/frame_001.jpg --sliding --frame-m-per-px 0.15 --map-m-per-px 0.5 --stride-ratio 0.5 --min-inlier-ratio 0.5
Перебирает тайлы по всей карте, размер окна автоматически подстраивается под соотношение масштабов кадра и карты. Выводит топ-5 кандидатов и сохраняет localization.jpg с найденной позицией.

Обработка последовательности кадров:
python pipeline.py --map data/map/fragment.jpg --frames data/frames/ --fps 1.0
Прогоняет все *.jpg / *.png из директории через пайплайн с общей историей фильтра.

## Структура проекта
- config.yaml — Конфигурация пайплайна.
- core.py — Типы, конфиг, матчеры, геометрия, фильтр.
- maps.py — Провайдеры карт (JPG, GeoTIFF, MBTiles).
- pipeline.py — Оркестратор, визуализация и CLI.
- req.txt — Зависимости Python.
- pyvenv.cfg — Параметры виртуального окружения.
- data/map/ — Карты (GeoTIFF, JPG).
- data/frames/ — Кадры дрона.
- output/ — Результаты работы (визуализации).

## Основные типы данных
- MatchResult — Точки на кадре и карте, уверенности, имя модели, время инференса.
- PositionEstimate — Оценка позиции в пикселях карты, уверенность, число инлайеров, флаг валидности.
- FilteredPosition — Итоговая позиция после фильтрации со статусом (accepted, rejected_vote, rejected_distance, no_estimate) и радиусом неопределённости.

## Источники
- LoFTR: Detector-Free Local Matcher with Transformer — https://github.com/zju3dv/LoFTR
- Kornia: Computer Vision Library for PyTorch — https://kornia.readthedocs.io
- EfficientLoFTR (HuggingFace) — https://huggingface.co/zju-community/efficientloftr
- OpenCV: Homography estimation and RANSAC — https://docs.opencv.org
- rasterio: Geospatial raster I/O — https://rasterio.readthedocs.io
