"""Точка входа: HTTP-сервис коррекции координат дрона.

Как это работает в целом:
1. При старте сервера (lifespan) создаются все "рабочие" объекты:
   камера, селектор GeoTIFF-снимков, нейросеть, связь с PixHawk.
2. Они собираются в один AICorrectionManager, который умеет делать
   полный цикл коррекции.
3. Когда кто-то делает запрос GET /correct, сервер вызывает этот цикл
   и возвращает исправленные координаты в формате JSON.
"""

import logging
import threading
from contextlib import asynccontextmanager

import rospy  # библиотека для работы с ROS (через неё мы говорим с PixHawk)
import uvicorn  # ASGI-сервер, который реально принимает HTTP-запросы
from fastapi import FastAPI, HTTPException, Request  # веб-фреймворк

from settings import settings  # настройки из файла .env

from interface.AICorrectionManager import AICorrectionManager
from realizations.CameraImpl import CameraImpl
from realizations.GeoTiffFrameSelector import GeoTiffFrameSelector
from realizations.MavrosManagerImpl import MavrosManagerImpl
from superpoint_lightglue.superpoint_lightglue_corrector import (
    SuperPointLightGlueCorrector,
)

# Логгер нужен, чтобы писать сообщения в консоль (logger.info, logger.exception).
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Код "жизненного цикла" приложения.

    Всё, что написано ДО слова yield, выполняется один раз при запуске сервера.
    Всё, что ПОСЛЕ yield (в блоке finally), выполняется при остановке сервера.
    """

    # Регистрируем наш процесс как "ноду" (участника) в системе ROS.
    # Без этого Publisher и wait_for_message из MavrosManagerImpl не заработают.
    # disable_signals=True: не даём rospy перехватывать Ctrl+C, иначе он
    # будет конфликтовать с uvicorn при остановке сервера.
    rospy.init_node("coordinate_corrector", anonymous=True, disable_signals=True)

    # Объект, который по координатам дрона находит нужный GeoTIFF-снимок
    # и вырезает из него кусок карты под дроном.
    frame_selector = GeoTiffFrameSelector(
        settings.frame_base_dir,  # папка со снимками .tif
        settings.frame_source_crs,  # система координат GPS дрона (например EPSG:4326)
        settings.frame_target_crs,  # система координат для расчётов (например EPSG:3857)
        settings.frame_fov_v,  # вертикальный угол обзора камеры
        settings.frame_fov_h,  # горизонтальный угол обзора камеры
        settings.frame_crop_margin,  # запас при вырезании кропа
    )

    # Нейросеть: находит одинаковые точки на кадре дрона и на спутниковой карте.
    ai_corrector = SuperPointLightGlueCorrector(
        settings.ai_max_keypoints,  # сколько ключевых точек искать максимум
        settings.ai_device,  # "cuda" (видеокарта) или "cpu"
    )

    # Связь с полётным контроллером PixHawk: читает и отправляет координаты.
    mavros_manager = MavrosManagerImpl(
        settings.ros_source_topic,  # откуда читаем координаты
        settings.ros_target_topic,  # куда отправляем исправленные
        settings.ros_timeout,  # сколько секунд ждать ответа
    )

    # Камера. Поток захвата кадров запускается прямо в конструкторе,
    # поэтому отдельно вызывать __enter__() не нужно.
    camera = CameraImpl(
        settings.cam_id,
        settings.cam_timeout,
        settings.cam_frame_size,
        settings.cam_fps,
    )

    # Менеджер объединяет все четыре части в один рабочий процесс коррекции.
    correction_manager = AICorrectionManager(
        camera,
        frame_selector,
        ai_corrector,
        mavros_manager,
    )

    # app.state — "общая полка" приложения. Всё, что мы туда положим,
    # будет доступно из любой ручки через request.app.state.
    app.state.camera = camera
    app.state.correction_manager = correction_manager

    # Замок, который не даёт запустить две коррекции одновременно.
    # (Камера, GPU и ROS не рассчитаны на параллельное использование.)
    app.state.correction_lock = threading.Lock()

    logger.info("Correction service started")
    try:
        yield  # <-- здесь сервер работает и принимает запросы
    finally:
        # Сюда попадаем при остановке: освобождаем камеру.
        camera.close()
        logger.info("Correction service stopped")


# Создаём приложение и подключаем к нему lifespan.
app = FastAPI(title="Coordinate correction service", lifespan=lifespan)


@app.get("/correct")
def correct(request: Request):
    """Ручка GET /correct: один раз корректирует координаты дрона.

    Возвращает JSON с итоговыми координатами (широта, долгота, высота).
    """
    lock: threading.Lock = request.app.state.correction_lock

    # Пытаемся занять замок, НЕ дожидаясь. Если он занят, значит другая
    # коррекция ещё идёт, и мы сразу отвечаем кодом 409 (Conflict).
    if not lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Correction is already in progress",
        )

    try:
        # Главный вызов: камера -> GPS -> карта -> нейросеть -> координаты.
        location = request.app.state.correction_manager.correct_position()
    except TimeoutError as exc:
        # PixHawk не прислал координаты вовремя.
        raise HTTPException(status_code=504, detail=str(exc))
    except RuntimeError as exc:
        # Нет кадра с камеры или нет подходящего снимка карты под дроном.
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        # Не удалось вычислить гомографию (мало совпавших точек).
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        # Любая непредвиденная ошибка: пишем полный текст в лог и отвечаем 500.
        logger.exception("Correction failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # finally выполняется всегда: и при успехе, и при ошибке.
        # Обязательно освобождаем замок, иначе сервис "зависнет" на 409.
        lock.release()

    # FastAPI сам превратит словарь в JSON.
    return {
        "status": "completed",
        "latitude": location.latitude,
        "longitude": location.longitude,
        "altitude": location.altitude,
    }


@app.get("/health")
def health():
    """Простая проверка, что сервер жив (удобно для мониторинга)."""
    return {"status": "ok"}


# Этот блок выполняется только при запуске командой `python main.py`.
if __name__ == "__main__":
    # host="0.0.0.0" — принимать запросы с любых адресов сети, не только с localhost.
    # workers=1 — обязательно один процесс: одна камера, одна ROS-нода, одна модель.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)