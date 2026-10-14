import cv2
import numpy as np
from superpoint_lightglue_corrector import SuperPointLightGlueCorrector


def visualize_matches(frame_drone_path: str, frame_satellite_path: str, output_path: str):
    #Визуализация совпадений между двумя изображениями
    # Загрузка изображений
    frame_drone = cv2.imread(frame_drone_path)
    frame_satellite = cv2.imread(frame_satellite_path)

    if frame_drone is None or frame_satellite is None:
        print("Ошибка загрузки изображений")
        return

    # Соответствия
    corrector = SuperPointLightGlueCorrector(max_keypoints=1024)
    result = corrector.correct(frame_satellite, frame_drone)

    # Линии между совпавшими точками
    # Объединённое изображение
    h1, w1 = frame_satellite.shape[:2]
    h2, w2 = frame_drone.shape[:2]
    combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined[:h1, :w1] = frame_satellite
    combined[:h2, w1:w1 + w2] = frame_drone

    # Линии
    for i in range(min(50, result.num_matches)):
        pt1 = (int(result.points_map[i][0]), int(result.points_map[i][1]))
        pt2 = (int(result.points_frame[i][0] + w1), int(result.points_frame[i][1]))

        # Цвет зависит от score (зелёный = высокий, красный = низкий)
        score = result.scores[i]
        color = (0, int(255 * score), int(255 * (1 - score)))
        thickness = 1

        cv2.line(combined, pt1, pt2, color, thickness)
        cv2.circle(combined, pt1, 3, color, -1)
        cv2.circle(combined, pt2, 3, color, -1)

    # Сохранение результата
    cv2.imwrite(output_path, combined)
    print(f"   Найдено совпадений: {result.num_matches}")


if __name__ == "__main__":
    # Пример использования
    visualize_matches(
        'my_data/drone/frame_192s_mapscale.png',
        'my_data/satellite/search_192s.tif',
        'matches_visualization.png'
    )