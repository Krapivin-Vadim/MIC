import json
import cv2
from superpoint_lightglue_corrector import SuperPointLightGlueCorrector, MatchResult


class DroneSimulation:

    def __init__(self, metadata_path: str, max_keypoints: int = 1024):

        # Загрузка данных
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        # Инициализация модели
        self.corrector = SuperPointLightGlueCorrector(max_keypoints=max_keypoints)

    def run_simulation(self):
        # Прогон всех пар кадров через модель
        results = []

        for i, pair in enumerate(self.metadata['pairs']):
            print(f"\n Кадр #{i + 1}")
            print(f"  Дрон: {pair['drone_frame']}")
            print(f"  Спутник: {pair['satellite_frame']}")

            # Загрузка изображения
            frame_drone = cv2.imread(pair['drone_frame'])
            frame_satellite = cv2.imread(pair['satellite_frame'])

            if frame_drone is None or frame_satellite is None:
                print(f"  Ошибка загрузки изображений!")
                continue

            # Прогон через модель
            result = self.corrector.correct(frame_satellite, frame_drone)

            # Результаты
            print(f"  Найдено совпадений: {result.num_matches}")
            print(f"  Время обработки: {result.processing_time_ms:.2f} мс")
            print(f"  Средний score: {result.scores.mean():.3f}" if len(result.scores) > 0 else "   Score: N/A")

            results.append({
                'frame_index': i,
                'num_matches': result.num_matches,
                'processing_time_ms': result.processing_time_ms,
                'avg_score': float(result.scores.mean()) if len(result.scores) > 0 else 0.0,
                'drone_location': pair['drone_location']
            })

        # Итоговая статистика
        print("\n Итоговая статистика:")

        total_matches = sum(r['num_matches'] for r in results)
        avg_time = sum(r['processing_time_ms'] for r in results) / len(results) if results else 0
        avg_score = sum(r['avg_score'] for r in results) / len(results) if results else 0

        print(f"  Всего кадров: {len(results)}")
        print(f"  Всего совпадений: {total_matches}")
        print(f"  Среднее время: {avg_time:.2f} мс")
        print(f"  Средний score: {avg_score:.3f}")
        print(f"  FPS: {1000 / avg_time:.1f}" if avg_time > 0 else "  FPS: N/A")

        # Сохранение результатов в JSON
        output_path = 'simulation_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results

if __name__ == "__main__":
    # Путь к данным
    metadata_path = 'my_data/metadata.json'

    # Создание симуляции
    simulation = DroneSimulation(metadata_path, max_keypoints=512)

    # Запуск
    simulation.run_simulation()


