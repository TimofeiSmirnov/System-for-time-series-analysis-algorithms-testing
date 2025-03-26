import pandas as pd
import os
from typing import Dict
from app.data_controller.data_loader import TimeSeriesGenerator


class DataController:
    def __init__(self):
        self.generator = None
        self.storage_path = "app/data"
        self.series_metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, int]]:
        """Загружает метаданные о хранящихся временных рядах во всех подпапках."""
        metadata = {}
        for root, _, files in os.walk(self.storage_path):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    # Читаем CSV-файл
                    data = pd.read_csv(file_path)
                    # Добавляем метаданные о файле
                    metadata[file] = {
                        "length": data.shape[0],  # Количество строк
                        "dimensions": data.shape[1],  # Количество столбцов
                        "path": file_path,  # Абсолютный путь к файлу
                        "category": os.path.relpath(root, self.storage_path)  # Путь от каталога данных
                    }
        return metadata

    def load_user_series_from_data(self, file_path: str):
        """Загружает пользовательский временной ряд из CSV и проверяет его."""
        try:
            loaded_time_series = pd.read_csv(file_path)
            timestamps = loaded_time_series["timestamp"]
            time_series = loaded_time_series["value"]
            if loaded_time_series.isnull().values.any():
                raise ValueError("Загруженный ряд содержит пропущенные значения или NaN.")
            return timestamps, time_series
        except Exception as e:
            print(f"Ошибка при загрузке ряда: {e}")
            return None

    def get_available_series(self):
        """Возвращает список доступных временных рядов с их характеристиками."""
        return self.series_metadata

    def generate_series(self, time_series_length, time_series_dim):
        self.generator = TimeSeriesGenerator(time_series_length, time_series_dim)
        return self.generator.generate_time_series()

    def get_series(self, series_name: str):
        """Возвращает конкретный временной ряд по имени, если он есть в хранилище."""
        metadata = self.series_metadata.get(series_name)
        loaded_time_series = pd.read_csv(metadata["path"])
        timestamps = loaded_time_series.iloc[:, 0]
        df_filtered = loaded_time_series.drop(loaded_time_series.columns[0], axis=1)
        time_series = [df_filtered[col].map(float).tolist() for col in df_filtered.columns]
        return timestamps, time_series
