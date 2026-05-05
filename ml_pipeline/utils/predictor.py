# ml_pipeline/utils/predictor.py

import numpy as np
import torch
import torch.nn as nn
import pickle
import os

# Определяем путь к корню проекта
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PropertyPriceNN(nn.Module):
    """Та же архитектура, что и при обучении"""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PropertyPricePredictor:
    """Класс для предсказания цен на недвижимость"""

    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = None
        self.is_loaded = False

    def load_model(self,
                   model_path=None,
                   scaler_X_path=None,
                   scaler_y_path=None):
        """Загружает модель и скейлеры"""

        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'ml_pipeline', 'models', 'property_model_best.pth')
        if scaler_X_path is None:
            scaler_X_path = os.path.join(PROJECT_ROOT, 'ml_pipeline', 'models', 'scaler_X.pkl')
        if scaler_y_path is None:
            scaler_y_path = os.path.join(PROJECT_ROOT, 'ml_pipeline', 'models', 'scaler_y.pkl')

        try:
            # Загружаем скейлеры
            with open(scaler_X_path, 'rb') as f:
                self.scaler_X = pickle.load(f)
            with open(scaler_y_path, 'rb') as f:
                self.scaler_y = pickle.load(f)

            # Определяем количество признаков
            input_dim = self.scaler_X.mean_.shape[0]

            # Создаем и загружаем модель
            self.model = PropertyPriceNN(input_dim=input_dim)
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()

            # Определяем имена признаков (если есть датасет, можно загрузить)
            self.feature_names = ['total_meters', 'living_meters', 'kitchen_meters',
                                  'rooms_count', 'floor', 'floors_count',
                                  'year_of_construction', 'author_type', 'district']

            self.is_loaded = True
            print("✓ Модель успешно загружена")
            return True

        except Exception as e:
            print(f"✗ Ошибка загрузки модели: {e}")
            self.is_loaded = False
            return False

    def predict(self, features_dict):
        """Предсказывает цену на основе словаря с признаками"""

        if not self.is_loaded:
            if not self.load_model():
                return None

        try:
            # Подготавливаем признаки в правильном порядке
            features = np.array([[features_dict[name] for name in self.feature_names]])

            # Масштабируем
            features_scaled = self.scaler_X.transform(features)
            features_tensor = torch.FloatTensor(features_scaled)

            # Предсказываем
            with torch.no_grad():
                pred_scaled = self.model(features_tensor)
                pred_log = self.scaler_y.inverse_transform(pred_scaled.numpy())
                pred_price = np.expm1(pred_log)

            # Округляем до тысяч
            return round(float(pred_price[0][0]) / 1000) * 1000

        except Exception as e:
            print(f"✗ Ошибка предсказания: {e}")
            return None


# Глобальный экземпляр (создается один раз при импорте)
_predictor_instance = None


def get_predictor():
    """Возвращает глобальный экземпляр предиктора (синглтон)"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = PropertyPricePredictor()
    return _predictor_instance