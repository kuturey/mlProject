import numpy as np
import torch
import torch.nn as nn
import os
from django.conf import settings


class PropertyNN(nn.Module):
    def __init__(self, input_dim=6):
        super(PropertyNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


class PropertyPricePredictor:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(
            settings.BASE_DIR,
            'property_evaluation',
            'ml_models',
            'model.pth'
        )
        # Порядок признаков (должен совпадать с обучением)
        self.features_names = [
            'total_meters',
            'kitchen_meters',
            'rooms_count',
            'floor',
            'floors_count',
            'building_age'
        ]

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"❌ Модель не найдена: {self.model_path}")
            return False

        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')

            # Пытаемся определить input_dim
            if 'input_dim' in checkpoint:
                input_dim = checkpoint['input_dim']
            else:
                input_dim = len(self.features_names)

            self.model = PropertyNN(input_dim)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("✅ Модель загружена (без scaler)")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, features_dict):
        if self.model is None:
            if not self.load_model():
                return None

        try:
            # Собираем признаки в правильном порядке
            features = []
            for fname in self.features_names:
                val = float(features_dict.get(fname, 0.0))
                features.append(val)

            # Преобразуем в тензор
            X_tensor = torch.tensor([features], dtype=torch.float32)

            # Предсказываем
            with torch.no_grad():
                prediction = self.model(X_tensor).numpy()[0][0]

            # Если модель обучалась на логарифме — раскомментируй:
            # prediction = np.expm1(prediction)

            # Округляем до тысяч
            return max(0, round(prediction, -3))

        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")
            import traceback
            traceback.print_exc()
            return None


# Глобальный экземпляр
predictor = PropertyPricePredictor()
