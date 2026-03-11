import numpy as np
import torch
import torch.nn as nn
import joblib
import os
from django.conf import settings


class PropertyPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.features_names = None
        self.model_path = os.path.join(settings.BASE_DIR, 'property_evaluation', 'ml_models', 'model.pkl')

    def load_model(self):
        """Загружает модель из файла"""
        if not os.path.exists(self.model_path):
            print(f"❌ Модель не найдена по пути: {self.model_path}")
            return False

        try:
            # Загружаем данные модели
            model_data = joblib.load(self.model_path)

            # Восстанавливаем архитектуру модели
            self.model = nn.Linear(model_data['input_dim'], 1)
            self.model.load_state_dict(model_data['model_state'])
            self.model.eval()

            # Загружаем скейлеры
            self.scaler_X = model_data['scaler_X']
            self.scaler_y = model_data['scaler_y']
            self.features_names = model_data['features_names']

            print("✅ Модель успешно загружена")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    def predict(self, features_dict):
        """
        Предсказание цены по словарю признаков
        """
        if self.model is None:
            if not self.load_model():
                return None

        try:
            # Подготавливаем признаки в правильном порядке
            features = []
            missing_features = []

            print("\n🔍 ДАННЫЕ ДЛЯ МОДЕЛИ:")
            for feature in self.features_names:
                if feature in features_dict:
                    value = float(features_dict[feature])
                    features.append(value)
                    print(f"  {feature}: {value}")
                else:
                    missing_features.append(feature)
                    features.append(0.0)
                    print(f"  {feature}: 0 (НЕТ ДАННЫХ!)")

            if missing_features:
                print(f"⚠️ Отсутствуют признаки: {missing_features}")

            print(f"📊 Итоговый вектор признаков: {features}")

            # Масштабируем
            features_scaled = self.scaler_X.transform([features])
            print(f"📈 После масштабирования: {features_scaled[0]}")

            # Конвертируем в тензор и предсказываем
            with torch.no_grad():
                X_tensor = torch.tensor(features_scaled, dtype=torch.float32)
                y_pred_scaled = self.model(X_tensor).numpy()
                print(f"🤖 Предсказание (масштабированное): {y_pred_scaled[0][0]}")

            # Обратное преобразование
            y_pred_log = self.scaler_y.inverse_transform(y_pred_scaled)
            print(f"📉 После обратного масштабирования (log): {y_pred_log[0][0]}")

            y_pred_rub = float(np.expm1(y_pred_log).flatten()[0])
            print(f"💰 ИТОГОВАЯ ЦЕНА: {y_pred_rub:.2f} руб.")

            return max(0, round(y_pred_rub, -3))  # Округляем до тысяч
        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")
            import traceback
            traceback.print_exc()
            return None


# Создаем глобальный экземпляр
predictor = PropertyPricePredictor()