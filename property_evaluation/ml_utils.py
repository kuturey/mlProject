import numpy as np
import torch
import pickle
import os
from django.conf import settings


class PropertyPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.features_names = [
            'total_meters',
            'kitchen_meters',
            'rooms_count',
            'floor',
            'floors_count',
            'building_age'
        ]
        self.model_path = os.path.join(
            settings.BASE_DIR,
            'property_evaluation',
            'ml_models',
            'model.pth'
        )
        self.scaler_X_path = os.path.join(
            settings.BASE_DIR,
            'property_evaluation',
            'ml_models',
            'scaler_X.pkl'
        )
        self.scaler_y_path = os.path.join(
            settings.BASE_DIR,
            'property_evaluation',
            'ml_models',
            'scaler_y.pkl'
        )

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Модель не найдена: {self.model_path}")
            return False

        if not os.path.exists(self.scaler_X_path):
            print(f"Scaler X не найден: {self.scaler_X_path}")
            return False

        if not os.path.exists(self.scaler_y_path):
            print(f"Scaler Y не найден: {self.scaler_y_path}")
            return False

        try:
            with open(self.scaler_X_path, 'rb') as f:
                self.scaler_X = pickle.load(f)
            with open(self.scaler_y_path, 'rb') as f:
                self.scaler_y = pickle.load(f)

            state_dict = torch.load(self.model_path, map_location='cpu')
            from .train_model import PropertyPriceNN
            self.model = PropertyPriceNN(input_dim=len(self.features_names))
            self.model.load_state_dict(state_dict)
            self.model.eval()

            print("Модель и скейлеры успешно загружены")
            return True
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            return False

    def predict(self, features_dict):
        if self.model is None:
            if not self.load_model():
                return None

        try:
            features = []
            for fname in self.features_names:
                val = float(features_dict.get(fname, 0.0))
                features.append(val)

            features_scaled = self.scaler_X.transform([features])
            X_tensor = torch.FloatTensor(features_scaled)

            with torch.no_grad():
                y_pred_scaled = self.model(X_tensor).numpy()[0][0]

            y_pred_log = self.scaler_y.inverse_transform([[y_pred_scaled]])
            y_pred_rub = float(np.expm1(y_pred_log)[0][0])

            return max(0, round(y_pred_rub, -3))

        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return None


predictor = PropertyPricePredictor()