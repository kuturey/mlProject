import os
import json
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from django.conf import settings
from catboost import CatBoostRegressor


class PropertyPricePredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.load_success = False

        self.model_path = os.path.join(settings.BASE_DIR, 'property_evaluation', 'ml_models', 'best_catboost_model.cbm')
        self.feature_names_path = os.path.join(settings.BASE_DIR, 'property_evaluation', 'ml_models', 'feature_names.json')

        self.center_nsk = (55.030165, 82.920436)
        self.nsu_coords = (54.8436, 83.0941)

        self.district_centers = {
            "Советский": (54.85, 83.05), "Центральный": (55.030, 82.920),
            "Октябрьский": (55.01, 82.94), "Ленинский": (54.96, 82.90),
            "Кировский": (54.95, 82.92), "Заельцовский": (55.06, 82.91),
            "Дзержинский": (55.04, 82.97), "Калининский": (55.10, 82.94),
            "Железнодорожный": (55.04, 82.90), "Первомайский": (54.97, 83.08),
        }

        self.load_model()

    def load_model(self):
        try:
            self.model = CatBoostRegressor()
            self.model.load_model(self.model_path)

            with open(self.feature_names_path, 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)

            print(f"Model loaded successfully. Features: {len(self.feature_names)}")
            self.load_success = True
            return True
        except Exception as e:
            print(f"Critical error loading model: {e}")
            self.load_success = False
            return False

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return 2 * R * atan2(sqrt(a), sqrt(1-a))

    def get_nsu_zone(self, dist):
        if pd.isna(dist) or dist > 50:
            return 'unknown'
        if dist < 1.0:
            return 'akadem_core'
        if dist < 3.0:
            return 'akadem_near'
        if dist < 7.0:
            return 'nsu_mid'
        return 'nsu_far'

    def predict(self, features_dict):
        if not self.load_success or self.model is None:
            print("Model not loaded")
            return None

        try:
            df = pd.DataFrame([features_dict])

            district = str(features_dict.get('district', 'Центральный'))
            lat, lon = self.district_centers.get(district, (55.03, 82.92))

            df['latitude'] = lat
            df['longitude'] = lon
            df['distance_to_center_km'] = self.haversine_distance(lat, lon, *self.center_nsk)
            df['distance_to_NSU_km'] = self.haversine_distance(lat, lon, *self.nsu_coords)

            df['district'] = district
            df['street'] = str(features_dict.get('street', ''))
            df['author_type'] = str(features_dict.get('author_type', 'realtor'))
            df['underground'] = 'unknown'

            d_nsu = float(df['distance_to_NSU_km'].iloc[0])
            is_akadem = int(d_nsu <= 3.0)

            df['is_akademgorodok'] = is_akadem
            df['nsu_zone'] = self.get_nsu_zone(d_nsu)
            df['akadem_premium_sqm'] = is_akadem * float(df['total_meters'].iloc[0])
            df['akadem_premium_sqm2'] = is_akadem * (float(df['total_meters'].iloc[0]) ** 1.3)
            df['akadem_distance_interaction'] = is_akadem / (d_nsu + 0.3) if is_akadem else 0
            df['log_distance_nsu'] = is_akadem * np.log1p(d_nsu) if is_akadem else 0
            df['district_price_index'] = 0.0
            df['has_metro'] = int(district in {"Центральный", "Железнодорожный", "Заельцовский", "Дзержинский", "Октябрьский", "Калининский", "Ленинский"})
            df['metro_zone'] = "Комфортная"
            df['distance_to_metro_km'] = 2.0

            for col in ['street', 'nsu_zone', 'metro_zone', 'author_type', 'district']:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            for col in self.feature_names:
                if col not in df.columns:
                    if col in ['is_studio', 'has_metro', 'is_akademgorodok']:
                        df[col] = 0
                    else:
                        df[col] = 0.0

            df = df[self.feature_names]

            pred_log = self.model.predict(df)[0]
            price_mln = float(np.expm1(pred_log))

            if district == "Советский" and d_nsu <= 8.0:
                total_m = float(features_dict.get('total_meters', 0))
                if total_m >= 50:
                    price_mln *= 1.23
                elif total_m >= 40:
                    price_mln *= 1.17
                elif total_m >= 30:
                    price_mln *= 1.10
                else:
                    price_mln *= 1.06

            final_price = round(price_mln * 1_000_000, -3)
            return max(3_000_000, final_price)

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None


predictor = PropertyPricePredictor()