import os
import json
import time
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from urllib.parse import quote
import requests
from django.conf import settings
from catboost import CatBoostRegressor


BASE_DIR = settings.BASE_DIR
YANDEX_API_KEY = "343b79b6-07bd-4ea8-a8cc-b1d443629c35"

NOVOSIBIRSK_METRO = {
    "Заельцовская": (55.0590, 82.9126), "Гагаринская": (55.0509, 82.9148),
    "Красный проспект": (55.0416, 82.9174), "Площадь Ленина": (55.0302, 82.9204),
    "Октябрьская": (55.0188, 82.9392), "Речной вокзал": (55.0093, 82.9377),
    "Спортивная": (54.9947, 82.9200), "Студенческая": (54.9890, 82.9068),
    "Площадь Маркса": (54.9823, 82.8962), "Площадь Гарина-Михайловского": (55.0354, 82.8996),
    "Сибирская": (55.0401, 82.9217), "Маршала Покрышкина": (55.0446, 82.9342),
    "Березовая Роща": (55.0433, 82.9519), "Золотая Нива": (55.0375, 82.9756),
}

DISTRICTS_WITH_METRO = {"Центральный", "Железнодорожный", "Заельцовский", "Дзержинский",
                        "Октябрьский", "Калининский", "Ленинский"}

DISTRICT_CENTERS = {
    "Советский": (54.85, 83.05), "Центральный": (55.030, 82.920), "Октябрьский": (55.01, 82.94),
    "Ленинский": (54.96, 82.90), "Кировский": (54.95, 82.92), "Заельцовский": (55.06, 82.91),
    "Дзержинский": (55.04, 82.97), "Калининский": (55.10, 82.94),
    "Железнодорожный": (55.04, 82.90), "Первомайский": (54.97, 83.08),
}

RAW_FLAT_COLUMNS = {"author_type", "district", "floor", "floors_count", "kitchen_meters",
                    "rooms_count", "street", "total_meters", "is_studio", "house_age", "house_number"}

center_of_novosibirsk = (55.030165, 82.920436)
NSU_coords = (54.8436, 83.0941)

DISTRICT_MEAN_PRICE = {
    "Советский": 8.2,
    "Центральный": 7.1,
    "Октябрьский": 5.8,
    "Ленинский": 5.4,
    "Кировский": 4.9,
    "Заельцовский": 5.6,
    "Дзержинский": 5.7,
    "Калининский": 5.2,
    "Железнодорожный": 6.0,
    "Первомайский": 6.3,
}


def geocode_yandex(address, max_retries=3):
    encoded_address = quote(str(address))
    url = f"https://geocode-maps.yandex.ru/1.x/?apikey={YANDEX_API_KEY}&geocode={encoded_address}&format=json&results=1&lang=ru_RU"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                geo_objects = data["response"]["GeoObjectCollection"]["featureMember"]
                if geo_objects:
                    point = geo_objects[0]["GeoObject"]["Point"]["pos"]
                    lon, lat = map(float, point.split())
                    return lat, lon
        except:
            time.sleep(1 * (2 ** attempt))

    address_str = str(address).lower()
    for district, coords in DISTRICT_CENTERS.items():
        if district.lower() in address_str:
            return coords
    return 55.03, 82.92


def haversine_distance(lat1, lon1, lat2, lon2):
    radius_km = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return radius_km * c


def distance_to_nearest_metro(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return None, np.nan
    min_dist = float("inf")
    nearest = None
    for station, (mlat, mlon) in NOVOSIBIRSK_METRO.items():
        dist = haversine_distance(lat, lon, mlat, mlon)
        if dist < min_dist:
            min_dist = dist
            nearest = station
    return nearest, min_dist


def get_metro_zone(distance):
    if pd.isna(distance):
        return "Удалённая"
    if distance <= 0.5:
        return "Пешеходная"
    if distance <= 1.5:
        return "Комфортная"
    if distance <= 3.0:
        return "Транспортная"
    if distance <= 7.0:
        return "Автомобильная"
    return "Удалённая"


def get_nsu_zone(dist):
    if pd.isna(dist) or dist > 50:
        return 'unknown'
    if dist < 1.0:
        return 'akadem_core'
    if dist < 3.0:
        return 'akadem_near'
    if dist < 7.0:
        return 'nsu_mid'
    return 'nsu_far'


def build_novosibirsk_address(flat_dict):
    parts = ["Новосибирск"]
    if flat_dict.get("street"):
        parts.append(str(flat_dict["street"]))
    if flat_dict.get("house_number"):
        parts.append(str(flat_dict["house_number"]))
    return ", ".join(parts)


def validate_raw_flat(flat_dict):
    missing = RAW_FLAT_COLUMNS - set(flat_dict.keys())
    if missing:
        raise ValueError(f"Missing fields: {missing}")


class PropertyPricePredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.load_success = False

        model_path = os.path.join(BASE_DIR, 'property_evaluation', 'ml_models', 'best_catboost_model.cbm')
        feature_names_path = os.path.join(BASE_DIR, 'property_evaluation', 'ml_models', 'feature_names.json')

        try:
            self.model = CatBoostRegressor()
            self.model.load_model(model_path)

            with open(feature_names_path, 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)

            print(f"Model loaded successfully. Features: {len(self.feature_names)}")
            self.load_success = True
        except Exception as e:
            print(f"Critical error loading model: {e}")
            self.load_success = False

    def predict(self, features_dict):
        if not self.load_success or self.model is None:
            print("Model not loaded")
            return None

        try:
            validate_raw_flat(features_dict)
            flat_df = pd.DataFrame([features_dict]).copy()

            address = build_novosibirsk_address(features_dict)
            lat, lon = geocode_yandex(address)

            station, metro_dist = distance_to_nearest_metro(lat, lon)

            flat_df.loc[0, "latitude"] = lat
            flat_df.loc[0, "longitude"] = lon
            flat_df.loc[0, "underground"] = station
            flat_df.loc[0, "distance_to_metro_km"] = round(metro_dist, 3) if pd.notna(metro_dist) else 999.0
            flat_df.loc[0, "distance_to_center_km"] = haversine_distance(lat, lon, *center_of_novosibirsk)
            flat_df.loc[0, "distance_to_NSU_km"] = haversine_distance(lat, lon, *NSU_coords)

            district_val = flat_df["district"].iloc[0]
            flat_df.loc[0, "has_metro"] = int(district_val in DISTRICTS_WITH_METRO)
            flat_df.loc[0, "metro_zone"] = get_metro_zone(flat_df.loc[0, "distance_to_metro_km"])

            d_nsu = flat_df.loc[0, "distance_to_NSU_km"]
            is_akadem = int(d_nsu <= 3.0)

            flat_df.loc[0, "is_akademgorodok"] = is_akadem
            flat_df.loc[0, "nsu_zone"] = get_nsu_zone(d_nsu)
            flat_df.loc[0, "akadem_premium_sqm"] = is_akadem * flat_df.loc[0, "total_meters"]
            flat_df.loc[0, "akadem_premium_sqm2"] = is_akadem * (flat_df.loc[0, "total_meters"] ** 1.3)
            flat_df.loc[0, "akadem_distance_interaction"] = is_akadem / (d_nsu + 0.3)
            flat_df.loc[0, "log_distance_nsu"] = is_akadem * np.log1p(d_nsu)

            district = flat_df.loc[0, "district"]
            flat_df.loc[0, "district_price_index"] = DISTRICT_MEAN_PRICE.get(district, 5.5)

            for col in self.feature_names:
                if col not in flat_df.columns:
                    flat_df[col] = 0 if col in ['is_studio', 'has_metro', 'is_akademgorodok'] else np.nan

            flat_df = flat_df[self.feature_names]

            pred_log = self.model.predict(flat_df)
            pred = float(np.expm1(pred_log)[0])

            district = features_dict.get("district")
            total_m = features_dict.get("total_meters", 0)
            street = str(features_dict.get("street", "")).lower()
            d_nsu = flat_df["distance_to_NSU_km"].iloc[0]

            if district == "Советский" and d_nsu <= 8.0:
                if "морской" in street:
                    pred *= 1.235
                elif "терешковой" in street:
                    pred *= 1.185
                elif total_m >= 50:
                    pred *= 1.24
                elif total_m >= 40:
                    pred *= 1.17
                elif total_m >= 30:
                    pred *= 1.10
                else:
                    pred *= 1.06

            return round(pred * 1_000_000, -3)

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None


predictor = PropertyPricePredictor()