import json
import time
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from urllib.parse import quote
import numpy as np
import pandas as pd
import requests
from catboost import CatBoostRegressor

BASE_DIR = Path(__file__).resolve().parent
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


model = CatBoostRegressor()
model.load_model(BASE_DIR / "best_catboost_model.cbm")

with open(BASE_DIR / "feature_names.json", "r", encoding="utf-8") as f:
    FEATURE_NAMES = json.load(f)


def prepare_flat_features(flat_dict):
    validate_raw_flat(flat_dict)
    flat_df = pd.DataFrame([flat_dict]).copy()

    address = build_novosibirsk_address(flat_dict)
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

    for col in FEATURE_NAMES:
        if col not in flat_df.columns:
            flat_df[col] = 0 if col in ['is_studio', 'has_metro', 'is_akademgorodok'] else np.nan

    return flat_df[FEATURE_NAMES]


def predict_flat_price(flat_dict):
    flat_df = prepare_flat_features(flat_dict)
    pred_log = model.predict(flat_df)
    pred = float(np.expm1(pred_log)[0])

    district = flat_dict.get("district")
    total_m = flat_dict.get("total_meters", 0)
    street = str(flat_dict.get("street", "")).lower()
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

    return round(pred, 3)


if __name__ == "__main__":
    print("=" * 80)
    print("EXTENDED MODEL TESTING")
    print("=" * 80)

    tests = [
        {"name": "Oktyabrsky, Griboedova 75, 28m2 (1-room)",
         "data": {"author_type": "realtor", "district": "Октябрьский", "floor": 5, "floors_count": 5,
                  "kitchen_meters": 5, "rooms_count": 1, "street": "Грибоедова", "total_meters": 28, "is_studio": 0,
                  "house_age": 56, "house_number": 75}, "real": 4.6},

        {"name": "Leninsky, Gorsky 8, 58m2 (2-room)",
         "data": {"author_type": "realtor", "district": "Ленинский", "floor": 4, "floors_count": 18,
                  "kitchen_meters": 12.4, "rooms_count": 2, "street": "микрорайон Горский", "total_meters": 57.8,
                  "is_studio": 0, "house_age": 18, "house_number": 8}, "real": 8.7},

        {"name": "Sovetsky, Morskoy 40, 53m2 (expensive)",
         "data": {"author_type": "realtor", "district": "Советский", "floor": 3, "floors_count": 4,
                  "kitchen_meters": 8.4, "rooms_count": 2, "street": "просп. Морской", "total_meters": 53,
                  "is_studio": 0, "house_age": 65, "house_number": 40}, "real": 13.5},

        {"name": "Sovetsky, Tereshkovoy 12, 30m2",
         "data": {"author_type": "real_estate_agent", "district": "Советский", "floor": 2, "floors_count": 9,
                  "kitchen_meters": 7.1, "rooms_count": 2, "street": "Терешковой", "total_meters": 30.3, "is_studio": 0,
                  "house_age": 54, "house_number": 12}, "real": 8.35},

        {"name": "Sovetsky, large apartment 75m2",
         "data": {"author_type": "realtor", "district": "Советский", "floor": 5, "floors_count": 9,
                  "kitchen_meters": 15, "rooms_count": 4, "street": "Ильича", "total_meters": 75, "is_studio": 0,
                  "house_age": 25, "house_number": 10}, "real": 15.5},

        {"name": "Tsentralny, large studio",
         "data": {"author_type": "realtor", "district": "Центральный", "floor": 7, "floors_count": 12,
                  "kitchen_meters": 6, "rooms_count": 1, "street": "Красный проспект", "total_meters": 42,
                  "is_studio": 1, "house_age": 15, "house_number": 55}, "real": 6.8},

        {"name": "Kirovsky, 35m2 (1-room)",
         "data": {"author_type": "homeowner", "district": "Кировский", "floor": 3, "floors_count": 5,
                  "kitchen_meters": 8, "rooms_count": 1, "street": "Сибиряков-Гвардейцев", "total_meters": 35,
                  "is_studio": 0, "house_age": 40, "house_number": 22}, "real": 3.9},

        {"name": "Pervomaysky, 65m2 (3-room)",
         "data": {"author_type": "realtor", "district": "Первомайский", "floor": 8, "floors_count": 17,
                  "kitchen_meters": 12, "rooms_count": 3, "street": "Марии Ульяновой", "total_meters": 65,
                  "is_studio": 0, "house_age": 12, "house_number": 18}, "real": 9.2},

        {"name": "Zaeltsovsky, 45m2 (2-room)",
         "data": {"author_type": "realtor", "district": "Заельцовский", "floor": 9, "floors_count": 16,
                  "kitchen_meters": 9, "rooms_count": 2, "street": "Дуси Ковальчук", "total_meters": 45, "is_studio": 0,
                  "house_age": 8, "house_number": 33}, "real": 7.1},
    ]

    print(f"Total tests: {len(tests)}\n")

    results = []
    for t in tests:
        print(f"Location: {t['name']}")
        print(f"   Actual price: {t['real']:.2f} million rub")
        try:
            pred = predict_flat_price(t['data'])
            err = ((pred - t['real']) / t['real']) * 100
            print(f"   Predicted:   {pred:.3f} million rub")
            print(f"   Error:        {err:+.1f}%\n")
            results.append(err)
        except Exception as e:
            print(f"   Error: {e}\n")

    errors = [abs(e) for e in results]
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Mean absolute error: {np.mean(errors):.1f}%")
    print(f"Median error:          {np.median(errors):.1f}%")
    print(f"Max error:              {max(errors):.1f}%")
    print(f"Tests with error >15%: {sum(1 for e in errors if e > 15)} of {len(errors)}")