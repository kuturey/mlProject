import json
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2-lat1
    dlon = lon2-lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R*c

def distance_to_nearest_metro(lat, lon, metro_dict):
    if pd.isna(lat) or pd.isna(lon):
        return None, np.nan

    min_distance = float('inf')
    nearest_station = None
    
    for station, (mlat, mlon) in metro_dict.items():
        distance = haversine_distance(lat, lon, mlat, mlon)
        if distance < min_distance:
            min_distance = distance
            nearest_station = station
    
    return nearest_station, min_distance

def add_metro_distance(df, metro_dict):
    df = df.copy()
    
    results = df.apply(lambda row: distance_to_nearest_metro(row['latitude'], row['longitude'], metro_dict), axis=1)
    df['underground'] = [r[0] for r in results]
    df['distance_to_metro_km'] = [round(r[1], 3) if r[1] else np.nan for r in results]
    
    return df

def get_metro_zone(distance):
    if distance <= 0.5:
        return 'Пешеходная'
    elif distance <= 1.5:
        return 'Комфортная'
    elif distance <= 3.0:
        return 'Транспортная'
    elif distance <= 7.0:
        return 'Автомобильная'
    else:
        return 'Удалённая'

def add_metro_features(df, districs_with_metro):
    df = df.copy()
    
    df['has_metro'] = df['district'].isin(districs_with_metro).astype(int)
    
    df['metro_zone'] = df['distance_to_metro_km'].apply(get_metro_zone)
    
    df['effective_metro_distance'] = df['distance_to_metro_km'] * df['has_metro']
    
    df = df.drop('distance_to_metro_km', axis=1)
    
    return df

model = CatBoostRegressor()
model.load_model('best_catboost_model.cbm')

with open('feature_names.json', 'r', encoding='utf-8') as f:
    feature_names = json.load(f)
    
def predict_flat_price(flat_dict):
    flat_df = pd.DataFrame([flat_dict])
    missing_cols = set(feature_names) - set(flat_df.columns)
    if missing_cols:
        raise ValueError(f'Не хватает колонок: {missing_cols}')

    flat_df = flat_df[feature_names]
    
    pred_log = model.predict(flat_df)
    pred_price = np.expm1(pred_log)[0]
    
    return pred_price

flat_example1 = {
    'author_type': 'real_estate_agent',
    'district': 'Заельцовский',
    'floor': 5,
    'floors_count': 5,
    'kitchen_meters': 7,
    'rooms_count': 2,
    'street': 'Кубовая',
    'total_meters': 37.1,
    'underground': 'Заельцовская',
    'is_studio': 0,
    'house_age': 39,
    'house_number': 111,
}

flat_example2 = {
    'author_type': 'real_estate_agent',
    'district': 'Советский',
    'floor': 5,
    'floors_count': 5,
    'kitchen_meters': 6.4,
    'rooms_count': 1,
    'street': 'Иванова',
    'total_meters': 33,
    'underground': 'Маршала Покрышкина',
    'is_studio': 0,
    'house_age': 51,
    'house_number': 40,
    'distance_to_metro_km': 14.6
}

flat_example3 = {
    'author_type': 'real_estate_agent',
    'district': 'Октябрьский',
    'floor': 14,
    'floors_count': 25,
    'kitchen_meters': 11.6,
    'rooms_count': 2,
    'street': 'Закаменский микрорайон',
    'total_meters': 59.4,
    'underground': 'Березовая Роща',
    'is_studio': 0,
    'house_age': 16,
    'house_number': 15,
    'distance_to_metro_km': 0.9
}

price = predict_flat_price(flat_example3)

print(f'Предсказанная цена: {price:.3f} млн руб')



