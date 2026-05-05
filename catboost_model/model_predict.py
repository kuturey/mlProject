import json
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

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
    'district': 'Кировский',
    'floor': 4,
    'floors_count': 9,
    'kitchen_meters': 2,
    'rooms_count': 2,
    'street': 'Немировича-Данченко',
    'total_meters': 24.6,
    'underground': 'Площадь Маркса',
    'is_studio': 1,
    'house_age': 47
}

flat_example2 = {
    'author_type': 'real_estate_agent',
    'district': 'Дзержинский',
    'floor': 2,
    'floors_count': 5,
    'kitchen_meters': 8,
    'rooms_count': 2,
    'street': 'Промышленная',
    'total_meters': 27,
    'underground': 'Маршала Покрышкина',
    'is_studio': 0,
    'house_age': 66
}

flat_example2 = {
    'author_type': 'real_estate_agent',
    'district': 'Советский',
    'floor': 3,
    'floors_count': 5,
    'kitchen_meters': 7,
    'rooms_count': 2,
    'street': 'Барьерная',
    'total_meters': 42.8,
    'underground': 'Речной вокзал',
    'is_studio': 0,
    'house_age': 58
}

price = predict_flat_price(flat_example2)

print(f'Предсказанная цена: {price:.3f} млн руб')



