import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calc_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2


df = pd.read_csv('final_output.csv')
df['distance_to_NSU_km'] = df['distance_to_NSU_km'].fillna(999.0)

def get_nsu_zone(dist):
    if pd.isna(dist) or dist >= 999:
        return 'unknown'
    if dist < 1.0:
        return 'akadem_core'
    if dist < 3.0:
        return 'akadem_near'
    if dist < 7.0:
        return 'nsu_mid'
    return 'nsu_far'

df['is_akademgorodok'] = (df['distance_to_NSU_km'] <= 3.0).astype(int)
df['nsu_zone'] = df['distance_to_NSU_km'].apply(get_nsu_zone)

district_mean = df.groupby('district')['price'].transform('mean')
df['district_price_index'] = district_mean

df['akadem_premium_sqm'] = df['is_akademgorodok'] * df['total_meters']
df['akadem_premium_sqm2'] = df['is_akademgorodok'] * (df['total_meters'] ** 1.3)
df['akadem_distance_interaction'] = df['is_akademgorodok'] / (df['distance_to_NSU_km'] + 0.3)
df['log_distance_nsu'] = df['is_akademgorodok'] * np.log1p(df['distance_to_NSU_km'])

X = df.drop('price', axis=1)
y = np.log1p(df['price'])

cat_features = ['author_type', 'district', 'underground', 'is_studio', 'metro_zone', 'street', 'nsu_zone']
feature_names = list(X.columns)

with open('../property_evaluation/ml_models/feature_names.json', 'w', encoding='utf-8') as f:
    json.dump(feature_names, f, ensure_ascii=False, indent=4)
with open('../property_evaluation/ml_models/cat_features.json', 'w', encoding='utf-8') as f:
    json.dump(cat_features, f, ensure_ascii=False, indent=4)

sample_weights = df.apply(
    lambda r: 10.0 if r['district'] == 'Советский' and r['distance_to_NSU_km'] <= 4 else
              6.0 if r['district'] == 'Советский' else
              2.0 if r['district'] in ['Первомайский'] else 1.0, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=X['district'])
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.18, random_state=42)

model = CatBoostRegressor(
    iterations=12000,
    learning_rate=0.028,
    depth=8,
    l2_leaf_reg=6,
    early_stopping_rounds=350,
    random_seed=42,
    verbose=400,
    task_type="CPU"
)

model.fit(X_train, y_train,
          cat_features=cat_features,
          eval_set=(X_valid, y_valid),
          sample_weight=sample_weights.loc[X_train.index].values,
          use_best_model=True)

model.save_model('best_catboost_model.cbm')

valid_mape = calc_metrics(y_valid, model.predict(X_valid))[2]
test_mape = calc_metrics(y_test, model.predict(X_test))[2]
print(f"VALID MAPE: {valid_mape:.2f}%")
print(f"TEST MAPE:  {test_mape:.2f}%")