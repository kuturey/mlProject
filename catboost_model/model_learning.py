import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import json

def calc_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return mae, rmse, mape, r2


df = pd.read_csv('final_output.csv')

cat_features = [
    'author_type',
    'district',
    'underground',
    'is_studio',
    'metro_zone'
]

# print(df)
# Разделение данных на признаки и целевую переменную
X = df.drop('price', axis=1)
y = df['price']

feature_names = list(X.columns)

with open('feature_names.json', 'w', encoding='utf-8') as f:
    json.dump(feature_names, f, ensure_ascii=False, indent=4)

with open('cat_features.json', 'w', encoding='utf-8') as f:
    json.dump(cat_features, f, ensure_ascii=False, indent=4)


y_log = np.log1p(y)


# Разделение данных на обучающую и тестовую выборку
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y_log, test_size=0.15, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.1765, random_state=42)


configs = [
    {"name": "depth5_l2_5_lr003", "depth": 5, "l2_leaf_reg": 5, "learning_rate": 0.03},
    {"name": "depth5_l2_10_lr003", "depth": 5, "l2_leaf_reg": 10, "learning_rate": 0.03},
    {"name": "depth6_l2_5_lr003", "depth": 6, "l2_leaf_reg": 5, "learning_rate": 0.03},
    {"name": "depth6_l2_10_lr003", "depth": 6, "l2_leaf_reg": 10, "learning_rate": 0.03},
    {"name": "depth7_l2_10_lr003", "depth": 7, "l2_leaf_reg": 10, "learning_rate": 0.03},
    {"name": "depth6_l2_10_lr002", "depth": 6, "l2_leaf_reg": 10, "learning_rate": 0.02},
    {"name": "depth6_l2_10_lr005", "depth": 6, "l2_leaf_reg": 10, "learning_rate": 0.05},
]

results = []
models = {}

for config in configs:
    print("=" * 50)
    print(f"Обучаем конфиг : {config['name']}")
    print("=" * 50)

    model = CatBoostRegressor(
        iterations=10000,
        learning_rate=config["learning_rate"],
        depth=config["depth"],
        l2_leaf_reg=config["l2_leaf_reg"],
        early_stopping_rounds=100,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )

    valid_pred_log = model.predict(X_valid)

    valid_mae, valid_rmse, valid_mape, valid_r2 = calc_metrics(y_valid, valid_pred_log)
    
    result = {
        "name":config["name"],
        "depth": config["depth"],
        "l2_leaf_reg": config["l2_leaf_reg"],
        "learning_rate": config["learning_rate"],
        "best_iteration": model.get_best_iteration(),
        "valid_mae": valid_mae,
        "valid_rmse": valid_rmse,
        "valid_mape": valid_mape,
        "valid_r2": valid_r2
    }
    
    results.append(result)
    models[config["name"]] = model
    
    print(f"VALID MAE:  {valid_mae:.3f} млн руб.")
    print(f"VALID RMSE: {valid_rmse:.3f} млн руб.")
    print(f"VALID MAPE: {valid_mape:.2f}%")
    print(f"VALID R2:   {valid_r2:.3f}")
    print(f"Best iteration: {model.get_best_iteration()}")
    
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("valid_mape", ascending=True)

best_result = results_df.iloc[0]
best_config_name = best_result["name"]
best_model = models[best_config_name]
best_model.save_model('best_catboost_model.cbm')

print("=" * 70)
print("ЛУЧШАЯ КОНФИГУРАЦИЯ ПО VALID")
print("=" * 70)
print(best_result)

test_pred_log = best_model.predict(X_test)

test_mae, test_rmse, test_mape, test_r2 = calc_metrics(y_test, test_pred_log)

print("=" * 50)
print("РЕЗУЛЬТАТЫ ЛУЧШЕЙ МОДЕЛИ CATBOOST НА TEST")
print("=" * 50)
print(f"BEST CONFIG: {best_config_name}")
print(f"MAE:  {test_mae:.3f} млн руб.")
print(f"RMSE: {test_rmse:.3f} млн руб.")
print(f"MAPE: {test_mape:.2f}%")
print(f"R2:   {test_r2:.3f}")



# # Вывод анализа
# plt.figure(figsize=(15, 5))

# # График обучения
# plt.subplot(1, 2, 1)
# plt.plot(history)
# plt.title("Процесс обучения (MSE)")
# plt.xlabel("Эпоха")
# plt.ylabel("Loss")

# # Анализ остатков
# with torch.no_grad():
#     # ИСПРАВЛЕНИЕ: используем масштабированные данные X_test_t вместо X_test
#     test_preds_scaled = model(X_test_t).numpy().flatten()
#     # Преобразуем y_test в numpy, если это еще не сделано
#     y_test_np = y_test_t.numpy().flatten()
#     residuals = y_test_np - test_preds_scaled

# plt.subplot(1, 2, 2)
# plt.scatter(test_preds_scaled, residuals, alpha=0.3, color='teal')
# plt.axhline(0, color='red', linestyle='--')
# plt.title("Анализ остатков (Residual Analysis)")
# plt.xlabel("Предсказанная цена (масштабированная)")
# plt.ylabel("Ошибка (масштабированная)")
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()
    