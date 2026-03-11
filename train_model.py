import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Создаем папки, если их нет
os.makedirs('property_evaluation/data', exist_ok=True)
os.makedirs('property_evaluation/ml_models', exist_ok=True)

# Загружаем датасет
df = pd.read_csv('property_evaluation/data/dataset.csv')
print(f"Загружено строк: {len(df)}")
print(f"Колонки: {list(df.columns)}")

# Признаки и целевая переменная
X = df.drop('price', axis=1)
y = df['price']

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Логарифмируем цены
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Масштабируем признаки
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Масштабируем целевую переменную
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_log.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test_log.values.reshape(-1, 1))

# Конвертация в тензоры
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

# Создаем модель
model = nn.Linear(X_train_t.shape[1], 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0.0001)
criterion = nn.MSELoss()

# Обучение
epochs = 1000
for epoch in range(epochs):
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Оценка модели
with torch.no_grad():
    y_pred_scaled = model(X_test_t)

    # Обратное преобразование
    y_pred_log = scaler_y.inverse_transform(y_pred_scaled.numpy())
    y_test_log_really = scaler_y.inverse_transform(y_test_t.numpy())
    y_pred_rub = np.expm1(y_pred_log).flatten()
    y_test_rub = np.expm1(y_test_log_really).flatten()

    # Метрики
    mse = np.mean((y_pred_rub - y_test_rub) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred_rub - y_test_rub))
    mape = np.mean(np.abs(y_test_rub - y_pred_rub) / y_test_rub) * 100

    print("=" * 50)
    print("РЕЗУЛЬТАТЫ МОДЕЛИ")
    print("=" * 50)
    print(f"RMSE: {rmse:.2f} руб.")
    print(f"MAE: {mae:.2f} руб.")
    print(f"MAPE: {mape:.2f}%")

# Сохраняем модель и скейлеры
model_data = {
    'model_state': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'input_dim': X_train_t.shape[1],
    'features_names': list(X.columns)
}

# Сохраняем
joblib.dump(model_data, 'property_evaluation/ml_models/model.pkl')
print("✅ Модель сохранена в property_evaluation/ml_models/model.pkl")
print(f"Признаки модели: {list(X.columns)}")