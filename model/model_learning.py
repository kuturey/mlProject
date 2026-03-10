import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

###########################################
# ТОЧНОСТЬ МОДЕЛИ В ЗАВИСИМОСТИ ОТ ПРИЗНАКОВ
# total_meters и rooms_count вместе -- Loss = 0.2958, MAPE = 25.97%
# total_metes, без rooms_count -- loss = 0.3260, MAPE = 27.92%
# без total_meters, rooms_count -- loss = 0.3526, MAPE = 27.84%
# убрали kitchen_meters -- loss = 0.3072, MAPE 26.27%
# убрали в ценах все разряды до тысяч -- Loss:0.2958, MAPE = 25.97%
# привели все цены к тысячам -- Loss:0.2958, MAPE = 25.98% (после этого ниже так и будет)
# убрали районы, тип автора - Loss:0.3010, MAPE = 26.28%
# оставил все признаки, кроме underground и street -- Loss:0.2722, MAPE = 24.43%


###########################################
df = pd.read_csv('dataset9.csv')
# print(df)

# Разделение данных на признаки и целевую переменную
X = df.drop('price', axis=1)
y = df['price']

# Разделение данных на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабируем признаки

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Конвертация в тензоры PyTorch
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

model = nn.Linear(X_train_t.shape[1], 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
criterion = nn.MSELoss()

epochs = 100000
history = []
for epoch in range(epochs):
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    history.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss:{loss.item():.4f}")
        
with torch.no_grad():
    y_pred = model(X_test_t)
    
    # y_pred_log = scaler_y.inverse_transform(y_pred_scaled)
    # y_test_log_really = scaler_y.inverse_transform(y_test_t)
    # y_pred_rub = np.expm1(y_pred_log)
    # y_test_rub = np.expm1(y_test_log_really)
    y_pred_rub = y_pred.numpy()
    y_test_rub = y_test_t.numpy()
    mse = np.mean((y_pred_rub - y_test_rub)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred_rub - y_test_rub))
    mape = np.mean(np.abs(y_test_rub - y_pred_rub) / y_test_rub) * 100
    print("=" * 50)
    print("РЕЗУЛЬТАТЫ МОДЕЛИ")
    print("=" * 50)
    print(f"MSE (в квадрате рублей): {mse:.2f}")
    print(f"RMSE (средняя ошибка в RUB): {rmse:.2f}")
    print(f"MAE (средняя абсолютная ошибка в RUB): {mae:.2f}")
    print(f"MAPE (средняя процентная ошибка): {mape:.2f}%")
    