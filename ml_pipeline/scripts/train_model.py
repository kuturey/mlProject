import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PropertyPriceNN(nn.Module):
    """Нейросеть для оценки недвижимости"""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(csv_path, model_save_path, scaler_X_path, scaler_y_path):

    print("=" * 60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"Загружено {len(df)} объектов")
    print(f"Доступные колонки: {list(df.columns)}")

    df = df[(df['price'] > 500000) & (df['price'] < 50000000)]
    print(f"После удаления выбросов: {len(df)} объектов")

    feature_columns = [
        'total_meters',
        'kitchen_meters',
        'rooms_count',
        'floor',
        'floors_count',
        'year_of_construction'
    ]

    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Ошибка: отсутствуют колонки: {missing_cols}")
        return

    X = df[feature_columns].copy()
    y = df['price']

    current_year = datetime.now().year
    X['building_age'] = current_year - X['year_of_construction']
    X = X.drop('year_of_construction', axis=1)

    final_features = ['total_meters', 'kitchen_meters', 'rooms_count', 'floor', 'floors_count', 'building_age']
    X = X[final_features]

    print(f"Используемые признаки: {final_features}")
    print(f"Всего признаков: {len(final_features)}")

    y_log = np.log1p(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_log, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    X_train_t = torch.FloatTensor(X_train_scaled)
    X_val_t = torch.FloatTensor(X_val_scaled)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_train_t = torch.FloatTensor(y_train_scaled)
    y_val_t = torch.FloatTensor(y_val_scaled)
    y_test_t = torch.FloatTensor(y_test_scaled)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = PropertyPriceNN(input_dim=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5)

    epochs = 300
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0

    print("\n" + "=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            with open(scaler_X_path, 'wb') as f:
                pickle.dump(scaler_X, f)
            with open(scaler_y_path, 'wb') as f:
                pickle.dump(scaler_y, f)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping на эпохе {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"Эпоха [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print("\n" + "=" * 60)
    print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t)
        y_pred_log = scaler_y.inverse_transform(y_pred_scaled.numpy())
        y_test_log = scaler_y.inverse_transform(y_test_t.numpy())
        y_pred_rub = np.expm1(y_pred_log)
        y_test_rub = np.expm1(y_test_log)

    mape = np.mean(np.abs(y_test_rub - y_pred_rub) / y_test_rub) * 100
    print(f"MAPE на тесте: {mape:.2f}%")
    print(f"Модель сохранена в: {model_save_path}")
    print(f"Scaler X сохранён в: {scaler_X_path}")
    print(f"Scaler Y сохранён в: {scaler_y_path}")
    print("=" * 60)

    return model


if __name__ == "__main__":
    BASE_ML_DIR = os.path.join(PROJECT_ROOT, 'property_evaluation', 'ml_models')
    os.makedirs(BASE_ML_DIR, exist_ok=True)

    csv_path = os.path.join(PROJECT_ROOT, 'ml_pipeline', 'data', 'dataset9.csv')
    model_path = os.path.join(BASE_ML_DIR, 'model.pth')
    scaler_X_path = os.path.join(BASE_ML_DIR, 'scaler_X.pkl')
    scaler_y_path = os.path.join(BASE_ML_DIR, 'scaler_y.pkl')

    train_model(csv_path, model_path, scaler_X_path, scaler_y_path)