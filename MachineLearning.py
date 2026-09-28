import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import joblib

PARAM_DIR = '/Users/zxfg0/Automation/params'

# WAPE (Weighted Absolute Percentage Error, %)
def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


# 1. 데이터 불러오기
df = pd.read_csv(r'/Users/zxfg0/Automation/TalkFile_generated_data.csv')

X = df[['idle_time']]
y_ws = df['ws_gap_diff']
y_ds = df['ds_gap_diff']

# 2. Train / Test Split
X_train, X_test, y_ws_train, y_ws_test, y_ds_train, y_ds_test = train_test_split(
    X, y_ws, y_ds, test_size=0.2, random_state=42
)

# 신경망 및 알고리즘용 스케일러
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 모델 정의
models = {
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

# [추가] 그래프용 지표 저장 리스트
results = []

# 4. ws_gap_diff 예측 모델 비교 및 평가
print("=" * 70)
print(" [Task 1] ws_gap_diff 예측 모델 비교")
print("=" * 70)

best_ws_model = None
best_ws_score = -1

for name, model in models.items():
    if 'Neural' in name:
        model.fit(X_train_scaled, y_ws_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_ws_train)
        preds = model.predict(X_test)

    r2 = r2_score(y_ws_test, preds)
    mae = mean_absolute_error(y_ws_test, preds)
    mse = mean_squared_error(y_ws_test, preds)
    rmse = np.sqrt(mse)
    wape_v = wape(y_ws_test, preds)

    print(f"[{name:20s}] R²: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | WAPE: {wape_v:.2f}%")

    # [추가] 지표 기록
    results.append({'Target': 'ws_gap_diff', 'Model': name,
                    'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'WAPE': wape_v})

    if r2 > best_ws_score:
        best_ws_score = r2
        best_ws_model = (name, model)

# 5. ds_gap_diff 예측 모델 비교 및 평가
print("\n" + "=" * 70)
print(" [Task 2] ds_gap_diff 예측 모델 비교")
print("=" * 70)

best_ds_model = None
best_ds_score = -1

for name, model in models.items():
    if 'Neural' in name:
        model.fit(X_train_scaled, y_ds_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_ds_train)
        preds = model.predict(X_test)

    r2 = r2_score(y_ds_test, preds)
    mae = mean_absolute_error(y_ds_test, preds)
    mse = mean_squared_error(y_ds_test, preds)
    rmse = np.sqrt(mse)
    wape_v = wape(y_ds_test, preds)

    print(f"[{name:20s}] R²: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | WAPE: {wape_v:.2f}%")

    # [추가] 지표 기록
    results.append({'Target': 'ds_gap_diff', 'Model': name,
                    'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'WAPE': wape_v})

    if r2 > best_ds_score:
        best_ds_score = r2
        best_ds_model = (name, model)

# 6. 최적 모델 저장
print("\n" + "=" * 70)
print(f"최적의 ws_gap_diff 예측 모델: {best_ws_model[0]} (R² = {best_ws_score:.4f})")
print(f"최적의 ds_gap_diff 예측 모델: {best_ds_model[0]} (R² = {best_ds_score:.4f})")

joblib.dump(best_ws_model[1], f'{PARAM_DIR}/best_ws_gap_model.pkl')
joblib.dump(best_ds_model[1], f'{PARAM_DIR}/best_ds_gap_model.pkl')
print("모델 저장 완료 ('best_ws_gap_model.pkl', 'best_ds_gap_model.pkl')")

# 7. [추가] 모델별 지표 저장 (그래프용)
metrics_df = pd.DataFrame(results)
metrics_df.to_csv(f'{PARAM_DIR}/metrics_ml.csv', index=False)
print(f"지표 저장 완료 ('{PARAM_DIR}/metrics_ml.csv')")
print(metrics_df.round(4).to_string(index=False))