import numpy as np
import pandas as pd
import joblib

from data import data_load

PARAM_DIR = '/Users/zxfg0/Automation/params'

# WAPE (Weighted Absolute Percentage Error, %)
def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


class TestCode():
    def __init__(self):
        self.train_ws_x, self.test_ws_x, self.train_ws_y, self.test_ws_y, self.train_ds_x, self.test_ds_x, self.train_ds_y, self.test_ds_y = data_load()
        self.ws_params = np.load(f'{PARAM_DIR}/params_ws.npy', allow_pickle=True)
        self.ds_params = np.load(f'{PARAM_DIR}/params_ds.npy', allow_pickle=True)
        # MinMaxScaler (열 순서: idle_time=0, ws_gap_diff=1, ds_gap_diff=2)
        self.scaler = joblib.load(f'{PARAM_DIR}/scaler.pkl')

    # ---------- 예측 ----------
    def predict(self, params, x):
        a, b, c = params
        return a * np.exp(b * np.asarray(x, dtype=float)) + c

    # 0~1 스케일 → 원래 단위
    def inverse_y(self, y_scaled, col):
        return np.asarray(y_scaled, dtype=float) * self.scaler.data_range_[col] + self.scaler.data_min_[col]

    # ---------- 지표 ----------
    @staticmethod
    def metrics(y, y_pred):
        y = np.asarray(y, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
        return {'R2': r2, 'MSE': mse, 'RMSE': np.sqrt(mse), 'MAE': mae, 'WAPE': wape(y, y_pred)}

    def evaluate_models(self):
        self.results = {}
        for name, params, x, y, col in [
            ('ws_gap_diff', self.ws_params, self.test_ws_x, self.test_ws_y, 1),
            ('ds_gap_diff', self.ds_params, self.test_ds_x, self.test_ds_y, 2),
        ]:
            y_pred = self.predict(params, x)                        # test 예측 (0~1 스케일)
            y_true_raw = self.inverse_y(y, col)                     # 원래 단위로 역변환
            y_pred_raw = self.inverse_y(y_pred, col)
            self.results[name] = self.metrics(y_true_raw, y_pred_raw)   # 모든 지표를 원래 단위로 계산
        return self.results

    def print_results(self):
        print('\nEvaluating models...\n')
        for name, params in [('ws_gap_diff', self.ws_params), ('ds_gap_diff', self.ds_params)]:
            m = self.results[name]
            print('=' * 50)
            print(f'{name} model (test)')
            print('=' * 50)
            print(f'params: {params}')
            print(f"R²: {m['R2']:.4f} | MSE: {m['MSE']:.2f} | RMSE: {m['RMSE']:.2f} | MAE: {m['MAE']:.2f} | WAPE: {m['WAPE']:.2f}%\n")

    # 그래프용 지표 저장
    def save_results(self, path=f'{PARAM_DIR}/metrics_exp_fit.csv'):
        rows = [{'Target': t, 'Model': 'Exponential Curve Fit', **m} for t, m in self.results.items()]
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f'지표 저장 완료: {path}')


if __name__ == '__main__':
    test = TestCode()
    test.evaluate_models()
    test.print_results()
    test.save_results()