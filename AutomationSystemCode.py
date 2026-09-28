import numpy as np
import pandas as pd
from scipy.optimize import minimize
import joblib

from data import data_load


# WAPE (Weighted Absolute Percentage Error, %)
def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


class AutomationSystemCode():
    p0_ws = [0.001, 0.001, 0.001]
    p0_ds = [0.001, 0.001, 0.001]

    def __init__(self):
        self.train_ws_x, self.test_ws_x, self.train_ws_y, self.test_ws_y, self.train_ds_x, self.test_ds_x, self.train_ds_y, self.test_ds_y = data_load()
        # data_load()에서 저장한 MinMaxScaler (열 순서: idle_time=0, ws_gap_diff=1, ds_gap_diff=2)
        self.scaler = joblib.load('/Users/joyongjae/Automation/params/scaler.pkl')

    def mse_loss_ws(self, params, x, y):
        a, b, c = params
        y_pred = a * np.exp(b * x) + c
        return np.mean((self.train_ws_y - y_pred) ** 2)

    def mse_loss_ds(self, params, x, y):
        a, b, c = params
        y_pred = a * np.exp(b * x) + c
        return np.mean((self.train_ds_y - y_pred) ** 2)

    def fit_model(self):
        self.result_ws = minimize(self.mse_loss_ws, self.p0_ws, args=(self.train_ws_x, self.train_ws_y), method='L-BFGS-B')
        self.a_ws, self.b_ws, self.c_ws = self.result_ws.x

        self.result_ds = minimize(self.mse_loss_ds, self.p0_ds, args=(self.train_ds_x, self.train_ds_y), method='L-BFGS-B')
        self.a_ds, self.b_ds, self.c_ds = self.result_ds.x

        return self.test_ws_y, self.test_ds_y

    # ---------- test 평가 (WAPE) ----------
    def predict(self, params, x):
        a, b, c = params
        return a * np.exp(b * np.asarray(x, dtype=float)) + c

    def inverse_y(self, y_scaled, col):
        # 0~1 스케일 → 원래 단위 (col: ws=1, ds=2)
        return np.asarray(y_scaled, dtype=float) * self.scaler.data_range_[col] + self.scaler.data_min_[col]

    def evaluate_wape(self):
        ws_pred = self.predict(self.result_ws.x, self.test_ws_x)
        ds_pred = self.predict(self.result_ds.x, self.test_ds_x)

        # WAPE는 원래 단위로 계산 (MinMax의 min 이동 때문에 스케일 값으로 계산하면 달라짐)
        self.wape_ws = wape(self.inverse_y(self.test_ws_y, 1), self.inverse_y(ws_pred, 1))
        self.wape_ds = wape(self.inverse_y(self.test_ds_y, 2), self.inverse_y(ds_pred, 2))

        return self.wape_ws, self.wape_ds

    def print_results(self):
        print('fiting....')
        print('\n')
        print('='*50)
        print('ws_gap_diff result')
        print('='*50)
        print(f'p0(초기값): {self.p0_ws}')
        print(f"a={self.a_ws:.4f}, b={self.b_ws:.4f}, c={self.c_ws:.4f}")
        print(f"MSE: {self.result_ws.fun:.4f}")
        print(f"Test WAPE: {self.wape_ws:.2f}%")

        print('\n')
        print('='*50)
        print('ds_gap_diff result')
        print('='*50)
        print(f'p0(초기값): {self.p0_ds}')
        print(f"a={self.a_ds:.4f}, b={self.b_ds:.4f}, c={self.c_ds:.4f}")
        print(f"MSE: {self.result_ds.fun:.4f}")
        print(f"Test WAPE: {self.wape_ds:.2f}%")

    def save_params(self):
        np.save('/Users/joyongjae/Automation/params/params_ws.npy', self.result_ws.x)
        np.save('/Users/joyongjae/Automation/params/params_ds.npy', self.result_ds.x)


if __name__ == '__main__':
    automation = AutomationSystemCode()
    automation.fit_model()
    automation.evaluate_wape()
    automation.print_results()
    automation.save_params()