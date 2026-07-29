import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data import data_load

class AutomationSystemCode():
    p0_ws = [0.001, 0.001, 0.001]
    p0_ds = [0.001, 0.001, 0.001]

    def __init__(self):
        self.train_ws_x, self.test_ws_x, self.train_ws_y, self.test_ws_y, self.train_ds_x, self.test_ds_x, self.train_ds_y, self.test_ds_y = data_load()

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

    def print_results(self):
        print('fiting....')
        print('\n')
        print('='*50)
        print('ws_gap_diff result')
        print('='*50)
        print(f'p0(초기값): {self.p0_ws}')
        print(f"a={self.a_ws:.4f}, b={self.b_ws:.4f}, c={self.c_ws:.4f}")
        print(f"MSE: {self.result_ws.fun:.4f}")

        print('\n')
        print('='*50)
        print('ds_gap_diff result')
        print('='*50)
        print(f'p0(초기값): {self.p0_ds}')
        print(f"a={self.a_ds:.4f}, b={self.b_ds:.4f}, c={self.c_ds:.4f}")
        print(f"MSE: {self.result_ds.fun:.4f}")

    def save_params(self):
        np.save('/Users/joyongjae/Automation/params/params_ws.npy', self.result_ws.x)
        np.save('/Users/joyongjae/Automation/params/params_ds.npy', self.result_ds.x)


if __name__ == '__main__':
    automation = AutomationSystemCode()
    automation.fit_model()
    automation.print_results()
    automation.save_params()