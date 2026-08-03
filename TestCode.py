import numpy as np
import pandas as pd
from scipy.optimize import minimize
from data import data_load
import joblib

class TestCode():
    def __init__(self):
        self.train_ws_x, self.test_ws_x, self.train_ws_y, self.test_ws_y, self.train_ds_x, self.test_ds_x, self.train_ds_y, self.test_ds_y = data_load()
        self.ds_params = np.load('/Users/joyongjae/Automation/params/params_ds.npy', allow_pickle=True)
        self.ws_params = np.load('/Users/joyongjae/Automation/params/params_ws.npy', allow_pickle=True)
        self.scaler = joblib.load('/Users/joyongjae/Automation/params/scaler.pkl')
        
    def mse_loss_ws(self, params, x, y):
        a, b, c = params
        y_pred = a * np.exp(b * x) + c

        return np.mean((self.test_ws_y - y_pred) ** 2)

    def mse_loss_ds(self, params, x, y):
        a, b, c = params
        y_pred = a * np.exp(b * x) + c

        return np.mean((self.test_ds_y - y_pred) ** 2)

    def evaluate_models(self):
        ws_model = self.mse_loss_ws(self.ws_params, self.test_ws_x, self.test_ws_y)
        ds_model = self.mse_loss_ds(self.ds_params, self.test_ds_x, self.test_ds_y)

        return ws_model, ds_model

    def print_results(self, ws_model, ds_model):
        print('\n')
        print('Evaluating models...')
        print('\n')
        print('='*50)
        print('result')
        print('='*50)
        print('\n')
        print('='*50)
        print('ws_gap_diff model')
        print('='*50)
        print(f"ws params: {self.ws_params}")
        print(f"MSE: {ws_model:.4f}")
        print('\n')
        print('='*50)
        print('ds_gap_diff model')
        print('='*50)
        print(f"ds params: {self.ds_params}")
        print(f"MSE: {ds_model:.4f}")
        print('\n')

if __name__ == '__main__':
    test = TestCode()
    ws_model, ds_model = test.evaluate_models()
    test.print_results(ws_model, ds_model)