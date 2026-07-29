import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data import data_load

train_ws_x, test_ws_x, train_ws_y, test_ws_y, train_ds_x, test_ds_x, train_ds_y, test_ds_y = data_load()

def mse_loss_ws(params,x,y):
    a, b, c = params
    y_pred = a * np.exp(b * x) + c
    return np.mean((train_ws_y - y_pred) ** 2)

def mse_loss_ds(params,x,y):
    a, b, c = params
    y_pred = a * np.exp(b * x) + c
    return np.mean((train_ds_y - y_pred) ** 2)

p0 = [0.001, 0.001, 0.001]

result_ws = minimize(mse_loss_ws, p0, args=(train_ws_x, train_ws_y), method='L-BFGS-B')
a_ws, b_ws, c_ws = result_ws.x

result_ds = minimize(mse_loss_ds, p0, args=(train_ds_x, train_ds_y), method='L-BFGS-B')
a_ds, b_ds, c_ds = result_ds.x

print('fiting....')
print('\n')
print('='*50)
print('ws_gap_diff result')
print('='*50)
print(f'p0(초기값): {p0}')
print(f"a={a_ws:.4f}, b={b_ws:.4f}, c={c_ws:.4f}")
print(f"MSE: {result_ws.fun:.4f}")

print('\n')
print('='*50)
print('ds_gap_diff result')
print('='*50)
print(f'p0(초기값): {p0}')
print(f"a={a_ds:.4f}, b={b_ds:.4f}, c={c_ds:.4f}")
print(f"MSE: {result_ds.fun:.4f}")

np.save('params_ws.npy',result_ws)
np.save('params_ds.npy',result_ds)
print('============params saved=============')
