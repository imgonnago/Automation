import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data import data_load

ws_y, ds_y, x = data_load()

def mse_loss(params,x,y):
    a, b, c = params
    y_pred = a * np.exp(b * x) + c
    return np.mean((ws_y - y_pred) ** 2)

p0 = [0.1, 0.01, 0.1]

result = minimize(mse_loss, p0, args=(x, ws_y), method='L-BFGS-B')
a, b, c = result.x

print(f"a={a:}, b={b}, c={c}")
print(f"MSE: {result.fun}")
