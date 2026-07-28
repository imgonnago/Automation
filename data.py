import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def data_load():
    data = pd.read_csv("/Users/joyongjae/Automation/TalkFile_generated data.csv")

    scaler = MinMaxScaler()

    data_scaler = scaler.fit_transform(data)

    data_scaler = pd.DataFrame(data_scaler, columns=['ws_gap_diff', 'ds_gap_diff','idle_time'])

    print(data_scaler.info())
    print('='*50)
    print(data_scaler.describe())
    print('='*50)
    print(data_scaler.head())

    ws_gap_diff = data_scaler["ws_gap_diff"]
    ds_gap_diff = data_scaler["ds_gap_diff"]
    x = data_scaler["idle_time"]

    return ws_gap_diff, ds_gap_diff, x


"""data = pd.read_csv("/Users/joyongjae/Automation/TalkFile_generated data.csv")
scaler = MinMaxScaler()
data_scaler = scaler.fit_transform(data)
data_scaler = pd.DataFrame(data_scaler, columns=['ws_gap_diff', 'ds_gap_diff','idle_time'])
plt.scatter(x = data_scaler["idle_time"], y = data_scaler["ws_gap_diff"])
plt.xlabel("idel_time")
plt.ylabel("ws_gap_diff")
plt.show()
"""
