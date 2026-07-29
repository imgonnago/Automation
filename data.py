import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def data_load():
    data = pd.read_csv("/Users/joyongjae/Automation/TalkFile_generated_data.csv")

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

    train_ws_x, test_ws_x, train_ws_y, test_ws_y = train_test_split(x, ws_gap_diff, test_size=0.2, random_state=42)

    train_ds_x, test_ds_x, train_ds_y, test_ds_y = train_test_split(x, ds_gap_diff, test_size=0.2, random_state=42)

    return train_ws_x, test_ws_x, train_ws_y, test_ws_y, train_ds_x, test_ds_x, train_ds_y, test_ds_y  


"""data = pd.read_csv("/Users/joyongjae/Automation/TalkFile_generated data.csv")
scaler = MinMaxScaler()
data_scaler = scaler.fit_transform(data)
data_scaler = pd.DataFrame(data_scaler, columns=['ws_gap_diff', 'ds_gap_diff','idle_time'])
plt.scatter(x = data_scaler["idle_time"], y = data_scaler["ws_gap_diff"])
plt.xlabel("idel_time")
plt.ylabel("ws_gap_diff")
plt.show()
"""
