import matplotlib.pyplot as plt
from data import data_load
import pandas as pd
import numpy as np

data = pd.read_csv("/Users/joyongjae/Automation/TalkFile_generated_data.csv")
train_ws_x, test_ws_x, train_ws_y, test_ws_y, train_ds_x, test_ds_x, train_ds_y, test_ds_y = data_load()

def boxplot_old_data():
    plt.boxplot([data['ws_gap_diff'], data['ds_gap_diff']], labels=['WS Gap Diff', 'DS Gap Diff'])
    plt.ylabel('Gap Diff')
    plt.title('Box Plot of WS Gap Diff and DS Gap Diff')
    plt.savefig('/Users/joyongjae/Automation/BoxPlot.png')
    plt.show()

def plot_scatter_old_data():
    plt.scatter(data['idle_time'], data['ws_gap_diff'], label='WS Gap Diff', color='blue')
    plt.scatter(data['idle_time'], data['ds_gap_diff'], label='DS Gap Diff', color='orange')
    plt.xlabel('Idle Time')
    plt.ylabel('Gap Diff')
    plt.title('Scatter Plot of WS Gap Diff and DS Gap Diff vs Idle Time')
    plt.legend()
    plt.savefig('/Users/joyongjae/Automation/ScatterPlot.png')
    plt.show()

def plot_scatter_scaled_data():
    plt.scatter(train_ws_x, train_ws_y, label='WS Gap Diff', color='blue')
    plt.scatter(train_ds_x, train_ds_y, label='DS Gap Diff', color='orange')
    plt.xlabel('Idle Time')
    plt.ylabel('Gap Diff')
    plt.title('Scatter Plot of WS Gap Diff and DS Gap Diff vs Idle Time')
    plt.legend()
    plt.savefig('/Users/joyongjae/Automation/ScatterPlot_Scaled.png')
    plt.show()

def fitting_model_graph():
    x_range = np.linspace(0, 1, 100)  # 0~1 균일하게 100개 점
    # Fit the model for WS Gap Diff
    a_ws, b_ws, c_ws = np.load('/Users/joyongjae/Automation/params/params_ws.npy', allow_pickle=True)
    y_pred_ws = a_ws * np.exp(b_ws * x_range) + c_ws

    # Fit the model for DS Gap Diff
    a_ds, b_ds, c_ds = np.load('/Users/joyongjae/Automation/params/params_ds.npy', allow_pickle=True)
    y_pred_ds = a_ds * np.exp(b_ds * x_range) + c_ds
    plt.scatter(test_ws_x, test_ws_y, label='WS Gap Diff (Test)', color='blue', alpha=0.5)
    plt.scatter(test_ds_x, test_ds_y, label='DS Gap Diff (Test)', color='orange', alpha=0.5)
    plt.plot(x_range, y_pred_ws, label='Fitted WS Model', color='blue')
    plt.plot(x_range, y_pred_ds, label='Fitted DS Model', color='orange')
    plt.xlabel('Idle Time')
    plt.ylabel('Gap Diff')
    plt.title('Fitted Models for WS and DS Gap Diff vs Idle Time')
    plt.legend()
    plt.savefig('/Users/joyongjae/Automation/FittedModels.png')
    plt.show()

"""plot_scatter_old_data()
plot_scatter_scaled_data()
fitting_model_graph()"""
boxplot_old_data()