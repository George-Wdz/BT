"""
回归任务评估指标：仅保留 MAE 和 MSE。
"""
import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def metric(pred, true):
    """返回 (mae, mse)"""
    return MAE(pred, true), MSE(pred, true)
