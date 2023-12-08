import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def np_rrmse(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.float64:
    squared_diffs = np.square(Y_true - Y_pred)
    row_means = squared_diffs.mean(axis=1)
    return np.sqrt(row_means)


def np_mrrmse(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.float64:
    root_means = np_rrmse(Y_true, Y_pred)
    return root_means.mean()


def mrrmse(y_true, y_pred):
    squared_diffs = K.square(y_true - y_pred)
    row_means = K.mean(squared_diffs, axis=1)
    sq = K.sqrt(row_means)
    return K.mean(sq)
