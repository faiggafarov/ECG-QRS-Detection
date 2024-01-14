import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[TP, FP], [FN, TN]])

def evaluate_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    precision = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    f1_score = 2 * precision * sensitivity / (precision + sensitivity)

    print("Confusion Matrix:")
    print(cm)
    print("\nAccuracy:", accuracy)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Precision:", precision)
    print("F1 Score:", f1_score)

def rms_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def auc_roc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)