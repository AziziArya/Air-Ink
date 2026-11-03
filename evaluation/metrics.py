import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def classification_metrics(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_f1": float(f1_score(y_true, y_pred, average="macro")), "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0), "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(10))).tolist()}
