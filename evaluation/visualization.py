from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

def save_confusion_matrix(matrix, output_path):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(matrix)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10): ax.text(j, i, str(matrix[i,j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(output_path, dpi=160); plt.close(fig)
