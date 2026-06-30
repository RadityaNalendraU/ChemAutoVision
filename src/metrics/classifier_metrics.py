from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def create_roc_curve(y_score: np.array, y_test: List[int], model_name: str) -> str:
    fprs, tprs, _ = roc_curve(y_test, y_score)

    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fprs, tprs, label=model_name)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    file_path = f"../mlruns/images/{datetime.now().strftime('%Y%m%d%H%M%S')}_{model_name}_roc_curve.png"
    plt.savefig(file_path)
    plt.close()
    return file_path
