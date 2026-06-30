import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

def calc_weights_for_binary(train_ds) -> dict:
    y_list = []
    for _, y in train_ds:
        y_np = y.numpy()
        y_list.append(np.reshape(y_np, (-1,)))  # ★ここがポイント（スカラーでもOK）

    y_train = np.concatenate(y_list, axis=0)

    y_train = y_train[y_train != -1]

    # 0/1 が両方ないと class_weight が不安定なのでチェック推奨
    uniq = set(np.unique(y_train).tolist())
    if uniq != {0, 1}:
        raise ValueError(f"Training labels must contain both 0 and 1. Got {uniq}")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train
    )

    return {0: weights[0], 1: weights[1]}
