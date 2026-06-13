from __future__ import annotations

from sklearn.svm import LinearSVC

from common import RANDOM_STATE, train_best_candidate


MODEL_NAME = "Linear SVM"


def candidates() -> list[tuple[str, object]]:
    return [
        ("C=0.5", LinearSVC(C=0.5, random_state=RANDOM_STATE)),
        ("C=1.0", LinearSVC(C=1.0, random_state=RANDOM_STATE)),
        ("C=2.0", LinearSVC(C=2.0, random_state=RANDOM_STATE)),
    ]


def train_model(x_train, x_val, x_test, y_train, y_val, y_test):
    return train_best_candidate(
        MODEL_NAME,
        candidates(),
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
    )
