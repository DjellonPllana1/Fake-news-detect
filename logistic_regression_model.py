from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from common import RANDOM_STATE, train_best_candidate


MODEL_NAME = "Logistic Regression"


def candidates() -> list[tuple[str, object]]:
    return [
        (
            "C=0.5",
            LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE),
        ),
        (
            "C=1.0",
            LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
        ),
        (
            "C=2.0",
            LogisticRegression(C=2.0, max_iter=1000, random_state=RANDOM_STATE),
        ),
    ]


def train_model(x_train, x_val, x_test, y_train, y_val, y_test):
    return train_best_candidate(
        MODEL_NAME,
        candidates(),
        x_train,)