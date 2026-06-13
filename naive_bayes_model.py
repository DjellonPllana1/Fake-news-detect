from __future__ import annotations

from sklearn.naive_bayes import MultinomialNB

from common import train_best_candidate


MODEL_NAME = "Naive Bayes"


def candidates() -> list[tuple[str, object]]:
    return [
        ("alpha=0.1", MultinomialNB(alpha=0.1)),
        ("alpha=0.5", MultinomialNB(alpha=0.5)),
        ("alpha=1.0", MultinomialNB(alpha=1.0)),
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
