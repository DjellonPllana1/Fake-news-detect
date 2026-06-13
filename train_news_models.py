from __future__ import annotations

import pandas as pd

from common import (
    RESULTS_DIR,
    load_news_data,
    prepare_output_dirs,
    save_dataset_summary,
    split_data,
    to_markdown_table,
)
from logistic_regression_model import train_model as train_logistic_regression
from naive_bayes_model import train_model as train_naive_bayes
from svm_model import train_model as train_svm


def main() -> None:
    prepare_output_dirs()

    data = load_news_data()
    save_dataset_summary(data)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(data)

    trainers = [
        train_naive_bayes,
        train_logistic_regression,
        train_svm,
    ]

    result_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

    for train_model in trainers:
        result_row, model_tuning_rows = train_model(
            x_train,
            x_val,
            x_test,
            y_train,
            y_val,
            y_test,
        )
        result_rows.append(result_row)
        tuning_rows.extend(model_tuning_rows)

    results = pd.DataFrame(result_rows).sort_values("f1_score", ascending=False)
    tuning = pd.DataFrame(tuning_rows).sort_values(["model", "validation_f1"])

    results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    tuning.to_csv(RESULTS_DIR / "hyperparameter_tuning.csv", index=False)
    (RESULTS_DIR / "model_comparison.md").write_text(
        to_markdown_table(results),
        encoding="utf-8",
    )
    (RESULTS_DIR / "hyperparameter_tuning.md").write_text(
        to_markdown_table(tuning),
        encoding="utf-8",
    )

    print("\nFinal comparison:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
