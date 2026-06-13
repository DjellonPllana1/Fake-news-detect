from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
RANDOM_STATE = 42
MAX_ROWS_PER_CLASS = 7000


def prepare_output_dirs() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def to_markdown_table(dataframe: pd.DataFrame) -> str:
    headers = [str(column) for column in dataframe.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in dataframe.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_news_data() -> pd.DataFrame:
    fake = pd.read_csv(BASE_DIR / "Fake.csv")
    true = pd.read_csv(BASE_DIR / "True.csv")

    fake["label"] = "fake"
    true["label"] = "true"

    data = pd.concat([fake, true], ignore_index=True)
    data = data.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)

    for column in ["title", "text", "subject"]:
        data[column] = data[column].fillna("")

    data["content"] = (
        data["title"].str.strip()
        + " "
        + data["subject"].str.strip()
        + " "
        + data["text"].str.strip()
    )
    data["target"] = data["label"].map({"fake": 0, "true": 1})

    sampled_parts = []
    for _, part in data.groupby("target"):
        sampled_parts.append(
            part.sample(
                n=min(MAX_ROWS_PER_CLASS, len(part)),
                random_state=RANDOM_STATE,
            )
        )

    return (
        pd.concat(sampled_parts, ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def save_dataset_summary(data: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        {
            "metric": [
                "rows_used_after_balanced_sampling",
                "fake_news",
                "true_news",
                "empty_text_rows",
            ],
            "value": [
                len(data),
                int((data["label"] == "fake").sum()),
                int((data["label"] == "true").sum()),
                int((data["content"].str.strip() == "").sum()),
            ],
        }
    )
    summary.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)


def split_data(data: pd.DataFrame):
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        data["content"],
        data["target"],
        test_size=0.2,
        stratify=data["target"],
        random_state=RANDOM_STATE,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.2,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def make_pipeline(model: object) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=10000,
                    ngram_range=(1, 1),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("model", model),
        ]
    )


def save_confusion_matrix(y_true: pd.Series, y_pred, model_name: str) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["fake", "true"],
        yticklabels=["fake", "true"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    filename = model_name.lower().replace(" ", "_")
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{filename}.png", dpi=180)
    plt.close()


def train_best_candidate(
    model_name: str,
    candidates: list[tuple[str, object]],
    x_train,
    x_val,
    x_test,
    y_train,
    y_val,
    y_test,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    print(f"\nTraining {model_name}...")
    best_f1 = -1.0
    best_label = ""
    best_pipeline: Pipeline | None = None
    tuning_rows: list[dict[str, object]] = []

    for candidate_label, model in candidates:
        pipeline = make_pipeline(model)
        pipeline.fit(x_train, y_train)
        val_predictions = pipeline.predict(x_val)
        val_f1 = f1_score(y_val, val_predictions)
        tuning_rows.append(
            {
                "model": model_name,
                "candidate": candidate_label,
                "validation_f1": val_f1,
            }
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_label = candidate_label
            best_pipeline = pipeline

    assert best_pipeline is not None
    test_predictions = best_pipeline.predict(x_test)
    report = classification_report(
        y_test,
        test_predictions,
        target_names=["fake", "true"],
        output_dict=True,
    )
    report_path = RESULTS_DIR / (
        f"classification_report_{model_name.lower().replace(' ', '_')}.csv"
    )
    pd.DataFrame(report).transpose().to_csv(report_path)
    save_confusion_matrix(y_test, test_predictions, model_name)

    result_row = {
        "model": model_name,
        "selected_params": best_label,
        "validation_f1": best_f1,
        "accuracy": accuracy_score(y_test, test_predictions),
        "precision": precision_score(y_test, test_predictions),
        "recall": recall_score(y_test, test_predictions),
        "f1_score": f1_score(y_test, test_predictions),
    }
    return result_row, tuning_rows
