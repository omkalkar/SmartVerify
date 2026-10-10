from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "fake_content_model.pkl"


def combine_text(df):
    """Combine available title and article text columns."""
    title = (
        df["title"].fillna("").astype(str)
        if "title" in df.columns
        else pd.Series("", index=df.index)
    )

    body = (
        df["text"].fillna("").astype(str)
        if "text" in df.columns
        else pd.Series("", index=df.index)
    )

    return (title.str.strip() + " " + body.str.strip()).str.strip()


def main():
    fake_path = DATA_DIR / "Fake.csv"
    real_path = DATA_DIR / "True.csv"

    if not fake_path.exists() or not real_path.exists():
        print("ERROR: data folder madhye Fake.csv ani True.csv pahijet.")
        return

    print("Loading dataset...")

    fake_data = pd.read_csv(fake_path)
    real_data = pd.read_csv(real_path)

    print("Fake.csv columns:", list(fake_data.columns))
    print("True.csv columns:", list(real_data.columns))

    fake = pd.DataFrame({
        "text": combine_text(fake_data),
        "label": "FAKE"
    })

    real = pd.DataFrame({
        "text": combine_text(real_data),
        "label": "REAL"
    })

    data = pd.concat([fake, real], ignore_index=True)

    # Remove empty articles and exact duplicate text.
    data["text"] = (
        data["text"].fillna("").astype(str).str.strip()
    )
    data = data[data["text"].str.len() > 0]
    data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("\nTotal usable articles:", len(data))
    print("\nLabel counts:")
    print(data["label"].value_counts())

    if data["label"].nunique() < 2:
        print("ERROR: FAKE ani REAL donhi labels avashyak aahet.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"]
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=100000,
                stop_words="english",
                min_df=2,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy:.2%}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["FAKE", "REAL"],
            zero_division=0
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["FAKE", "REAL"]
        )
    )

    joblib.dump(model, MODEL_PATH)
    print("\nModel saved successfully:", MODEL_PATH)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()