
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "fake_content_model.pkl"

FAKE_PATH = DATA_DIR / "Fake.csv"
REAL_PATH = DATA_DIR / "True.csv"


def load_articles(path, label):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    required = {"title", "text"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{path.name} must contain title and text columns."
        )

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["content"] = (
        df["title"].str.strip() + " " + df["text"].str.strip()
    ).str.strip()

    df = df[df["content"].str.split().str.len() >= 20].copy()
    df["label"] = label

    return df[["content", "label"]]


def main():
    print("Loading dataset...")

    fake_df = load_articles(FAKE_PATH, "FAKE")
    real_df = load_articles(REAL_PATH, "REAL")

    data = pd.concat([fake_df, real_df], ignore_index=True)
    data = data.drop_duplicates(subset=["content"]).reset_index(drop=True)

    print("Usable articles:", len(data))
    print("\nLabel counts:")
    print(data["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        data["content"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
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
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nHeld-out test accuracy:", round(
        accuracy_score(y_test, predictions) * 100, 2
    ), "%")

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(
        y_test, predictions, labels=["FAKE", "REAL"]
    ))

    joblib.dump(model, MODEL_PATH)
    print("\nModel saved successfully:", MODEL_PATH)

    print("\nPrediction examples (sanity check only):")
    examples = [
        "BREAKING: Share this message now to receive guaranteed free money. "
        "No application or official approval is required.",
        "The local education department announced that schools will conduct "
        "their annual examinations next month. School authorities said the "
        "detailed timetable would be shared with students and parents through "
        "official notices. Students have been advised to contact their schools "
        "for updates about examination dates, subjects, and reporting times.",
    ]

    for text in examples:
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        confidence = max(probabilities) * 100

        print(f"\nPrediction: {prediction}")
        print(f"Model confidence: {confidence:.2f}%")
        print("Text:", text)

    print(
        "\nImportant: These example predictions do not prove factual truth. "
        "Verify real news with reliable sources."
    )


if __name__ == "__main__":
    main()
