
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "fake_content_model.pkl"

def main():
    if not MODEL_PATH.exists():
        print("ERROR: fake_content_model.pkl file sapadli nahi.")
        return

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        print("Model load zala nahi.")
        print("Error:", exc)
        return

    samples = [
        "The government announced a new public transport policy today.",
        "BREAKING! Everyone will receive free money tomorrow. Share immediately!",
        "The school announced its examination schedule for next month.",
        "A miracle cure guarantees that every disease disappears overnight."
    ]

    print("\nSMARTVERIFY MODEL TEST")
    print("-" * 40)

    for text in samples:
        try:
            prediction = model.predict([text])[0]
            print("\nText:", text)
            print("Prediction:", prediction)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([text])[0]
                confidence = max(probabilities) * 100
                print(f"Model confidence: {confidence:.2f}%")

        except Exception as exc:
            print("\nPrediction failed for:", text)
            print("Error:", exc)

    print("\nTesting complete.")
    print("Note: Sample predictions do not prove model accuracy.")

if __name__ == "__main__":
    main()
