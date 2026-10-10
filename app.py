
from flask import Flask, render_template, request
import joblib
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fake_content_model.pkl"

# Predictions below this confidence are shown as uncertain.
CONFIDENCE_THRESHOLD = 60.0

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
    print("Model classes:", getattr(model, "classes_", "Not available"))
except Exception as error:
    model = None
    print("Model loading error:", repr(error))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/fake-detector", methods=["GET", "POST"])
def fake_detector():
    result = None
    confidence = None
    warning = None
    content = ""

    if request.method == "POST":
        content = request.form.get("content", "").strip()
        word_count = len(content.split())

        if not content:
            warning = "Please enter a news article."

        elif word_count < 20:
            warning = (
                f"Your text has only {word_count} words. "
                "Please enter the full article with at least 20 words."
            )

        elif model is None:
            warning = (
                "The model could not be loaded. "
                "Check that fake_content_model.pkl exists."
            )

        else:
            try:
                prediction = model.predict([content])[0]
                label = str(prediction).strip().upper()

                print("RAW MODEL PREDICTION:", repr(prediction))

                classes = list(model.classes_)
                probabilities = None

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([content])[0]
                    predicted_index = classes.index(prediction)
                    confidence = round(
                        float(probabilities[predicted_index]) * 100, 2
                    )

                # Handle uncertain predictions before displaying a label.
                if (
                    confidence is not None
                    and confidence < CONFIDENCE_THRESHOLD
                ):
                    result = "Uncertain - Needs Verification"
                    warning = (
                        f"The model confidence is {confidence:.2f}%, "
                        "which is below the 60% display threshold. "
                        "The article cannot be reliably classified by "
                        "this model. Verify it using trusted sources."
                    )

                elif label == "FAKE":
                    result = "Potentially Fake Content"
                    warning = (
                        "This is an ML model prediction, not verified "
                        "fact-checking. Check reliable sources before "
                        "deciding whether the article is false."
                    )

                elif label == "REAL":
                    result = "Likely Genuine Content"
                    warning = (
                        "This prediction does not prove that the article "
                        "is true. Verify it using reliable sources."
                    )

                else:
                    result = "Unable to classify content"
                    warning = (
                        "The model returned an unrecognized label. "
                        "Check the model training labels."
                    )

                print("DISPLAY RESULT:", result)
                print("CONFIDENCE:", confidence)

            except Exception as error:
                result = None
                confidence = None
                warning = (
                    "An error occurred while analyzing the article. "
                    "Please try again and check the server logs."
                )
                print("Prediction error:", repr(error))

    return render_template(
        "fake_detector.html",
        result=result,
        confidence=confidence,
        warning=warning,
        content=content
    )


@app.route("/product-recommendation", methods=["GET", "POST"])
def product_recommendation():
    recommendations = []
    selected_category = ""
    search_query = ""

    products = {
        "electronics": [
            {"name": "Wireless Earphones", "price": "₹1,499", "rating": "4.4"},
            {"name": "Bluetooth Speaker", "price": "₹1,999", "rating": "4.5"},
            {"name": "Smart Watch", "price": "₹2,499", "rating": "4.3"}
        ],
        "fitness": [
            {"name": "Yoga Mat", "price": "₹799", "rating": "4.5"},
            {"name": "Resistance Bands", "price": "₹599", "rating": "4.4"},
            {"name": "Dumbbells", "price": "₹1,299", "rating": "4.6"}
        ],
        "books": [
            {"name": "Python Programming Book", "price": "₹499", "rating": "4.7"},
            {"name": "AI & ML Book", "price": "₹699", "rating": "4.6"},
            {"name": "Data Science Book", "price": "₹599", "rating": "4.5"}
        ],
        "fashion": [
            {"name": "T-Shirt", "price": "₹699", "rating": "4.3"},
            {"name": "Jeans", "price": "₹1,299", "rating": "4.4"},
            {"name": "Sneakers", "price": "₹1,999", "rating": "4.5"}
        ],
        "home": [
            {"name": "Mixer Grinder", "price": "₹2,499", "rating": "4.4"},
            {"name": "Electric Kettle", "price": "₹999", "rating": "4.5"},
            {"name": "Storage Organizer", "price": "₹499", "rating": "4.2"}
        ]
    }

    if request.method == "POST":
        selected_category = request.form.get(
            "category", ""
        ).strip().lower()

        search_query = request.form.get(
            "search", ""
        ).strip().lower()

        category_products = products.get(selected_category, [])

        if search_query:
            recommendations = [
                product for product in category_products
                if search_query in product["name"].lower()
            ]
        else:
            recommendations = category_products

    return render_template(
        "product_recommendation.html",
        recommendations=recommendations,
        selected_category=selected_category,
        search_query=search_query
    )


if __name__ == "__main__":
    app.run(debug=True)
