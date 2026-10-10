
from flask import Flask, render_template, request
import joblib
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fake_content_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as error:
    model = None
    print("Model loading error:", error)


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
            warning = "Please enter at least 20 words for analysis."

        elif model is None:
            warning = "Model could not be loaded. Check fake_content_model.pkl."

        else:
            try:
                probabilities = None
                prediction = model.predict([content])[0]

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([content])[0]
                    classes = list(model.classes_)
                    predicted_index = classes.index(prediction)
                    confidence = round(
                        float(probabilities[predicted_index]) * 100, 2
                    )

                if str(prediction).upper() == "FAKE":
                    result = "Potentially Fake Content"
                elif str(prediction).upper() == "REAL":
                    result = "Likely Genuine Content"
                else:
                    result = "Unable to classify content"

                # A low model score means the prediction needs extra caution.
                if confidence is not None and confidence < 70:
                    warning = (
                        "The model is uncertain about this prediction. "
                        "Check reliable sources before deciding whether "
                        "the news is true or false."
                    )
                else:
                    warning = (
                        "This is only an ML model prediction, not verified "
                        "fact-checking. Confirm the article using reliable "
                        "official sources before sharing."
                    )

                print("\n--- SMARTVERIFY RESULT ---")
                print("Prediction:", prediction)
                print("Confidence:", confidence)

            except Exception as error:
                warning = "An error occurred while analyzing the article."
                print("Prediction error:", error)

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
