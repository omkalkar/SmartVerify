from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("fake_content_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/fake-detector", methods=["GET", "POST"])
def fake_detector():

    result = None
    confidence = None

    if request.method == "POST":

        content = request.form["content"]
        prediction = model.predict([content])[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba([content])[0].max()
            confidence = round(probability * 100, 2)

        if prediction == "FAKE":
            result = "Potentially Fake Content"
        else:
            result = "Likely Genuine Content"

    return render_template(
        "fake_detector.html",
        result=result,
        confidence=confidence
    )


@app.route("/product-recommendation", methods=["GET", "POST"])
def product_recommendation():

    recommendations = []
    selected_category = ""

    if request.method == "POST":

        selected_category = request.form.get("category", "")

        products = {
            "electronics": [
                {
                    "name": "Wireless Earphones",
                    "price": "₹1,499",
                    "rating": "4.4"
                },
                {
                    "name": "Bluetooth Speaker",
                    "price": "₹1,999",
                    "rating": "4.5"
                },
                {
                    "name": "Smart Watch",
                    "price": "₹2,499",
                    "rating": "4.3"
                }
            ],

            "fitness": [
                {
                    "name": "Yoga Mat",
                    "price": "₹799",
                    "rating": "4.5"
                },
                {
                    "name": "Resistance Bands",
                    "price": "₹599",
                    "rating": "4.4"
                },
                {
                    "name": "Dumbbells",
                    "price": "₹1,299",
                    "rating": "4.6"
                }
            ],

            "books": [
                {
                    "name": "Python Programming Book",
                    "price": "₹499",
                    "rating": "4.7"
                },
                {
                    "name": "AI & ML Book",
                    "price": "₹699",
                    "rating": "4.6"
                },
                {
                    "name": "Data Science Book",
                    "price": "₹599",
                    "rating": "4.5"
                }
            ],

            "fashion": [
                {
                    "name": "T-Shirt",
                    "price": "₹699",
                    "rating": "4.3"
                },
                {
                    "name": "Jeans",
                    "price": "₹1,299",
                    "rating": "4.4"
                },
                {
                    "name": "Sneakers",
                    "price": "₹1,999",
                    "rating": "4.5"
                }
            ],

            "home": [
                {
                    "name": "Mixer Grinder",
                    "price": "₹2,499",
                    "rating": "4.4"
                },
                {
                    "name": "Electric Kettle",
                    "price": "₹999",
                    "rating": "4.5"
                },
                {
                    "name": "Storage Organizer",
                    "price": "₹499",
                    "rating": "4.2"
                }
            ]
        }

        recommendations = products.get(selected_category, [])

    return render_template(
        "product_recommendation.html",
        recommendations=recommendations,
        selected_category=selected_category
    )


if __name__ == "__main__":
    app.run(debug=True)