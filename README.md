
# SmartVerify
### Fake Content Detection and Intelligent Product Recommendation System

SmartVerify is a Python Flask web application that combines machine learning-based content classification with a category-based product recommendation system.

## Features

- **Fake Content Detection:** Predicts whether submitted content is likely genuine based on a trained machine learning model.
- **Model Confidence:** Displays the model's prediction confidence when available.
- **Product Recommendation:** Displays products based on the selected category.
- **Web Interface:** Provides a simple interface for interacting with both features.
- **Input Handling:** Accepts user-submitted content for analysis.

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- JavaScript

## Project Structure

```text
FakeContentRecommendation/
├── app.py
├── train_model.py
├── fake_content_model.pkl
├── requirements.txt
├── data/
│   └── news.csv
└── templates/
    ├── index.html
    ├── fake_detector.html
    └── product_recommendation.html
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/omkalkar/SmartVerify.git
cd SmartVerify
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment on Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

Open this address in your browser:

http://127.0.0.1:5000

## Important Note

The fake content detector provides a machine learning prediction, not a guarantee that content is true or false. Always verify important claims with reliable sources.

## Future Improvements

- Improve model accuracy with a suitable, verified dataset.
- Add more product categories and improve recommendations.
- Add automated testing and stronger input validation.
- Deploy the application online.

## Author

Om Kalkar
