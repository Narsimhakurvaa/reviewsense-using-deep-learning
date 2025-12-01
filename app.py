from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

print("📥 Loading model + vectorizer...")
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("✅ Loaded.")


def predict_sentiment(text):
    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0][1]
    label = "Positive" if prob >= 0.5 else "Negative"
    confidence = round((prob if prob >= 0.5 else 1 - prob) * 100, 2)
    return label, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    user_text = ""
    prediction = None
    confidence = None

    if request.method == "POST":
        user_text = request.form.get("review", "")
        if user_text.strip():
            prediction, confidence = predict_sentiment(user_text)

    return render_template("index.html",
                           user_text=user_text,
                           prediction=prediction,
                           confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
