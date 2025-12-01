Sentiment Analyzer for Movies - ReviewSense IMDB Top 1000
 A Deep Learning (MLP Neural Network) Project for Text Sentiment Classification

ReviewSense is a movie sentiment prediction system trained on the IMDB Top 1000 dataset. It uses movie plot descriptions to classify sentiment as Positive or Negative.

It includes a neural network model (MLPClassifier), TF-IDF vectorization, and a fully functional Flask web application.

 Features:
Sentiment classification using a neural network (MLP)
✔ Trained on movie overviews from the IMDB Top 1000 dataset
✔️ Automatic label generation based on IMDB ratings
✔️ TF-IDF text vectorization
✔️ Clean UI with Bootstrap
✔️ 100% compatible with macOS M1/M2 & Python 3.10+
✔️ No TensorFlow required

PROJECT STRUCTURE :

reviewsense/
│
├── train_model.py          # Model training script (MLP Neural Network)
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── imdb_top_1000.csv       # Dataset used for training
│
├── templates/
│   ├── base.html           # Layout template
│   └── index.html          # Main UI
│
└── static/
    └── style.css           # Custom CSS
————————————————————————————————————

Dataset Description:
Dataset used: IMDB Top 1000 Movies
File: imdb_top_1000.csv
Used columns:

Column	Description
Overview	Short plot summary used for text sentiment analysis
IMDB_Rating	Numeric rating used to create labels
label	1 = Positive (rating ≥ 8.0) 0 = Negative (rating < 8.0)
—————————————————————————————————————
Model Details

We use: MLPClassifier (Multi-Layer Perceptron Neural Network) :

Hidden Layers: (128, 64)

Activation: ReLU

Optimizer: Adam

Loss: Cross-Entropy

Vectorization: TF-IDF, with:

10,000 max features

1–2 n-grams

English stopwords removed

The trained model is saved as:

sentiment_model.pkl

vectorizer.pkl
—————————————————————————————————————
Training the Model:

Run:


pip3 install -r requirements.txt

python3 train_model.py

Expected output:

Training progress logs

Accuracy score

Sentiment_model.pkl and vectorizer.pkl created


Web App - Run 


Run:

python3 app.py


Then open in browser:

http://127.0.0.1:5000

Enter any movie-style text, for example:

"A beautiful emotional journey with powerful performances."

Your result:

Prediction: Positive 

Confidence: 93.4%

Area	Technology
Backend	Flask
ML Model	Multi-Layer Perceptron (scikit-learn)
NLP	TF-IDF Vectorization
Frontend	HTML + CSS + Bootstrap
Dataset	IMDB Top 1000
Language	Python

Installation:
Clone the repository:

git clone https://github.com/narsimhakurvaa/reviewsense using deep learning.git
cd reviewsense
Install dependencies:

pip3 install -r requirements.txt
Train model:

python3 train_model.py
Run app:

python3 app.py
—————————————————————————————————————
How Sentiment is Determined
To convert ratings → sentiment:

IMDB Rating ≥ 8.0 → Positive (1)
IMDB Rating < 8.0 → Negative (0)
This allows you to train a real sentiment classifier using plot descriptions.

- Future Improvements
🔹 Add movie recommendation engine 🔹 Improve model accuracy using BERT / DistilBERT 🔹 Add real-time movie review scraping 🔹 Deploy on Render / Railway 🔹 Add user accounts & history
