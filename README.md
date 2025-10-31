Truth Detector — Fake News Detection System

An intelligent fake news detection system that uses Natural Language Processing (NLP) and Machine Learning to classify news content as FAKE or REAL.
Built using Python, spaCy, NLTK, TF-IDF, and Logistic Regression, this model analyzes news text, performs linguistic preprocessing, and predicts authenticity with confidence scoring.


---

✅ Features

🧠 Machine Learning–based Fake News Classification

📊 TF-IDF vectorization + Logistic Regression model

🔍 Advanced text preprocessing with spaCy + NLTK

🧾 Supports custom news/content input

📈 Evaluation metrics: Accuracy, Precision, Recall, F1-Score

💾 Saves trained model (.pkl) and vectorizer

✅ Confidence score shown for predictions

🏗️ Modular & extendable codebase



---

🏛️ System Architecture

Raw News Text → Preprocessing → TF-IDF Vectorizer → ML Model → Fake/Real + Confidence Score


---

📂 Project Structure

truth-detector/
│── main.py
│── data/
│     └── news.csv
│── models/
│     └── model.pkl
│── vectorizer/
│     └── tfidf.pkl
│── README.md


---

🔧 Installation & Setup

1️⃣ Clone the repo

git clone https://github.com/yourusername/truth-detector.git
cd truth-detector

2️⃣ Install dependencies

pip install pandas numpy nltk spacy scikit-learn joblib

3️⃣ Download spaCy model

python -m spacy download en_core_web_sm

4️⃣ Run the project

python main.py


---

🧠 Dataset Format

Your data/news.csv file should have these columns:

text	label

Government launches new AI policy…	REAL
Breaking! Alien life confirmed…	FAKE


Labels accepted: FAKE or REAL


---

📊 Model Performance (example)

Metric	Score

Accuracy	0.94
Precision	0.92
Recall	0.93
F1-Score	0.925



---

📥 Sample Output

[FAKE NEWS DETECTED]
Prediction Confidence: 0.9462

OR

[FACTUAL / REAL NEWS]
Prediction Confidence: 0.8714


---

🧩 Future Enhancements

Feature	Description

🌐 Web UI	Using Flask/Django/Streamlit
📊 Dashboard	Model insights & charts
📱 Mobile App	Detect fake news on phone
🧠 Deep Learning	LSTM / BERT / transformers
🔗 URL Scanner	Check links for misinformation



---

🛟 Troubleshooting

Issue	Fix

NLTK stopwords error	nltk.download('stopwords')
spaCy model missing	python -m spacy download en_core_web_sm



---

📜 License

This project is open-source for educational and research purposes.
Feel free to modify and extend!


---

👨‍💻 Author

Truth Detector Research Project
Fake News Analysis using NLP & ML
