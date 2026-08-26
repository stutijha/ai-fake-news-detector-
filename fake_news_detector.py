# Fake News Detection using Machine Learning
# Uses a real dataset of ~6,335 news articles labeled REAL or FAKE.
# Same pipeline as sir's movie review example: TF-IDF -> Logistic Regression
# -> accuracy report -> interactive prediction, with cleaner spaced-out output.

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# DATASET
# ============================================================

print("\n")
print("=" * 55)
print("   FAKE NEWS DETECTOR - AI Powered")
print("=" * 55)
print()

data = pd.read_csv("news.csv")

print(f"Loaded {len(data)} articles.")
print()
print("Label breakdown:")
print(data["label"].value_counts().to_string())
print()

# ============================================================
# X AND Y
# ============================================================

X = data["text"]
y = data["label"]

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ============================================================
# MODEL
# ============================================================

print("-" * 55)
print("Training the model... please wait.")
print("-" * 55)
print()

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ============================================================
# PREDICTION + EVALUATION
# ============================================================

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 55)
print(f"   MODEL ACCURACY: {accuracy * 100:.2f}%")
print("=" * 55)
print()

print("Classification Report:")
print("-" * 55)
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print("-" * 55)
print(confusion_matrix(y_test, y_pred))
print()

# ============================================================
# PRETTY PREDICTION DISPLAY
# ============================================================

def show_confidence_bar(confidence):
    # Builds a simple text progress bar like: [██████████░░░░░░░░░░] 52%
    bar_length = 20
    filled = int(confidence / 100 * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"   Confidence: [{bar}] {confidence:.1f}%")

def predict_news(article_text):
    article_tfidf = vectorizer.transform([article_text])
    prediction = model.predict(article_tfidf)[0]
    probability = model.predict_proba(article_tfidf).max() * 100

    print()
    print("-" * 55)
    print(f"   RESULT: {prediction}")
    show_confidence_bar(probability)
    print("-" * 55)

# ============================================================
# USER INPUT
# ============================================================

print("=" * 55)
print("Paste a news article's text below to check if it's REAL or FAKE.")
print("Type 'exit' to stop.")
print("=" * 55)

while True:
    article = input("\nEnter news text (or 'exit' to stop) :: ")

    if article.lower() == "exit":
        print("\nGoodbye!")
        break

    predict_news(article)
