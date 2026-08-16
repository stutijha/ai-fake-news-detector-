# Fake News Detection using Machine Learning

A machine learning model that classifies news articles as REAL or FAKE,
using TF-IDF text vectorization and Logistic Regression.

## Dataset

This project uses the ["Fake or Real News"](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news)
dataset from Kaggle — about 6,335 news articles labeled REAL or FAKE.

To run this project yourself:
1. Download the CSV from the Kaggle link above.
2. Rename it to `news.csv` (if it isn't already) and place it in the same
   folder as `fake_news_detector.py`.

## How it works

1. The dataset is loaded using `pandas`.
2. The article text is converted into numeric vectors using `TfidfVectorizer`,
   which scores words by how important/unique they are, while ignoring
   common filler words like "the" and "is".
3. The data is split into a training set (80%) and test set (20%).
4. A `LogisticRegression` model is trained to recognize word patterns that
   separate real news from fake news.
5. Accuracy, a classification report, and a confusion matrix show how well
   the model performs on articles it hasn't seen before.
6. You can then paste in any news text and get a live REAL/FAKE prediction
   with a confidence score.

## Requirements

```
pip install pandas scikit-learn
```

## How to run

```
python fake_news_detector.py
```

## Example

```
Enter news text (or 'exit' to stop): Scientists confirm the earth is flat and NASA has been lying for decades

Prediction: FAKE
Confidence: 87.42 %
```

## Future improvements

- Combine title + text for potentially better accuracy
- Try other models (Naive Bayes, Random Forest) and compare results
- Build a simple web interface with Flask or Streamlit
