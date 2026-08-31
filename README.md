✿ ⋆｡°✩ *.✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿ ⋆｡°✩

# 🌸 Fake News Detector 🌸
### *!! a smart little AI that finds out whether a news article title is fake or not !!* 🕵️‍♀️💖

✿ ⋆｡°✩ *.✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿ ⋆｡°✩

A machine learning model that classifies the entered news articles as **REAL** 🌷 or
**FAKE** 🥀, using TF-IDF text vectorization and Logistic Regression — with
a cute little confidence percentage bar to show how sure it is.

<br>

## 🌷 Dataset ::

This project uses the ["Fake or Real News"](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news)
dataset from Kaggle — about 6,335 news articles labeled REAL or FAKE.

**To run this project yourself:**
1. 🌸 Download the CSV from the Kaggle link above
2. 🌸 Rename it to `news.csv` (if it isn't already)
3. 🌸 Place it in the same folder as `fake_news_detector.py`

<br>

✿ ⋆｡°✩ *.✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿ ⋆｡°✩

## 🌼 How it works !!

1. 💮 The dataset is loaded using `pandas`
2. 💮 Article text is converted into numeric vectors using `TfidfVectorizer`,
   which scores words by how important/unique they are, while ignoring
   common filler words like "the" and "is"
3. 💮 The data is split into a training set (80%) and the test set (20%)
4. 💮 A `LogisticRegression` model is trained to recognize word patterns
   that separates real news from fake news
5. 💮 Accuracy, a classification report, and a confusion matrix show how
   well the model performs on articles it hasn't seen before
6. 💮 You can then paste in any news text and get a live REAL/FAKE
   prediction with a pretty confidence bar 🎀

<br>

✿ ⋆｡°✩ *.✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿ ⋆｡°✩

## 🌸 Requirements !!

```
pip install pandas scikit-learn
```

## 🌸 How to run

```
python fake_news_detector.py
```

<br>

## 🎀 Example

```
Enter news text (or 'exit' to stop): monster was seen eating a mango popsicle

-------------------------------------------------------
   RESULT: FAKE
   Confidence: [████████████████░░░░] 83.5%
-------------------------------------------------------
```

<br>

✿ ⋆｡°✩ *.✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿ ⋆｡°✩

## 🌷 Future improvements !!

- 🌸 Combine title + text for potentially better accuracy
- 🌸 Try other models (Naive Bayes, Random Forest) and compare results
- 🌸 Build a simple web interface with Flask or Streamlit

✿ ⋆｡°✩ *.✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿*.｡✧*.｡✿ ⋆｡°✩
