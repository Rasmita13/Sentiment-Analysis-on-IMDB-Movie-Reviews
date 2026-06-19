import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("IMDB Dataset.csv")

print(df.head())

# -----------------------------
# Text Cleaning
# -----------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text


df['review'] = df['review'].apply(clean_text)

# -----------------------------
# Tokenization + Stopword Removal
# -----------------------------

stop_words = set(stopwords.words('english'))

def preprocess(text):

    tokens = word_tokenize(text)

    tokens = [word for word in tokens
              if word not in stop_words]

    return " ".join(tokens)


df['review'] = df['review'].apply(preprocess)

# -----------------------------
# Convert Labels
# -----------------------------

df['sentiment'] = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})

# -----------------------------
# TF-IDF
# -----------------------------

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df['review'])

y = df['sentiment']

# -----------------------------
# Split Data
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Logistic Regression
# -----------------------------

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\nLOGISTIC REGRESSION")

print("Accuracy:",
      accuracy_score(y_test, lr_pred))

# -----------------------------
# Naive Bayes
# -----------------------------

nb = MultinomialNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)

print("\nNAIVE BAYES")

print("Accuracy:",
      accuracy_score(y_test, nb_pred))

# -----------------------------
# Classification Report
# -----------------------------

print("\nClassification Report")

print(classification_report(y_test, lr_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# -----------------------------
# Dashboard
# -----------------------------

positive_count = np.sum(lr_pred == 1)
negative_count = np.sum(lr_pred == 0)

total = len(lr_pred)

positive_percent = positive_count / total * 100
negative_percent = negative_count / total * 100

print("\nDashboard")

print("Positive %:",
      round(positive_percent,2))

print("Negative %:",
      round(negative_percent,2))

# -----------------------------
# Pie Chart
# -----------------------------

plt.figure(figsize=(6,6))

plt.pie(
    [positive_percent,
     negative_percent],
    labels=["Positive",
            "Negative"],
    autopct="%1.1f%%"
)

plt.title("Sentiment Dashboard")

plt.show()

# -----------------------------
# Prediction Function
# -----------------------------

def predict_sentiment(review):

    review = clean_text(review)

    review = preprocess(review)

    review = tfidf.transform([review])

    prediction = lr.predict(review)

    if prediction[0] == 1:
        return "Positive"
    else:
        return "Negative"


# -----------------------------
# User Input
# -----------------------------

while True:

    review = input(
        "\nEnter Review (quit to exit): "
    )

    if review.lower() == "quit":
        break

    result = predict_sentiment(review)

    print("Sentiment:", result)