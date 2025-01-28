import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Load data
df = pd.read_csv(r"C:\Users\Rishi\Desktop\Data-sci\sentiment analysis pipeline\data\IMDB_Dataset.csv")

df['sentiment'].replace({'positive':1,'negative':0},inplace=True)


# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

df['review'] = df['review'].apply(clean_text)

# Split data
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open(r"C:\Users\Rishi\Desktop\Data-sci\sentiment analysis pipeline\data\sentiment_model.pkl", "wb") as model_file:
    pickle.dump((vectorizer, model), model_file)
