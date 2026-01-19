import pandas as pd
import nltk
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# ✅ LOAD DATASET
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# preprocess text
df['transformed_text'] = df['text'].apply(transform_text)

X_text = df['transformed_text']
y = df['label']

# ✅ TF-IDF (IMPROVED)
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2)
)

X = tfidf.fit_transform(X_text)

# ✅ BETTER MODEL
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X, y)

# ✅ SAVE TRAINED OBJECTS
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

print("✅ Improved model trained and saved successfully")
