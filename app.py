import streamlit as st
import pickle
import string
import nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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


# ✅ LOAD TRAINED OBJECTS
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("📩 Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])

    prediction = model.predict(vector_input)[0]
    confidence = model.predict_proba(vector_input).max()

    if prediction == 1:
        st.header("🚨 Spam")
    else:
        st.header("✅ Not Spam")

    st.write(f"**Confidence:** {confidence:.2f}")
