import  streamlit as st
import pickle 
import nltk
from nltk.corpus import stopwords
from nltk.stemmer import PorterStemmer
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl', 'rb'))
st.title('Email/SMS Spam Classifier')
st.text_input('Enter the message')
input_sms = st.text_input('Enter the message')
import string

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
    if i not in stopwords.words('english') and i not in string.punctuation:
        y.append(i)

  text = y[:]
  y.clear()

  ps = PorterStemmer()
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)
transformed_sms = transform_text(input_sms)
vector_input = tfidf.transform([transformed_sms])
result = model.predict(vector_input)[0]
if(result == 1):
   st.header("Spam")
else:
   st.header("Not Spam")