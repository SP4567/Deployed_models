import streamlit as st
import pickle
import nltk
import nltk.data
nltk.download('stopwords')
import string
import keras
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.saving import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
import keras.src.saving
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit.web.cli as stcli
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
vectorizer = pickle.load(
    open('C:\\Users\\Suyash Pandey\\PycharmProjects\\Spam_SMS_Prediction\\vectorizer (3).pkl', 'rb'))
model = load_model('Spam_classifier.h5')
st.title('Email/SMS Spam Classifier')
st.text_input('Enter the message')
if st.button('Predict'):
    input_sms = st.text_input('Enter the message', key='')
    transformed_sms = transform_text(input_sms)
    vector_input = vectorizer.transform([transformed_sms]).toarray()
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
