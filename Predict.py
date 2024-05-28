import pickle
import streamlit as st
import keras
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.web.cli as stcli
from keras.src.saving import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
import keras.src.saving


def return_prediction(model, Scaler, sample_json):
    a = sample_json['age']
    s = sample_json['sex']
    cp = sample_json['cp']
    c = sample_json['trestbps']
    o = sample_json['chol']
    f = sample_json['fbs']
    r = sample_json['restecg']
    t = sample_json['thalach']
    e = sample_json['exang']
    p = sample_json['oldpeak']
    st = sample_json['slope']
    cc = sample_json['ca']
    l = sample_json['thal']
    dc = [[c, o, f, r, t, e, p, st, cc, l]]
    dc = Scaler.fit_transform(dc)
    predict = model.predict(dc)
    classes = np.argmax(predict, axis=1)
    return classes


Scaler = pickle.load(open("C:\\Users\\Suyash Pandey\\PycharmProjects\\Heart_Disease_Predictor\\scaler (2).pkl", "rb"))
model = load_model("Heart_Disease_Predictor.h5")
st.title("Heart_Disease_Predictor")
a = st.number_input('Enter the age', step=1., format="%.2f")
s = st.number_input('Enter the sex', step=1., format="%.2f")
cp = st.number_input('Enter the cp',  step=1., format="%.2f")
c = st.number_input('Enter the trestbps', step=1., format="%.2f")
o = st.number_input('Enter the chol', step=1., format="%.2f")
f = st.number_input('Enter the fbs', step=1., format="%.2f")
r = st.number_input('Enter the restecg', step=1., format="%.2f")
t = st.number_input('Enter the thalach', step=1., format="%.2f")
e = st.number_input('Enter the exang',  step=1., format="%.2f")
p = st.number_input('Enter the oldpeak', step=1., format="%.2f")
slope = st.number_input('Enter the slope', step=1., format="%.2f")
v = st.number_input('Enter the ca',  step=1., format="%.2f")
l = st.number_input('Enter the thal', step=1., format="%.2f")
if st.button('Predict'):
    dc = [[a, s, cp, c, o, f, r, t, e, p, slope, v, l]]
    inp = Scaler.fit_transform(dc)
    res = model.predict(inp)
    class_x = np.argmax(res, axis=1)
    if class_x == 1:
        st.header("Patient is suffering from heart disease")
    else:
        st.header("No Heart Disease")
