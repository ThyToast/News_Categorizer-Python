import pandas as pd 
import numpy as np 
import streamlit as st
import pickle
import sklearn
import json

st.markdown(
        """
    <style>
    .st-br {
            background-color: #90909066;
        }
    </style>""", unsafe_allow_html=True)

with open('logistic_regression.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pk', 'rb') as file:
    vectorizer = pickle.load(file)

with open('id_to_category.json') as json_file: 
    id_to_category = json.load(json_file) 
  

st.write("# Natural Language Processing - Group Assignment")
input_text = st.text_area("Input News title here: ")
st.write("Input text: " + " \n ** " + input_text + " **")
input_text = [input_text]
text_features = vectorizer.transform(input_text)

predictions = model.predict(text_features)
predictions = str(predictions[0])


st.write("Predicted category: " + " \n ** "+ id_to_category[predictions] + " **")

