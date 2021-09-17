import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, sigmoid_kernel

index = pd.read_csv('sig_wines.csv')

st.title('Wine Recommender System 1.0')

def recommend_wine(name, sig_kern=sig_kern):
    indx = index[name]
    sigmoid_score = list(enumerate(sig_kern[indx]))
    sigmoid_score = sorted(sigmoid_score, key = lambda x:x[1], reverse = True)
    sigmoid_score = sigmoid_score[1:4]
    position = [i[0] for i in sigmoid_score]
    return predictors.iloc[position]

user_input = st.text_input('Show me wines similar to: ')
recommend_wine(user_input)
