import pandas as pd
import numpy as np
import streamlit as st
import pickle
import boto3
from s3fs.core import S3FileSystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, sigmoid_kernel

## Importing and cleaning data

#predictors = pd.read_csv('wine_pred_matrix.csv')

## Loading model from pickle file
#@st.cache(allow_output_mutation=True)
#def load_model():
#    with open('wine_model.pkl', 'rb') as file:
#        data = pickle.load(file)
#    return data

s3 = boto3.resource('s3', aws_access_key_id=st.secrets["key_id"], aws_secret_access_key=st.secrets["secret_key"])
data = pickle.loads(s3.Bucket("wineproj").Object("wine_model_ita.pkl").get()['Body'].read())
predictors = pd.read_csv(s3.Bucket("wineproj").Object("wine_pred_matrix_ita.csv").get()['Body'].read())

#data = load_model()
sig_kern = data["model"]

## Creating function to display streamlit page
def show_page():
    st.title('Wine Recommendation System 1.0')

def recommend_wine(sig_kern=sig_kern):
    variety = st.sidebar.selectbox("Filter Wines by variety:", np.unique(predictors['variety']))
    variety_filtered = predictors[(predictors['variety'] == variety)]
    #st.dataframe(variety_filtered[['name', 'variety']])
    user_wine_input = st.selectbox('Recommend me a wine similar to the:', variety_filtered['name'].sort_values(ascending=True))


    index = pd.Series(predictors.index, index=predictors['name']).drop_duplicates()
    indx = index[user_wine_input]
    sigmoid_score = list(enumerate(sig_kern[indx]))
    sigmoid_score = sorted(sigmoid_score, key = lambda x:x[1], reverse = True)
    sigmoid_score = sigmoid_score[1:4]
    position = [i[0] for i in sigmoid_score]

    pd.set_option('display.max_colwidth', None)
    if st.button("Recommend Wine"):
        st.header(f"Other wines to consider are: ")

        name1 = predictors[['name']].iloc[position[0]]
        desc1 = predictors[['description']].iloc[position[0]]
        st.subheader("The " + name1.to_string(header=False, index=False))
        st.markdown(desc1.to_string(header=False, index=False))
        st.markdown("____")

        name2 = predictors[['name']].iloc[position[1]]
        desc2 = predictors[['description']].iloc[position[1]]
        st.subheader("The " + name2.to_string(header=False, index=False))
        st.markdown(desc2.to_string(header=False, index=False))
        st.markdown("____")

        name3 = predictors[['name']].iloc[position[2]]
        desc3 = predictors[['description']].iloc[position[2]]
        st.subheader("The " + name3.to_string(header=False, index=False))
        st.markdown(desc3.to_string(header=False, index=False))
