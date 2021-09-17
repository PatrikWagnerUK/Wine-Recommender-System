import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, sigmoid_kernel
import warnings

warnings.filterwarnings(action='ignore')

st.title('Wine Recommendation System 1.0')
## Importing and cleaning data

#df = pd.read_csv('../../../Data/wine_data.csv')
predictors = pd.read_csv('wine_pred_matrix.csv')

# ## Vectorizing With Tfidf

vectors = TfidfVectorizer(min_df = 3,
                         max_features = None,
                         strip_accents = 'unicode',
                         analyzer = 'word',
                         token_pattern = '\w{2,}',
                         ngram_range = (1,3),
                         stop_words = 'english')

#@st.cache
vectors_matrix = vectors.fit_transform(predictors['description'])

# ## Calculating Similarity

sig_kern = sigmoid_kernel(vectors_matrix, vectors_matrix)

index = pd.Series(predictors.index, index=predictors['name']).drop_duplicates()

def recommend_wine(name, sig_kern=sig_kern):
    indx = index[name]
    sigmoid_score = list(enumerate(sig_kern[indx]))
    sigmoid_score = sorted(sigmoid_score, key = lambda x:x[1], reverse = True)
    sigmoid_score = sigmoid_score[1:4]
    position = [i[0] for i in sigmoid_score]
    return st.write(predictors.iloc[position])

user_input = st.text_input('Show me wines similar to: ')
recommend_wine(user_input)
