import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, sigmoid_kernel

st.title('Wine Recommendation System 1.0')
## Importing and cleaning data

df = pd.read_csv('../../Data/wine_data.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('region_2', axis=1, inplace=True)

predictors = df[['country', 'description', 'designation', 'province', 'region_1', 'variety', 'winery']]

## Missing Data
# For the first iteration of this recommender system, I will drop observations with missing values across the board instead of being more selective. This cuts the available data in half. Next iterations could try modelling using fewer features, but more observations.

predictors.dropna(inplace=True)

## Feature Engineering & Unique Names for Wines

## Creating a more detailed name for each wine by combining Winery and Designation

predictors['name'] = predictors['winery'] + ', ' + predictors['designation']

#predictors.drop('index', axis=1, inplace=True)

## Removing Duplicate Values

predictors.drop_duplicates(inplace=True)

# In order for the end user to recieve recommendations, using this model and approach, they need to enter a unique name for a wine they like. With so many duplicates this becomes tricky. For now I will drop duplicated wine values, which significantly reduces the volume of data but solved the uniqueness issue. There is almost certainly a better way around this!

predictors.drop_duplicates(subset='name', keep='last', inplace=True)

predictors.reset_index(inplace=True)

predictors.drop('index', axis=1, inplace=True)


# ## Vectorizing With Tfidf

vectors = TfidfVectorizer(min_df = 3,
                         max_features = None,
                         strip_accents = 'unicode',
                         analyzer = 'word',
                         token_pattern = '\w{2,}',
                         ngram_range = (1,3),
                         stop_words = 'english')

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
    return predictors.iloc[position]

user_input = st.text_input('Show me wines similar to: ')
recommend_wine(user_input)


# In[ ]:





# In[ ]:
