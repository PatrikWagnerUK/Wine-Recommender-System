#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, sigmoid_kernel 


# In[ ]:





# # Building a Wine Recommendation System
# 
# Creating a content-based recommendation system through using NLP modelinng on sommellier reviews.

# In[ ]:





# In[31]:


df = pd.read_csv('../../Data/wine_data.csv')


# In[32]:


df.info()


# In[33]:


df.sample(10)


# In[34]:


df.drop('Unnamed: 0', axis=1, inplace=True)


# In[35]:


df.drop('region_2', axis=1, inplace=True)


# In[36]:


df.sample(10)


# In[37]:


predictors = df[['country', 'description', 'designation', 'province', 'region_1', 'variety', 'winery']]


# In[38]:


predictors.info()


# ## Missing Data
# 
# For the first iteration of this recommender system, I will drop observations with missing values across the board instead of being more selective. This cuts the available data in half. Next iterations could try modelling using fewer features, but more observations.

# In[39]:


predictors.dropna(inplace=True)


# In[40]:


predictors.info()


# In[41]:


predictors = predictors.reset_index()


# In[42]:


predictors.description.duplicated().value_counts()


# In[43]:


predictors.duplicated().value_counts()


# In[44]:


predictors[(predictors.description.duplicated() == True)]


# In[45]:


predictors.sample(10)


# ## Feature Engineering & Unique Names for Wines

# In[47]:


## Creating a more detailed name for each wine by combining Winery and Designation

predictors['name'] = predictors['winery'] + ', ' + predictors['designation']


# In[48]:


predictors.drop('index', axis=1, inplace=True)


# In[98]:


predictors.sample(10)


# In[387]:


## Leaving this in for future ideas

## Creating a UID for each wine by combining all data into one variable

#predictors['uid'] = predictors['winery'] + ', ' + predictors['designation'] + ', ' + predictors['country'] + ', ' + predictors['description'] + ', ' + predictors['province'] + ', ' + predictors['region_1'] + ', ' + predictors['variety']


# In[ ]:





# ### Removing Duplicate Values

# In[50]:


predictors.drop_duplicates(inplace=True)


# In[51]:


predictors.info()


# In[52]:


predictors.country.value_counts()


# In[54]:


predictors.name.duplicated().value_counts()


# In order for the end user to recieve recommendations, using this model and approach, they need to enter a unique name for a wine they like. With so many duplicates this becomes tricky. For now I will drop duplicated wine values, which significantly reduces the volume of data but solved the uniqueness issue. There is almost certainly a better way around this!

# In[57]:


predictors.drop_duplicates(subset='name', keep='last', inplace=True)


# In[58]:


predictors.info()


# In[80]:


predictors.reset_index(inplace=True)


# In[82]:


predictors.drop('index', axis=1, inplace=True)


# ## Vectorizing With Tfidf

# In[83]:


vectors = TfidfVectorizer(min_df = 3,
                         max_features = None,
                         strip_accents = 'unicode',
                         analyzer = 'word',
                         token_pattern = '\w{2,}',
                         ngram_range = (1,3),
                         stop_words = 'english')


# In[84]:


vectors_matrix = vectors.fit_transform(predictors['description'])


# In[85]:


vectors_matrix.shape


# ## Calculating Similarity

# In[86]:


sig_kern = sigmoid_kernel(vectors_matrix, vectors_matrix)


# In[87]:


sig_kern


# In[88]:


index = pd.Series(predictors.index, index=predictors['name']).drop_duplicates()


# In[89]:


index['Sobon Estate, Fiddletown']


# In[92]:


def recommend_wine(name, sig_kern=sig_kern):
    indx = index[name]
    sigmoid_score = list(enumerate(sig_kern[indx]))
    sigmoid_score = sorted(sigmoid_score, key = lambda x:x[1], reverse = True)
    sigmoid_score = sigmoid_score[1:4]
    position = [i[0] for i in sigmoid_score]
    return predictors.iloc[position]


# In[99]:


recommend_wine('Castelli del Grevepesa, Riserva Castelgreve')


# In[ ]:





# In[ ]:




