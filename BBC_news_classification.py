#!/usr/bin/env python
# coding: utf-8

# ## News Classification

# ### Classifying news based on their content onto a category

# In[2]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/bbc-news-data.csv", sep='\t')


# In[3]:


data


# In[4]:


# check null values
data.isnull().sum()


# In[5]:


data["category"].value_counts()


# In[6]:


data = data[["title", "category"]]

x = np.array(data["title"])
y = np.array(data["category"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[7]:


model = MultinomialNB()
model.fit(X_train,y_train)


# In[8]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)


# In[21]:


def user():
    user = input("Enter a Text: ")
    print(model.predict(cv.transform([user]).toarray()))


# In[23]:


user()

