#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing Project
# 
# ## by, Konark Pahuja
# 
# In this NLP project we will be attempting to classify Yelp Reviews into 1 star or 5 star categories based on the text content in the reviews. We will utilize pipeline for the more complex tasks.
# 
# We will use the [Yelp Review Data Set from Kaggle](https://www.kaggle.com/c/yelp-recsys-2013).
# 
# Each observation in this dataset is a review of a particular business by a particular user.
# 
# The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.
# 
# The "cool" column is the number of "cool" votes this review received from other Yelp users. 
# 
# All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. In other words, it is a rating of the review itself, not a rating of the business.
# 
# The "useful" and "funny" columns are similar to the "cool" column.
# 
# Let's get started! 

# ## Imports
#  **Import the libraries **

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# ## The Data
# 
# **Read the yelp.csv file and set it as a dataframe called yelp.**

# In[2]:


yelp = pd.read_csv("yelp.csv")


# ** Check the head, info , and describe methods on yelp.**

# In[3]:


yelp.head()


# In[4]:


yelp.info()


# In[5]:


yelp.describe()


# **Create a new column called "text length" which is the number of words in the text column.**

# In[37]:


yelp["text length"] = yelp["text"].apply(len)


# # Exploratory Data Analysis
# 
# Let's explore the data

# **Use FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings. Reference the seaborn documentation for hints on this**

# In[40]:


g = sb.FacetGrid(data=yelp,col="stars")
g.map(plt.hist, "text length")


# **Create a boxplot of text length for each star category.**

# In[44]:


sb.boxplot(x='stars',y='text length',data=yelp)


# **Create a countplot of the number of occurrences for each type of star rating.**

# In[45]:


sb.countplot(x='stars',data=yelp)


# ** Use groupby to get the mean values of the numerical columns, you should be able to create this dataframe with the operation:**

# In[47]:


stars = yelp.groupby('stars').mean()


# In[48]:


stars.head()


# **Use the corr() method on that groupby dataframe to produce this dataframe:**

# In[51]:


stars_corr = stars.corr()


# **Then use seaborn to create a heatmap based off that .corr() dataframe:**

# In[57]:


sb.heatmap(stars_corr, cmap='coolwarm',annot=True)


# ## NLP Classification Task
# 
# Let's move on to the actual task. To make things a little easier, go ahead and only grab reviews that were either 1 star or 5 stars.
# 
# **Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.**

# In[82]:


yelp_class = yelp[ (yelp['stars']== 1) | (yelp['stars']==5) ]
yelp_class.head()


# ** Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)**

# In[84]:


X = yelp_class['text']
y = yelp_class['stars']


# **Import CountVectorizer and create a CountVectorizer object.**

# In[87]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# ** Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X.**

# In[88]:


X = cv.fit_transform(X)


# ## Train Test Split
# 
# Let's split our data into training and testing data.
# 
# ** Use train_test_split to split up the data into X_train, X_test, y_train, y_test. Use test_size=0.3 and random_state=101 **

# In[89]:


from sklearn.model_selection import train_test_split
    


# In[90]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training a Model
# 
# Time to train a model!
# 
# ** Import MultinomialNB and create an instance of the estimator and call is nb **

# In[92]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# **Now fit nb using the training data.**

# In[93]:


nb.fit(X_train,y_train)


# ## Predictions and Evaluations
# 
# Time to see how our model did!
# 
# **Use the predict method off of nb to predict labels from X_test.**

# In[94]:


preds = nb.predict(X_test)


# ** Create a confusion matrix and classification report using these predictions and y_test **

# In[95]:


from sklearn.metrics import classification_report, confusion_matrix


# In[96]:


print(confusion_matrix(y_test,preds))
print("/n")
print(classification_report(y_test,preds))


# **Great! Let's see what happens if we try to include TF-IDF to this process using a pipeline.**

# # Using Text Processing
# 
# ** Import TfidfTransformer from sklearn. **

# In[97]:


from sklearn.feature_extraction.text import TfidfTransformer


# ** Import Pipeline from sklearn. **

# In[98]:


from sklearn.pipeline import Pipeline


# ** Now create a pipeline with the following steps:CountVectorizer(), TfidfTransformer(),MultinomialNB()**

# In[103]:


pipeline = Pipeline([
    ('countVectorizer',CountVectorizer()),
    ('tfidfTransformer',TfidfTransformer()),
    ('NBClassifer',MultinomialNB())
])


# ## Using the Pipeline
# 
# **Time to use the pipeline! Remember this pipeline has all your pre-process steps in it already, meaning we'll need to re-split the original data (Remember that we overwrote X as the CountVectorized version. What we need is just the text)**

# ### Train Test Split
# 
# **Redo the train test split on the yelp_class object.**

# In[104]:


X = yelp_class['text']
y = yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# **Now fit the pipeline to the training data. Remember you can't use the same training data as last time because that data has already been vectorized. We need to pass in just the text and labels**

# In[105]:


pipeline.fit(X_train,y_train)


# ### Predictions and Evaluation
# 
# ** Now use the pipeline to predict from the X_test and create a classification report and confusion matrix. You should notice strange results.**

# In[106]:


tfidf_preds = pipeline.predict(X_test)


# In[107]:


print(confusion_matrix(y_test,tfidf_preds))
print("/n")
print(classification_report(y_test,tfidf_preds))


# In[154]:





# ## CONCLUSION:
# 
# Looks like Tf-Idf actually made things worse and resulted in lower acuracy that simple word count.

# # End of Project
