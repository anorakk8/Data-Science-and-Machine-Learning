#!/usr/bin/env python
# coding: utf-8

# 
# # Logistic Regression Project 
# ## by, Konark Pahuja
# 
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries
# 
# **Import a few libraries you think you'll need (Or just import them as you go along!)**

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# In[4]:


ad_data= pd.read_csv("advertising.csv")


# **Check the head of ad_data**

# In[5]:


ad_data.head()


# ** Use info and describe() on ad_data**

# In[6]:


ad_data.info()


# In[7]:


ad_data.describe()


# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# ** Create a histogram of the Age**

# In[20]:


sb.histplot(ad_data['Age'],bins=30)
sb.set_style(style='whitegrid')


# **Create a jointplot showing Area Income versus Age.**

# In[21]:


sb.jointplot(x='Age',y='Area Income',data=ad_data)


# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

# In[26]:


sb.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde',color='red',fill=True)


# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[29]:


sb.jointplot(x='Daily Time Spent on Site',y = 'Daily Internet Usage', data=ad_data,color='green')


# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

# In[32]:


sb.pairplot(data=ad_data,hue='Clicked on Ad')


# # Logistic Regression
# 

# ** Split the data into training set and testing set using train_test_split**

# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


ad_data.head()


# In[42]:


X = ad_data.drop(['Ad Topic Line', 'City','Country',
       'Timestamp', 'Clicked on Ad'], axis=1)
y = ad_data['Clicked on Ad']


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ** Train and fit a logistic regression model on the training set.**

# In[44]:


from sklearn.linear_model import LogisticRegression


# In[45]:


lrmodel = LogisticRegression()


# In[46]:


lrmodel.fit(X_train,y_train)


# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

# In[47]:


predictions = lrmodel.predict(X_test)


# ** Create a classification report for the model.**

# In[48]:


from sklearn.metrics import classification_report


# In[50]:


print(classification_report(y_test,predictions))


# ## End of Project
