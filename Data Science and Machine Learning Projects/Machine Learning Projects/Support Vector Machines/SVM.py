#!/usr/bin/env python
# coding: utf-8

# 
# # Support Vector Machines Project 
# ## by, Konark Pahuja
# 
# ## The Data
# For this series of lectures, we will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
# 
# Here's a picture of the three different Iris types:

# In[17]:


# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[18]:


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[19]:


# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# The iris dataset contains measurements for 150 iris flowers from three different species.
# 
# The three classes in the Iris dataset:
# 
#     Iris-setosa (n=50)
#     Iris-versicolor (n=50)
#     Iris-virginica (n=50)
# 
# The four features of the Iris dataset:
# 
#     sepal length in cm
#     sepal width in cm
#     petal length in cm
#     petal width in cm
# 
# ## Get the data
# 
# **Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') **

# In[6]:


import numpy as np 
import pandas as pd 
import matplotlib as plt 
import seaborn as sb 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


iris = sb.load_dataset('iris')


# Let's visualize the data and get you started!
# 
# ## Exploratory Data Analysis
# 
# Time to put your data viz skills to the test! Try to recreate the following plots, make sure to import the libraries you'll need!
# 
# **Import some libraries you think you'll need.**

# In[25]:





# ** Create a pairplot of the data set. Which flower species seems to be the most separable?**

# Setosa

# In[12]:


iris.head()


# In[14]:


sb.pairplot(iris,hue='species')


# In[37]:





# **Create a kde plot of sepal_length versus sepal width for setosa species of flower.**

# In[57]:


sb.set_style('whitegrid')
setosa = iris[iris['species']=='setosa']
sb.kdeplot(x='sepal_width',y='sepal_length',data=setosa,cmap='plasma',shade=True)


# In[44]:





# # Train Test Split
# 
# ** Split your data into a training set and a testing set.**

# In[22]:


from sklearn.model_selection import train_test_split


# In[25]:


X=iris.drop('species',axis=1)
y=iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# # Train a Model
# 
# Now its time to train a Support Vector Machine Classifier. 
# 
# **Call the SVC() model from sklearn and fit the model to the training data.**

# In[26]:


from sklearn.svm import SVC


# In[27]:


svc = SVC()


# In[28]:


svc.fit(X_train,y_train)


# ## Model Evaluation
# 
# **Now get predictions from the model and create a confusion matrix and a classification report.**

# In[29]:


from sklearn.metrics import classification_report, confusion_matrix


# In[31]:


preds = svc.predict(X_test)


# In[32]:


print(confusion_matrix(y_test,preds))


# In[33]:


print(classification_report(y_test,preds))


# Let us try to use gridsearch to tune the parameters and improve the result

# ## Gridsearch Practice
# 
# ** Import GridsearchCV from SciKit Learn.**

# In[34]:


from sklearn.model_selection import GridSearchCV


# **Create a dictionary called param_grid and fill out some parameters for C and gamma.**

# In[36]:


param_grid = {'C':[0.1,1,10,100], 'gamma' : [1,0.1,0.01,0.001,0.0001]}


# ** Create a GridSearchCV object and fit it to the training data.**

# In[39]:


grid = GridSearchCV(SVC(),param_grid,verbose=3)


# In[40]:


grid.fit(X_train,y_train)


# In[41]:


grid.best_params_


# ** Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?**

# In[42]:


grid_preds = grid.predict(X_test)


# In[43]:


print(confusion_matrix(y_test,grid_preds))


# In[44]:


print(classification_report(y_test,grid_preds))


# Our result after re-tuning parameters is nearly same as before. This makes sense, there is basically just one point that is too noisey to grab, which makes sense, we don't want to have an overfit model that would be able to grab that.

# ## End of Project
