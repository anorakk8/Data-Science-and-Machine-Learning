#!/usr/bin/env python
# coding: utf-8

# # K Nearest Neighbors Project 
# ## by, Konark Pahuja 
# In this project we will work with an unlabelled/anonymized data set, preprocess it, and train a KNN model to predit the target class. We will choose an appropriate K value using the elbow method.
# 
# ## Import Libraries
# **Import pandas,seaborn, and the usual libraries.**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb 


# ## Get the Data
# ** Read the 'KNN_Project_Data csv file into a dataframe **

# In[2]:


df = pd.read_csv("KNN_Project_Data")


# **Check the head of the dataframe.**

# In[3]:


df.head()


# # Exploratory Data Analysis
# 
# Since this data is artificial, we'll just do a large pairplot with seaborn.
# 
# **Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**

# In[5]:


sb.pairplot(df, hue = 'TARGET CLASS')


# In[ ]:





# # Standardize the Variables
# 
# Time to standardize the variables.
# 
# ** Import StandardScaler from Scikit learn.**

# In[6]:


from sklearn.preprocessing import StandardScaler


# ** Create a StandardScaler() object called scaler.**

# In[7]:


scaler = StandardScaler()


# ** Fit scaler to the features.**

# In[10]:


scaler.fit(df.drop('TARGET CLASS', axis=1))


# **Use the .transform() method to transform the features to a scaled version.**

# In[12]:


scaled_feats = scaler.transform(df.drop('TARGET CLASS', axis=1))


# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[13]:


df_feats = pd.DataFrame(scaled_feats,columns=df.columns[:-1])


# In[14]:


df_feats.head()


# # Train Test Split
# 
# **Use train_test_split to split your data into a training set and a testing set.**

# In[15]:


from sklearn.model_selection import train_test_split


# In[29]:


X= df_feats
y= df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# # Using KNN
# 
# **Import KNeighborsClassifier from scikit learn.**

# In[17]:


from sklearn.neighbors import KNeighborsClassifier


# **Create a KNN model instance with n_neighbors=1**

# In[18]:


knn = KNeighborsClassifier(n_neighbors=1)


# **Fit this KNN model to the training data.**

# In[30]:


knn.fit(X_train,y_train)


# # Predictions and Evaluations
# Let's evaluate our KNN model!

# **Use the predict method to predict values using your KNN model and X_test.**

# In[31]:


preds = knn.predict(X_test)


# ** Create a confusion matrix and classification report.**

# In[21]:


from sklearn.metrics import confusion_matrix,classification_report


# In[39]:


print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))


# # Choosing a K Value
# Let's go ahead and use the elbow method to pick a good K Value!
# 
# ** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**

# In[45]:


error_rate=[]

for i in range(1,60):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    
    error_rate.append(np.mean(pred_i != y_test))
    
    


# **Now create the following plot using the information from your for loop.**

# In[46]:


plt.figure(figsize=(10,6))
plt.plot(range(1,60),error_rate, color='blue',linestyle = '--', marker='o', markerfacecolor='red',markersize=10)
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.title("Error Rate vs K Value")


# ## Retrain with new K Value
# 
# **Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix to compare the results before and after.**

# In[40]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
preds = knn.predict(X_test)


print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))


# In[47]:


knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train,y_train)
preds = knn.predict(X_test)


print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))


# # End of Project

# In[ ]:




