#!/usr/bin/env python
# coding: utf-8

# # 911 Calls Data Science Project 

# This project is based on a dataset of 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:
# 
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
# 
# The project consists of various operations of analysis and visualisation carried out on the same dataset.

# ## Data and Setup

# ### Data Analysis Libraries:

# In[12]:


import numpy as np
import pandas as pd


# ### Data Viz Libraries:

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Read in the csv file as a dataframe called df **

# In[14]:


df = pd.read_csv('911.csv')


# ** Check the info() of the df **

# In[15]:


df.info()


# ** Check the head of df **

# In[16]:


df.head()


# ## Basic Operations:

# ** What are the top 5 zipcodes for 911 calls? **

# In[17]:


df['zip'].value_counts().head(5)


# ** What are the top 5 townships (twp) for 911 calls? **

# In[18]:


df['twp'].value_counts().head(5)


# ** Take a look at the 'title' column, how many unique title codes are there? **

# In[19]:


df['title'].nunique()


# ## Creating New Features with Dataset:

# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# In[20]:


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])


# ** What is the most common Reason for a 911 call based off of this new column? **

# In[21]:


df['Reason'].value_counts()


# ** Now use seaborn to create a countplot of 911 calls by Reason. **

# In[63]:


plt.figure(figsize=(10,6))
sns.countplot(x='Reason',data=df)


# ___
# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **

# In[23]:


type(df['timeStamp'].iloc[0])


# ** These timestamps are still strings. Convert the column from strings to DateTime objects. **

# In[24]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. 

# In[25]:


df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)


# ** Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week: **
# 
#     dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[26]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[27]:


df['Day of Week'] = df['Day of Week'].map(dmap)


# ** Now create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[46]:


plt.figure(figsize=(10,6))

sns.countplot(x='Day of Week',data=df,hue='Reason')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ** Now do the same for Month:**

# In[47]:


plt.figure(figsize=(10,6))
sns.countplot(x='Month',data=df,hue='Reason')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ** Did you notice something strange about the Plot? **

# In[30]:


# It is missing some months! 9,10, and 11 are not there.


# ** You should have noticed it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas...**

# ** Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame. **

# In[31]:


byMonth = df.groupby('Month').count()
byMonth.head()


# ** Now create a simple plot off of the dataframe indicating the count of calls per month. **

# In[48]:


plt.figure(figsize=(10,6))

# Could be any column
byMonth['twp'].plot()


# ** Now we use seaborn's lmplot() to create a linear fit on the number of calls per month. 

# In[51]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. ** 

# In[34]:


df['Date']=df['timeStamp'].apply(lambda t: t.date())


# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[52]:


plt.figure(figsize=(10,6))
df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# ** Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[64]:



plt.figure(figsize=(10,6))
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[65]:


plt.figure(figsize=(10,6))
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[66]:


plt.figure(figsize=(10,6))
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# ____
# ** Now let's move on to creating  heatmaps with our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. We use groupby with an unstack method. 

# In[39]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# ** Now create a HeatMap using this new DataFrame. **

# In[40]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='coolwarm')


# ** Now create a clustermap using this DataFrame. **

# In[41]:


sns.clustermap(dayHour,cmap='coolwarm')


# ** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[42]:


dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()


# In[43]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='coolwarm')


# In[44]:


sns.clustermap(dayMonth,cmap='coolwarm')


# # End of Project
