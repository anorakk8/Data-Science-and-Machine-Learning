#!/usr/bin/env python
# coding: utf-8

# # Finance Data Project 
# 
# In this data project we will focus on exploratory data analysis & visualization of stock prices.
# ____
# We'll focus on bank stocks and see how they progressed throughout the [financial crisis](https://en.wikipedia.org/wiki/Financial_crisis_of_2007%E2%80%9308) all the way to early 2016.
# 
# The data used is downloaded from Yahoo Finance (HTML Format)

# ## Get the Data
# 
# In this section we will use pandas to directly read data from Google finance using pandas!
# 

# In[1]:


from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import matplotlib as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data
# 
# We need to get data using pandas datareader. We will get stock information for the following banks:
# *  Bank of America
# * CitiGroup
# * Goldman Sachs
# * JPMorgan Chase
# * Morgan Stanley
# * Wells Fargo

# In[2]:


start = datetime.datetime(2006,1,1)
end = datetime.datetime(2016,1,1)


# In[3]:


# Bank of America
BAC = data.DataReader('BAC','yahoo',start,end)

#CitiGroup
C = data.DataReader('C','yahoo',start,end)


#Goldman Sachs
GS = data.DataReader('GS','yahoo',start,end)


#JPMorgan Chase
JPM = data.DataReader('JPM','yahoo',start,end)


#Morgan Stanley
MS = data.DataReader('MS','yahoo',start,end)

#Wells Fargo

WFC = data.DataReader('WFC','yahoo',start,end)


# In[4]:


WFC


# ** Create a list of the ticker symbols (as strings) in alphabetical order. Call this list: tickers**

# In[5]:


tickers = ['BAC','C','GS','JPM','MS','WFC']


# ** Use pd.concat to concatenate the bank dataframes together to a single data frame called bank_stocks. Set the keys argument equal to the tickers list. 

# In[6]:


bank_stocks = pd.concat(objs=[BAC,C,GS,JPM,MS,WFC], keys=tickers,axis=1)
bank_stocks.head()


# ** Set the column name levels (this is filled out for you):**

# In[7]:


bank_stocks.columns.names = ['Bank Ticker','Stock Info']


# ** Check the head of the bank_stocks dataframe.**

# In[8]:


bank_stocks.head()


# # Exploratory Data Analysis
# 
# Let's explore the data a bit!  
# 
# ** What is the max Close price for each bank's stock throughout the time period?**

# In[9]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()


# ** Create a new empty DataFrame called returns. This dataframe will contain the returns for each bank's stock. returns are typically defined by:**
# 
# $$r_t = \frac{p_t - p_{t-1}}{p_{t-1}} = \frac{p_t}{p_{t-1}} - 1$$

# In[10]:


returns = pd.DataFrame()


# ** We can use pandas pct_change() method on the Close column to create a column representing this return value. Create a for loop that goes and for each Bank Stock Ticker creates this returns column and set's it as a column in the returns DataFrame.**

# In[11]:


for tick in tickers:
    returns[tick + ' Return'] = bank_stocks[tick]['Close'].pct_change()


# In[12]:


returns.head()


# ** Create a pairplot using seaborn of the returns dataframe. What stock stands out to you? Can you figure out why?**

# In[13]:


sb.pairplot(returns[1:])


# * See solution for details about Citigroup behavior....

# ** Using this returns DataFrame, figure out on what dates each bank stock had the best and worst single day returns. You should notice that 4 of the banks share the same day for the worst drop, did anything significant happen that day?**

# In[14]:


returns.idxmin()


# ** You should have noticed that Citigroup's largest drop and biggest gain were very close to one another, did anythign significant happen in that time frame? **

# * See Solution for details

# In[15]:


returns.idxmax()


# ** Take a look at the standard deviation of the returns, which stock would you classify as the riskiest over the entire time period? Which would you classify as the riskiest for the year 2015?**

# In[16]:


returns.std()


# In[17]:


returns['2015-01-01':'2015-12-31'].std()


# ** Create a distplot using seaborn of the 2015 returns for Morgan Stanley **

# In[29]:


plt.figure(figsize = (10,6))
sb.distplot(returns['2015-01-01':'2015-12-31']['MS Return'], bins=50)


# ** Create a distplot using seaborn of the 2008 returns for CitiGroup **

# In[30]:


plt.figure(figsize = (10,6))
sb.distplot(returns['2008-01-01':'2008-12-31']['C Return'], bins=50)


# ____
# # More Visualization
# 
# ### Imports

# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Optional Plotly Method Imports
import plotly
import cufflinks as cf
cf.go_offline()


# ** Create a line plot showing Close price for each bank for the entire index of time. 

# In[43]:


plt.figure(figsize = (12,6))
bank_stocks.xs(key="Close",axis=1,level="Stock Info").plot(label=tick,figsize=(12,4))
plt.tight_layout


# In[22]:


bank_stocks.xs(key="Close",axis=1,level="Stock Info").iplot()


# In[ ]:





# In[ ]:





# ## Moving Averages
# 
# Let's analyze the moving averages for these stocks in the year 2008. 
# 
# ** Plot the rolling 30 day average against the Close Price for Bank Of America's stock for the year 2008**

# In[44]:


plt.figure(figsize=(14,8))
BAC['Close']['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Moving Average')
BAC['Close']['2008-01-01':'2009-01-01'].plot(label='BAC Close')
plt.legend()
plt.tight_layout


# ** Create a heatmap of the correlation between the stocks Close Price.**

# In[45]:


plt.figure(figsize = (10,6))
sb.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True,cmap='coolwarm')


# ** Optional: Use seaborn's clustermap to cluster the correlations together:**

# In[25]:



sb.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True,cmap='coolwarm')


# In[ ]:





# ## Technical Analysis
# 
# In this second part of the project we will rely on the cufflinks library to create some Technical Analysis plots. This part of the project is experimental due to its heavy reliance on the cuffinks project.

# ** Use .iplot(kind='candle) to create a candle plot of Bank of America's stock from Jan 1st 2015 to Jan 1st 2016.**

# In[26]:


bac15 = BAC[['Open','High','Low','Close']]['2015-01-01':'2016-01-01']
bac15.iplot(kind='candle')


# ** Use .ta_plot(study='sma') to create a Simple Moving Averages plot of Morgan Stanley for the year 2015.**

# In[27]:


MS['Close']['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55])


# **Use .ta_plot(study='boll') to create a Bollinger Band Plot for Bank of America for the year 2015.**

# In[28]:


BAC['Close']['2015-01-01':'2016-01-01'].ta_plot(study='boll')


# # End of Project
# 
# 
