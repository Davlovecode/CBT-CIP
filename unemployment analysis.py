#!/usr/bin/env python
# coding: utf-8

# #  NAME  - DEVANSHU KUMAR
# 
# # TASK - 2
# # PROJECT NAME - UNEMPLOYMENT ANALYSIS 

# In[1]:


# imporat the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[5]:


data = pd.read_csv('F:/chip/Unemployment_Rate_upto_11_2020.csv')


# In[6]:


data


# In[7]:


# data information

data.info()


# In[20]:


data.isnull().sum()


# In[8]:


# data describe

data.describe()


# In[10]:


# rename columns 

data.columns = ['States','Date','Frequency','Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate'
               ,'Region','Longitude','Latitude']


# In[11]:


data.head()


# In[12]:


# plotting histogram


data.columns = ['States','Date','Frequency','Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate'
               ,'Region','Longitude','Latitude']
plt.title('Indian Unemployment ')
sns.histplot(x='Estimated Employed',hue='Region',data=data)
plt.show()


# In[15]:


#plotting 

plt.figure(figsize=(10,8))
plt.title('Indian Unemployment ')
sns.histplot(x='Estimated Unemployment Rate',hue='Region',data=data)
plt.show()


# In[16]:


# plotting sunburst

unemployment = data[['States','Region','Estimated Unemployment Rate']]
figure = px.sunburst(unemployment,path=['Region','States'],
                    values='Estimated Unemployment Rate',width = 650,height=600, color_continuous_scale='RdY1Gn',
                    title = 'Unemployment Rate in India')
figure.show()


# In[26]:


# checking the correlation 

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(8,6))

numerical_data = data.select_dtypes(include=[int, float])

# Create the heatmap
sns.heatmap(numerical_data.corr(), annot=True, linewidth=3)

# Set tick parameters
plt.tick_params(size=10, color='w', labelsize=10, labelcolor='w')

plt.show()


# In[ ]:




