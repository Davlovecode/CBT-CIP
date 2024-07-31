#!/usr/bin/env python
# coding: utf-8

# # NAME - DEVANSHU
# # DATA SCIENCE INTERNSHIP @CIPHERBYTE TECHNOLOGY 
# # TASK-4 
# 

# # PROJECT NAME - EMAIL SPAM DETECTON USING MACHINE LEARNING

# In[3]:


#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#convert text into features vector or numericc value\
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# #  Data collection and preprocessing 

# In[4]:


#loading the data
mail = pd.read_csv('F:/chip/Spam Email Detection - spam.csv')


# In[5]:


mail.head()


# In[6]:


mail.isnull().sum()


# In[7]:


#replace the null data with a value
# creating a frame

mail_data = mail.where((pd.notnull(mail)),'')


# In[8]:


mail_data.head()


# In[9]:


#checking the number of rows and columns 

mail_data.shape


# In[10]:


#rename the columns

mail_data = mail_data.rename(columns={'v1':'category','v2':'message'})


# In[11]:


mail_data.head()


# In[12]:


#label spam mail:0 and ham mail:1

mail_data.loc[mail_data['category']=='spam','category']=0
mail_data.loc[mail_data['category']=='ham','category']=1


# In[13]:


#seperataing the data as texts andd lebels

x = mail_data['message']
y= mail_data['category']


# In[14]:


print(x)


# In[15]:


print(y)


# # spliting the data 
# 

# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)


# In[17]:


print(x.shape)
print(x_train.shape)
print(x_test.shape)


# # feature extraction

# In[18]:


# transformation the text data to features vecctors 

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase=True)

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

#convert y_train and y_test values as integer

y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[19]:


y_train


# In[20]:


y_test


# In[21]:


x_train


# In[22]:


x_train_features


# In[23]:


print(x_train_features)


# # training the model
# 
# # LogisticRegression

# In[24]:


model = LogisticRegression()


# In[25]:


#training

model.fit(x_train_features,y_train)


# In[26]:


#prediction training data

prediction_on_trainig_data = model.predict(x_train_features)
accuracy_on_trainig_data = accuracy_score(y_train, prediction_on_trainig_data)


# In[27]:


print('accuracy_on_trainig_data:',accuracy_on_trainig_data)


# In[28]:


#prediction test dataa

prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)


# In[29]:


print('accuracy_on_test_data:',accuracy_on_test_data)


# # Building a predictive system
# 
# 

# In[30]:


input = ["I am devanshu from bihar"]

input_data_fextures = feature_extraction.transform(input)

prediction = model.predict(input_data_fextures)
print(prediction)

if(prediction[0]==1):
    print('Ham Mail')
    
else:
    print('Spam Mail')


# In[ ]:




