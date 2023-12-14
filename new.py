#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df=pd.read_csv("IRIS.csv")
df.head()


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


df['species'].value_counts()


# In[11]:


df.isnull().sum()


# In[12]:


df['sepal_length'].hist()


# In[13]:


df['sepal_width'].hist()


# In[14]:


df['petal_length'].hist()


# In[15]:


df['petal_width'].hist()


# In[16]:


colors=['red','orange','blue']
species=['virginica','versicolor','setosa']


# In[17]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['sepal_length'],x['sepal_width'],c=colors[i],label=species[i])
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.legend()


# In[18]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['petal_length'],x['petal_width'],c=colors[i],label=species[i])
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.legend()


# In[19]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['sepal_length'],x['petal_length'],c=colors[i],label=species[i])
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend()


# In[20]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['sepal_width'],x['petal_width'],c=colors[i],label=species[i])
plt.xlabel("Sepal width")
plt.ylabel("Petal width")
plt.legend()


# In[21]:


df.corr()


# In[24]:


corr=df.corr()
fig,ax= plt.subplots(figsize=(5,4))
sns.heatmap(corr,annot=True,ax=ax,cmap='coolwarm')


# In[25]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[26]:


df['species']=le.fit_transform(df['species'])
df.head()


# In[27]:


from sklearn.model_selection import train_test_split
X=df.drop(columns=['species'])
Y=df['species']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30)


# In[28]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[29]:


model.fit(x_train,y_train)


# In[31]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[32]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()


# In[33]:


model.fit(x_train,y_train)


# In[34]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[35]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()


# In[36]:


model.fit(x_train,y_train)


# In[37]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[ ]:


# train_model.py




