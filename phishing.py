#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df=pd.read_csv('phishing.csv')


# In[2]:


df.head()


# In[3]:


df.isnull().sum()


# In[4]:


X= df.drop(columns='class')
X.head()


# In[5]:


from sklearn.model_selection import train_test_split,cross_val_score
Y=df['class']
Y=pd.DataFrame(Y)
Y.head()


# In[6]:


train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.3,random_state=2)


# In[7]:


print(train_X.shape)
print(test_X.shape)
print(train_Y.shape)
print(test_Y.shape)


# In[8]:


from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[9]:


logreg=LogisticRegression()
model_1=logreg.fit(train_X,train_Y)


# In[10]:


logreg_predict= model_1.predict(test_X)
accuracy_score(logreg_predict,test_Y)


# In[11]:


print(classification_report(logreg_predict,test_Y))


# In[12]:


def plot_confusion_matrix(test_Y, predict_y):
 C = confusion_matrix(test_Y, predict_y)
 A =(((C.T)/(C.sum(axis=1))).T)
 B =(C/C.sum(axis=0))
 plt.figure(figsize=(20,4))
 labels = [1,2]
 cmap=sns.light_palette("green")
 plt.subplot(1, 3, 1)
 sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Confusion matrix")
 plt.subplot(1, 3, 2)
 sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Precision matrix")
 plt.subplot(1, 3, 3)
 sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Recall matrix")
 plt.show()


# In[13]:


plot_confusion_matrix(test_Y, logreg_predict)


# In[14]:


#Lets apply K-Nearest Neighbors Classifier and check its accuracy
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
model_2= knn.fit(train_X,train_Y)
knn_predict=model_2.predict(test_X)


# In[15]:


accuracy_score(knn_predict,test_Y)


# In[16]:


print(classification_report(test_Y,knn_predict))


# In[17]:


plot_confusion_matrix(test_Y, knn_predict)


# In[18]:


#Lets apply Random Forest Classifier and check its accuracy

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
model_4=rfc.fit(train_X,train_Y)


# In[19]:


rfc_predict=model_4.predict(test_X)


# In[20]:


accuracy_score(rfc_predict,test_Y)


# In[21]:


print(classification_report(rfc_predict,test_Y))


# In[22]:


plot_confusion_matrix(test_Y, rfc_predict)


# In[ ]:




