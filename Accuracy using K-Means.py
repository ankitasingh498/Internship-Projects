
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')
import pandas as pd
import random


# In[2]:


df = pd.read_csv('breast-cancer-wisconsin.data.csv')
df.replace('?',np.nan, inplace=True) 
df.dropna(axis=0,inplace=True)
df.drop(['id'], 1, inplace=True)


# In[3]:


df.loc[df.label==2,'label']=0
df.loc[df.label==4,'label']=1


# In[4]:


df.head()


# In[5]:


full_data = df.astype(float).values.tolist()
test_size = 0.2
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]


# In[6]:


class K_Means:
    def __init__(self, k=2,tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):
        self.centroids = {}
        self.all_centroids={}
        for i in range(self.k):
            self.all_centroids[i]=[]

        for i in range(self.k):
            self.centroids[i] = data[i][:-1]
            self.all_centroids[i].append(data[i][:-1])
        
        for i in range(self.max_iter):
            self.classifications = {} 
            
            for i in range(self.k):
                self.classifications[i] = []
         
            for featureset in data:  
                distances = [np.linalg.norm(np.array(featureset[:-1]) - np.array(self.centroids[centroid])) for centroid in self.centroids]                                                                 
                classification = distances.index(min(distances))

                self.classifications[classification].append(featureset[:-1])
    

            prev_centroids = dict(self.centroids)
              
            for classification in self.classifications:

                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
                self.all_centroids[classification].append(np.average(self.classifications[classification],axis=0))
            optimized = True
            
                
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((np.array(current_centroid)-np.array(original_centroid))/np.array(original_centroid)*100.0)>self.tol:
                    optimized = False
                   
            
            if optimized:
                break
            
            
                
    def predict(self,data):
        distances = [np.linalg.norm(np.array(row[:-1])-np.array(self.centroids[centroid])) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
        
    
    def path(self):
        for i in range(self.k):
            x,y=[],[]
            for j in range(len(self.all_centroids[i])):
                x.append(self.all_centroids[i][j][0])
                y.append(self.all_centroids[i][j][1])
            print(x,y)
            plt.plot(x,y,marker='*')
           
               
            


# In[7]:


accuracy,k=[],[]
for j in range(1,4):
    k.append(j)
    clf=K_Means(j)
    clf.fit(train_data)
    correct=0
    total=0
    for row in test_data:
        ans=clf.predict(row)
        if ans==row[-1]:
            correct+=1
        total+=1
    temp=(correct/total)*100
    accuracy.append(temp)
    print("For k=",j,", accuracy is ",accuracy[j-1])       

plt.plot(k,accuracy)    

