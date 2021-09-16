# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 03:51:36 2021

@author: MohammadMahdi
"""

from keras.datasets import mnist
import numpy as np
import pandas as pd
from math import sqrt
import random
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA

def _PCA(X,k):
    #normalization
    X=np.array(X).astype(float)
    m=X.shape[0]
    n=X.shape[1]
    mu=np.sum(X,axis=0)*(1.0/m)
    for i in range(m):
        X[i]=X[i]-mu
    for i in range(n):
        x=X[:,i]
        zigmaj=sqrt(np.dot(x,x)*(1.0/m))
        if zigmaj==0.0:
            continue
        for j in range(m):
            X[j,i]=X[j,i]*(1.0/zigmaj)
            
    #generating sigma
    sigma=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            sigma[i,j]=np.dot(X[:,i],X[:,j])/m
    
    #finding eigenvectors
    eigenValues, eigenVectors = linalg.eig(sigma)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    b=[]
    for i in range(k):
        b.append(eigenVectors[:,i])
    
    #new data
    newX=np.zeros([m,k])
    for i in range(m):
        for j in range(k):
            newX[i][j]=np.dot(X[i],b[j])
    
    return pd.DataFrame(newX)
    

col={0:'red',
     1:'blue',
     2:'green',
     3:'orange',
     4:'purple',
     5:'brown',
     6:'pink',
     7:'gray',
     8:'black',
     9:'cyan',}

#loading
(train_X, train_y), (test_X, test_y) = mnist.load_data()
rand=random.sample(range(0,60000),58500)
X=pd.DataFrame(np.array([x.reshape(784) for x in train_X])).drop(rand)
Y=pd.DataFrame(train_y).drop(rand)
X=_PCA(X,50)
print("PCA Done, features reduced to: ",50)
X = np.array(TSNE(n_components=2, perplexity=30.0, n_iter=1000).fit_transform(X))
for i in range(X.shape[0]):
    plt.scatter(X[i][0],X[i][1],color=col[int(Y.iloc[i])])

#pca=PCA(n_components=2)
#pca.fit_transform(X)
#for i in range(X.shape[0]):
#    plt.scatter(pca.components_[0][i],pca.components_[1][i],color=col[int(Y.iloc[i])])
    
plt.savefig('mnist_pca_tsne.png')
plt.show()
