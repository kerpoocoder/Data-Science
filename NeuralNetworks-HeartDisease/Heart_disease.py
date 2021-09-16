# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:35:21 2021

@author: MohammadMahdi
"""
import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from random import random
from math import exp

class unit:
    def __init__(self,inp):
        self.weights=[random()/10-0.05 for i in range(inp+1)]
    
    def update(self,x,delta,zeta):
        self.weights[0]=self.weights[0]+delta*zeta
        for i in range(len(x)):
            self.weights[i+1]=self.weights[i+1]+x[i]*zeta*delta
            
    def run(self,x):
        d=self.weights[0]
#        print('x',len(x))
#        print('w',len(self.weights))
        for i in range(len(x)):
            d=d+x[i]*self.weights[i+1]
        return self.sigmoid(d)
    
    def sigmoid(self,net):
#        print(net)
        if abs(net)>25:
            return 0.00001
        return 1.0/(1.0+exp(-net))
        
class network:
    def __init__(self,nin,zeta,layer_size):
        self.layers=[]
        self.zeta=zeta
        lst=[]
        for s in layer_size:
            for n in range(s):
                u=unit(nin)
                lst.append(u)
            nin=s
            self.layers.append(lst)
            lst=[]
    
    def train(self,xtrain,ytrain,xval,yval):
        cnt=0
        ex=0
        while ex<0.8:
            for i in range(xtrain.shape[0]):
                self.update(list(xtrain.iloc[i]),ytrain.iloc[i],0)
            cnt=cnt+1
            ex=self.acc(xval,yval)
            print("epoch ",cnt," ",ex)
    
    def update(self,x,y,lev):
        if lev==len(self.layers)-1:
            o=self.layers[-1][0].run(x)
            delta=o*(1.0-o)*(float(y)-o)
            self.layers[-1][0].update(x,delta,self.zeta)
            return [delta]
        lst=[]
        o=0
        for i in range(len(self.layers[lev])):
            o=self.layers[lev][i].run(x)
            lst.append(o)
        delta=self.update(lst,y,lev+1)
        ret=[]
        d=0
        for i in range(len(self.layers[lev])):
            for j in range(len(self.layers[lev+1])):
                d=d+delta[j]*self.layers[lev+1][j].weights[i]
            ret.append(o*(1.0-o)*d)
            self.layers[lev][i].update(x,d*o*(1.0-o),self.zeta)
        return ret
    
    def acc(self,xval,yval):
        total=xval.shape[0]
        corr=0
        for i in range(total):
            if self.run(list(xval.iloc[i]))==yval.iloc[i]:
                corr=corr+1
        return corr/total
    
    def run(self,x):
        for lay in self.layers:
            x2=[]
            for u in lay:
                x2.append(u.run(x))
            x=x2
        if x[0]>=0.5:
            return 1
        return 0
    
db=pd.read_excel('Heart_Disease.xls')
X=db.loc[:,'age':'thal']
Y=db.loc[:,'num']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size= 0.3,random_state=1)
X_test,X_val,Y_test,Y_val=train_test_split(X_test,Y_test,test_size=0.5,random_state=1)
print(len(X.columns),X.head())
nn=network(len(X.columns),0.1,[105,1])
nn.train(X_train,Y_train,X_val,Y_val)
print("test accuracy ",nn.acc(X_test,Y_test))
        
                