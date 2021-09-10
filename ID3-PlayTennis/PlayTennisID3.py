import pandas as pd
from math import log2
import json


target=''
clist=[]
mark={}
C=int(0)

def Entropy(df):
    cnt=df.count()[target]
    E=0.
    for c in clist:
       d=df.loc[df[target]==c]
       p=d.count()[target]/cnt
       if p==0. or p==1.:
           return 0.
       E+=-p*log2(p)
    return E

def IG(df,f):
    ret=Entropy(df)
    for obj in df[f].unique():
        d=df.loc[df[f]==obj]
        ret-=d.count()[f]*Entropy(d)/df.count()[f]
    return ret

def build_tree(df,par,edgelabel):
    ret={}
    global C
    maxig=0.
    nownode=''
    for f in features:
        if not mark[f] and IG(df,f)>maxig:
            maxig=IG(df,f)
            nownode=f
    if maxig==0.:
        C+=1
        if df.iloc[0][target]=='Yes':
            ret['node name']='Yes'
        else:
            ret['node name']='No'
        return ret
    C+=1
    v=C
    ret['node name']=nownode
    childs=list(df[nownode].unique())
    mark[nownode]=True
    for c in childs:
        subdf=df.loc[df[nownode]==c]
        ret[c]=build_tree(subdf,v,c)
    return ret

data=pd.read_csv("PlayTennis.csv")
features=list(data.columns)
target=features[-1]
clist=list(data[target].unique())

for f in features:
    if f!=target:
        mark[f]=False
mark[target]=True

with open("result.json",'w') as fp:
    json.dump(build_tree(data,-1,''),fp,indent=4)

with open("result.json",'r') as fp:
    dt=json.load(fp)

import graphshow
graphshow.show(dt)
