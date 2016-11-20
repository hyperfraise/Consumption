# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:42:41 2016

@author: Damien
"""
#general purpose
from __future__ import division
import math
import time
t_start = time.time()
import pandas as pd
import datetime
import boto3
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine
import pytz
import numpy as np
import csv
import datetime as dt
import numpy as np
from datetime import timedelta
from functions import *
from sklearn.svm import SVR

####download data
now = dt.utcnow() - timedelta(days=1) 
local = pytz.timezone('Africa/Casablanca')
utc = pytz.timezone("utc")
end = utc.localize(now).astimezone(local).replace(tzinfo=None)
start = dt.datetime.strptime('2015 10 1','%Y %m %d').replace(tzinfo=None)
raw_data = download_range(start,end)
n_download = int(abs(start-end).days)
#on met la data en kW aussi avec *3.6/60
raw_data = repair(raw_data,n_download)['total']*3.6/60

####shape data
raw_data = np.mean(raw_data.reshape(-1, 10), axis=1)
feat=np.array([0,7,14,18,19,20])*144
feat_m=np.array(range(21*144-20,21*144))
toapp=range(-14,1)
ind=raw_data
size=0.01

#predire une journee
def day(start,end,base):
    m=len(raw_data)-start*144
    running=raw_data[range(m-1,m-21,-1)]
    results = [daymin(m,g,running,end,base) for g in range(144)]
    return results

#accorder de l'importance en fonction de l'eloignement
#il faudrait que l'importance optimale soit cross validee independamment pour chaque point
#pour l'instant j'ai cross valide la meilleure exponentielle
def kernel(dec,end,base):
    return base*math.exp(end*dec)

#predire chaque point
def daymin(m,g,running,end,base):
    c=g+m
    dat=raw_data[:c]
    pred=np.array(dat[21*144:])
    preds=np.array([])
    inds=[]
    d=c%1008
    dec=(c-len(raw_data))%144
    for i in range(len(pred)):
        k=i%1008
        if k==d:
            preds=np.append(preds,pred[[i+x for x in toapp]])
            inds=inds+[np.ravel(np.append(ind[[i+x+y for x in feat]],ind[[i+n+y-g for n in feat_m]]*kernel(dec,end,base))) for y in toapp]
    preds=np.array(preds)
    inds=np.array(inds)
    pr1 = SVR(kernel='linear',C=0.1,epsilon=0.01).fit(size*inds,size*preds)
    X=np.append(dat[[c-21*144+x for x in feat]],running*kernel(dec,end,base))
    pt1=pr1.predict(size*X.reshape(1,-1))/size
    return pt1[0]


resultsn = day(0,-0.01,0.2)
resultsn=np.array(resultsn)
f=resultsn
#on corrige les points negatifs
for i in range(len(f)):
    f[i]=max(f[i],0)

print "uploading now"
upload(now+timedelta(minutes = 10),now+timedelta(minutes = 1440),f)

t_end= time.time()
duration = t_end - t_start
print(duration)
