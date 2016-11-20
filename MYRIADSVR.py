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
import random
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
print("predicting...")
raw_data = np.mean(raw_data.reshape(-1, 10), axis=1)
feat=np.array([0,7,14,18,19,20])*144
toapp=range(-14,1)
ind=raw_data
size=0.01

#on predit chaque point independamment avec les meme features
def day(g):
    dat=raw_data[:len(raw_data)-144+g]
    pred=np.array(dat[21*144:])
    preds=np.array([])
    inds=[]
    z=len(raw_data)+g
    d=z%1008
    for i in range(len(pred)):
        k=i%1008
        if k==d:
            preds=np.append(preds,pred[[i+x for x in toapp]])
            inds=inds+[np.ravel(ind[[i+x+y for x in feat]]) for y in toapp]
    preds=np.array(preds)
    inds=np.array(inds)
    pr1 = SVR(kernel='linear',C=0.6,epsilon=0.003).fit(size*inds,size*preds)
    X=raw_data[[z-21*144+x for x in feat]]
    pt1=pr1.predict(size*X.reshape(1,-1))/size
    return pt1[0]


results=np.array([day(g) for g in range(144)])

#on empeche le forecast d'etre negatif
f=np.ravel(results)
for i in range(len(f)):
    f[i]=max(f[i],0)

print "uploading now"
upload(now,now+timedelta(minutes = 1430),f)

t_end= time.time()
duration = t_end - t_start
print(duration)
