# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:42:41 2016

@author: Damien
"""
#general purpose
from __future__ import division
print('importing...')
import math
import time
t_start = time.time()
import pandas as pd
import datetime
import boto3
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine
import numpy as np
import csv
import datetime as dt
from datetime import timedelta
import pytz
from kmeans import *
from SVR import *
from Trees import *
from functions import *


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
print("shaping data")
raw_data = np.mean(raw_data.reshape(-1, 10), axis=1)
data = np.reshape(np.array(raw_data), (n_download,-1))
norms = np.array([np.mean(i) for i in data])
data_res = np.array([resize(i) for i in data])

#taille de la validation
val = min(40,int(math.floor(0.20*n_download)))

print("clustering days")
#nombre de clusters, a ameliorer
n_cluster = min(14,n_download)
classes = mainkmeans(data_res,n_cluster)
usual_norms = [0.0]*n_cluster

#on reclusterise tant qu'il y a des clusters vides
#on recueille la norme typique d'un cluster
for i in range(n_cluster):
    b = np.array([unbinarize(np.array((classes[j]))) == i for j in range(len(classes))])
    usual_norms[i] = np.mean(norms[b])
while np.isnan(usual_norms).any():
    print 'bad clusters, reclustering'    
    classes = mainkmeans(data_res,n_cluster)
    usual_norms = [0.0]*n_cluster
    for i in range(n_cluster):
        b = np.array([unbinarize(classes[j]) == i for j in range(len(classes))])
        usual_norms[i] = np.mean(norms[b])


####execute modules
print("Tree for cluster prediction")
#on predit la prochaine classe
classe_test = TreeClusters(classes)

print("decision tree for clusters")
#on y associe une forme de journee
day_pred = mainTreePredict(classes,classe_test,data_res)

print("predicting l1 norm")
#on predit la prochaine norme
size = mainSVR(classes,norms,val,classe_test)

####assemble predictions
print("assembling predictions")
tomorrow = np.array(day_pred*size)

print "uploading now"
upload(now, now + timedelta(minutes = 1430), tomorrow)


t_end= time.time()
duration = t_end - t_start
print(duration)
