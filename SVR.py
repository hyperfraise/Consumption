# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:06:40 2016

@author: Damien
"""

#general purpose
from __future__ import division
import math
import time
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
from functions import createTrainList

#svr
from sklearn.svm import SVR
from sklearn.svm import SVR
from sklearn.base import BaseEstimator
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint as sp_randint

###########################  SVR  ########################
#pour cross valider
def onerun(n,epsilon,norms,classes,val):
    size=max(norms)
    sizes=norms/size
    res=[]
    preds=[]
    used=[]
    for k in range(val,0,-1):
        siz = sizes[:len(sizes)-k]
        clas = classes[:len(classes)-k]
        X=np.concatenate((createTrainList(siz,n,1)[0],clas[n:]),axis=1)
        y=siz[n:]
        
        model = SVR(epsilon=epsilon,kernel='linear').fit(X,y)
        
        X=np.append(siz[len(siz)-n:],classes[len(clas)]).reshape(1, -1)
        y=sizes[len(siz)]
        pred = model.predict(X)
        acc=size*np.linalg.norm(pred-y,ord=1)
        res.append(acc)
        preds.append(pred)
        used.append(y)
    acur=np.mean(res)

    return n,epsilon,acur


#je cross valide un peu mon SVR ici
def mainSVR(classes,norms,val,classe_test):
    resultsn =[onerun(n,epsilon,norms,classes,val) for epsilon in np.arange(0.001,0.04,0.001) for n in range(1,9)]
    resultsn = np.array(resultsn)
    n,epsilon,accuracy = resultsn[np.argmin(resultsn[:,2])]
    n=int(n)
    size=max(norms)
    sizes=norms/size
    X=np.concatenate((createTrainList(sizes,n,1)[0],classes[n:]),axis=1)
    y=sizes[n:]
    
    model = SVR(epsilon=epsilon,kernel='linear').fit(X,y)
    
    X=np.append(sizes[len(sizes)-n:],classe_test)    
    pred=size*model.predict([X])
    return pred
