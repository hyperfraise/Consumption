# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:06:46 2016

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
from functions import *

#tree
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier as dtc
from kmeans import *
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.svm import SVC

###########################  TREE FOR THE DAY  ########################

#cacluler les  distances entre journees typique
def treedistances(classes,usual_norms,data_res,n_cluster,data):
    X_train = classes
    y_train = data_res
    
    clf = DecisionTreeRegressor()
    clf = clf.fit(X_train,y_train)
    
    classe_test = np.array([binarize(i,n_cluster) for i in range(n_cluster)])
    prediction = clf.predict(classe_test)
    prediction = np.array([usual_norms[i]*prediction[i] for i in range(len(prediction))])
    distances = np.array([[0.0]*n_cluster]*n_cluster)
    for i in range(n_cluster):
        for j in range(n_cluster):
            distances[i][j] = np.linalg.norm(prediction[i]-prediction[j],ord=1)/len(prediction[0])
    #on calcule aussi l'ecart-type au sein des clusters
    for i in range(n_cluster):
        b = np.array([unbinarize(classes[r]) == i for r in range(len(classes))])
        days = data[b]    
        var = np.array([k-prediction[i] for k in days]) 
        distances[i][i]=np.linalg.norm(np.ravel(var),ord=1)/len(np.ravel(var))
    return distances
    
def treenorms(classes,classes_test,norms,n_cluster,data_res):
    i=1
    j=4
    k=41
    k_f=6
    classes = mainkmeans(data_res,n_cluster)
    usual_norms = [0.0]*n_cluster
    for q in range(n_cluster):
        b = np.array([unbinarize(classes[r]) == q for r in range(len(classes))])
        usual_norms[q] = np.mean(norms[b])            
    
    data_svr = np.array([np.append(norms[s],usual_norms[unbinarize(classes[s])]) for s in range(len(norms))])
    
    sizes = data_svr[:,0]/k
    clas = data_svr[:,1]/k_f
    n_features = i
    cv_sizes,dummy = createTrainList(sizes,n_features,0)
    cv_data = np.array([np.append(cv_sizes[p],clas[p+1]) for p in range(len(cv_sizes)-1)])
    
    c_train = cv_data[:len(cv_data) - val]
    X_train,y_train = createTrainList(c_train,1,1)
    X_train = np.squeeze(X_train,1)
    y_train = np.squeeze(y_train,1)
    y_train = y_train[:,n_features-1]
    
    X_pred = np.append(sizes[len(sizes)-i:],usual_norms[unbinarize(classes_test)])
    clf = rf(n_estimators = 100, max_depth = j)
    clf = clf.fit(X_train,y_train)
    prediction = clf.predict(X_pred)*k
    
    return prediction[0]



def mainTreePredict(classes,classes_test,data_res):
    X_train = classes
    y_train = data_res
    
    clf = DecisionTreeRegressor()
    clf = clf.fit(X_train,y_train)
    
    prediction = clf.predict([classes_test])
    return prediction[0]


def TreeClusters(classes):
    nf=7
    n_cluster=14
    ct=classes[:len(classes)-1]
    X,y=createTrainList(ct,nf,1)
    X = np.array([np.ravel(i) for i in X])
    y=np.squeeze(y,1)
    y=np.array([unbinarize(i) for i in y])
    
    clf = SVC(kernel='linear').fit(X,y)
    
    ct=classes[len(classes)-nf:]
    X,y=createTrainList(ct,nf,0)
    X = np.array([np.ravel(i)  for i in X])
    
    preds=clf.predict(X)
    
    preds=binarize(preds[0],n_cluster)
    return preds
