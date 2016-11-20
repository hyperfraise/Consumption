# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:07:26 2016

@author: Damien
"""
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
import cStringIO

#telecharger les donnees de la base
def download_range(d_start, d_end):
    t_start = time.time()
    
    #print("connection to engine...")
    engine = create_engine('postgresql+psycopg2://elum_dev:Frenchtech25@bonima-data.c0unq3v5cib6.us-west-2.rds.amazonaws.com:5432/dashboard_data_prod1')
    conn = engine.connect()    
    #print("connected to engine")
    
    
    params  = {"t1":d_start,"t2":d_end}
    
    #je recupere presque toujoursseulement les colonnes time et total
    SQLrequest = conn.execute("SELECT DISTINCT time,total FROM bonima_data WHERE time >= %(t1)s AND time <= %(t2)s",params)
    
    #on retrie en utilisant time au cas ou
    df = pd.DataFrame(SQLrequest.fetchall())
    df.columns = SQLrequest.keys()
    #data = df.set_index('time')
    df = df.sort_values(by='time')
    df = df.reset_index()
    df = df.drop('index',1)
    #print("Data downloaded")
    
    t_end= time.time()
    duration = t_end - t_start
    print(duration)
    conn.close()
    return df

#creer facilement un dataset input/output avec une time series
def createTrainList(timeserie, nb_features, n_samples):
    l = len(timeserie)
    tab = [timeserie[i-nb_features:i] for i in range(nb_features,l-n_samples+1)]
    y = [timeserie[i:i+n_samples] for i in range(nb_features,l-n_samples+1)]
    tab = np.array(tab)
    y=np.array(y)
    return (tab,y)

#normaliser un vecteur par sa moyenne
def resize(vector):
    size = np.mean(vector)
    return [i/size for i in vector]

#transformer un nombre en vecteur binaire le representant
def binarize(k,size):
    res = [0]*size
    res[k] = 1
    return res

#reparer les données telechargees : a tester/ameliorer
def repair(raw_data,n_download):
    start = raw_data['time'][0]
    normal = pd.date_range(start,start+timedelta(minutes=n_download*1440-1),freq="min")
    copy = np.array(raw_data.as_matrix())
    for i in range(len(copy)):
        norm = normal[i]
        if copy[i][0] != norm:
            copy = np.insert(copy, i,[norm]+[copy[i][1]], 0)
    if len(copy)!=len(normal):
        for i in range(len(copy,len(normal))):
            norm = normal[i]
            copy = np.insert(copy,len(copy),[norm]+[copy[len(copy)-1][1]], 0)
    repaired = pd.DataFrame(columns=raw_data.columns.values.tolist(),data = copy)
    return repaired

#transformer un vecteur binaire en nombre
def unbinarize(v):
    return np.argmax(v)

#transformer un vecteur de probabilites en vecteur binaire en choisissant le max de vraisemblance
def choose(array):
    choice = [0]*len(array)
    k = np.argmax(np.array(array))
    choice[k] = 1
    return choice

#outrepasser la limite de print
def fullprint(*args, **kwargs):
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold='nan')
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)

#upload des datas dans la base de donnees
def upload(d_start,d_end,data):
    
    #print("connection to engine...")
    engine = create_engine('postgresql+psycopg2://elum_dev:Frenchtech25@bonima-data.c0unq3v5cib6.us-west-2.rds.amazonaws.com:5432/predictions_modeles')
    #print("connected to engine")    
    raw_connection = engine.raw_connection()
    cursor = raw_connection.cursor()
    output = cStringIO.StringIO()

    #construire le dataframe a envoyer, suppose ici à granularite de 10 minutes
    #et de taillle 1 jour
    raw_firstcol = [d_start]*144
    raw_secondcol = pd.date_range(d_start,d_end,freq = "10T")
    secondcol=[]    
    firstcol = []
    thirdcol = []
    for i in raw_secondcol:
        secondcol.append(str(i))
    for i in raw_firstcol:
        firstcol.append(str(i))
    for i in data:
        thirdcol.append(str(i))
    #cols = np.concatenate((firstcol,secondcol), axis = 0)
    #cols = np.concatenate((cols,thirdcol), axis = 0)
    #cols = np.reshape(cols, (-1, 3))
    
    #mes uploads ont ces trois colonnes
    df = pd.DataFrame()
    #une qui indique le moment où la prediction est faite
    df['time_start'] = firstcol
    #une qui indique les moments que la prediction predit
    df['time_forecast'] = secondcol
    #une qui stocke les valeurs predites
    df['forecast'] = thirdcol

    df.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    
    #print("importing data to POSTGRE...")
    #importer dans la table souhaitee
    cursor.copy_from(output, 'my_table', null="")    
    raw_connection.commit()    
    #print ("data imported to SQL")
    cursor.close()
    raw_connection.close()
