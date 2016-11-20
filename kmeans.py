# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:06:45 2016

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

#kmeans
from scipy import cluster
from sklearn.cluster import KMeans



############################ KMEANS  #############################

def mainkmeans(data_res,elbow):
    cent, var = cluster.vq.kmeans(data_res,elbow)
    #use vq() to get as assignment for each obs.
    assignment,cdist = cluster.vq.vq(data_res,cent)
    classes = np.array([binarize(i,elbow) for i in assignment])
    return classes
