# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:30:05 2016

@author: Damien
"""
import sys
import numpy as np
import pandas as pd
import csv
import subprocess
from threading import Timer
import pytz
from functions import *
import datetime
import time as t
from datetime import timedelta

t_start = t.time()

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


raw_data = np.array(np.mean(raw_data.reshape(-1, 10), axis=1))

#on lance le fichier en R
#cette fonction est dans function.py, qui a donc une version specifique a cette prediction
prediction = predictionConsommation(raw_data)


upload(now, now + timedelta(minutes = 1430), prediction)


t_end= t.time()
duration = t_end - t_start
print(duration)
