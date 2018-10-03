import re, sys, glob
from os.path import expanduser
import numpy as np
import pandas as pd
from lxml import etree
import matplotlib.pyplot as plt

def training_cycles(dataset, columns_analyze, training_cycle_length):

    # Set datetime as index
    dataset['Datetime'] = pd.to_datetime(dataset['actdate'])
    dataset = dataset.set_index('Datetime')
    dataset = dataset.drop(['actdate'], axis=1)

    # Sort by datetime index
    dataset = dataset.sort_index()

    # Restdays: difference between activities in days
    datesvals = dataset.index.values
    datediffs = [ ]
    for i in range(1,len(datesvals)):
        datediffs.append(datesvals[i]-datesvals[i-1])
    datediffs = ( [ datediffs[0]] + datediffs ) / np.timedelta64(1,'D')
    dataset['restdays'] = datediffs
 
    # Resample dataset into training cycle time

    # add total time
    dataout = dataset.resample(training_cycle_length, label='right').agg({'tottime': np.sum})

    # time spend in previous and previous previous training cycle
    dataout['pre_tottime'] = np.insert(dataout.tottime.values[:-1], 0, 0.)
    dataout['pre_pre_tottime'] = np.insert(dataout.tottime.values[:-2], 0, [0.,0.])

    # mean of number of restdays
    datarest = dataset.resample(training_cycle_length, label='right').agg({'restdays': np.mean})
    dataout = pd.concat([dataout,datarest], axis=1)

    # build the weighted average of watts, hr, cad by activity time
    avcols = ['avwatts', 'avhr', 'avcad']
    for av in avcols: 
        dataoutav = dataset.groupby(pd.Grouper(freq=training_cycle_length, label='right')).apply(lambda x: weighted_mean(x, av))
        dataout = pd.concat([dataout, dataoutav.rename(av)], axis=1)

    # find max of best 20 min power
    datamax = dataset.resample(training_cycle_length, label='right').agg({'best20minpower': np.amax})
    dataout = pd.concat([dataout,datamax], axis=1)

    # compare columns with input columns to check if they agree
    if list(dataout.columns.values) != columns_analyze:
        print("Parameter COLUMNS_ANALYZE and Columns used in training_cycle do not agree. Abort!")
        sys.exit(0)

    return dataout

def weighted_mean(x, colname):
    return ( (x[colname] * x['tottime']).sum() ) / x['tottime'].sum()
