import re, sys, glob
from os.path import expanduser
import numpy as np
import pandas as pd
from lxml import etree
import matplotlib.pyplot as plt

def training_cycles(dataset, columns_analyze, training_cylce_earliestdate, training_cylce_lastdate, training_cycle_length):

    # Set datetime as index
    dataset['Datetime'] = pd.to_datetime(dataset['actdate'])
    dataset = dataset.set_index('Datetime')
    dataset = dataset.drop(['actdate'], axis=1)

    # temporarily sort by datetime index
    dataset = dataset.sort_index()
    print(dataset)
 
    # Resample dataset
    #dataout = dataset.resample('10D', label='right').apply(myresampler)
    dataout1 = dataset.resample('10D', label='right').agg({'tottime': np.sum})
    dataout2 = dataset.groupby(pd.Grouper(freq='10D', label='right')).apply(weighted_mean)

    dataout = pd.concat([dataout1, dataout2], axis=1)
    #dataout = dataout.rename(columns={'':'wavwatts'})
    dataout.rename(columns = {list(dataout)[1]: 'wavwatts'}, inplace = True)

    # Define new dataframe, index?
    #dataout = pd.DataFrame(columns=columns_analyze)

    # Loop over all entries of dataset
    #N = dataset.shape[0]
    #for i in range(N):

    return dataout

def weighted_mean(x):
    return ( (x['avwatts'] * x['tottime']).sum() ) / x['tottime'].sum()
