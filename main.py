from lxml import etree
from os.path import expanduser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re, sys, glob

import process_tcx
# TODO(mrumm): Change like process_tcx.
from estimator import normalize_array, linreg_rough, linreg_detailed, linreg_multidim
from resample import resample, weighted_mean

# first and last date of training cycle
firstdate  = np.datetime64('2018-02-14')
lastdate  = np.datetime64('2018-06-16')

# columns, features, labels
COLUMNS_READ = ['actdate', 'tottime', 'avwatts', 'avhr', 'avcad', 'best20minpower']
COLUMNS_ANALYZE = ['tottime', 'pre_tottime', 'pre_pre_tottime', 'restdays', 'avwatts', 'avhr', 'avcad', 'best20minpower']
FEATURES = COLUMNS_ANALYZE[:-1]
LABEL = COLUMNS_ANALYZE[-1]

# Power zones.
powerzones = []
deltapow = 50.
maxpow = 1500.
for i in range(int(maxpow/deltapow)):
    powerzones.append(deltapow*i)

# Heartrate zones.
hrzones = []
deltahr = 10.
minhr = 40.
maxhr = 200.
for i in range(int((maxhr-minhr)/deltahr)):
    hrzones.append(minhr+deltahr*i)

# Cadence zones.
cadzones = [0.]
deltacad = 10.
mincad = 30.
maxcad = 140.
for i in range(int((maxcad-mincad)/deltacad)):
    cadzones.append(mincad+deltacad*i)

def main(argv=None):
    #process_tcx.print_tags(inputfile,100)
    #inputfile = "GuelphSwimBike.tcx" 
    #print(process_tcx.process_file(inputfile, powerzones, hrzones, cadzones))

    #inputfolder = "/home/rummelm/local_files/tapiriik/"
    #inputfolder = expanduser("~")+"/local_files/tapiriik/"
    #inputfiles = glob.glob(inputfolder+"*2018-08-14*")
    #inputfiles = glob.glob(inputfolder+"*half*")
    #inputfiles = glob.glob(inputfolder+"*Pennsyl*") 
    
    inputfolder = expanduser("~")+"/local_files/tapiriik/"
    #inputfolder = expanduser("~")+"/Dropbox/MLTraining/3month/"
    inputfiles = glob.glob(inputfolder+"*")
    #inputfiles = glob.glob(inputfolder+"*2016-09-01_10-35-24_Morning Ride_Cycling*")
    #process_tcx.print_tags(inputfiles[0],100)
    
    #print(inputfiles)
    datasetdate = process_tcx.process_files(inputfiles, COLUMNS_READ,
            powerzones, hrzones, cadzones) 
    print(datasetdate)

    dataset = resample(datasetdate, COLUMNS_ANALYZE, '10D')
    print(dataset)

    # Print some Pearson Correlation coefficients
    #print(" ")
    """print("Pearson Correlation coefficients between features and Label:"+LABEL)
    for col in FEATURES:
        corrcoef = np.corrcoef(dataset[col].values, dataset[LABEL].values)[0][1]
        print(col+": "+str(corrcoef))"""

    # Onedim Linear Regression
    #print(linreg_rough(dataset, 0.85, FEATURES, LABEL))
    #FEATURE = FEATURES[2]
    #linreg_detailed(dataset, 0.85, FEATURE, LABEL)
    #print("1dim linreg with feature "+FEATURE)

    # Multidim linear Regression
    linreg_multidim(dataset, 0.85, FEATURES, LABEL)

if __name__ == "__main__":
    sys.exit(main())
