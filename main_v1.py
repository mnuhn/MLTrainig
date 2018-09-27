import re, sys, glob
from os.path import expanduser
import numpy as np
from lxml import etree
import matplotlib.pyplot as plt

from process_tcx_v4 import process_file, process_files, printtags
from estimator_v2 import normalize_array, linreg_rough, linreg_detailed, linreg_multidim

# columns, features, labels
COLUMNS = ['tottime', 'avwatts', 'avhr', 'avcad', 'best20minpower']
FEATURES = COLUMNS[:-1]
LABEL = COLUMNS[-1]

# power zones
powerzones = []
deltapow = 50.
maxpow = 1500.
for i in range(int(maxpow/deltapow)):
    powerzones.append(deltapow*i)

# heartrate zones
hrzones = []
deltahr = 10.
minhr = 40.
maxhr = 200.
for i in range(int((maxhr-minhr)/deltahr)):
    hrzones.append(minhr+deltahr*i)

# cadence zones
cadzones = [0.]
deltacad = 10.
mincad = 30.
maxcad = 140.
for i in range(int((maxcad-mincad)/deltacad)):
    cadzones.append(mincad+deltacad*i)

def main(argv=None):

    #printtags(inputfile,100)

    #inputfile = "GuelphSwimBike.tcx" 
    #print(process_file(inputfile, powerzones, hrzones, cadzones))

    #inputfolder = "/home/rummelm/local_files/tapiriik/"
    inputfolder = expanduser("~")+"/local_files/tapiriik/"
    #inputfiles = glob.glob(inputfolder+"*2018-08-14*")
    #inputfiles = glob.glob(inputfolder+"*half*")
    inputfiles = glob.glob(inputfolder+"*Pennsyl*") 
    #print(inputfiles)
    dataset = process_files(inputfiles, COLUMNS, powerzones, hrzones, cadzones) 
    print(dataset)
    #print(np.random.sample((4,6)))
    
    # Linear Regression
    #print(linreg_rough(dataset, 0.85, FEATURES, LABEL))
    #FEATURE = FEATURES[2]
    #linreg_detailed(dataset, 0.85, FEATURE, LABEL)
    #print("1dim linreg with feature "+FEATURE)

    # Multidim linear Regression
    linreg_multidim(dataset, 0.85, FEATURES, LABEL)

if __name__ == "__main__":
    sys.exit(main())
