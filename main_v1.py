import re, sys, glob
import numpy as np
from lxml import etree

from process_tcx_v4 import process_file, process_files, printtags

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

    inputfolder = "/home/rummelm/local_files/tapiriik/"
    #inputfiles = glob.glob(inputfolder+"*2018-08-14*")
    #inputfiles = glob.glob(inputfolder+"*half*")
    inputfiles = glob.glob(inputfolder+"*Pennsyl*")
    print(process_files(inputfiles, powerzones, hrzones, cadzones))
    #print(np.random.sample((4,6)))
    

if __name__ == "__main__":
    sys.exit(main())
