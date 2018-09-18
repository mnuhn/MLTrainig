from __future__ import print_function
import re, sys
import numpy as np
from docopt import docopt
#import xml.etree.cElementTree as etree
from lxml import etree

#namespaces
ns1 = "{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}"
ns2 = "{http://www.garmin.com/xmlschemas/ActivityExtension/v2}"

# tags under which desired quantities are stored
acttag = ns1+"Activity"
tottimetag = ns1+"TotalTimeSeconds"
watttag = ns2+"Watts"
cadtag = ns1+"Cadence"
hrtagcat = ns1+"HeartRateBpm"
hrtagval = ns1+"Value"

# power zones
powerzones = []
powweights = []
deltapow = 50.
maxpow = 1500.
for i in range(int(maxpow/deltapow)):
    powerzones.append(deltapow*i)
    powweights.append(0.)
powweights = powweights[:-1]
npz = len(powerzones)

# heartrate zones
hrzones = []
hrweights = []
deltahr = 10.
minhr = 40.
maxhr = 200.
for i in range(int((maxhr-minhr)/deltahr)):
    hrzones.append(minhr+deltahr*i)
    hrweights.append(0.)
hrweights = hrweights[:-1]
nhr = len(hrzones)

# cadence zones
cadzones = [0.]
cadweights = [0.]
deltacad = 10.
mincad = 30.
maxcad = 140.
for i in range(int((maxcad-mincad)/deltacad)):
    cadzones.append(mincad+deltacad*i)
    cadweights.append(0.)
cadweights = cadweights[:-1]
ncad = len(cadzones)


# values to be calculated
tottime = 0.0
avwatts = 0.0
avhr = 0.0
avcad = 0.0
acttype = " "
watts = []
cadences = []
heartrates = []
 
# process tcx file, use power and heartrate and cadence zones as input(TODO!) and return(TODO!) vector with all desired quantities
def process_file(tcxfile):

    # make variables global so they can be modified within routine
    global acttype
    global tottime
    global avwatts
    global avhr    
    global avcad
    global powweights
    global hrweights
    global cadweights

    tree = etree.parse(tcxfile)
    root = tree.getroot()

    # hrnext is zero when previous element was not "HeartRateBpm"
    # this is due to different structure of xml file for heartrate
    # than for the other quantities
    hrnext = 0

    # loop through all entries of tcx file, storing all desired entries
    for element in root.iter():
        eltag = element.tag
        elval = element.text
        elatt = element.attrib

        if eltag == watttag:
            watts.append(float(elval))
            for i in range(npz-1):
                if float(elval) >= powerzones[i] and float(elval) < powerzones[i+1]:
                    powweights[i] += 1
                    break

        elif eltag == cadtag:
            cadences.append(float(elval))
            for i in range(ncad-1):
                if float(elval) >= cadzones[i] and float(elval) < cadzones[i+1]:
                    cadweights[i] += 1
                    break
 
        elif eltag == hrtagval and hrnext == 1:
            heartrates.append(float(elval))
            for i in range(nhr-1):
                if float(elval) >= hrzones[i] and float(elval) < hrzones[i+1]:
                    hrweights[i] += 1
                    break

        elif eltag == acttag:
            acttype = elatt

        # total time = lap time so we have to add up all lap times    
        elif eltag == tottimetag:
            tottime += float(elval)
            #print("Total time (s): "+str(tottime))
       
        # set hrnext to one if entry is "HeartRateBpm" so in next iteration
        # heartrate value can be saved
        if eltag == hrtagcat:
            hrnext = 1
        else:
            hrnet = 0

    # Calculate averages
    avwatts = np.average(watts)
    avhr = np.average(heartrates)
    avcad = np.average(cadences)

    # Normalize power- and heartrate-zones
    powweights = powweights/np.sum(powweights)
    hrweights = hrweights/np.sum(hrweights)
    cadweights = cadweights/np.sum(cadweights)

# help routine to find out what the tags are etc
def printtags(tcxfile,nprint):

    tree = etree.parse(tcxfile)
    root = tree.getroot()
    ncount  = 0
    for element in root.iter():
        print(element.tag+", "+str(element.attrib)+", "+element.text)
        ncount+=1
        if ncount > nprint:
            break

def main(argv=None):
    #arguments = docopt(__doc__)
    #inputfile = arguments["INPUT"]
    inputfile = "GuelphSwimBike.tcx" 

    #printtags(inputfile,100)
    
    process_file(inputfile)
    print(acttype)
    #print(tottime)
    #print(avwatts)
    #print(avhr)
    #print(avcad)
    print(cadzones)
    print(cadweights)
    #print(hrzones)
    #print(watts)
    #print(heartrates[0:8])
    #print(cadences)

if __name__ == "__main__":
    sys.exit(main())
