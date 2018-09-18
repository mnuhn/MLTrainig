import re, sys, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from lxml import etree

#TODO:
#1. do linear regression with dataset in tensorflow

# process a bunch of tcx files and return pandas data frame
def process_files(filelist, powzones, hrzones, cadzones):

    output = []
    for ifile in filelist:
        output.append(process_file(ifile, powzones, hrzones, cadzones))

    # Create numpy array and feed it into tensorflow dataset 
    outputnp = np.array(output)
    #pdoutput = pd.data_frame(output, index=index)
    #dataset = tf.data.Dataset.from_tensor_slices(outputnp)
    
    return outputnp

# process tcx file, use power and heartrate and cadence zones as input and 
# return vector with all desired quantities
def process_file(tcxfile, powzones, hrzones, cadzones):

    #namespaces
    ns1 = "{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}"
    ns2 = "{http://www.garmin.com/xmlschemas/ActivityExtension/v2}"

    # tags under which desired quantities are stored
    acttag = ns1+"Activity"
    timetag = ns1+"Time"
    tottimetag = ns1+"TotalTimeSeconds"
    watttag = ns2+"Watts"
    cadtag = ns1+"Cadence"
    hrtagcat = ns1+"HeartRateBpm"
    hrtagval = ns1+"Value"

    # data taking frequency in seconds
    deltat = 1.
    time1 = 0.
    time2 = 0.
    N20 = deltat*60.*20.
    min20array = []
    min20power = 0.
    best20minpower = 0.
    n20 = 0

    # values to be calculated
    tottime = 0.0
    avwatts = 0.0
    avhr = 0.0
    avcad = 0.0
    acttype = " "
    watts = []
    cadences = []
    heartrates = []
    npz = len(powzones)
    nhr = len(hrzones)
    ncad = len(cadzones)
    powweights = [0.]*(npz-1)
    hrweights = [0.]*(nhr-1)
    cadweights = [0.]*(ncad-1)

    # etree variables
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
 
        # Activity Type
        if eltag == acttag:
            acttype = elatt
            # at the moment only cycling:
            if str(acttype) != "{'Sport': 'Biking'}":
                print("Warning! Input file is not cycling!")
                return

        # check time interval of tcx file
        elif eltag == timetag:
            if time1 == 0. and time2 == 0.:
                time1 = float(elval[-7:-1])
            elif time2 == 0.:
                time2 = float(elval[-7:-1])

        # Power
        elif eltag == watttag:
            watts.append(float(elval))
            for i in range(npz-1):
                if float(elval) >= powzones[i] and float(elval) < powzones[i+1]:
                    powweights[i] += 1
                    break

            # best 20 min average
            min20array = np.append(min20array,float(elval))
            min20power += float(elval)/float(N20)
            if n20 == N20:
                best20minpower = min20power
            elif n20 > N20:
                # substract 0th entry
                min20power -= min20array[0]/float(N20)
                # update best 20 min power if 20 min power greater
                if min20power > best20minpower:
                    best20minpower = min20power
                # delete first entry of 20 min array
                min20array = np.delete(min20array,0)
            # increase counter
            n20 = n20+1

        # Heartrate
        elif eltag == cadtag:
            cadences.append(float(elval))
            for i in range(ncad-1):
                if float(elval) >= cadzones[i] and float(elval) < cadzones[i+1]:
                    cadweights[i] += 1
                    break

        # Cadence
        elif eltag == hrtagval and hrnext == 1:
            heartrates.append(float(elval))
            for i in range(nhr-1):
                if float(elval) >= hrzones[i] and float(elval) < hrzones[i+1]:
                    hrweights[i] += 1
                    break


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

    # Check if time difference is set right, otherwise abort and print error
    if time2-time1 != deltat:
        print("Time interval in tcx file is "+str(time2-time1)+" s. But set time interval is "+str(deltat)+" s. Abort!!!")
        sys.exit(0)

    # if there is no power data (= watts empty or zero) throw away
    if watts == [] or np.average(watts) == 0.:
        print("No wattage data")
        return

    # Calculate averages
    avwatts = np.average(watts)
    avhr = np.average(heartrates)
    # exclude zeroes for cadence
    cadencesnp = np.array(cadences)
    cadencesnonzero = cadencesnp.ravel()[np.flatnonzero(cadencesnp)]
    avcad = np.average(cadencesnonzero) 

    # Normalize power- and heartrate-zones
    powweights = powweights/np.sum(powweights)
    hrweights = hrweights/np.sum(hrweights)
    cadweights = cadweights/np.sum(cadweights)

    # Return vector of desired quantities
    #return [acttype, tottime, avwatts, avhr, avcad, powweights, hrweights, cadweights, best20minpower]
    return [tottime, avwatts, avhr, avcad, best20minpower]

# test routine to find out what the tags are etc
def printtags(tcxfile,nprint):

    tree = etree.parse(tcxfile)
    root = tree.getroot()
    ncount  = 0
    for element in root.iter():
        print(element.tag+", "+str(element.attrib)+", "+element.text)
        ncount+=1
        if ncount > nprint:
            break


