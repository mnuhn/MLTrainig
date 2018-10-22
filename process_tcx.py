import re, sys, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from lxml import etree

class TcxStatus:
    OK = 0
    NOT_CYCLING = 1
    TOO_SHORT = 2
    WRONG_TIME_INTERVAL = 3
    NO_POWER_DATA = 4
    NO_HEART_RATE_DATA = 5
    NO_CADENCE_DATA = 6


# process a bunch of tcx files and return pandas data frame
def process_files(filelist, cols, powzones, hrzones, cadzones):

    #count how many files failed and for what reason
    allgood = 0
    notcycling = 0
    less20min = 0
    wrongtimeint = 0
    nowatts = 0
    noheartrate = 0
    nocadence = 0

    # process files
    output = []
    num_nocad = 0
    for ifile in filelist:
        processfile, filestatus = process_file(ifile, powzones, hrzones, cadzones)
        # append if output is not empty, i.e. file didn't fail
        if processfile != None:
            output.append(processfile)
        # count failures/successes
        if filestatus == Tcx.OK:
            allgood += 1
        elif filestatus == "notcycling":
            notcycling += 1
        elif filestatus == "less20min":
            less20min += 1
        elif filestatus == "wrongtimeint":
            wrongtimeint += 1
        elif filestatus == "nowatts":
            nowatts += 1
        elif filestatus == "noheartrate":
            noheartrate += 1
        elif filestatus == "nocadence":
            nocadence += 1

    # Create numpy array and feed it into tensorflow dataset 
    outputnp = np.array(output) 
    pdoutput = pd.DataFrame(outputnp, columns=cols)
    #dataset = tf.data.Dataset.from_tensor_slices(outputnp)

    # Convert datatypes to numeric except dates
    #colstonumeric = pdoutput.columns.drop('actdate')
    cols = pdoutput.columns
    pdoutput[cols] = pdoutput[cols].apply(pd.to_numeric, errors='ignore')

    # Print summary of failures/successes
    print(" ")
    print("Files successfully processed: "+str(allgood))
    print("Files that are not cycling: "+str(notcycling))
    print("Files that are less than 20 min: "+str(less20min))
    print("Files that have the wrong time delta: "+str(wrongtimeint))
    print("Files that have no watts: "+str(nowatts))
    print("Files that have no heartrate: "+str(noheartrate))
    print("Files that have no cadence: "+str(nocadence))
    print(" ")

    return pdoutput
    #return outputnp

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
    datetag = ns1+"Id"
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
    actdate = " "
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
                print("Warning! Input file" +tcxfile+" is not cycling!")
                return None, "notcycling"
        
        # Date of activity: time is 4 hours ahead of Westcoast winter time 
        # so in Greenwich time(?)
        elif eltag == datetag:
            actdate = elval

        # check time interval of tcx file
        elif eltag == timetag:
            if time1 == 0. and time2 == 0.:
                time1 = float(elval[-7:-1])
            elif time2 == 0.:
                time2 = float(elval[-7:-1])

        # total time = lap time so we have to add up all lap times    
        elif eltag == tottimetag:
            tottime += float(elval)
            # if total time less than 20 minutes discard:
            if tottime < 60.*20.:
                print("Total time less than 20 minutes  - file discarded: "+tcxfile)
                return None, "less20min"

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
      
        # set hrnext to one if entry is "HeartRateBpm" so in next iteration
        # heartrate value can be saved
        if eltag == hrtagcat:
            hrnext = 1
        else:
            hrnet = 0

    # Check if time difference is set right, otherwise abort and print error
    if time2-time1 != deltat:
        print("Time interval in tcx file "+tcxfile+" is "+str(time2-time1)+" s. But set time interval is "+str(deltat)+" s. File discarded.")
        #sys.exit(0)
        return None, "wrongtimeint"

    # if there is no power data (= watts empty or zero) throw away
    if watts == [] or np.average(watts) == 0.:
        print("No wattage data - file discarded: "+tcxfile)
        return None, "nowatts"

    # if there is no heartrate data (= hr empty or zero) throw away
    if heartrates == [] or np.average(heartrates) == 0.:
        print("No heartrate data - file discarded: "+tcxfile)
        return None, "noheartrate"

    # if there is no cadence data (= cadence empty or zero) throw away
    if cadences == [] or np.average(cadences) == 0.:
        print("No cadence data - file discarded: "+tcxfile)
        return None, "nocadence"


    """if np.sum(hrweights) == 0.:
        print(tcxfile+" something wrong with it")
        sys.exit(0)"""

    # Calculate averages.
    avwatts = np.average(watts)
    avhr = np.average(heartrates)
    # Exclude zeroes for cadence.
    cadencesnp = np.array(cadences)
    cadencesnonzero = cadencesnp.ravel()[np.flatnonzero(cadencesnp)]
    avcad = np.average(cadencesnonzero) 

    # Normalize power- and heartrate-zones.
    powweights = powweights / np.sum(powweights)
    hrweights = hrweights / np.sum(hrweights)
    cadweights = cadweights / np.sum(cadweights)

    # Return vector of desired quantities.
    return [actdate, tottime, avwatts, avhr, avcad, best20minpower], TcxStatus.OK

# test routine to find out what the tags are etc
def print_tags(tcxfile, nprint):
    tree = etree.parse(tcxfile)
    root = tree.getroot()
    ncount  = 0
    for element in root.iter():
        print(element.tag + ", " + str(element.attrib) + ", " + element.text)
        ncount+=1
        if ncount > nprint:
            break
