import xml.etree.ElementTree as ET
#from scipy.misc import imread
import os

tree = ET.parse('Wednesday_Classic.gpx')
#tree = ET.parse('GuelphSwimBike.tcx')
root = tree.getroot()
xAx = []
yAx = []

for (i,j) in zip(root.iter('trk'),root.iter('ele')):
	#get the lat and lon values from the i list. Might have got these back to front
	xAx.append(float(i.get('lon')))
	yAx.append(float(i.get('lat')))

print yAx
