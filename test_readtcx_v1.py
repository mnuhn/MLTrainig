from __future__ import print_function
import re, sys
from docopt import docopt
import xml.etree.cElementTree as etree

watttag = "{http://www.garmin.com/xmlschemas/ActivityExtension/v2}Watts"

def process_wattage(element):
    wattval = 0
    if element.tag==watttag:
        wattval = element.text
    return wattval

def process_file(tcxfile):

    tree = etree.parse(tcxfile)
    root = tree.getroot()
    watts = []
    for element in root.iter():
        watts.append(process_wattage(element))
    return watts

def main(argv=None):
    #arguments = docopt(__doc__)
    #inputfile = arguments["INPUT"]
    inputfile = "GuelphSwimBike.tcx" 
    print(process_file(inputfile))

if __name__ == "__main__":
    sys.exit(main())
