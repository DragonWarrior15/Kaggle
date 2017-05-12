#-------------------------------------------------------------------------------
# Name:        EDD_L_G for gzip infile
# Purpose:
#
# Author:      lei.chen
#
# Created:     17/01/2013
# Copyright:   (c) lei.chen 2013
# Licence:     lei.chen@operasolutions.com
#-------------------------------------------------------------------------------

#!/usr/bin/env python2.7

import csv
import os
import sys
from numpy import *
#from collections import Counter
from optparse import OptionParser
import gzip

"""
    Usage: pipe a csv file with headers into EDD.py
    Output: EDD of the data to stdout
    Example:
        python EDD_L_G.py -i in.csv.gz -o EDD.csv
"""

##################################################################
# 1.Parse options
##################################################################
parser = OptionParser(
    description="""EDD for Gzip File....""")
parser.add_option("-d","--delimiter",
          action="store", type="string", dest="deli", default=',',
          help="Name of csv file to be processed. (string) Format: [INFILE]")
parser.add_option("-i","--in_file",
          action="store", type="string", dest="in_file", default='error',
          help="Name of csv file to be processed. (string) Format: [INFILE]")
parser.add_option("-o","--out_file",
          action="store", type="string", dest="out_file", default='error',
          help=" Name of output file. ")

(options,args) = parser.parse_args()

##################################################################
# 2.Process data
##################################################################
reader = csv.DictReader(open(options.in_file,'rbU'),delimiter=options.deli)
#reader = csv.DictReader(gzip.open(options.in_file,'rb'),delimiter=options.deli)
numeric = set(reader.fieldnames)
print 'var list:' , numeric

categoric = set()
numStats = dict([(i, []) for i in numeric])
catStats = {}
blankStats = dict([(i, 0) for i in numeric])
# Read in lines
print >> sys.stderr, 'Reading file'
for (ct, row) in enumerate(reader):
    if ct % 10000 == 0:
        print >> sys.stderr, '\r%i'%(ct),
    for (k, v) in row.iteritems():
        if v == '':
            blankStats[k] += 1
            continue
        if k in numeric:
            try:
                numStats[k].append(float(v))
            except ValueError:
                # This field is now categoric
                numeric.remove(k)
                categoric.add(k)
                catStats[k] = [str(i) for i in numStats[k]]
                catStats[k].append(v)
                del numStats[k]
        else:
            try:
                catStats[k].append(v)
            except:
                print k,v
print >> sys.stderr, '\r%i'%(ct)
# Calculate statistics
print >> sys.stderr, 'Calculating statistics'
mins = {}
maxs = {}
means = {}
meds = {}
stds = {}
modes = {}
hist = {}
Pct1 = {}
Pct5 = {}
Pct25 = {}
Pct75 = {}
Pct95 = {}
Pct99 = {}
for f in numeric:
    if len(numStats[f]) == 0:
        mins[f] = maxs[f] = means[f] = meds[f] = stds[f] = None
        Pct1[f] = Pct5[f] = Pct25[f] = Pct75[f] = Pct95[f] = Pct99[f] = None
    else:
        x = array(numStats[f])
        mins[f] = x.min()
        maxs[f] = x.max()
        means[f] = mean(x)
        meds[f] = median(x)
        stds[f] = std(x)
        #hist[f] = hist_num(x)
        Pct1[f]  = percentile(x,1)
        Pct5[f]  = percentile(x,5)
        Pct25[f] = percentile(x,25)
        Pct75[f] = percentile(x,75)
        Pct95[f] = percentile(x,95)
        Pct99[f] = percentile(x,99)

catVals = {}
for f in categoric:
    #hist[f] = hist_cat(array(catStats[f]))
    vals = {}
    for v in catStats[f]:
        vals[v] = vals.get(v,0) + 1
    vals = vals.items()
    vals.sort(key = lambda x: x[1], reverse = True)
    if vals[0][1] == 1:
        catVals[f] = ['All Unique' for i in range(20)]
    else:
        catVals[f] = ['%s:%i'%(i[0], i[1]) for i in vals]

##################################################################
# Output EDD Results
##################################################################
headers = ['Field Num', 'Field Name', 'Type', 'Num Blanks', 'Num Entries',
    'Num Unique', 'Stddev', 'Mean_or_Top1', 'Min_or_Top2',
    'P1_or_Top3', 'P5_or_Top4', 'P25_or_Top5', 'Median_or_Bot5','P75_or_Bot4', 'P95_or_Bot3', 'P99_or_Bot2', 'Max_or_Bot1']
writer = csv.DictWriter(open(options.out_file,'wb'), headers, lineterminator = '\n')
writer.writeheader()
for (ct, f) in enumerate(reader.fieldnames):
    print
    if f in numeric:
        writer.writerow({'Field Num' : ct+1, 'Field Name' : f, 'Type' : 'Num',
            'Num Blanks' : blankStats[f], 'Num Entries' : len(numStats[f]),
            'Num Unique' : len(set(numStats[f])), 'Min_or_Top2' : mins[f],
            'Max_or_Bot1' : maxs[f], 'Mean_or_Top1' : means[f], 'Median_or_Bot5' : meds[f],
            'Stddev' : stds[f], 'P1_or_Top3' : Pct1[f], 'P5_or_Top4' : Pct5[f],
            'P25_or_Top5' : Pct25[f], 'P75_or_Bot4' : Pct75[f], 'P95_or_Bot3' : Pct95[f],
            'P99_or_Bot2' : Pct99[f]})
    else:
        writer.writerow({'Field Num' : ct+1, 'Field Name' : f, 'Type' : 'Cat',
            'Num Blanks' : blankStats[f], 'Num Entries' : len(catStats[f]),
            'Num Unique' : len(set(catStats[f])), 'Min_or_Top2' : catVals[f][min(1,len(catVals[f])-1)],
            'Max_or_Bot1' : catVals[f][max(-1,-len(catVals[f]))], 'Mean_or_Top1' : catVals[f][0], 'Median_or_Bot5' : catVals[f][max(-5,-len(catVals[f]))],
            'P1_or_Top3' : catVals[f][min(2,len(catVals[f])-1)], 'P5_or_Top4' : catVals[f][min(3,len(catVals[f])-1)],
            'P25_or_Top5' : catVals[f][min(4,len(catVals[f])-1)], 'P75_or_Bot4' : catVals[f][max(-4,-len(catVals[f]))], 'P95_or_Bot3' : catVals[f][max(-3,-len(catVals[f]))],
            'P99_or_Bot2' : catVals[f][max(-2,-len(catVals[f]))]})




