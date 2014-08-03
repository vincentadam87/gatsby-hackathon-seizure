"""
script to check if columns 2 and 3 are identical
"""

import sys

filename = '/nfs/data3/kaggle_seizure/scratch/Stiched_data/output.csv'

f = open(filename, 'r')

for n, line in enumerate(f):
    line = line.rstrip('\n')
    parts = line.split(',')
    if float(parts[1]) < float(parts[2]):
        print 'prob(seizure) < prob(early seizure) in line = %s' % parts
