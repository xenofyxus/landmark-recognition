# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:10:13 2019

@author: vaken
"""

import csv

with open('data/labels.csv') as inp, open('data/labels2.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    label = -1
    index = -1
    for row in csv.reader(inp):
        if row[1] != label:
            label = row[1]
            index += 1
        row[1] = index            
        writer.writerow(row)