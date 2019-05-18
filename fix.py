# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:10:13 2019

@author: vaken
"""

import csv
i = 0
rows = 0
with open('data/labels.csv') as inp, open('data/labels2.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        rows += 1
        try:
            single_image_name = 'data/top100labels/' + row[0] + '.jpg'
            x = open(single_image_name, 'r')
            writer.writerow(row)
        except FileNotFoundError:
            i += 1
            print('file ' + single_image_name + ' does not exist at row ' + str(rows))
print(i + ' files not found, deleted them')