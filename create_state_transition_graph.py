# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:44:15 2022

@author: gyshi
"""


import csv

file = open("Coords.csv")
csvreader = csv.reader(file)
header = next(csvreader)
print(header)

rows = []
for row in csvreader:
    rows.append(row)
print(rows)
file.close()
