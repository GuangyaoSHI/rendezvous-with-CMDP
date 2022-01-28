# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 12:16:33 2021

@author: gyshi
"""

from mcts import MctsSim
from simulator import State
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
