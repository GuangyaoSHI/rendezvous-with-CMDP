# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:44:15 2022

@author: gyshi
"""


import csv
import networkx as nx
import numpy as np

def distance(a1, a2):
    return np.linalg.norm(a1-a2)

file = open("Coords.csv")
csvreader = csv.reader(file)
header = next(csvreader)
print(header)

rows = []
for row in csvreader:
    rows.append(row)
print(rows)

names = []
G = nx.Graph()
for i in range(0, len(rows)):
    if i < 7:
        G.add_node(rows[i][0], pos=np.array([float(rows[i][1]), float(rows[i][2])]))  
        names.append(rows[i][0])
    else:
        G.add_node(i-7, pos=np.array([float(rows[i][1]), float(rows[i][2])]))



for node in G.nodes:
    dis = (distance(G.nodes['Depot A']['pos'], G.nodes[node]['pos'])
    +distance(G.nodes[node]['pos'], G.nodes['AOI A']['pos'])
    +distance(G.nodes['AOI A']['pos'], G.nodes['Depot B']['pos']))
    if  18 <= dis <= 19:
        print("node is {} dis is {}".format(node, dis))
        print("node position is {}".format(G.nodes[node]['pos']))



road_network = nx.Graph()
for i in range(7, len(rows)-1):
    node1 = (float(rows[i][1]), float(rows[i][2]))
    node2 = (float(rows[i+1][1]), float(rows[i+1][2]))
    dis = np.linalg.norm(np.array(node1)-np.array(node2))
    road_network.add_edge(node1, node2, dis=dis)

((6.29, 11.14), (1.0, 13.4)) in road_network.edges
road_network.remove_edge((6.29, 11.14), (1.0, 13.4))
((6.29, 11.14), (17.5, 1.5)) in road_network.edges
road_network.remove_edge((6.29, 11.14), (17.5, 1.5))

pos = dict(zip(road_network.nodes, road_network.nodes))
nx.draw(road_network, pos=pos, alpha=1, node_color='r', node_size=20)



file.close()
