# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:59:38 2022

@author: gyshi
"""

import numpy as np 
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import random
import matplotlib.pyplot as plt


class rendezvous():
    def __init__(self, UAV_task, UGV_task, road_network):
        # a sequence of nodes in 2D plane, represented as a directed graph
        self.UAV_task = UAV_task 
        
        # a sequence of nodes in road network, represented as a directed graph
        # rendezvous action may change the task, some nodes may be inserted into tha task 
        self.UGV_task = UGV_task 
        
        # road network can be viewed as a fine discretization of 2D continuous road network
        self.road_network = road_network 

        self.check_UGV_task()
        
    def check_UGV_task(self):
        # check whether each task point is in road network
        for node in self.UGV_task.nodes:
            assert node in self.road_network.nodes, "UGV task node not in road network"
        
        
    def transit(self, state, action):
        # return probability distribution P(s' | s, a)
        return
    
    def power_consumption(self, SoC, action, duration):
        # return power distribution after taking action with SoC
        return
    
    def rendezvous_point(self, state, action):
        # return rendezvous point
        return
        

def generate_road_network():
    G = nx.DiGraph()
    # a simple straight line network
    for i in range(1, 30*60*5):
        G.add_edge((0, (i-1)*5), (0, i*5))  
    return G

def generate_UAV_task():
    angle = 70 / 180 * np.pi
    length = 13*60*20 / 2
    height = 0.5*np.math.sin(angle)*(length)
    segments = 5
    vector_plus = np.array([np.math.cos(angle), np.math.sin(angle)]) * length/segments
    vector_minus = np.array([np.math.cos(-angle), np.math.sin(-angle)]) * length/segments
    G = nx.DiGraph()
    G.add_node((0, height))
    for i in range(4):
        leaf = [x for x in G.nodes() if (G.out_degree(x)==0 and G.in_degree(x)==1) or (G.out_degree(x)==0 and G.in_degree(x)==0)]
        assert len(leaf) == 1
        curr_node = leaf[0]
        for t in range(1, segments+1):
            next_node = (curr_node[0]+vector_minus[0], curr_node[1]+vector_minus[1])
            G.add_edge(curr_node, next_node)
            # watch out deep copy
            curr_node = next_node
            
        for t in range(1, segments+1):
            next_node = (curr_node[0]+vector_plus[0], curr_node[1]+vector_plus[1])
            G.add_edge(curr_node, next_node)
            # watch out deep copy
            curr_node = next_node
            
    pos = dict(zip(G.nodes, G.nodes))
    nx.draw(G, pos=pos)
    return G
    
    
    
     
    
    
    