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
        
        # uav velocity
        self.velocity_uav = {'v_be' : 9.8, 'v_br' : 16}
        
        # ugv velocity
        self.velocity_ugv = 4.5

        self.check_UGV_task()
                
    def check_UGV_task(self):
        # check whether each task point is in road network
        for node in self.UGV_task.nodes:
            assert node in self.road_network.nodes, "UGV task node not in road network"
        
        
    def transit(self, state, action, UGV_task_state):
        # return probability distribution P(s' | s, a)
        # state = (xa, ya, xg, yg, SoC)
        # action: {'v_be', 'v_br', 'v_be_be', 'v_be_br', 'v_br_be', 'v_br_br'}
        # UGV_task_state: UGV is transiting from which node to which
        # (x, y, x1, y1)
        UAV_state, UGV_state, battery_state = self.get_states(state)
        UAV_state_next = []
        UGV_state_next = []
        battery_state_next = []
        
        if action == 'v_be':
            # UAV choose to go to next task node with best endurance velocity
            descendants = list(self.UGV_task.neighbors[UAV_state])
            assert len(descendants) == 1
            UAV_state_next = descendants[0]
            # compute next state for UGV
            duration = self.UAV_task.edges[UAV_state, UAV_state_next]['dis']/self.velocity_uav['v_be']
            UGV_state_next = self.UGV_transit(UGV_state, UGV_task_state, duration)
            
            
            
        return UAV_state_next, UGV_state_next
    
    def UGV_transit(self, UGV_state, UGV_task_state, duration):
        #last_task_state = (UGV_task_state[0], UGV_task_state[1])
        # Todo: check UGV_state is indeed between two task nodes
        next_task_state = (UGV_task_state[2], UGV_task_state[3])
        
        # UGV will move duration * velocity distance along the task path
        total_dis = self.velocity_ugv * duration
        
        state_before_stop = next_task_state
        dis = np.linalg.norm(np.array(UGV_state)-np.array(state_before_stop))
        
        while (dis < total_dis):
            descendants = list(self.UGV_task.neighbors(state_before_stop))[0]
            dis += np.linalg.norm(np.array(descendants)-np.array(state_before_stop))
            state_before_stop = descendants
            
        previous_state = self.UGV_task.predecessors(state_before_stop)
        vector = np.array(previous_state) - np.array(state_before_stop)
        vector = vector/np.linalg.norm(vector)
        assert dis>= total_dis
        UGV_state_next = tuple(np.array(state_before_stop)-(dis-total_dis)*vector)
        return UGV_state_next
        
    def power_consumption(self, SoC, action, duration):
        # return power distribution after taking action with SoC
        return
    
    def rendezvous_point(self, state, action):
        # return rendezvous point
        return
    
    def get_states(self, state):
        # state = (xa, ya, xg, yg, SoC)
        UAV_state = (state[0], state[1])
        assert UAV_state in self.UAV_task, "UAV state is not in task"
        UGV_state = (state[2], state[3])
        battery_state = state[-1]
        return UAV_state, UGV_state, battery_state
        

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
    G.add_node((0, int(height)))
    for i in range(4):
        leaf = [x for x in G.nodes() if (G.out_degree(x)==0 and G.in_degree(x)==1) or (G.out_degree(x)==0 and G.in_degree(x)==0)]
        assert len(leaf) == 1
        curr_node = leaf[0]
        for t in range(1, segments+1):
            next_node = (int(curr_node[0]+vector_minus[0]), int(curr_node[1]+vector_minus[1]))
            dis = np.linalg.norm(np.array(curr_node) - np.array(next_node))
            G.add_edge(curr_node, next_node, dis=dis)
            # watch out deep copy
            curr_node = next_node
            
        for t in range(1, segments+1):
            next_node = (int(curr_node[0]+vector_plus[0]), int(curr_node[1]+vector_plus[1]))
            dis = np.linalg.norm(np.array(curr_node) - np.array(next_node))
            G.add_edge(curr_node, next_node, dis=dis)
            # watch out deep copy
            curr_node = next_node
            
    pos = dict(zip(G.nodes, G.nodes))
    nx.draw(G, pos=pos)
    return G

    
def generate_UGV_task():
    G = nx.DiGraph()
    # a simple straight line network
    for i in range(1, 30*60*4):
        G.add_edge((0, (i-1)*5), (0, i*5))  
    return G
    
    
     
    
    
    