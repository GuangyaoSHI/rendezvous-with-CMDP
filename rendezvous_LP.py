# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from utils import *

# generate state transition function

UAV_task = generate_UAV_task()
# UGV_task is a directed graph. Node name is an index
UGV_task = generate_UGV_task()
UGV_task_states = list(set([UGV_task.nodes[node]['pos'] for node in UGV_task.nodes])) 
road_network = generate_road_network()
actions = ['v_be', 'v_be_be']
rendezvous = Rendezvous(UAV_task, UGV_task, road_network)
rendezvous.display = False

P_s_a = {}

probs = [2.27e-2, 13.6e-2, 34.13e-2, 34.13e-2, 13.6e-2, 2.27e-2]
values = [-0.25, -0.15, -0.05, 0.05, 0.15, 0.25]

for uav_state in UAV_task.nodes:
    for ugv_state in road_network.nodes:
        for battery in range(0, 101):
            for ugv_task_state in UGV_task_states:
                battery_state = battery/100*rendezvous.battery
                state = uav_state + ugv_state + (battery_state, ) + ugv_task_state
                P_s_a[state] = {}
                for action in actions:
                    P_s_a[state][action] = {}
                if action == 'v_be':
                    power_consumptions = list(battery_state - np.array(values)+1)
                    power_distribution = {}
                    assert len(power_consumptions) == 6
                    for p_c in power_consumptions:
                        
                    UGV_road_state = ugv_state + ugv_state
                    UAV_state, UGV_state, UGV_road_state, UGV_task_node, battery_state = rendezvous.transit(state, action, UGV_road_state, ugv_task_node)
                    state_ = UAV_state + UGV_state + (int(battery_state/rendezvous.battery*100), )+UGV_task.nodes[UGV_task_node]['pos']
                    if state_ not in P_s_a[state][action]:
                        P_s_a[state][action][state_]

                    
            
