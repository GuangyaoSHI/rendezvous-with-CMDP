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
road_network = generate_road_network()
actions = ['v_be', 'v_br', 'v_be_be', 'v_be_br', 'v_br_be', 'v_br_br']
rendezvous = Rendezvous(UAV_task, UGV_task, road_network)
rendezvous.display = False

P_s_a = {}

for uav_state in UAV_task.nodes:
    for ugv_state in road_network.nodes:
        for battery in range(0, 101):
            for ugv_task_node in UGV_task.nodes:
                state = uav_state + ugv_state + (battery/100*rendezvous.battery, ) + UGV_task.nodes[ugv_task_node]['pos']
                P_s_a[state] = {}
                for action in actions:
                    P_s_a[state][action] = {}
                for action in actions:
                    UGV_road_state = ugv_state + ugv_state
                    UAV_state, UGV_state, UGV_road_state, UGV_task_node, battery_state = rendezvous.transit(state, action, UGV_road_state, ugv_task_node)
                    state_ = UAV_state + UGV_state + (int(battery_state/rendezvous.battery*100), )+UGV_task.nodes[UGV_task_node]['pos']
                    if state_ not in P_s_a[state][action]:
                        P_s_a[state][action][state_]

                    
            
