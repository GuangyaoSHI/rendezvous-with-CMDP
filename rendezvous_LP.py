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
            # Todo: different node may represent the same node, this can cause some problem
            for ugv_task_node in UGV_task.nodes:
                battery_state = battery/100*rendezvous.battery
                state = uav_state + ugv_state + (battery_state, ) + ugv_task_node
                P_s_a[state] = {}
                for action in actions:
                    P_s_a[state][action] = {}
                
                for action in actions:
                    if action == 'v_be':  
                        power_states = list(battery_state - np.array(values)-1)
                        power_distribution = dict(zip(power_states, probs))
                        assert len(power_states) == 6
                        
                        UGV_road_state = ugv_state + ugv_state
                        UAV_state, UGV_state, UGV_road_state, UGV_task_node, battery_state = rendezvous.transit(state, action, UGV_road_state, ugv_task_node)
                        
                        for p_c in power_states:
                            if p_c < 0:
                                state_ = ('f', 'f', 'f', 'f', 'f', 'f', 'f')
                            else:
                                state_ = UAV_state + UGV_state + (int(p_c/rendezvous.battery*100), )+UGV_task.nodes[UGV_task_node]['pos']
                            if state_ not in P_s_a[state][action]:
                                P_s_a[state][action][state_] = power_distribution[p_c]
                            else:
                                assert state_ == ('f', 'f', 'f', 'f', 'f', 'f', 'f')
                                P_s_a[state][action][state_] += power_distribution[p_c]
                        
                    if action == 'v_be_be':
                        ugv_road_state = ugv_state + ugv_state
                        v1 = action[0:4]
                        v2 = 'v'+action[4:]
                        uav_state_next = UAV_task.neighbors[uav_state]
                        rendezvous_state, t1, t2 = rendezvous.rendezvous_point(uav_state, uav_state_next, ugv_state, ugv_road_state, ugv_task_node, 10, 10)
                        rendezvous_road_state = rendezvous_state + rendezvous_state 
                        UGV_state_next, UGV_road_state_next, UGV_task_node_next = rendezvous.UGV_transit(rendezvous_state, rendezvous_road_state, ugv_task_node, t2)
                        dis1 = np.linalg.norm(np.array(uav_state)-np.array(rendezvous_state))
                        power_states = list(battery_state - np.array(values)-dis1/rendezvous.power_measure)
                        power_distribution = dict(zip(power_states, probs))
                        assert len(power_states) == 6
                        
                        failure_prob = 0
                        for p_c in power_states:
                            # first find the failure probability
                            if p_c < 0:
                                failure_prob += power_distribution[p_c]
                        
                        state_ = ('f', 'f', 'f', 'f', 'f', 'f', 'f')
                        P_s_a[state][action][state_] = failure_prob
                        dis2 = np.linalg.norm(np.array(uav_state_next)-np.array(rendezvous_state))
                        power_states2 = list(rendezvous.battery - np.array(values)-dis2/rendezvous.power_measure)
                        for p_c in power_states2:
                            if p_c < 0:
                                state_ = ('f', 'f', 'f', 'f', 'f', 'f', 'f')
                            else:
                                state_ = UAV_state + UGV_state + (int(p_c/rendezvous.battery*100), )+UGV_task.nodes[UGV_task_node]['pos']
                            if state_ not in P_s_a[state][action]:
                                P_s_a[state][action][state_] = power_distribution[p_c]*(1-failure_prob)
                            else:
                                assert state_ == ('f', 'f', 'f', 'f', 'f', 'f', 'f')
                                P_s_a[state][action][state_] += power_distribution[p_c]*(1-failure_prob)
                                



                    

                    
            
