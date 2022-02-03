# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:44:15 2022

@author: gyshi
"""


# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from utils import *
import pickle


# generate state transition function
UAV_task = generate_UAV_task()
UAV_goal = [x for x in UAV_task.nodes() if (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==1) or (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==0)]
UAV_goal = UAV_goal[0]
# UGV_task is a directed graph. Node name is an index
UGV_task = generate_UGV_task()
road_network = generate_road_network()
actions = ['v_be', 'v_be_be']
rendezvous = Rendezvous(UAV_task, UGV_task, road_network, battery=280e3)
rendezvous.display = False

# get power consumption distribution:
# best endurance velocity
stats = rendezvous.get_power_consumption_distribution(rendezvous.velocity_uav['v_be'])
powers_be = []
probs_be = []
for interval in stats:
    powers_be.append((interval[0]+interval[1])/2)
    probs_be.append(stats[interval])

# best range velocity
# stats = rendezvous.get_power_consumption_distribution(rendezvous.velocity_uav['v_br'])
# powers_br = []
# probs_br = []
# for interval in stats:
#     powers_br.append((interval[0]+interval[1])/2)
#     probs_br.append(stats[interval])
    
state_f = ('f', 'f', 'f', 'f', 'f', 'f')
state_l = ('l', 'l', 'l', 'l', 'l', 'l')
state_init = (6.8e3, 19.1e3, 6.8e3, 19.1e3, 100, 0)


G = nx.DiGraph()
        
def get_actions(state):
    uav_state = (state[0], state[1])
    if uav_state == UAV_goal:
        return ['l']
    if state == state_f:
        return ['l']
    if state == state_l:
        print("inquire actions for loop state")
        return ['l']
    return actions

def expand_graph(state):
    if state == state_l:
        return
    
    uav_state = (state[0], state[1])
    if (state == state_f or uav_state == UAV_goal):    
        G.add_edge(state, state+('l',))
        G.add_edge(state+('l',), state_l, action='l', prob=1)
        expand_graph(state_l)
        return
    
    
    ugv_state = (state[2], state[3])
    battery = state[4]
    ugv_task_node = state[5]
    energy_state = battery/100*rendezvous.battery
    
    actions = get_actions(state)
    G.add_node(state, actions=actions)
    
   
    for action in actions:
        if action == 'v_be':  
            # UAV choose to go to next task node with best endurance velocity
            descendants = list(UAV_task.neighbors(uav_state))
            assert len(descendants) == 1
            UAV_state_next = descendants[0]
            duration = UAV_task.edges[uav_state, UAV_state_next]['dis'] / rendezvous.velocity_uav[action]
            
            # compute the energy distribution
            energy_states = list(energy_state-np.array(powers_be)*duration)
            energy_distribution = dict(zip(energy_states, probs_be))
            
            UGV_road_state = ugv_state + ugv_state
            UGV_state_next, UGV_road_state_next, UGV_task_node_next = rendezvous.UGV_transit(ugv_state, UGV_road_state, ugv_task_node, duration)                        
            # use discrete UGV_state by assigning UGV to one road state
            rs1 = np.linalg.norm(np.array(UGV_state_next)-np.array([UGV_road_state_next[0], UGV_road_state_next[1]]))
            rs2 = np.linalg.norm(np.array(UGV_state_next)-np.array([UGV_road_state_next[2], UGV_road_state_next[3]]))
            if rs1<rs2:
                UGV_state_next = (UGV_road_state_next[0], UGV_road_state_next[1])
            else:
                UGV_state_next = (UGV_road_state_next[2], UGV_road_state_next[3])
            
            # next_state:transition_probability
            P_s_a_s = {}
            
            for p_c in energy_states:
                if p_c < 0:
                    state_ = ('f', 'f', 'f', 'f', 'f', 'f')
                else:
                    state_ = UAV_state_next + UGV_state_next + (round(p_c/rendezvous.battery*100), )+(UGV_task_node_next,)
                if state_ not in P_s_a_s:
                    P_s_a_s[state_] = energy_distribution[p_c]
                else:
                    # assert state_ == ('f', 'f', 'f', 'f', 'f', 'f')
                    P_s_a_s[state_] += energy_distribution[p_c]
                    
            for next_state in P_s_a_s:
                G.add_edge(state, state+(action,))
                G.add_edge(state+(action,), next_state, action=action, prob=P_s_a_s[next_state])
                expand_graph(next_state)
        
        
        if action == 'v_be_be':
            # compute UAV position after rendezvous
            descendants = list(UAV_task.neighbors(uav_state))
            assert len(descendants) == 1
            UAV_state_next = descendants[0]
            
            ugv_road_state = ugv_state + ugv_state
            v1 = action[0:4]
            v2 = 'v'+action[4:]
            rendezvous_state, t1, t2 = rendezvous.rendezvous_point(uav_state, UAV_state_next, ugv_state, 
                                                                   ugv_road_state, ugv_task_node, 
                                                                   rendezvous.velocity_uav[v1], 
                                                                   rendezvous.velocity_uav[v2])
            rendezvous_road_state = rendezvous_state + rendezvous_state 
            UGV_state_next, UGV_road_state_next, UGV_task_node_next = rendezvous.UGV_transit(rendezvous_state, rendezvous_road_state, ugv_task_node, t2)
            
            # use discrete UGV_state by assigning UGV to one road state
            rs1 = np.linalg.norm(np.array(UGV_state_next)-np.array([UGV_road_state_next[0], UGV_road_state_next[1]]))
            rs2 = np.linalg.norm(np.array(UGV_state_next)-np.array([UGV_road_state_next[2], UGV_road_state_next[3]]))
            if rs1 < rs2:
                UGV_state_next = (UGV_road_state_next[0], UGV_road_state_next[1])
            else:
                UGV_state_next = (UGV_road_state_next[2], UGV_road_state_next[3])
            
            # compute energy distribution after rendezvous
            energy_states = list(energy_state - np.array(powers_be)*t1)
            energy_distribution = dict(zip(energy_states, probs_be))
            
            failure_prob = 0
            for p_c in energy_states:
                # first find the failure probability
                if p_c < 0:
                    failure_prob += energy_distribution[p_c]
            
            P_s_a_s = {}
            if failure_prob > 0:
                P_s_a_s[state_f] = failure_prob
            
            if failure_prob == 1:
                G.add_edge(state, state+(action,))
                G.add_edge(state+(action,), state_f, action=action, prob=P_s_a_s[state_f])
                expand_graph(state_f)
                return
            
            energy_states2 = list(rendezvous.battery - np.array(powers_be)*t2)
            #Todo: probs_be need to change if br is used
            energy_distribution2 = dict(zip(energy_states2, probs_be))
            for p_c in energy_states2:
                if p_c < 0:
                    state_ = ('f', 'f', 'f', 'f', 'f', 'f')
                else:
                    state_ = UAV_state_next + UGV_state_next + (min(round(p_c/rendezvous.battery*100), 100), )+(UGV_task_node_next,)
                if state_ not in  P_s_a_s:
                    P_s_a_s[state_] = energy_distribution2[p_c]*(1-failure_prob)
                else:
                    # assert state_ == ('f', 'f', 'f', 'f', 'f', 'f')
                    P_s_a_s[state_] += energy_distribution2[p_c]*(1-failure_prob)
                    
            for next_state in P_s_a_s:
                G.add_edge(state, state+(action,))
                G.add_edge(state+(action,), next_state, action=action, prob=P_s_a_s[next_state])
                expand_graph(next_state)
                
                
expand_graph(state_init)
            
            
        
    
    