# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 17:04:26 2022

@author: gyshi
"""

import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from utils import *
import pickle
import logging
import copy
import time
import os
import sys

#experiment_name = '_velocity3'   '_risk_tolerance' '_risk_level_example'
# '_toy_example'
experiment_name = '_risk_tolerance'
print("we are doing experiments: {}".format(experiment_name))
state_f = ('f', 'f', 'f', 'f', 'f')
state_l = ('l', 'l', 'l', 'l', 'l')
state_init = (0, int(6.8e3), int(19.1e3), 100, 0)

threshold = 0.01
print("threshold is {}".format(threshold))
# generate state transition function
# remember to change this option !!!!!!!!!!!!!!!!!!!!!!!!!
UAV_task = generate_UAV_task(option='long')
UAV_goal = [x for x in UAV_task.nodes() if (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==1) or (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==0)]
UAV_goal = UAV_goal[0]
print("UAV task is {} and goal is {}".format(UAV_task.nodes, UAV_goal))
# UGV_task is a directed graph. Node name is an index
UGV_task = generate_UGV_task()
print("UGV task is {}".format(UGV_task.nodes))

road_network = generate_road_network()


actions = ['v_be', 'v_br', 'v_be_be', 'v_br_br']
rendezvous = Rendezvous(UAV_task, UGV_task, road_network, battery=240e3)
if experiment_name == '_velocity_comparison1':
    rendezvous.velocity_ugv = 5
    rendezvous.velocity_uav = {'v_be' : 7.5, 'v_br' : 7.5}

if experiment_name == '_velocity_comparison2':
    rendezvous.velocity_ugv = 5
    rendezvous.velocity_uav = {'v_be' : 10, 'v_br' : 10}    
    
if experiment_name == '_velocity_comparison3':
    rendezvous.velocity_ugv = 5
    rendezvous.velocity_uav = {'v_be' : 14, 'v_br' : 14} 


# file names to get transition information 
current_directory = os.getcwd()
target_directory = os.path.join(current_directory, r'transition_information')
P_s_a_name = os.path.join(target_directory, 'P_s_a'+experiment_name+'.obj')
transition_graph_name = os.path.join(target_directory, 'state_transition_graph'+experiment_name+'.obj')

# Getting back the objects:
with open(P_s_a_name , 'rb') as f:  # Python 3: open(..., 'rb')
    P_s_a = pickle.load(f)
with open(transition_graph_name, 'rb') as f:  # Python 3: open(..., 'rb')
    G = pickle.load(f)


# set directory and file name to save the policy
target_directory = os.path.join(current_directory, r'policy')
if not os.path.exists(target_directory):
   os.makedirs(target_directory)
policy_name = os.path.join(target_directory, 'policy'+str(threshold)+experiment_name+'.obj')
                     
# create transition function 
def transition_prob(s_a_s):
    state = tuple(list(s_a_s)[0:5])
    action = s_a_s[5]
    next_state = tuple(list(s_a_s)[6:])
    
    # unreachable state
    if (action not in P_s_a[state]) or (next_state not in P_s_a[state][action]):
        #print("trying to transit to an unreachable state")
        return 0
    
    # failure state
    if state == state_f:
        #print('transit to failure state')
        assert action == 'l' and (next_state == state_l)
        return 1

    # goal
    if state[0] == UAV_goal:
        print("reach the goal and transition is {}".format(s_a_s))
        assert action == 'l' and (next_state == state_l)
        return 1
    
    # loop state
    if state == state_l:
        assert action == 'l' and (next_state == state_l)
        return 1
    
    return P_s_a[state][action][next_state] 


def reward(s_a):
    state = tuple(list(s_a)[0:5])
    action = s_a[5]
    
    if state == state_f:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if state == state_l:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if state[0] == UAV_goal:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if action in ['v_be', 'v_br']:
        uav_state = state[0]
        uav_state_next = list(UAV_task.neighbors(uav_state))[0]
        duration = UAV_task.edges[uav_state, uav_state_next]['dis'] / rendezvous.velocity_uav[action]
        return -duration
    
    if action in ['v_be_be', 'v_br_br']:
        uav_state = state[0]
        uav_state_next = list(UAV_task.neighbors(uav_state))[0]
        ugv_state = state[1:3]
        ugv_road_state = ugv_state + ugv_state
        ugv_task_node = state[-1]
        v1 = action[0:4]
        v2 = 'v'+action[4:]
        uav_state_next = list(UAV_task.neighbors(uav_state))[0]
        rendezvous_state, t1, t2 = rendezvous.rendezvous_point(UAV_task.nodes[uav_state]['pos'], 
                                                               UAV_task.nodes[uav_state_next]['pos'], 
                                                               ugv_state, 
                                                               ugv_road_state, ugv_task_node, 
                                                               rendezvous.velocity_uav[v1], 
                                                                               rendezvous.velocity_uav[v2])
        return -(t1+t2+rendezvous.charging_time)
    


def cost(s_a):
    state = tuple(list(s_a)[0:5])
    action = s_a[5]
    
    if state == state_f:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if state == state_l:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if state[0] == UAV_goal:
        assert action == 'l', "should transit to loop state"
        return 0
    
    # compute cost
    C = 0
    for next_state in P_s_a[state][action]:
        if next_state == state_f:
            C += P_s_a[state][action][next_state]
    
    assert C <= (1+1e-9)
    
    return C

start_time = time.time()
y = {}
lambda_1 = 10000
lambda_sa = {}
nu = {}
C_sa_pri = {}
C_sa_sec = {} 
# transform nu to sa 
s2sa = []
P_sa_s_prime = {}
P_s_prime_sa = {}

states = list(P_s_a.keys())
N = len(states)
print("start to initialize")
for state in P_s_a:
    if state != state_l:
        nu[state] = 0.1
    for action in P_s_a[state]:
        y[state+(action,)] = 0.01
        lambda_sa[state+(action,)]  = 500
        C_sa_pri[state+(action,)] = reward(state+(action,))
        C_sa_sec[state+(action,)] = cost(state+(action,))
        
        # transformation matrix
        index = states.index(state)
        row = np.zeros((N-1,))
        if state != state_l:
            row[index] = 1
        s2sa.append(row)

s2sa = np.array(s2sa)
state_actions = list(y.keys())
print("--- it takes %s seconds to initialize---" % (time.time() - start_time))
        

# update step size
ay = 0.001
a_lambda = 0.1
a_nu = 0.1
C_pri = np.array(list(C_sa_pri.values()))
C_sec = np.array(list(C_sa_sec.values()))

y_last_step = np.array(copy.deepcopy(list(y.values())))
C_last_step = np.dot(y_last_step, C_pri)-1

lambda_1_last_step = lambda_1
lambda_sa_last_step = np.array(copy.deepcopy(list(lambda_sa.values())))
nu_last_step = np.array(copy.deepcopy(list(nu.values())))

y_current_step = np.array(copy.deepcopy(list(y.values())))
lambda_1_current_step = lambda_1
lambda_sa_current_step = np.array(copy.deepcopy(list(lambda_sa.values())))
nu_current_step = np.array(copy.deepcopy(list(nu.values())))
C_current_step = np.dot(y_current_step, C_pri)


indicator_initi = np.zeros(len(nu))
assert states.index(state_init) == list(nu.keys()).index(state_init) 
init_index = states.index(state_init)
indicator_initi[init_index] = 1

traces = [C_current_step]
print("start to do iterations")
while abs(C_current_step - C_last_step)>0.01:
    C_last_step = C_current_step
    dy = C_pri + lambda_1_last_step*C_sec - lambda_sa_last_step + s2sa@nu_last_step
    dy_last_term_vec = np.zeros(y_last_step.shape)
    for i in range(len(state_actions)):
        s_a = state_actions[i]
        state = s_a[0:4]
        action = s_a[4]
        temp = 0
        for state_next in P_s_a[state][action]:
            index = states.index(state_next)
            temp += nu_last_step[index]*P_s_a[state][action][state_next]
        dy_last_term_vec[i] = temp
    dy -= dy_last_term_vec
    # update y
    y_current_step = y_last_step - ay*dy
    # projection back to domain
    y_current_step = np.multiply(y_current_step, y_current_step>=0)
    
    d_lambda1 = np.dot(y_last_step, C_sec) - threshold
    # update lambda1
    lambda_1_current_step = lambda_1_last_step + a_lambda * d_lambda1
    # projection back to domain
    lambda_1_current_step = max(lambda_1_current_step, 0)
    
    d_lambda = -y_last_step
    # update lambda
    lambda_sa_current_step = lambda_sa_last_step + a_lambda*d_lambda
    # projection back to domain
    lambda_sa_current_step = np.multiply(lambda_sa_current_step, lambda_sa_current_step>=0)
    
    # update nu
    d_nu_last_term = 0
    for state in nu.keys():
        for s_a in G.predecessors(state):
            assert len(s_a) == 6
            #if s_a not in s_a_nodes:
            #s_a_nodes.append(s_a)
            index = state_actions.index(s_a)
            s_a_s = s_a + state
            d_nu_last_term += y_last_step[index] * transition_prob(s_a_s)
    d_nu = np.transpose(s2sa)@y_last_step - indicator_initi - d_nu_last_term
    nu_current_step = nu_last_step + a_nu*d_nu           
    
    C_current_step = np.dot(y_current_step, C_pri)
    traces.append(C_current_step)
    
plt.plot(traces)





        
