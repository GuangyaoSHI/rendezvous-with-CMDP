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
experiment_name = '_test_iteration_alg'
print("we are doing experiments: {}".format(experiment_name))
state_f = ('f', 'f', 'f', 'f', 'f')
state_l = ('l', 'l', 'l', 'l', 'l')
state_init = (0, int(6.8e3), int(19.1e3), 100, 0)

threshold = 0.1
print("threshold is {}".format(threshold))
# generate state transition function
# remember to change this option !!!!!!!!!!!!!!!!!!!!!!!!!
UAV_task = generate_UAV_task(option='LP_test')
UAV_goal = [x for x in UAV_task.nodes() if (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==1) or (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==0)]
UAV_goal = UAV_goal[0]
print("UAV task is {} and goal is {}".format(UAV_task.nodes, UAV_goal))
# UGV_task is a directed graph. Node name is an index
UGV_task = generate_UGV_task()
print("UGV task is {}".format(UGV_task.nodes))

road_network = generate_road_network()


actions = ['v_be', 'v_be_be']
rendezvous = Rendezvous(UAV_task, UGV_task, road_network, battery=240e3)


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

scale = 10000

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
        return -duration/scale 
    
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
        return (t1+t2+rendezvous.charging_time)/scale
    


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


y = {}
lambda_1 = 100
lambda_sa = {}
nu = {}
C_sa_pri = {}
C_sa_sec = {} 
print("start to initialize")
for state in P_s_a:
    if state != state_l:
        nu[state] = 10
    for action in P_s_a[state]:
        y[state+(action,)] = 1
        lambda_sa[state+(action,)]  = 1000
        C_sa_pri[state+(action,)] = reward(state+(action,))
        assert C_sa_pri[state+(action,)] >= 0
        C_sa_sec[state+(action,)] = cost(state+(action,))


# update step size
ay = 0.0001
a_lambda = 0.0001
a_nu = 0.0001



y_current_step = copy.deepcopy(y)
lambda_1_current_step = lambda_1
lambda_sa_current_step = copy.deepcopy(lambda_sa)
nu_current_step = copy.deepcopy(nu)
C_current_step = np.dot(np.array(list(y_current_step.values())), np.array(list(C_sa_pri.values())))


y_last_step = copy.deepcopy(y)
C_last_step = np.dot(np.array(list(y_last_step.values())), np.array(list(C_sa_pri.values())))-1
lambda_1_last_step = lambda_1
lambda_sa_last_step = copy.deepcopy(lambda_sa)
nu_last_step = copy.deepcopy(nu)
#Lagrangian_last_step = Lagrangian_current_step - 0.01



obj_traces = [C_current_step]
obj_mean = []
y_traces = [y_current_step]
y_mean = []
#Lagrangian_traces = [Lagrangian_current_step]
print("start to do iterations")
for j in range(1, 30000):
    print("iteration {}".format(j))
    ay = ay/1.0
    a_lambda = ay
    a_nu = ay
    #Lagrangian_last_step = Lagrangian_current_step
    for s_a in y:
        # dy
        state = s_a[0:5]
        last_term_dy = 0
        for state_next in P_s_a[state][s_a[-1]]:
            if state_next != state_l:
                last_term_dy += P_s_a[state][s_a[-1]][state_next]*nu_last_step[state_next]
        
        if state != state_l:
            dy = C_sa_pri[s_a] + lambda_1_last_step*C_sa_sec[s_a] - \
                lambda_sa_last_step[s_a] + nu_last_step[state] - last_term_dy
        else:
            dy = C_sa_pri[s_a] + lambda_1_last_step*C_sa_sec[s_a] - lambda_sa_last_step[s_a] - last_term_dy
        # update y
        y_current_step[s_a] = y_last_step[s_a] - ay*dy
        #y_current_step[s_a] = max(y_current_step[s_a], 0)
        # d lambda_sa
        d_lambda_sa = -y_last_step[s_a]
        # update lambda_sa
        lambda_sa_current_step[s_a] = lambda_sa_last_step[s_a] + a_lambda*d_lambda_sa
        if lambda_sa_current_step[s_a] < 0:
            lambda_sa_current_step[s_a] = 0
        
    y_traces.append(y_current_step)
    # d lambda1
    d_lambda1 = np.dot(np.array(list(y_last_step.values())), np.array(list(C_sa_sec.values()))) - threshold
    # update lambda1
    lambda_1_current_step = lambda_1_last_step + a_lambda*d_lambda1
    if lambda_1_current_step < 0:
        lambda_1_current_step = 0
        
    for state in P_s_a:
        # d nu
        if state != state_l:
            # first term
            d_nu_first_term = 0
            for action in P_s_a[state]:
                d_nu_first_term += y_last_step[state+(action,)]
            
            # indicator
            initial_state = 0
            if state == state_init:
                initial_state = 1
            
            # last term 
            d_nu_last_term = 0
            for s_a in G.predecessors(state):
                assert len(s_a) == 6
                #if s_a not in s_a_nodes:
                #s_a_nodes.append(s_a)
                s_a_s = s_a + state
                d_nu_last_term += y_last_step[s_a] * transition_prob(s_a_s)
                
            d_nu = d_nu_first_term - initial_state - d_nu_last_term
            
            # update nu
            nu_current_step[state] = nu_last_step[state] + a_nu*d_nu
    
    C_current_step = np.dot(np.array(list(y_current_step.values())), np.array(list(C_sa_pri.values())))
    obj_traces.append(C_current_step)
    
    # update last step information
    #y_last_step = copy.deepcopy(y_current_step)
    y_last_step = dict(zip(y_last_step.keys(), y_current_step.values()))
    #lambda_sa_last_step = copy.deepcopy(lambda_sa_current_step)
    lambda_sa_last_step = dict(zip(lambda_sa_last_step.keys(), lambda_sa_current_step.values()))
    
    lambda_1_last_step = lambda_1_current_step
    nu_last_step = dict(zip(nu_last_step.keys(), nu_current_step.values()))
    #nu_last_step = copy.deepcopy(nu_current_step)

# post processing
print("Iterations done!")
# post processing
obj_mean = [obj_traces[0]]
for i in range(1, len(obj_traces)):
    obj_mean.append(i*obj_mean[i-1]/(i+1) + obj_traces[i]/(i+1))
    
print("converge value is {}".format(obj_mean[-1]*1000))       

plt.plot(obj_mean)


temp = 0
for i in range(len(y_traces)):
    temp += np.array(list(y_traces.values()))
    
y_value = dict(zip(list(y.keys()), temp))
 
constraints = np.dot(np.array(list(y_value.values())), np.array(list(C_sa_sec.values())))
print("risk is {}".format(constraints))            

print("check equality constraint")
for state in nu_current_step:
    if state == state_l:
        continue
    temp = 0
    for action in P_s_a[state]:
        temp += y_current_step[state+(action,)]
    if state == state_init:
        temp -= 1
    for s_a in G.predecessors(state):
        assert len(s_a) == 6
        #if s_a not in s_a_nodes:
        #s_a_nodes.append(s_a)
        s_a_s = s_a + state
        temp -= y_current_step[s_a] * transition_prob(s_a_s)
    if temp > 0.0000001:
        print("state {} equality constraint value is {}".format(state, temp))




        
