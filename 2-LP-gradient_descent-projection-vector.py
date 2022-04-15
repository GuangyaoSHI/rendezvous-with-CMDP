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

scale = 1000

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
        return duration/scale
    
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



start_time = time.time()
y = {}
lambda_1 = 100

nu = {}
C_sa_pri = {}
C_sa_sec = {} 
# transform nu to sa 
s2sa = []
P_sa_s = []

states = list(P_s_a.keys())
N = len(states)

# probability vector
P = np.zeros(N-1)
row = np.zeros(N-1)
Ps = np.zeros(N-1)
print("start to initialize")
for state in P_s_a:
    if state != state_l:
        nu[state] = 0.1
    for action in P_s_a[state]:
        y[state+(action,)] = 0.0001
        C_sa_pri[state+(action,)] = reward(state+(action,))
        C_sa_sec[state+(action,)] = cost(state+(action,))
        
        # transformation matrix
        index = states.index(state)
        row = np.copy(P)
        if state != state_l:
            row[index] = 1
        s2sa.append(row)
        Ps = np.copy(P)
        for next_state in P_s_a[state][action]:
            if next_state == state_l:
                continue
            index = states.index(next_state)
            Ps[index] = P_s_a[state][action][next_state]
        P_sa_s.append(Ps)

      
print("initialization done")
s2sa = np.array(s2sa)
s2sa_transpose = np.copy(np.transpose(s2sa))

state_actions = list(y.keys())
assert s2sa.shape[0] == len(state_actions)

# the column of P_sa_s is equal to len(nu)  
P_sa_s = np.array(P_sa_s)
assert P_sa_s.shape[0] == len(state_actions)
assert P_sa_s.shape[1] == len(nu)
P_sa_s_transpose = np.copy(np.transpose(P_sa_s))

print("--- it takes %s seconds to initialize---" % (time.time() - start_time))
        

# update step size
ay = 0.003
a_lambda = 0.1
a_nu = 0.1
C_pri = np.array(list(C_sa_pri.values()))
C_sec = np.array(list(C_sa_sec.values()))

start_time = time.time()
y_last_step = np.array(copy.deepcopy(list(y.values())))
print("--- it takes %s seconds to copy an array---" % (time.time() - start_time))
test = copy.deepcopy(y_last_step)
start_time = time.time()
test = np.array(y_last_step)
print("--- it takes %s seconds to copy a predefined array with np.array---" % (time.time() - start_time))
start_time = time.time()
test = np.copy(y_last_step)
print("--- it takes %s seconds to copy a predefined array with np.copy---" % (time.time() - start_time))


lambda_1_last_step = lambda_1
nu_last_step = np.array(copy.deepcopy(list(nu.values())))

y_current_step = np.array(copy.deepcopy(list(y.values())))
lambda_1_current_step = lambda_1
nu_current_step = np.array(copy.deepcopy(list(nu.values())))
C_current_step = np.dot(y_current_step, C_pri)

indicator_initi = np.zeros(len(nu))
assert states.index(state_init) == list(nu.keys()).index(state_init) 
init_index = states.index(state_init)
indicator_initi[init_index] = 1


iterations = 20000
obj_traces = [C_current_step]
y_traces = np.zeros((y_current_step.shape[0], iterations))
y_traces[:, 0] = y_current_step
y_mean = []
obj_mean = [obj_traces[0]]



dy = np.zeros(y_last_step.shape)
d_lambda1 = 0
d_nu = np.zeros(nu_last_step.shape)

L_nu = len(nu)
nu_states = list(nu.keys())


# ----------test SVD to speed up matrix multiplication
start_time = time.time()
#a = np.matmul(P_sa_s, nu_last_step)
'''
print("--- it takes %s seconds to do matrix multiplication---" % (time.time() - start_time))
U, s, V = np.linalg.svd(P_sa_s) # Very slow, so precompute!
rank = len(s) / 3
start_time = time.time()
y = np.matmul(V[:rank,:], nu_last_step)
y *= s[:rank]
y = np.matmul(U[:,:rank], y)
print("--- it takes %s seconds to do matrix multiplication with SVD---" % (time.time() - start_time))
'''
print("start to do iterations")

for j in range(1, iterations):
    #print("iteration {}".format(j))
    
    ay = ay/1.0
    a_lambda = ay
    a_nu = ay
    
    
    dy = C_pri + lambda_1_last_step*C_sec + np.matmul(s2sa, nu_last_step) - np.matmul(P_sa_s, nu_last_step)
    
    # update y
    y_current_step = y_last_step - ay*dy
    # projection back to domain
    y_current_step = np.multiply(y_current_step, y_current_step>=0)
    y_traces[:, j] = y_current_step
    
    d_lambda1 = np.dot(y_last_step, C_sec) - threshold
    # update lambda1
    lambda_1_current_step = lambda_1_last_step + a_lambda * d_lambda1
    # projection back to domain
    lambda_1_current_step = max(lambda_1_current_step, 0)
    
    # update nu
    #start_time = time.time()
    d_nu = np.matmul(s2sa_transpose, y_last_step) - indicator_initi - np.matmul(P_sa_s_transpose, y_last_step)
    #print("--- it takes %s seconds to finish a loop---" % (time.time() - start_time))
    nu_current_step = nu_last_step + a_nu*d_nu           
    
    
    y_last_step = np.copy(y_current_step)
    lambda_1_last_step = lambda_1_current_step
    nu_last_step = np.copy(nu_current_step)
    
    C_current_step = np.dot(y_current_step, C_pri)
    obj_traces.append(C_current_step*scale)
    obj_mean.append(j*obj_mean[j-1]/(j+1) + obj_traces[j]/(j+1))
    #print('objective mean is {}'.format(obj_mean[-1]))
    

print("Iterations done!")
print("converge value is {}".format(obj_mean[-1]))       
plt.plot(obj_mean)




        
