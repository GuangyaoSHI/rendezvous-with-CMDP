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


y = {}
lambda_1 = 1
lambda_sa = {}
nu = {}
C_sa_pri = {}
C_sa_sec = {} 
print("start to initialize")
for state in P_s_a:
    if state != state_l:
        nu[state] = 1
    for action in P_s_a[state]:
        y[state+(action,)] = 0.01
        lambda_sa[state+(action,)]  = 500
        C_sa_pri[state+(action,)] = reward(state+(action,))
        C_sa_sec[state+(action,)] = cost(state+(action,))




# update step size
ay = 0.001
a_lambda = 0.1
a_nu = 0.1

'''
start_time = time.time()
C_last_step = 0
for s_a in y:
    C_last_step += y[s_a]*C_sa_pri[s_a]
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()    
np.dot(np.array(list(y.values())), np.array(list(C_sa_pri.values())))
print("--- %s seconds ---" % (time.time() - start_time))
'''

C_last_step = -sys.maxsize
y_last_step = copy.deepcopy(y)
lambda_1_last_step = lambda_1
lambda_sa_last_step = copy.deepcopy(lambda_sa)
nu_last_step = copy.deepcopy(nu)

y_current_step = copy.deepcopy(y)
lambda_1_current_step = lambda_1
lambda_sa_current_step = copy.deepcopy(lambda_sa)
nu_current_step = copy.deepcopy(nu)
C_current_step = np.dot(np.array(list(y_current_step.values())), np.array(list(C_sa_pri.values())))

traces = [C_current_step]
print("start to do iterations")
while abs(C_current_step - C_last_step)>0.1:
    C_last_step = C_current_step
    for s_a in y:
        # dy
        state = s_a[0:5]
        last_term_dy = 0
        for state_next in P_s_a[state][s_a[-1]]:
            if state_next != state_l:
                s_a_s = s_a + state_next
                last_term_dy += transition_prob(s_a_s)*nu_last_step[state_next]
        
        if state != state_l:
            dy = C_sa_pri[s_a] + lambda_1_last_step*C_sa_sec[s_a] - \
                lambda_sa_last_step[s_a] + nu_last_step[state] - last_term_dy
        else:
            dy = C_sa_pri[s_a] + lambda_1_last_step*C_sa_sec[s_a] - lambda_sa_last_step[s_a] - last_term_dy
        # update y
        y_current_step[s_a] = y_last_step[s_a] - ay*dy
        if y_current_step[s_a]<0:
            y_current_step[s_a] = 0
        if y_current_step[s_a] > 1:
            y_current_step[s_a] = 1
        y_last_step[s_a] = y_current_step[s_a]
       
        # d lambda_sa
        d_lambda_sa = -y_last_step[s_a]
        # update lambda_sa
        lambda_sa_current_step[s_a] = lambda_sa_last_step[s_a] + a_lambda*d_lambda_sa
        if lambda_sa_current_step[s_a] < 0:
            lambda_sa_current_step[s_a] = 0
        lambda_sa_last_step[s_a] = lambda_sa_current_step[s_a]
        
    # d lambda1
    d_lambda1 = np.dot(np.array(list(y_last_step.values())), np.array(list(C_sa_sec.values()))) - threshold
    # update lambda1
    lambda_1_current_step = lambda_1_last_step + a_lambda*d_lambda1
    if lambda_1_current_step < 0:
        lambda_1_current_step = 0
    lambda_1_last_step = lambda_1_current_step
        
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
    traces.append(C_current_step)    

print("optimal value is {}".format(C_current_step))            

plt.plot(traces)





        
#indices for variables first define as list and then transform to numpy array
'''
y_sa = []
lambda1 = 1
lambda_sa = []
nu_s = []

# primary and secondary cost matrix
C_sa_pri = []
C_sa_sec = [] 
P_sa_s_prime = []

transition_vector = {}
i=0
for state in P_s_a:
    # Todo: need to check whether the state_l is the last state
    if state == state_l:
        print("state_l index is {}".format(i))
        print("number of states is {}".format(len(list(P_s_a.keys()))))
    if state != state_l:
        transition_vector[state] = 0 
    i += 1
    
for state in P_s_a:
    # Todo: need to check whether the state_l is the last state
    if state != state_l:
        nu_s.append(100) # initialize nu
    
    y = []
    lamb = []
    Cp = []
    Cs = []
    P = []
    for action in P_s_a[state]:
        # initialize y_sa labmda_sa 
        y.append(0.5)
        lamb.append(100)
        Cp.append(reward(state+(action,)))
        Cs.append(cost(state+(action,)))
        
        temp = copy.deepcopy(transition_vector)
        assert state_l not in temp
        for state_next in P_s_a[state][action]:
            if state_next in temp:
                assert state_next != state_l
                temp[state_next] = P_s_a[state][action][state_next]
        P.append(np.array(temp.values()))
            
    y_sa.append(np.array(y))
    lambda_sa.append(np.array(lamb))
    C_sa_pri.append(np.array(Cp))
    C_sa_sec.append(np.array(Cs))
    P_sa_s_prime.append(np.array(P))
    
y_sa = np.array(y_sa, dtype=object)
lambda_sa = np.array(lambda_sa, dtype=object)
P_sa_s_prime = np.array(P_sa_s_prime, dtype=object)
'''  



'''
policy = {}
for s_a in indices:
    state = tuple(list(s_a)[0:5])
    actions = list(P_s_a[state].keys())
    # numerical issue, gurobi will sometimes return a very small negative number for zero
    if rho[s_a].x < 0:
        deno = 0
    else:
        deno = rho[s_a].x
    # if rho[s_a].x > 0:
        #print("s_a: {} rho_s_a: {}".format(s_a, rho[s_a].x ))
    num = 0
    for action in actions:
        #assert rho[state+(action, )].x >= 0
        if rho[state+(action, )].x < 0:
            num += 0
        else:
            num += rho[state+(action, )].x
        if rho[state+(action, )].x < 0:
            print("------------- numerical accuracy -------------")
            print(rho[state+(action, )].x)
    
    #assert num >= deno
    if num == 0:
        policy[s_a] = 0
    else:
        policy[s_a] = deno/num
        if policy[s_a]>0:
            print("s_a: {} pi_s_a: {}".format(s_a, policy[s_a]))
    #print("Optimal Prob of choosing action {} at state {} is {}".format((s_a[2], s_a[3]), state, deno/num))

print("objective value is {}".format(obj.getValue()))        
# Saving the objects:
       
with open(policy_name, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(policy, f)        
'''    
    
