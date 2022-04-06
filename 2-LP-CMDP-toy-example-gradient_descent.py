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


state_f = ('sf')
goal = 'sg'
state_l = ('sl')
state_init = ('s0')

threshold = 0.3
print("threshold is {}".format(threshold))
# generate state transition function

P_s_a = {}
P_s_a['s0'] = {}
P_s_a['s0']['a1'] = {}
P_s_a['s0']['a1']['sg'] = 0.8
P_s_a['s0']['a1']['sf'] = 0.2
P_s_a['s0']['a2'] = {}
P_s_a['s0']['a2']['sg'] = 0.6
P_s_a['s0']['a2']['sf'] = 0.4

P_s_a['sg'] = {}
P_s_a['sg']['al'] = {}
P_s_a['sg']['al']['sl'] = 1
P_s_a['sf'] = {}
P_s_a['sf']['al'] = {}
P_s_a['sf']['al']['sl'] = 1
P_s_a['sl'] = {}
P_s_a['sl']['al'] = {}
P_s_a['sl']['al']['sl'] = 1

G = nx.DiGraph()
for state in P_s_a:
    G.add_node(state, action=list(P_s_a[state].keys()))
    for action in P_s_a[state]:
        G.add_edge(state, (state, )+(action,))
        for next_state in P_s_a[state][action]:
            assert next_state in P_s_a
            G.add_edge((state,)+(action,), next_state, action=action, prob=P_s_a[state][action][next_state])


                     
# create transition function 
def transition_prob(s_a_s):
    state = tuple(list(s_a_s)[0])
    action = s_a_s[1]
    next_state = tuple(list(s_a_s)[2])
    
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
    if state[0] == goal:
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



model = gp.Model('LP_CMDP')

#indics for variables
indices = []

start_time = time.time()
for state in P_s_a:
    # if state == state_l:
    #     continue
    for action in P_s_a[state]:
        indices.append(state + (action,))


# add Non-negative continuous variables that lies between 0 and 1        
rho = model.addVars(indices, lb=0,  vtype=GRB.CONTINUOUS,  name='rho')
model.addConstrs((rho[s_a] >= 0.0 for s_a in indices), name='non-negative-ctrs')

# add constraints 
# cost constraints
C = 0
for s_a in indices:
    # if cost(s_a)>0:
    #     print("s_a {} cost is {}".format(s_a, cost(s_a)))
    C += rho[s_a] * cost(s_a)
cost_ctrs =  model.addConstr(C <= threshold, name='cost_ctrs')    

print("start to add equality constraint")
for state in P_s_a: 
    if state == state_l:
        continue
    lhs = 0
    rhs = 0
    for action in P_s_a[state]:
        #print(rho[state+(action,)].x)
        lhs += rho[state+(action,)]
        
    # for s_a in indices:
    #     s_a_s = s_a + state
    #     rhs += rho[s_a] * transition_prob(s_a_s)

    #s_a_nodes = []
    for s_a in G.predecessors(state):
        assert len(s_a) == 6
        #if s_a not in s_a_nodes:
        #s_a_nodes.append(s_a)
        s_a_s = s_a + state
        rhs += rho[s_a] * transition_prob(s_a_s)
        
    
    if state == state_init:
        print('initial state delta function')
        rhs += 1
    
    model.addConstr(lhs == rhs, name = str(state))
    model.update()
    #print("equality constraint for state {}".format(state))


    
obj = 0
for s_a in indices:
    obj += rho[s_a] * reward(s_a)
    
model.setObjective(obj, GRB.MAXIMIZE)
model.Params.FeasibilityTol = 1e-9    
print("--- %s seconds ---" % (time.time() - start_time))

model.optimize()

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
            print("numerical issue !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
    
    
