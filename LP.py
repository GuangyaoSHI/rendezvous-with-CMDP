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

logger = logging.getLogger(__name__) # Set up logger

state_f = ('f', 'f', 'f', 'f', 'f', 'f')
state_l = ('l', 'l', 'l', 'l', 'l', 'l')
state_init = (int(6.8e3), int(19.1e3), int(6.8e3), int(19.1e3), 100, 0)

# generate state transition function
UAV_task = generate_UAV_task()
UAV_goal = [x for x in UAV_task.nodes() if (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==1) or (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==0)]
UAV_goal = UAV_goal[0]
# UGV_task is a directed graph. Node name is an index
UGV_task = generate_UGV_task()
road_network = generate_road_network()

actions = ['v_be', 'v_br', 'v_be_be', 'v_br_br']
rendezvous = Rendezvous(UAV_task, UGV_task, road_network, battery=280e3)

# Getting back the objects:
with open('P_s_a.obj', 'rb') as f:  # Python 3: open(..., 'rb')
    P_s_a = pickle.load(f)

with open('state_transition_graph.obj', 'rb') as f:  # Python 3: open(..., 'rb')
    G = pickle.load(f)

#reduce the size of G
# G = copy.deepcopy(GG)
# for node in GG.nodes:
#     if not nx.has_path(GG, source=state_init, target=node):
#         G.remove_node(node)
                     
# create transition function 
def transition_prob(s_a_s):
    state = tuple(list(s_a_s)[0:6])
    action = s_a_s[6]
    next_state = tuple(list(s_a_s)[7:])
    
    # unreachable state
    if (action not in P_s_a[state]) or (next_state not in P_s_a[state][action]):
        #print("trying to transit to an unreachable state")
        return 0
    
    # failure state
    if state == state_f:     
        assert action == 'l' and (next_state == state_l)
        return 1

    # goal
    if (state[0], state[1]) == UAV_goal:
        assert action == 'l' and (next_state == state_l)
        return 1
    # loop state
    if state == state_l:
        assert action == 'l' and (next_state == state_l)
        return 1
    
    if P_s_a[state][action][next_state] < 1e-10:
        return 0
    return P_s_a[state][action][next_state] 


def reward(s_a):
    state = tuple(list(s_a)[0:6])
    action = s_a[6]
    
    if state == state_f:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if state == state_l:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if (state[0], state[1]) == UAV_goal:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if action in ['v_be', 'v_br']:
        uav_state = state[0:2]
        uav_state_next = list(UAV_task.neighbors(uav_state))[0]
        duration = UAV_task.edges[uav_state, uav_state_next]['dis'] / rendezvous.velocity_uav[action]
        return -duration
    
    if action in ['v_be_be', 'v_br_br']:
        uav_state = state[0:2]
        uav_state_next = list(UAV_task.neighbors(uav_state))[0]
        ugv_state = state[2:4]
        ugv_road_state = ugv_state + ugv_state
        ugv_task_node = state[-1]
        v1 = action[0:4]
        v2 = 'v'+action[4:]
        uav_state_next = list(UAV_task.neighbors(uav_state))[0]
        rendezvous_state, t1, t2 = rendezvous.rendezvous_point(uav_state, uav_state_next, ugv_state, 
                                                               ugv_road_state, ugv_task_node, 
                                                               rendezvous.velocity_uav[v1], 
                                                                               rendezvous.velocity_uav[v2])
        return -(t1+t2+rendezvous.charging_time)
    
    
def cost(s_a):
    state = tuple(list(s_a)[0:6])
    action = s_a[6]
    
    if state == state_f:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if state == state_l:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if (state[0], state[1]) == UAV_goal:
        assert action == 'l', "should transit to loop state"
        return 0
    
    # compute cost
    C = 0
    for next_state in P_s_a[state][action]:
        if next_state == state_f:
            C += P_s_a[state][action][next_state]
    
    assert C <= 1
    
    return C


threshold = 0.5
model = gp.Model('LP_CMDP')

#indics for variables
indices = []

for state in P_s_a:
    # if state == state_l:
    #     continue
    for action in P_s_a[state]:
        indices.append(state + (action,))

# add Non-negative continuous variables that lies between 0 and 1        
rho = model.addVars(indices, lb= 0.0,  vtype=GRB.CONTINUOUS,  name='rho')
model.addConstrs((rho[s_a] >= 0.0 for s_a in indices), name='non-negative-ctrs')
model.update()
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
        assert len(s_a) == 7
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
    logger.info("just added constraint %s" % (lhs == rhs))


    
obj = 0
for s_a in indices:
    obj += rho[s_a] * reward(s_a)

model.Params.FeasibilityTol = 1e-9    
model.setObjective(obj, GRB.MAXIMIZE)
model.optimize()

policy = {}
for s_a in indices:
    state = tuple(list(s_a)[0:6])
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
        assert rho[state+(action, )].x >= 0
        if rho[state+(action, )].x < 0:
            num += 0
        else:
            num += rho[state+(action, )].x
        if rho[state+(action, )].x < 0:
            print("numerical issue !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(rho[state+(action, )].x)
            print("all state action pairs")
            for a in actions:
                print(rho[state+(a, )].x)
    
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
with open('policy'+str(threshold)+'.obj', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(policy, f)        
    
    
# simulate the whole process


# debug for gurobi solver
# for s_a in indices:
#     state = tuple(list(s_a)[0:6])
#     actions = list(P_s_a[state].keys())  
#     #print("state is {}".format(state))
#     for action in actions:
#         if abs(rho[state+(action, )].x) >0:
#             print("state-action is {} rho is {}".format(state+(action, ), rho[state+(action, )].x))

            
for s_a in indices:
    if rho[s_a].x >0:
        print("state-action is {} rho is {}".format(s_a, rho[s_a].x))