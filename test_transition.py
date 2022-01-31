# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from utils import *
import pickle
import random

# Getting back the objects:
with open('P_s_a.obj', 'rb') as f:  # Python 3: open(..., 'rb')
    P_s_a = pickle.load(f)
    
# Getting back the objects:
with open('policy.obj', 'rb') as f:  # Python 3: open(..., 'rb')
    policy = pickle.load(f)


# Getting back the objects:
with open('state_transition_graph.obj', 'rb') as f:  # Python 3: open(..., 'rb')
    G = pickle.load(f)
    
state_f = ('f', 'f', 'f', 'f', 'f', 'f')
state_l = ('l', 'l', 'l', 'l', 'l', 'l')    
state_init = (0, 0, 3, 3, 100, 0)

# generate state transition function
UAV_task = generate_UAV_task()
UAV_goal = [x for x in UAV_task.nodes() if (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==1) or (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==0)]
UAV_goal = UAV_goal[0]
# UGV_task is a directed graph. Node name is an index
UGV_task = generate_UGV_task()
road_network = generate_road_network()
actions = ['v_be', 'v_be_be']
rendezvous = Rendezvous(UAV_task, UGV_task, road_network, battery=7)

state = state_init
UAV_state = (state[0], state[1])
UGV_state = (state[2], state[3])
energy_state = state[4]/100*rendezvous.battery
UGV_task_node = state[5]
state_traces = [state]
action_traces = []

while (UAV_state != UAV_goal and state != state_f):
    actions = []
    probs = []
    probs_uniform = []
    p = 0
    for action in P_s_a[state]:
        actions.append(action)
        s_a = state + (action,)
        probs.append(policy[s_a])
        p += policy[s_a]
        assert p <= 1
        probs_uniform.append(p)
    print("probs: {} probs_uniform:{}".format(probs, probs_uniform))    
    # sample 
    random_num = np.random.rand()
    for i in range(len(probs_uniform)):
        if random_num <= probs_uniform[i]:
            break
    action = actions[i]
    action_traces.append(action)
    # next_states = []
    # for neighbor in G.neighbors(state):
    #     print("neighbor is {} action is {}".format(neighbor, G.edges[state, neighbor]['action']))
    #     if G.edges[state, neighbor]['action'] == action:
    #         next_states.append(neighbor)
    
    
    # Todo: it shouldn't be uniform sampling
    next_state = random.sample(list(P_s_a[state][action].keys()), 1)[0]
    state_traces.append(next_state)
    state = next_state
    UAV_state = (state[0], state[1])

    # state_physical = UAV_state + UGV_state + (energy_state, ) + (UGV_task_node, )
    # UGV_road_state =  UGV_state +  UGV_state
    # UAV_state, UGV_state, UGV_road_state, UGV_task_node, energy_state = rendezvous.transit(state_physical, action, UGV_road_state, UGV_task_node)
    
    # if (energy_state == 'f') or (energy_state < 0):
    #     state = state_f
    #     print("transition to failure state from state {}".format(state))
    
    #  # use discrete UGV_state by assigning UGV to one road state
    # rs1 = np.linalg.norm(np.array(UGV_state)-np.array([UGV_road_state[0], UGV_road_state[1]]))
    # rs2 = np.linalg.norm(np.array(UGV_state)-np.array([UGV_road_state[2], UGV_road_state[3]]))
    # if rs1<rs2:
    #     UGV_state = (UGV_road_state[0], UGV_road_state[1])
    # else:
    #     UGV_state = (UGV_road_state[2], UGV_road_state[3])
    
    # # compute state in CMDP
    # state = UAV_state + UGV_state + (int(energy_state/rendezvous.battery*100), )+(UGV_task_node, )

UAV_traces = []
UGV_traces = []
battery_traces = []
for i in range(len(action_traces)):
    UAV_traces.append((state_traces[i][0], state_traces[i][1]))
    UGV_traces.append((state_traces[i][2], state_traces[i][3]))
    battery_traces.append(state_traces[i][4])
    UAV_traces.append(action_traces[i])

UAV_traces.append((state_traces[i+1][0], state_traces[i+1][1]))
UGV_traces.append((state_traces[i+1][2], state_traces[i+1][3]))  
battery_traces.append(state_traces[i+1][4])

print("UAV traces: {}".format(UAV_traces))
print("UGV traces: {}".format(UGV_traces))  
print("battery traces: {}".format(battery_traces))
    
       
    