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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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

state = state_init
UAV_state = (state[0], state[1])
UGV_state = (state[2], state[3])
energy_state = state[4]/100*rendezvous.battery
UGV_task_node = state[5]
state_traces = [state]
action_traces = []
duration_traces = []

i = 0
while (UAV_state != UAV_goal and state != state_f):
    actions = []
    probs = []
    for action in P_s_a[state]:
        actions.append(action)
        s_a = state + (action,)
        probs.append(policy[s_a])
    
    print("step {}".format(i))
    print("state {} policy {}".format(state, dict(zip(actions, probs))))
    # sample 
    action = np.random.choice(actions, 1, p=probs)[0]
    print("take action {}".format(action))
    action_traces.append(action)
    
    # compute UAV position after rendezvous
    descendants = list(UAV_task.neighbors(UAV_state))
    assert len(descendants) == 1
    UAV_state_next = descendants[0]
    if len(action)>4:
        
        ugv_road_state = UGV_state + UGV_state
        v1 = action[0:4]
        v2 = 'v'+action[4:]
        rendezvous_state, t1, t2 = rendezvous.rendezvous_point(UAV_state, UAV_state_next, UGV_state, 
                                                               ugv_road_state, UGV_task_node, 
                                                               rendezvous.velocity_uav[v1], 
                                                               rendezvous.velocity_uav[v2])
        print("rendezvous at {}!!!!!!!!!!!!!".format(rendezvous_state))
        duration_traces.append(t1+t2)
    else:
        duration = UAV_task.edges[UAV_state, UAV_state_next]['dis'] / rendezvous.velocity_uav[action]
        duration_traces.append(duration)
    # Todo: 
    next_states = list(P_s_a[state][action].keys())
    next_state_index = np.random.choice([i for i in range(len(next_states))], 1, p=list(P_s_a[state][action].values()))[0]
    next_state = next_states[next_state_index]
    print("transit to state {}".format(next_state))
    
    fig, axs = plt.subplots()
    # plot roadnetwork
    add_label = True
    for edge in road_network.edges:
        x = [edge[0][0], edge[1][0]]
        y = [edge[0][1], edge[1][1]]
        if add_label:
            axs.plot(x, y, marker='*', color='b', alpha=0.2, label='road network')
            add_label = False
        else:
            axs.plot(x, y, marker='*', color='b', alpha=0.2)
    
    add_label = True
    # plot UAV task
    for edge in UAV_task.edges:
        x, y = edge[0][0], edge[0][1] 
        u, v = edge[1][0]-edge[0][0], edge[1][1]-edge[0][1] 
        if add_label:
            axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=0.3, label="UAV task")
            add_label = False
        else:
            axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=0.3)

    # plot UAV UGV position now
    axs.plot(UAV_state[0], UAV_state[1], marker='o', color='r', label="UAV now")
    
    #plt.text(UAV_state[0], UAV_state[1], 'UAV now', horizontalalignment='left')
    axs.plot(UGV_state[0], UGV_state[1], marker='s', color='g', label="UGV now")
    #plt.text(UGV_state[0], UGV_state[1], 'UGV now', horizontalalignment='right')
    # plot UAV transition
    axs.legend()
    axs.set_title("policy:"+str(dict(zip(actions, probs))))
    axs.set_xlabel("battery:"+str(state[4]))
    #axs.set_aspect('equal', 'box')
    axs.axis('equal')
    fig.savefig("step"+str(i)+".pdf")
    
    
    
    
    
    fig, axs = plt.subplots()
    # plot roadnetwork
    add_label = True
    for edge in road_network.edges:
        x = [edge[0][0], edge[1][0]]
        y = [edge[0][1], edge[1][1]]
        if add_label:
            axs.plot(x, y, marker='*', color='b', alpha=0.2, label='road network')
            add_label = False
        else:
            axs.plot(x, y, marker='*', color='b', alpha=0.2)
    
    add_label = True
    # plot UAV task
    for edge in UAV_task.edges:
        x, y = edge[0][0], edge[0][1] 
        u, v = edge[1][0]-edge[0][0], edge[1][1]-edge[0][1] 
        if add_label:
            axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=0.3, label="UAV task")
            add_label = False
        else:
            axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=0.3)

    # plot UAV UGV position now
    axs.plot(UAV_state[0], UAV_state[1], marker='o', color='r', label="UAV now")
    
    #plt.text(UAV_state[0], UAV_state[1], 'UAV now', horizontalalignment='left')
    axs.plot(UGV_state[0], UGV_state[1], marker='s', color='g', label="UGV now")
    #plt.text(UGV_state[0], UGV_state[1], 'UGV now', horizontalalignment='right')
    # plot UAV transition
    axs.legend()
    axs.set_title("policy:"+str(dict(zip(actions, probs))))
    axs.set_xlabel("choose action "+str(action) + "       battery:"+str(state[4]))
    #axs.set_aspect('equal', 'box')
    axs.axis('equal')
    if len(action)>4:
        x, y = UAV_state[0], UAV_state[1]
        u, v = rendezvous_state[0] - UAV_state[0], rendezvous_state[1] - UAV_state[1] 
        axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=1, color='r')
        axs.plot(x, y, marker='p', color='m')
        #plt.text(x, y, 'rendezvous here')
        
        x, y = rendezvous_state[0], rendezvous_state[1]
        u, v = UAV_state_next[0]-rendezvous_state[0], UAV_state_next[1]-rendezvous_state[1]
        axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=1, color='r')
    else:
        if next_state == state_f:
            UAV_state_next = (0, 0)
        else:
            UAV_state_next = (next_state[0], next_state[1])
        x, y = UAV_state[0], UAV_state[1]
        u, v = UAV_state_next[0] - UAV_state[0], UAV_state_next[1] - UAV_state[1] 
        axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=1, color='r')
    fig.savefig("step"+str(i)+"action"+".pdf")
    
    
    
    
    
    i += 1
    state_traces.append(next_state)
    state = next_state
    UAV_state = (state[0], state[1])
    UGV_state = (state[2], state[3])
    UGV_task_node = state[5]



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
    
       
    