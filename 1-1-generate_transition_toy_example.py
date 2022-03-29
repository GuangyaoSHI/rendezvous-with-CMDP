# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from utils import *
import pickle
import logging
import time
import os


# generate state transition function
# UGV_task is a directed graph. Node name is an index
threshold = 0.1
experiment_name = '_toy_example'

print("threshold is {}, experiment name is {}".format(threshold, experiment_name))

# control parameter to decide whether we need to discretize the distribution
# if the distribution has been discretized before, the data is already saved, no need to generate again
discretize_power_distribution = False
# battery_interval 
battery_interval = 2

# file name
current_directory = os.getcwd()
target_directory = os.path.join(current_directory, r'transition_information')
if not os.path.exists(target_directory):
   os.makedirs(target_directory)

P_s_a_name = os.path.join(target_directory, 'P_s_a'+experiment_name+'.obj')
transition_graph_name = os.path.join(target_directory, 'state_transition_graph'+experiment_name+'.obj')



# different states
state_f = ('f', 'f', 'f', 'f', 'f')
state_l = ('l', 'l', 'l', 'l', 'l')
state_init = (0, 0, 1, 20, 0)

# actions
actions = ['v_be',  'v_be_be']

# generate UGV task
G = nx.DiGraph()
G.add_edge(0, 1)
G.add_edge(1, 0)
G.nodes[0]['pos'] = (0, 1)
G.nodes[1]['pos'] = (15, 1)
G.graph['UGV_goal'] = 1
pos = nx.get_node_attributes(G,'pos')
nx.draw(G, pos=pos,alpha=0.5, node_color='r', node_size=8)
UGV_task = G
print("UGV task is {}".format(UGV_task.nodes))

# generate road network
road_network = nx.Graph()
for i in range(0, 15):
    road_network.add_edge((i, 1), (i+1, 1), dis=1)
pos = dict(zip(road_network.nodes, road_network.nodes))
labels = dict(zip(road_network.nodes, np.arange(16)))
nx.draw(road_network, pos=pos, labels=labels, alpha=1, node_color='r', node_size=2)    

# generate UAV task
UAV_task = nx.DiGraph()
for i in range(0, 15):
    UAV_task.add_edge(i, i+1, dis=1)
    UAV_task.nodes[i]['pos'] = (i, 0)
    UAV_task.nodes[i+1]['pos'] = (i+1, 0)
pos = nx.get_node_attributes(UAV_task, 'pos')   
labels = dict(zip(UAV_task.nodes, UAV_task.nodes))
nx.draw(UAV_task, pos=pos, alpha=0.5, node_color='b', node_size=8, labels=labels)
UAV_goal = [x for x in UAV_task.nodes() if (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==1) or (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==0)]
UAV_goal = UAV_goal[0]
print("UAV task is {} and goal is {}".format(UAV_task.nodes, UAV_goal))


rendezvous = Rendezvous(UAV_task, UGV_task, road_network, battery=220e3)

# when we test the influences of different speed, we need to change the parameters below
rendezvous.velocity_ugv = 1
rendezvous.velocity_uav = {'v_be' : 1, 'v_br' : 1}  

    
rendezvous.display = False
print("extract transition with UAV speed {} and UGV speed {}".format(rendezvous.velocity_uav, rendezvous.velocity_ugv))

# get power consumption distribution:
# best endurance velocity
target_directory = os.path.join(current_directory, r'power_consumption')
power_stats_name = os.path.join(target_directory, 'power_stats_'+'v_be'+str(rendezvous.velocity_uav['v_be'])+
                                 'v_br'+str(rendezvous.velocity_uav['v_br'])+'.obj')


powers = {'v_be':[10, 12, 14], 'v_br':[10, 12, 14]}
probs = {'v_be':[0.8, 0.1, 0.1], 'v_br':[0.8, 0.1, 0.1]}


for action in probs:
    sum_prob = 0
    for i in probs[action]:
        sum_prob += i
    print("sum of prob for action {} is {}".format(action, sum_prob))

# transition probability
P_s_a = {}


start_time = time.time()
for uav_state in UAV_task.nodes:
    print("uav node {} and position {}".format(uav_state, UAV_task.nodes[uav_state]['pos']))
    for ugv_state in road_network.nodes:
        for battery in range(0, 101, battery_interval):
            # Todo: different node may represent the same node, this can cause some problem
            for ugv_task_node in UGV_task.nodes:
                # power state
                energy_state = battery
                #state_physical = uav_state + ugv_state + (energy_state, ) + (ugv_task_node, )
                state = (uav_state, ) + ugv_state + (battery, ) + (ugv_task_node, )
                P_s_a[state] = {}
                 
                if uav_state == UAV_goal:
                    P_s_a[state]['l'] = {}
                    P_s_a[state]['l'][state_l] = 1
                    continue
                
                for action in actions:
                    P_s_a[state][action] = {}
                
                for action in actions:
                    if action in ['v_be',  'v_br']:  
                        # UAV choose to go to next task node with best endurance velocity
                        descendants = list(UAV_task.neighbors(uav_state))
                        assert len(descendants) == 1
                        UAV_state_next = descendants[0]
                        duration = UAV_task.edges[uav_state, UAV_state_next]['dis'] / rendezvous.velocity_uav[action]
                        
                        # compute the energy distribution
                        energy_states = list(energy_state-np.array(powers[action])*duration)
                        energy_distribution = dict(zip(energy_states, probs[action]))
                        
                        UGV_road_state = ugv_state + ugv_state
                        UGV_state_next, UGV_road_state_next, UGV_task_node_next = rendezvous.UGV_transit(ugv_state, UGV_road_state, ugv_task_node, duration)                        
                        # use discrete UGV_state by assigning UGV to one road state
                        rs1 = np.linalg.norm(np.array(UGV_state_next)-np.array([UGV_road_state_next[0], UGV_road_state_next[1]]))
                        rs2 = np.linalg.norm(np.array(UGV_state_next)-np.array([UGV_road_state_next[2], UGV_road_state_next[3]]))
                        if rs1<rs2:
                            UGV_state_next = (UGV_road_state_next[0], UGV_road_state_next[1])
                        else:
                            UGV_state_next = (UGV_road_state_next[2], UGV_road_state_next[3])
                            
                        for p_c in energy_states:
                            if p_c < 0:
                                state_ = ('f',  'f', 'f', 'f', 'f')
                            else:
                                soc = p_c
                                temp = soc%battery_interval
                                if temp >= (battery_interval / 2):
                                    soc = soc - temp + battery_interval
                                    assert soc%battery_interval == 0
                                else:
                                    soc = soc - temp
                                    assert soc%battery_interval == 0
                                state_ = (UAV_state_next, ) + UGV_state_next + (soc, )+(UGV_task_node_next,)
                            if state_ not in P_s_a[state][action]:
                                P_s_a[state][action][state_] = energy_distribution[p_c]
                            else:
                                # assert state_ == ('f', 'f', 'f', 'f', 'f', 'f')
                                P_s_a[state][action][state_] += energy_distribution[p_c]
                        
                        # for debug
                        sum_debug = 0
                        for ns in P_s_a[state][action]:
                            assert P_s_a[state][action][ns]>=0, "negative prob in forward process"
                            sum_debug += P_s_a[state][action][ns]
                        assert abs(sum_debug-1)<0.001, "transition in forward process is not accurate"

                        
                    if action in ['v_be_be', 'v_br_br']:
                        # compute UAV position after rendezvous
                        descendants = list(UAV_task.neighbors(uav_state))
                        assert len(descendants) == 1
                        UAV_state_next = descendants[0]
                        
                        ugv_road_state = ugv_state + ugv_state
                        v1 = action[0:4]
                        v2 = 'v'+action[4:]
                        rendezvous_state, t1, t2 = rendezvous.rendezvous_point(UAV_task.nodes[uav_state]['pos'], UAV_task.nodes[UAV_state_next]['pos'], ugv_state, 
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
                        energy_states = list(energy_state - np.array(powers[v1])*t1)
                        energy_distribution = dict(zip(energy_states, probs[v1]))
                        
                        failure_prob = 0
                        for p_c in energy_states:
                            # first find the failure probability
                            if p_c < 0:
                                failure_prob += energy_distribution[p_c]
                        
                        if failure_prob > 1e-10:
                            P_s_a[state][action][state_f] = failure_prob
                        else:
                            failure_prob = 0
                            P_s_a[state][action][state_f] = failure_prob
                        
                        if abs(failure_prob - 1) < 1e-10:
                            continue
                        
                        failure_prob = min(failure_prob, 1)
                        energy_states2 = list(100 - np.array(powers[v2])*t2)
                        #Todo: probs_be need to change if br is used
                        if t2 == 0: 
                            energy_distribution2 = {100: 1}
                        else:
                            energy_distribution2 = dict(zip(energy_states2, probs[v2]))
                        for p_c in energy_distribution2:
                            if p_c < 0:
                                state_ = ('f',  'f', 'f', 'f', 'f')
                            else:
                                soc = p_c
                                temp = soc%battery_interval
                                if temp >= (battery_interval / 2):
                                    soc = soc - temp + battery_interval
                                    assert soc%battery_interval == 0
                                    assert soc >= 0
                                else:
                                    soc = soc - temp
                                    assert soc%battery_interval == 0
                                    assert soc >= 0
                                    
                                state_ = (UAV_state_next, ) + UGV_state_next + (soc, )+(UGV_task_node_next,)
                            
                            if state_ not in P_s_a[state][action]:
                                P_s_a[state][action][state_] = energy_distribution2[p_c]*(1-failure_prob)
                            else:
                                # assert state_ == ('f', 'f', 'f', 'f', 'f', 'f')
                                P_s_a[state][action][state_] += energy_distribution2[p_c]*(1-failure_prob)
                        # for debug
                        sum_debug = 0
                        for ns in P_s_a[state][action]:
                            assert P_s_a[state][action][ns]>=0, "negative prob in rendezvous process"
                            sum_debug += P_s_a[state][action][ns]
                        assert abs(sum_debug-1)<0.001, "transition in rendezvous process is not accurate"


action = 'l'
P_s_a[state_f] = {}
P_s_a[state_f][action] = {}
P_s_a[state_f][action][state_l] = 1
P_s_a[state_l] = {}
P_s_a[state_l][action] = {}
P_s_a[state_l][action][state_l] = 1

print("--- %s seconds ---" % (time.time() - start_time))


for state in P_s_a:
    for action in P_s_a[state]:
        sum_prob = 0
        for state_to_go in P_s_a[state][action]:
            P_s_a[state][action][state_to_go] = round(P_s_a[state][action][state_to_go], 8)
            if P_s_a[state][action][state_to_go] < 1e-10:
                P_s_a[state][action][state_to_go] = 0
            sum_prob += P_s_a[state][action][state_to_go]
        assert abs(sum_prob-1)<0.001, "transition is not accurate"
            
        next_states = list(P_s_a[state][action].keys())
        sum_prob = 0
        for state_to_go in next_states:
            sum_prob += P_s_a[state][action][state_to_go]
        
        for state_to_go in next_states:
            P_s_a[state][action][state_to_go] = P_s_a[state][action][state_to_go]/sum_prob
            #P_s_a[state][action][state_to_go] = round(P_s_a[state][action][state_to_go], 5)
        
        sum_prob = 0
        for state_to_go in next_states[1:]:
            sum_prob += P_s_a[state][action][state_to_go]
        
        P_s_a[state][action][next_states[0]] = max(1-sum_prob, 0)
        
        sum_prob = 0
        for state_to_go in P_s_a[state][action]:
            sum_prob += P_s_a[state][action][state_to_go]
        
        # add this line to double-check whether the transition probability is accurate enough for numpy to sample
        next_state_index = np.random.choice([i for i in range(len(next_states))], 1, p=list(P_s_a[state][action].values()))[0]
        assert abs(sum_prob-1)<0.0000001, "transition prob should sum to one"


print("\n------------np random choice works fine with transition distribution------\n")       
print("the transition probability has been normalized")

# Saving the state-transition graph:

with open(P_s_a_name, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(P_s_a, f)   
print("P_s_a is saved as "+P_s_a_name)


start_time = time.time()        
# construct a transition graph
G = nx.DiGraph()
for state in P_s_a:
    G.add_node(state, action=list(P_s_a[state].keys()))
    for action in P_s_a[state]:
        G.add_edge(state, state+(action,))
        for next_state in P_s_a[state][action]:
            assert next_state in P_s_a
            G.add_edge(state+(action,), next_state, action=action, prob=P_s_a[state][action][next_state])
print("--- %s seconds ---" % (time.time() - start_time))


# Saving the state-transition graph:
with open(transition_graph_name, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(G, f)
print("state transition graph is saved as "+transition_graph_name)

