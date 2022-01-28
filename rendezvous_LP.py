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
UGV_task_states = list(set([UGV_task.nodes[node]['pos'] for node in UGV_task.nodes])) 
road_network = generate_road_network()
actions = ['v_be', 'v_be_be']
rendezvous = Rendezvous(UAV_task, UGV_task, road_network)
rendezvous.display = False

# transition probability
P_s_a = {}

probs = [2.27e-2, 13.6e-2, 34.13e-2, 34.13e-2, 13.6e-2, 2.27e-2]
values = [-0.25, -0.15, -0.05, 0.05, 0.15, 0.25]

state_f = ('f', 'f', 'f', 'f', 'f', 'f')
state_l = ('l', 'l', 'l', 'l', 'l', 'l')
state_init = (0, 3664, 3731, 0, rendezvous.battery, 3731, 0)

for uav_state in UAV_task.nodes:
    for ugv_state in road_network.nodes:
        for battery in range(0, 101):
            # Todo: different node may represent the same node, this can cause some problem
            for ugv_task_node in UGV_task.nodes:
                battery_state = battery/100*rendezvous.battery
                state_physical = uav_state + ugv_state + (battery_state, ) + (ugv_task_node, )
                state = uav_state + ugv_state + (battery, ) + (ugv_task_node, )
                P_s_a[state] = {}
                P_s_a[state]['l'] = {}
                P_s_a[state]['l'][state_l] = 0
                if uav_state == UAV_goal:
                    P_s_a[state]['l'][state_l] = 1
                    continue
                
                for action in actions:
                    P_s_a[state][action] = {}
                
                for action in actions:
                    if action == 'v_be':  
                        power_states = list(battery_state - np.array(values)-1)
                        power_distribution = dict(zip(power_states, probs))
                        assert len(power_states) == 6
                        
                        UGV_road_state = ugv_state + ugv_state
                        UAV_state, UGV_state, UGV_road_state, UGV_task_node, battery_state = rendezvous.transit(state_physical, action, UGV_road_state, ugv_task_node)
                        
                        # use discrete UGV_state by assigning UGV to one road state
                        rs1 = np.linalg.norm(np.array(UGV_state)-np.array([UGV_road_state[0], UGV_road_state[1]]))
                        rs2 = np.linalg.norm(np.array(UGV_state)-np.array([UGV_road_state[2], UGV_road_state[3]]))
                        if rs1<rs2:
                            UGV_state = (UGV_road_state[0], UGV_road_state[1])
                        else:
                            UGV_state = (UGV_road_state[2], UGV_road_state[3])
                            
                        for p_c in power_states:
                            if p_c < 0:
                                state_ = ('f', 'f', 'f', 'f', 'f', 'f')
                            else:
                                state_ = UAV_state + UGV_state + (int(p_c/rendezvous.battery*100), )+(UGV_task_node,)
                            if state_ not in P_s_a[state][action]:
                                P_s_a[state][action][state_] = power_distribution[p_c]
                            else:
                                # assert state_ == ('f', 'f', 'f', 'f', 'f', 'f')
                                P_s_a[state][action][state_] += power_distribution[p_c]
                        
                    if action == 'v_be_be':
                        ugv_road_state = ugv_state + ugv_state
                        v1 = action[0:4]
                        v2 = 'v'+action[4:]
                        uav_state_next = list(UAV_task.neighbors(uav_state))[0]
                        rendezvous_state, t1, t2 = rendezvous.rendezvous_point(uav_state, uav_state_next, ugv_state, ugv_road_state, ugv_task_node, 10, 10)
                        rendezvous_road_state = rendezvous_state + rendezvous_state 
                        UGV_state_next, UGV_road_state_next, UGV_task_node_next = rendezvous.UGV_transit(rendezvous_state, rendezvous_road_state, ugv_task_node, t2)
                        
                        # use discrete UGV_state by assigning UGV to one road state
                        rs1 = np.linalg.norm(np.array(UGV_state)-np.array([UGV_road_state[0], UGV_road_state[1]]))
                        rs2 = np.linalg.norm(np.array(UGV_state)-np.array([UGV_road_state[2], UGV_road_state[3]]))
                        if rs1<rs2:
                            UGV_state = (UGV_road_state[0], UGV_road_state[1])
                        else:
                            UGV_state = (UGV_road_state[2], UGV_road_state[3])
                        
                        dis1 = np.linalg.norm(np.array(uav_state)-np.array(rendezvous_state))
                        power_states = list(battery_state - np.array(values)-dis1/rendezvous.power_measure)
                        power_distribution = dict(zip(power_states, probs))
                        assert len(power_states) == 6
                        
                        failure_prob = 0
                        for p_c in power_states:
                            # first find the failure probability
                            if p_c < 0:
                                failure_prob += power_distribution[p_c]
                        
                        state_ = ('f', 'f', 'f', 'f', 'f', 'f')
                        P_s_a[state][action][state_] = failure_prob
                        dis2 = np.linalg.norm(np.array(uav_state_next)-np.array(rendezvous_state))
                        power_states2 = list(rendezvous.battery - np.array(values)-dis2/rendezvous.power_measure)
                        power_distribution2 = dict(zip(power_states2, probs))
                        for p_c in power_states2:
                            if p_c < 0:
                                state_ = ('f', 'f', 'f', 'f', 'f', 'f')
                            else:
                                state_ = UAV_state + UGV_state + (int(p_c/rendezvous.battery*100), )+(UGV_task_node,)
                            if state_ not in P_s_a[state][action]:
                                P_s_a[state][action][state_] = power_distribution2[p_c]*(1-failure_prob)
                            else:
                                # assert state_ == ('f', 'f', 'f', 'f', 'f', 'f')
                                P_s_a[state][action][state_] += power_distribution2[p_c]*(1-failure_prob)


action = 'l'
P_s_a[state_f] = {}
P_s_a[state_f][action] = {}
P_s_a[state_f][action][state_l] = 1
P_s_a[state_l] = {}
P_s_a[state_l][action] = {}
P_s_a[state_l][action][state_l] = 1
         
# construct a transition graph
G = nx.DiGraph()
for state in P_s_a:
    G.add_node(state, action=list(P_s_a[state].keys()))
    for action in P_s_a[state]:
        for next_state in P_s_a[state][action]:
            assert next_state in P_s_a
            G.add_edge(state, next_state, action=action, prob=P_s_a[state][action][next_state])

for node in G:
    if node not in P_s_a.keys():
        print(node)
        

# Saving the objects:
with open('P_s_a.obj', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(P_s_a, f)

# Getting back the objects:
with open('P_s_a.obj', 'rb') as f:  # Python 3: open(..., 'rb')
    P_s_a = pickle.load(f)
                       
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
    if action == 'l' and (next_state == state_l):
        return 1
    else:
        return 0
    
    # goal
    if (state[0], state[1]) == UAV_goal:
        if action == 'l' and (next_state == state_l):
            return 1
        else:
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
    
    return -1


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


threshold = 0.1
model = gp.Model('LP_CMDP')

#indics for variables
indices = []

for state in P_s_a:
    for action in P_s_a[state]:
        indices.append(state + (action,))

# add Non-negative continuous variables that lies between 0 and 1        
rho = model.addVars(indices, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='rho')

# add constraints 
# cost constraints
C = 0
for s_a in indices:
    C += rho[s_a] * cost(s_a)
cost_ctrs =  model.addConstr(C <= threshold, name = 'cost_ctrs')    


for state in P_s_a:
    if state == state_l:
        continue
    lhs = 0
    rhs = 0
    for action in P_s_a[state]:
        lhs += rho[state+(action,)]
        
    # for s_a in indices:
    #     s_a_s = s_a + state
    #     rhs += rho[s_a] * transition_prob(s_a_s)
    for s in G.predecessors(state):
        action = G.edges[s, state]['action']
        s_a = s + (action,)
        s_a_s = s + (action,) + state
        rhs += rho[s_a] * transition_prob(s_a_s)
    
    if state == state_init:
        rhs += 1
    
    model.addConstr(lhs == rhs, name = str(state))

model.addConstrs((rho[s_a] >=0 for s_a in indices), name='non-negative-ctrs')
    
obj = 0
for s_a in indices:
    obj += rho[s_a] * reward(s_a)
    
model.setObjective(obj, GRB.MAXIMIZE)
model.optimize()

policy = {}
for s_a in indices:
    state = tuple(list(s_a)[0:6])
    actions = P_s_a[state]
    deno = rho[s_a].x
    num = 0
    for action in actions:
        num += rho[(s_a[0], s_a[1], action[0], action[1])].x
    
    policy[s_a] = deno/num
    #print("Optimal Prob of choosing action {} at state {} is {}".format((s_a[2], s_a[3]), state, deno/num))
        
# Saving the objects:
with open('policy.obj', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(policy, f)        
    
    
    
    

                                



                    

                    
            
