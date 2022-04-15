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


# generate state transition function
# UGV_task is a directed graph. Node name is an index
randomcase = False
threshold = 0.1
experiment_name = '_velocity3'
velocity_ugv = 5.5
velocity_uav = 15
if randomcase:
    road_network = generate_road_network_random()
    with open('road_network_random.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(road_network, f)
    UAV_task = generate_UAV_task_random()
    with open('UAV_task_random.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(UAV_task, f)
    UGV_task = generate_UGV_task_random(road_network)
    with open('UGV_task_random.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(UGV_task, f)
else:
    UGV_task = generate_UGV_task()
    road_network = generate_road_network()
    UAV_task = generate_UAV_task()

UAV_goal = [x for x in UAV_task.nodes() if (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==1) or (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==0)]
UAV_goal = UAV_goal[0]


actions = ['v_be', 'v_br','v_be_be', 'v_br_br']
rendezvous = Rendezvous(UAV_task, UGV_task, road_network, battery=240e3)
rendezvous.velocity_ugv = velocity_ugv
rendezvous.velocity_uav = {'v_be' : velocity_uav, 'v_br' : velocity_uav}
rendezvous.display = False

# get power consumption distribution:
# best endurance velocity

stats = {}
for action in ['v_be', 'v_br']:
    stats[action] = rendezvous.get_power_consumption_distribution(rendezvous.velocity_uav[action])
with open('power_stats.obj', 'wb') as f:  # Python 3: open(..., 'rb')
    pickle.dump(stats, f)
    
powers = {'v_be':[], 'v_br':[]}
probs = {'v_be':[], 'v_br':[]}

for action in ['v_be', 'v_br']:
    for interval in stats[action]:
        powers[action].append((interval[0]+interval[1])/2)
        probs[action].append(stats[action][interval])


# transition probability
P_s_a = {}

# battery_interval 
battery_interval = 4

#probs = [2.27e-2, 13.6e-2, 34.13e-2, 34.13e-2, 13.6e-2, 2.27e-2]
#values = [-0.25, -0.15, -0.05, 0.05, 0.15, 0.25]

state_f = ('f', 'f', 'f', 'f', 'f')
state_l = ('l', 'l', 'l', 'l', 'l')

if randomcase:
    state_init = (0, 0, 0, 0, 100, 0)
else:
    state_init = (0, int(6.8e3), int(19.1e3), 100, 0)


start_time = time.time()
for uav_state in UAV_task.nodes:
    print("uav node {} and position {}".format(uav_state, UAV_task.nodes[uav_state]['pos']))
    for ugv_state in road_network.nodes:
        for battery in range(0, 101, battery_interval):
            # Todo: different node may represent the same node, this can cause some problem
            for ugv_task_node in UGV_task.nodes:
                # power state
                energy_state = battery/100*rendezvous.battery
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
                                soc = round(p_c/rendezvous.battery*100)
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
                        
                        if failure_prob > 0:
                            P_s_a[state][action][state_f] = failure_prob
                        
                        if failure_prob == 1:
                            continue
                        
                        energy_states2 = list(rendezvous.battery - np.array(powers[v2])*t2)
                        #Todo: probs_be need to change if br is used
                        energy_distribution2 = dict(zip(energy_states2, probs[v2]))
                        for p_c in energy_states2:
                            if p_c < 0:
                                state_ = ('f',  'f', 'f', 'f', 'f')
                            else:
                                soc = min(round(p_c/rendezvous.battery*100), 100)
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


action = 'l'
P_s_a[state_f] = {}
P_s_a[state_f][action] = {}
P_s_a[state_f][action][state_l] = 1
P_s_a[state_l] = {}
P_s_a[state_l][action] = {}
P_s_a[state_l][action][state_l] = 1

print("--- %s seconds ---" % (time.time() - start_time))



# Getting back the objects:
# P_s_a = {}
# files = ['P_s_a_0-10', 'P_s_a_11-20', 'P_s_a_21-30', 'P_s_a_31-40', 'P_s_a_41-50',
#          'P_s_a_51-60', 'P_s_a_61-70', 'P_s_a_71-80', 'P_s_a_81-90', 'P_s_a_91-100']    
# for file in files:
#     filename = file+'.obj'
#     with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
#         P_s_a__ = pickle.load(f)
#     P_s_a = {**P_s_a, **P_s_a__}

# Saving the state-transition graph:
if randomcase:
    with open('P_s_a_random.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(P_s_a, f)
    
else:    
    with open('P_s_a'+experiment_name+'.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(P_s_a, f)   
    print("save as "+'P_s_a'+experiment_name+'.obj')


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
if randomcase:
    with open('state_transition_graph_random.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(G, f)
else:        
    with open('state_transition_graph'+experiment_name+'.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(G, f)
    print("save as "+'state_transition_graph'+experiment_name+'.obj')


# for node in G:
#     if node not in P_s_a.keys():
#         print(node)

# state = (3, 0, 3, 0, 99, 3)
# neighbors = G.neighbors(state)
# for neighbor in neighbors:
#     print([state, neighbor])
#     print(G.edges[state, neighbor])

'''
i = 0        
for state in P_s_a:
    for action in P_s_a[state]:
        for next_state in P_s_a[state][action]:
            if next_state == state_f:
                print("state: {} action:{} prob: {}".format(state, action, P_s_a[state][action][next_state]))
                i += 1
                if i>20:
                    break
# Saving the objects:
# with open('P_s_a.obj', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(P_s_a, f)

# Getting back the objects:
if randomcase:    
    with open('P_s_a_random.obj', 'rb') as f:  # Python 3: open(..., 'rb')
        P_s_a = pickle.load(f)
    
    with open('state_transition_graph_random.obj', 'rb') as f:  # Python 3: open(..., 'rb')
        G = pickle.load(f)
else:
    with open('P_s_a'+experiment_name+'.obj', 'rb') as f:  # Python 3: open(..., 'rb')
        P_s_a = pickle.load(f)

    with open('state_transition_graph'+experiment_name+'.obj', 'rb') as f:  # Python 3: open(..., 'rb')
        G = pickle.load(f)
                       
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
if randomcase:
    with open('policy_random.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(policy, f)  
else:        
    with open('policy'+str(threshold)+experiment_name+'.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(policy, f)        
    
    
# simulate the whole process
    
'''    

                                



                    

                    
            
