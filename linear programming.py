# -*- coding: utf-8 -*-


import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

G_ = nx.grid_2d_graph(3, 3)
obstacles = [(1,0)]
goals = [(2, 0)]
pos = dict(zip(G_.nodes, G_.nodes))
pos[('f', 'f')] = (1, -2)
pos[('l', 'l')] = (2, -3)
G_.add_node(('f', 'f')) # failure state
G_.add_node(('l', 'l'))
G_.add_edge(('f', 'f'), ('l', 'l'))

for obstacle in obstacles:
    G_.add_edge(obstacle, ('f', 'f'))

for goal in goals:
    G_.add_edge(goal, ('l', 'l'))

for node in G_.nodes:
    if node in obstacles:
        G_.nodes[node]['action'] = [('f', 'f')]
        continue
    
    if node in goals:
        G_.nodes[node]['action'] = [('l', 'l')]
        continue
    
    if node == ('l', 'l'):
        G_.nodes[node]['action'] = [('l', 'l')]
        continue
     
    if node == ('f', 'f'):
        G_.nodes[node]['action'] = [('l', 'l')]
        continue
    
    G_.nodes[node]['action'] = list(G_.neighbors(node))


nx.draw(G_, pos=pos, with_labels=True)

def transition_prob(s_a_s):
    # input state_action_nextstate a tuple (a, b, c, d, e, f)
    state = (s_a_s[0], s_a_s[1])
    action = (s_a_s[2], s_a_s[3])
    next_state = (s_a_s[4], s_a_s[5])
    neighbors = list(G_.neighbors(state))
    
    if (action not in neighbors) or (next_state not in neighbors):
        print("trying to transit to an unreachable state")
        return 0
    
    # obstacle
    if state in obstacles:
        print('obstacle state')
        #assert action==('f', 'f') and (next_state==('f', 'f')), "collision! can only transit to failure state"
        if action==('f', 'f') and (next_state==('f', 'f')):
            return 1
        else:
            return 0
    
    # goal
    if state in goals:
        print("goal state")
        #assert action == ('l', 'l') and (next_state==('l', 'l')), "goal! can only transit to loop state"
        if action==('l', 'l') and (next_state==('l', 'l')):
            return 1
        else:
            return 0
    
     
    # if there is only one neighbor in the current state
    # if len(neighbors)==1:
    #     print('only one neighbor')
    #     assert next_state == neighbors[0], "next state is not accessible"
    #     # with prob 1, the robot will transit to that state
    #     return 1
    
    # check whether next_state and action are in the neighbor list
    assert (action in neighbors), "action is not available in the state"
    assert (next_state in neighbors), "next_state is not in neighbor list"

    
    # transition model
    stochasticity = 0.01
    if next_state == action:
        prob = 1 - stochasticity * (len(neighbors) - 1)
        print('probability of {} is {}'.format(s_a_s, prob))
    else:
        prob = stochasticity
        print('probability of {} is {}'.format(s_a_s, prob))
    return prob



def reward(s_a):
    state = (s_a[0], s_a[1])
    action = (s_a[2], s_a[3])
    
    if state in obstacles:
        assert action == ('f', 'f'), "should transit to failure state"
        return 0
    
    if state in goals:
        assert action == ('l', 'l'), "should transit to loop state"
        return 0
    
    if state == ('f', 'f'):
        assert action == ('l', 'l'), "should transit to loop state"
        return 0
    
    if state == ('l', 'l'):
        assert action == ('l', 'l'), "should loop over loop state"
        return 0
    
    return -1


def cost(s_a):
    state = (s_a[0], s_a[1])
    action = (s_a[2], s_a[3])
    if state in obstacles:
        assert action == ('f', 'f'), "should transit to failure state"
        return 0
    
    if state in goals:
        assert action == ('l', 'l'), "should transit to loop state"
        return 0
    
    if state == ('f', 'f'):
        assert action == ('l', 'l'), "should transit to loop state"
        return 0
    
    if state == ('l', 'l'):
        assert action == ('l', 'l'), "should loop over loop state"
        return 0
    
    print('computing cost')
    neighbors = list(G_.neighbors(state))
    print('s_a is {} and neighbors are {}'.format(s_a, neighbors))

    C = 0
    for neighbor in list(neighbors):
        s_a_s = (s_a[0], s_a[1], s_a[2], s_a[3], neighbor[0], neighbor[1])
        prob = transition_prob(s_a_s)
        print("prob of transitting from {} to {}".format((s_a[0], s_a[1]), (s_a[2], s_a[3])))
        # if next state is collision state, it will incur a cost 1
        if neighbor in obstacles:
            C += prob*1
        print("average cost increase since with prob {} neighbor {} is an obstacl".format(prob, neighbor))
    
    print("average cost is {}".format(C))
    
    return C
    
    
        
threshold = 0.1

model = gp.Model('LP_CMDP')

#indics for variables
indices = []
for state in G_.nodes:
    if state in obstacles:
        indices.append((state[0], state[1], 'f', 'f'))
        continue
        
    if state in goals:
        indices.append((state[0], state[1], 'l', 'l'))
        continue
    
    if state == ('f', 'f'):
        indices.append((state[0], state[1], 'l', 'l'))
        continue
    # didn't define state-action variable for loop state
    # Todo
    
    if state == ('l', 'l'):
        indices.append((state[0], state[1], 'l', 'l'))
        continue
    
    for action in G_.neighbors(state):
        indices.append((state[0], state[1], action[0], action[1]))

# add Non-negative continuous variables that lies between 0 and 1        
rho = model.addVars(indices, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='rho')

# add constraints 
# cost constraints
C = 0
for s_a in indices:
    C += rho[s_a] * cost(s_a)
cost_ctrs =  model.addConstr(C <= threshold, name = 'cost_ctrs')

for state in G_.nodes:
    if state == ('l', 'l'):
        continue
    lhs = 0
    rhs = 0
    for action in G_.nodes[state]['action']:
        lhs += rho[(state[0], state[1], action[0], action[1])]
    
    for s_a in indices:
        s_a_s = (s_a[0], s_a[1], s_a[2], s_a[3], state[0], state[1])
        rhs += rho[s_a] * transition_prob(s_a_s)
    
    if state == (0, 0):
        rhs += 1
    
    model.addConstr(lhs == rhs, name = str(state))
    
model.addConstrs((rho[s_a] >=0 for s_a in indices), name='non-negative-ctrs')
        

obj = 0
for s_a in indices:
    obj += rho[s_a] * reward(s_a)
    
model.setObjective(obj, GRB.MAXIMIZE)
model.optimize()
    

for s_a in indices:
    state = (s_a[0], s_a[1])
    actions = G_.nodes[state]['action']
    deno = rho[s_a].x
    num = 0
    for action in actions:
        num += rho[(s_a[0], s_a[1], action[0], action[1])].x
    
    print("Optimal Prob of choosing action {} at state {} is {}".format((s_a[2], s_a[3]), state, deno/num))


    
