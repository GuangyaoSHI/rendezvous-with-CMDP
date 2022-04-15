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
P_s_a['sg']['l'] = {}
P_s_a['sg']['l']['sl'] = 1
P_s_a['sf'] = {}
P_s_a['sf']['l'] = {}
P_s_a['sf']['l']['sl'] = 1
P_s_a['sl'] = {}
P_s_a['sl']['l'] = {}
P_s_a['sl']['l']['sl'] = 1

# create a transition graph
G = nx.DiGraph()
for state in P_s_a:
    G.add_node(state, action=list(P_s_a[state].keys()))
    for action in P_s_a[state]:
        G.add_edge(state, (state, )+(action,))
        for next_state in P_s_a[state][action]:
            assert next_state in P_s_a
            G.add_edge((state,)+(action,), next_state, action=action, prob=P_s_a[state][action][next_state])

#nx.draw(G, with_labels=True)

                     
# create transition function 
def transition_prob(s_a_s):
    state = s_a_s[0]
    action = s_a_s[1]
    next_state = s_a_s[2]
    
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
    if state == goal:
        print("reach the goal and transition is {}".format(s_a_s))
        assert action == 'l' and (next_state == state_l)
        return 1
    
    # loop state
    if state == state_l:
        assert action == 'l' and (next_state == state_l)
        return 1
    
    return P_s_a[state][action][next_state] 


def reward(s_a):
    state = s_a[0]
    action = s_a[1]
    
    if state == state_f:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if state == state_l:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if state == goal:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if action == 'a1':
        return 1
    
    if action == 'a2':
        return 5
    

def cost(s_a):
    state = s_a[0]
    action = s_a[1]
    
    if state == state_f:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if state == state_l:
        assert action == 'l', "should transit to loop state"
        return 0
    
    if state == goal:
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
        indices.append((state,) + (action,))


# add Non-negative continuous variables that lies between 0 and 1        
rho = model.addVars(indices, lb=0,  vtype=GRB.CONTINUOUS,  name='rho')
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
        #print(rho[(state,)+(action,)].x)
        lhs += rho[(state,)+(action,)]
        
    # for s_a in indices:
    #     s_a_s = s_a + state
    #     rhs += rho[s_a] * transition_prob(s_a_s)

    #s_a_nodes = []
    for s_a in G.predecessors(state):
        #assert len(s_a) == 6
        #if s_a not in s_a_nodes:
        #s_a_nodes.append(s_a)
        s_a_s = s_a + (state,)
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
    
model.setObjective(obj, GRB.MINIMIZE)
model.Params.FeasibilityTol = 1e-9    
print("--- %s seconds ---" % (time.time() - start_time))

model.optimize()

print("objective value is {}".format(obj.getValue()))        
# Saving the objects:
       



# Create a new model
m = gp.Model('toy_CMDP')

# Create variables
x1 = m.addVar(name="x1", lb=0)
x2 = m.addVar(name="x2", lb=0)
x3 = m.addVar(name="x3", lb=0)
x4 = m.addVar(name="x4", lb=0)
x5 = m.addVar(name="x5", lb=0)

# Set objective function
m.setObjective(x1+5*x2 , GRB.MINIMIZE)
m.Params.FeasibilityTol = 1e-9    

#Add constraints
m.addConstr(0.2*x1 + 0.4*x2 <= 0.3, "c1")
m.addConstr(x1 + x2 == 1, "c2")
m.addConstr(x3 == 0.8*x1 + 0.6*x2, "c3")
m.addConstr(x4 == 0.2*x1 + 0.4*x2, "c4")
m.addConstr(x1 >= 0, "c5")
m.addConstr(x2 >= 0, "c6")
m.addConstr(x3 >= 0, "c7")
m.addConstr(x4 >= 0, "c8")
m.addConstr(x5 >= 0, "c9")


# Optimize model
m.optimize()

#Print values for decision variables
for v in m.getVars():
    print(v.varName, v.x)

#Print maximized profit value
print('Maximized profit:',  m.objVal)




y = {}
lambda_1 = 0
lambda_sa = {}
nu = {}
C_sa_pri = {}
C_sa_sec = {} 
print("start to initialize")
for state in P_s_a:
    if state != state_l:
        nu[state] = 10
    for action in P_s_a[state]:
        y[(state,)+(action,)] = 0
        lambda_sa[(state,)+(action,)]  = 0
        C_sa_pri[(state,)+(action,)] = reward((state,)+(action,))
        assert C_sa_pri[(state,)+(action,)] >= 0
        C_sa_sec[(state,)+(action,)] = cost((state,)+(action,))


# update step size
ay = 0.2
a_lambda = 0.001
a_nu = 0.001


y_current_step = copy.deepcopy(y)
lambda_1_current_step = lambda_1
lambda_sa_current_step = copy.deepcopy(lambda_sa)
nu_current_step = copy.deepcopy(nu)
C_current_step = np.dot(np.array(list(y_current_step.values())), np.array(list(C_sa_pri.values())))


y_last_step = copy.deepcopy(y)
C_last_step = np.dot(np.array(list(y_last_step.values())), np.array(list(C_sa_pri.values())))-1
lambda_1_last_step = lambda_1
lambda_sa_last_step = copy.deepcopy(lambda_sa)
nu_last_step = copy.deepcopy(nu)
#Lagrangian_last_step = Lagrangian_current_step - 0.01



obj_traces = [C_current_step]
y_traces = [y_current_step]
y_mean = []
#Lagrangian_traces = [Lagrangian_current_step]
print("start to do iterations")
for j in range(1, 8000):
    #print("iteration {}".format(j))
    ay = ay/1.0
    a_lambda = ay
    a_nu = ay
    #Lagrangian_last_step = Lagrangian_current_step
    for s_a in y:
        # dy
        state = s_a[0]
        last_term_dy = 0
        for state_next in P_s_a[state][s_a[-1]]:
            if state_next != state_l:
                last_term_dy += P_s_a[state][s_a[-1]][state_next]*nu_last_step[state_next]
        
        if state != state_l:
            dy = C_sa_pri[s_a] + lambda_1_last_step*C_sa_sec[s_a] - \
                0 + nu_last_step[state] - last_term_dy
        else:
            dy = C_sa_pri[s_a] + lambda_1_last_step*C_sa_sec[s_a] - 0 - last_term_dy
        # update y
        y_current_step[s_a] = y_last_step[s_a] - ay*dy
        y_current_step[s_a] = max(y_current_step[s_a], 0)
        # d lambda_sa
        #d_lambda_sa = -y_last_step[s_a]
        # update lambda_sa
        #lambda_sa_current_step[s_a] = 0 + a_lambda*d_lambda_sa
        #if lambda_sa_current_step[s_a] < 0:
        #    lambda_sa_current_step[s_a] = 0
        
    y_traces.append(y_current_step)
    # d lambda1
    d_lambda1 = np.dot(np.array(list(y_last_step.values())), np.array(list(C_sa_sec.values()))) - threshold
    # update lambda1
    lambda_1_current_step = lambda_1_last_step + a_lambda*d_lambda1
    if lambda_1_current_step < 0:
        lambda_1_current_step = 0
        
    for state in P_s_a:
        # d nu
        if state != state_l:
            # first term
            d_nu_first_term = 0
            for action in P_s_a[state]:
                d_nu_first_term += y_last_step[(state,)+(action,)]
            
            # indicator
            initial_state = 0
            if state == state_init:
                initial_state = 1
            
            # last term 
            d_nu_last_term = 0
            for s_a in G.predecessors(state):
                #assert len(s_a) == 6
                #if s_a not in s_a_nodes:
                #s_a_nodes.append(s_a)
                s_a_s = s_a + (state,)
                d_nu_last_term += y_last_step[s_a] * transition_prob(s_a_s)
                
            d_nu = d_nu_first_term - initial_state - d_nu_last_term
            
            # update nu
            nu_current_step[state] = nu_last_step[state] + a_nu*d_nu
    
    C_current_step = np.dot(np.array(list(y_current_step.values())), np.array(list(C_sa_pri.values())))
    obj_traces.append(C_current_step)
    
    # update last step information
    #y_last_step = copy.deepcopy(y_current_step)
    y_last_step = dict(zip(y_last_step.keys(), y_current_step.values()))
    #lambda_sa_last_step = copy.deepcopy(lambda_sa_current_step)
    #lambda_sa_last_step = dict(zip(lambda_sa_last_step.keys(), lambda_sa_current_step.values()))
    
    lambda_1_last_step = lambda_1_current_step
    nu_last_step = dict(zip(nu_last_step.keys(), nu_current_step.values()))
    #nu_last_step = copy.deepcopy(nu_current_step)

print("Iterations done!")
# post processing
obj_mean = [obj_traces[0]]
for i in range(1, len(obj_traces)):
    obj_mean.append(i*obj_mean[i-1]/(i+1) + obj_traces[i]/(i+1))
    
print("optimal value is {}".format(obj_mean[-1]))       
plt.plot(obj_mean)
print("objective value from Gurobi is {}".format(obj.getValue()))        

    
