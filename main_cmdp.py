# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:40:31 2021

@author: gyshi
"""

import random
import numpy as np
import networkx as nx
import copy
import sys
from simulator import State
from simulator import Simulator

from mcts import MctsSim
from mcts import search

from scipy.io import savemat


# initial state
root_state = State()
root_state.state = (0, 1)
# initialize a simulator to check terminal state
simulator = Simulator()
# initial cost threshold
c_hat = 0.2
# traces
paths =[root_state.state]
# policies along the path
policies = []
# current state
curr_state = root_state
while not simulator.is_terminal(curr_state):
    mcts_policy = search(curr_state, c_hat)
    Vc = mcts_policy.tree.nodes[0]['Vc']
    action = mcts_policy.GreedyPolicy(0, 0)
    print('transition from state {} by taking action {} in simulate'.format(curr_state.state, action))
    print('Display policy')
    MinCostAction = mcts_policy.latest_policy['minCostAction']
    MaxCostAction = mcts_policy.latest_policy['maxCostAction']
    prob_MinCost = mcts_policy.latest_policy['prob_minCost']
    print('MinCostAction:{}, MaxCostAction:{}, prob_MinCost:{}'.format(MinCostAction, MaxCostAction, prob_MinCost))
    Qc = mcts_policy.tree.nodes[0]['Qc'][action]
    print('Qc: {}'.format(mcts_policy.tree.nodes[0]['Qc']))
    print('Qr: {}'.format(mcts_policy.tree.nodes[0]['Qr']))
    # Todo is this the right way to check violation?
    if Qc > c_hat:
        print('Constraint is violated')
        print('cost threshold is {}, and Qc from mcts is {}'.format(c_hat, Qc))
        break
    else:
        print('cost threshold is {}, and Qc from mcts is {}'.format(c_hat, Qc))
   
    print('transition from state {} by taking action {} in simulate'.format(curr_state.state, action))
    next_state, reward, cost, done = simulator.transition(curr_state, action)
    paths.append(next_state.state)
    # Todo: what if the next_state has not been visited before?
    # c_hat = mcts_policy.update_admissble_cost(action, next_state)
    # print('updated c_hat is {}'.format(c_hat))
    curr_state = next_state
    break
    
if simulator.is_collision(curr_state):
    print('It collides with the obstacle')
    
print(paths)
savemat("paths.mat", {"paths":paths})



