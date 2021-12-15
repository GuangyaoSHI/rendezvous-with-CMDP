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
# initialize a simulator to check terminal state
simulator = Simulator()
# initial cost threshold
c_hat = 0.2
# traces
paths =[root_state.state]

# current state
curr_state = root_state
while not simulator.is_terminal(curr_state):
    mcts_policy = search(curr_state, c_hat)
    Vc = mcts_policy.tree.nodes[0]['Vc']
    # Todo is this the right way to check violation?
    if Vc > c_hat:
        print('Constraint is violated')
        print('cost threshold is {}, and Vc from mcts is {}'.format(c_hat, Vc))
        #break
    action = mcts_policy.GreedyPolicy(0, 0)
    next_state, reward, cost, done = simulator.transition(root_state, action)
    paths.append(next_state.state)
    # Todo: what if the next_state has not been visited before?
    # c_hat = mcts_policy.update_admissble_cost(action, next_state)
    # print('updated c_hat is {}'.format(c_hat))
    curr_state = next_state
    
if simulator.is_collision(curr_state):
    print('It collides with the obstacle')
    
print(paths)
savemat("paths.mat", {"paths":paths})
