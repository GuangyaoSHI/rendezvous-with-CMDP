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


# initial state
root_state = State()
# initialize a simulator to check terminal state
simulator = Simulator()
# initial cost threshold
c_hat = 0.9
# traces
paths =[root_state.state]

while not simulator.is_terminal(root_state):
    mcts_policy = search(root_state, c_hat)
    Vc = mcts_policy.tree.nodes[(root_state, 0)]['Vc']
    # Todo is this the right way to check violation?
    if Vc > c_hat:
        print('Constraint is violated')
        print('cost threshold is {}, and Vc from mcts is {}'.format(c_hat, Vc))
        break
    action = mcts_policy.GreedyPolicy(root_state)
    next_state, reward, cost = simulator.transition(root_state, action)
    paths.append(next_state.state)
    # Todo: what if the next_state has not been visited before?
    threshold = mcts_policy.tree.nodes[(next_state, 1)]['Vc']
    state = next_state
    
if simulator.is_collision(state):
    print('It collides with the obstacle')
