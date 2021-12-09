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
state = State()
# initialize a simulator to check terminal state
simulator = Simulator()
# initial cost threshold
threshold = 0.8

while not simulator.is_terminal(state):
    mcts_policy = search(state, threshold)
    Vc = mcts_policy.tree.nodes[(state, 0)]['Vc']
    # Todo is this the right way to check violation?
    if Vc > threshold:
        print('Constraint is violated')
        print('cost threshold is {}, and Vc from mcts is {}'.format(threshold, Vc))
        break
    action = mcts_policy.GreedyPolicy(state)
    next_state, reward, cost = simulator.transition(state, action)
    # Todo: what if the next_state has not been visited before?
    threshold = mcts_policy.tree.nodes[(next_state, 1)]['Vc']
    state = next_state
    