# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:59:35 2021

@author: gyshi
"""

import random
import numpy as np
import networkx as nx
import copy
import sys
from simulator import State
from simulator import Simulator

class DefaultPolicy:
    def __init__(self):
        next

class MctsPolicy:
    def __init__(self, root_state):
        # initialize a sample search tree
        self.tree = nx.DiGraph()
        # Prevents division by 0 in calculation of UCT
        self.EPSILON = 10e-6
        # UCB coefficient
        self.uct_c = np.sqrt(2)
        self.simulator = Simulator()
        # N is the number of times that this state has been visited
        # Na is a dictionary to track the the number of times of (s, a) 
        # Vc is the value for cost, Qr is Q-value for reward, Qc is for cost
        
        actions = self.simulator.actions(root_state)
        action_dict = dict(zip(actions, np.zeros(actions.shape)))
        
        self.tree.add_node((root_state, 0), N = 0, Na = action_dict, 
                           Vc=0, Qr=0, Qc=0)
        
        
        
        
        
        
    
    