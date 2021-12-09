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


class MctsSim:
    def __init__(self, lambda_):
        # initialize a sample search tree
        self.tree = nx.DiGraph()
        # Prevents division by 0 in calculation of UCT
        self.EPSILON = 10e-6
        # UCB coefficient
        self.uct_k = np.sqrt(2)
        # maximum depth
        self.max_depth = 100
        # shrinking factor
        self.gamma = 1
        # lambda
        self.lambda_ = lambda_
        self.simulator = Simulator()
        
        
    # corresponds to the GreedyPolicy in the paper    
    def GreedyPolicy(self, state):
        action = 0
        return action
    # default policy for rollout
    def default_policy(self, state):
        # actions is a list 
        actions = self.simulator.actions(state)
        action = random.sample(actions, 1)[0]
        return action
        
    def roll_out(self, state, depth):
        if depth == self.max_depth:
            return np.array([0, 0])
        action = self.default_policy(state)
        next_state, reward, cost = self.simulator.transtion(state, action)
        return np.array([reward, cost]) + self.gamma*self.roll_out(next_state, depth+1)
    
    # Simulate 
    def simulate(self, state, depth):
        if depth == self.max_depth:
            return np.array([0, 0])
        
        # expansion
        if not (state, depth) in self.tree.nodes:
            # find all action:N(a, s) pairs
            actions = self.simulator.actions(state)
            action_dict = dict(zip(actions, np.zeros(actions.shape)))
            # N is the number of times that this state has been visited
            # Na is a dictionary to track the the number of times of (s, a) 
            # Vc is the value for cost, Qr is Q-value for reward, Qc is for cost
            self.tree.add_node((state, depth), N = 0, Na = action_dict, 
                               Vc=0, Qr=action_dict, Qc=action_dict)
            return self.roll_out(state, depth)
        
        action = self.GreedyPolicy(state)
        next_state, reward, cost = self.simulator.transtion(state, action)
        # RC = np.array([reward, cost])
        RC = np.array([reward, cost]) + self.gamma*self.simulate(next_state, depth+1)
        R = RC[0]
        C = RC[1]
        # backpropagation
        self.tree.nodes[(state, depth)]['N'] += 1
        Vc = self.tree.nodes[(state, depth)]['Vc']
        self.tree.nodes[(state, depth)]['Vc'] = Vc + (C-Vc)/self.tree.nodes[(state, depth)]['N']
        
        for action in self.tree.nodes[(state, depth)]['Na']:
            self.tree.nodes[(state, depth)]['Na'][action] += 1
            Qr = self.tree.nodes[(state, depth)]['Qr'][action]
            Qc = self.tree.nodes[(state, depth)]['Qc'][action]
            self.tree.nodes[(state, depth)]['Qr'][action] = Qr + (R-Qr)/self.tree.nodes[(state, depth)]['Na'][action]
            self.tree.nodes[(state, depth)]['Qc'][action] = Qc + (C-Qc)/self.tree.nodes[(state, depth)]['Na'][action]
        return RC

        
        
def search(state, threshold):
    # initialize lambda
    lambda_ = 10
    # number of iterations
    iters = 100
    for i in range(iters):
        mcts = MctsSim(lambda_)
        mcts.simulate(state, 0)
        action = mcts.GreedyPolicy(state)
        if (mcts.tree.nodes[(state, 0)]['Qc'][action] - threshold > 0):
            # Todo: need to fine tune and check the implementation
            at = 1
        else:
            at = -1
        lambda_ = lambda_ + at * (mcts.tree.nodes[(state, 0)]['Qc'][action] - threshold)
    return mcts
        
        
        
        
        
        
        
        
        
    
    