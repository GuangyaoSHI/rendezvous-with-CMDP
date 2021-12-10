# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 21:00:50 2021

@author: gyshi
"""
import numpy as np
import networkx as nx
import random

class State:
    def __init__(self):
        # use (x, y) to represent the state
        self.state = np.array([0,0]) 
    
    # State needs to be hashable so that it can be used as a unique graph
    # node in NetworkX
    def __key(self):
        return self.__str__()
    
    def __eq__(x, y):
        return x.__key() == y.__key()
    
    def __hash__(self):
        return hash(self.__key())
    
    def __str__(self):                                                                                                          
        return np.array2string(self.state)

class Simulator:
    def __init__(self, start=(0, 0), goal=(6, 0)):
        self.start = start
        self.goal = goal
        self.G = nx.grid_2d_graph(7, 7)
        pos = dict(zip(self.G.nodes, self.G.nodes))
        self.G.graph['pos'] = pos
        # obstacle list
        self.obstacles = [(3,0), (3, 1), (3, 2)]
        colors = []
        for node in self.G.nodes:
            if node in self.obstacles:
                colors.append('r')
            else:
                colors.append('#1f78b4')
        self.G.graph['node_color'] = colors
    
    # return available actions in state, represented as a list
    def actions(self, state):
        # if it is a terminal/absorbing state, robot can only stay in this state
        if self.is_terminal(state):
            return [state]
        neighbors = [neighbor for neighbor in self.G.neighbors(state)]
        return [state] + neighbors
    
    # return (next_state, reward, cost)
    def transition(self, state, action):
        actions = self.actions(state)
        if np.random.binomial(1, 0.99):
            next_state = action
        else:
            actions_ = actions.remove(action)
            next_state = random.sample(actions_, 1)[0]
            if next_state == self.goal:
                reward = 0
                cost = 0
            elif self.is_collision(next_state):
                reward = -1
                cost = 1
            else:
                reward = -1
                cost = 0
            
        return (next_state, reward, cost)
    
    def is_terminal(self, state):
        if state == self.goal or state in self.obstacles:
            return True
        return False
    
    def is_collision(self, state):
        if state in self.obstacles:
            return True
        else:
            return False