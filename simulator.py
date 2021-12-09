# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 21:00:50 2021

@author: gyshi
"""
import numpy as np
import networkx as nx


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
    def __init__(self, start, goal):
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
        # if the state is a terminal/absorbing state, it can only stay in this state
        if self.is_terminal(state):
            return [state]
        neighbors = [neighbor for neighbor in self.G.neighbors(state)]
        return [state] + neighbors
    
    def transition(self, state, action):
        
        # return (next_state, reward, cost)
        return (0, 0, 0)
    
    def is_terminal(self, state):
        return True