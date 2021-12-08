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
    def __init__(self):
        next
    
    def actions(self, state):
        # return available actions in state, represented in a numpy array
        return