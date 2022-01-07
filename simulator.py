# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 21:00:50 2021

@author: gyshi
"""
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

class State:
    def __init__(self, state = (0, 0)):
        # use (x, y) to represent the state
        self.state = state
    
    # State needs to be hashable so that it can be used as a unique graph
    # node in NetworkX
    def __key(self):
        return self.__str__()
    
    def __eq__(x, y):
        return x.__key() == y.__key()
    
    def __hash__(self):
        return hash(self.__key())
    
    def __str__(self): 
        # we need to return a string here                                                                                                         
        # return np.array2string(np.array(self.state))
        return str(self.state)

class Simulator:
    def __init__(self, start=(0, 0), goal=(4, 0)):
        self.start = start
        self.goal = goal
        #self.G = nx.DiGraph()
        self.G = nx.grid_2d_graph(5, 5)
        #self.G.add_nodes_from(self.G_.nodes)
        pos = dict(zip(self.G.nodes, self.G.nodes))
        self.G.graph['pos'] = pos
        # obstacle list
        self.obstacles = [(2, 0)]
        colors = []
        for node in self.G.nodes:
            if node in self.obstacles:
                colors.append('r')
            else:
                colors.append('#1f78b4')
        self.G.graph['node_color'] = colors
        #self.generate_map()
        nx.draw(self.G, pos, with_labels=True)
        plt.show()
        
    # return available actions in state, represented as a list
    def actions(self, state):
        # if it is a terminal/absorbing state, robot can only stay in this state
        if self.is_goal(state):
            #print('in the goal state, no actions available')
            return [state.state]
        
        if self.is_collision(state):
            #print('collision state, no actions available')
            return [state.state]
        
        neighbors = [neighbor for neighbor in self.G.neighbors(state.state)]
        return neighbors
    
    # return (next_state, reward, cost)
    def transition(self, state, action):
        next_state = State()
        reward = -1
        cost = 0
        actions = self.actions(state)
        if action not in actions:
            print('action is {} while actions are {}'.format(action, actions))
        assert action in actions
        done = False
        # check whether robot is already in goal or has collided with obstacles
        if state.state == self.goal:
            #print('already reach goal state, no transition needed')
            reward = 0
            cost = 0
            done = True
            return (state, reward, cost, done)
        
        if self.is_collision(state):
            #print('already collide with obstacles, no transition needed')
            reward = 0
            cost = 0
            done = True
            return (state, reward, cost, done)
        
        # Todo: a better motion model
        if np.random.binomial(1, 0.95):
            next_state.state = action
        else:
            if len(actions) == 1:
                #print('state is {} and actions are {}'.format(state.state, actions))
                next_state.state = random.sample(actions, 1)[0]
            else:
            #print('avaible actions are {}'.format(actions))
            #print('remove action {} and sample the rest'.format(action))
            #print('state is {}; action is {}; actions are {}'.format(state.state, action, actions))
                actions.remove(action)
                assert len(actions) >= 1
                next_state.state = random.sample(actions, 1)[0]
                #print('take action {} in state {} but transit to {}'.format(action, state.state, next_state.state))
    
        if next_state.state == self.goal:
            reward = -1
            cost = 0
        elif self.is_collision(next_state):
            if nx.has_path(self.G, source=state.state, target=self.goal):
                #reward = -nx.shortest_path_length(self.G, source=state.state, target=self.goal)
                #print('next state {} is collision state'.format(next_state.state))
                #print('shortest path length is {} from {} to {}'.format(reward, state.state, self.goal))
                reward = -1
                cost = 1
            else:
                #Todo: if there is no path to the goal, how to define reward 
                print('no path to the goal')
                reward = -1
                cost = 1
        else:
            reward = -1
            cost = 0
            
        return (next_state, reward, cost, done)
    
    def is_terminal(self, state):
        if state.state == self.goal or (state.state in self.obstacles):
            return True
        return False
    
    def is_collision(self, state):
        if state.state in self.obstacles:
            return True
        else:
            return False
        
    def is_goal(self, state):
        if state.state == self.goal:
            return True
        else:
            return False
    
    def generate_map(self):   
        for i in range(0, 3):
            # add vertical edges to the graph G
            for j in range(0, 6):
                self.G.add_edge((i, j), (i, j+1))
            # add horizontal edges
            for j in range(0, 7):
                self.G.add_edge((i, j), (i+1, j))
        
        for i in range(4, 7):
            for j in range(0, 6):
                self.G.add_edge((i, j+1), (i, j))
            
            for j in range(0, 7):
                if i<6:
                    self.G.add_edge((i, j), (i+1, j))
        
        for i in range(3, 6):
            self.G.add_edge((3, i), (3, i+1))
        self.G.add_edge((3, 3), (3, 2))
        
        for i in range(3, 7):
            self.G.add_edge((3, i), (4, i))
        
        for i in range(0, 3):
            self.G.add_edge((4, i), (3, i))
        
        
                      
        
if __name__ == "__main__":
    print('test simulator')
    root = State()
    simulator = Simulator()
    print('state is {}'.format(root.state))
    actions = simulator.actions(root)
    print('available actions: {}'.format(actions))
    action = actions[-1]
    next_state, reward, cost, done = simulator.transition(root, action)
    print('take action {} and transit to {}'.format(action, next_state.state))
    print('reward is {}'.format(reward))
    print('cost is {}'.format(cost))
    
    
    print('test collision state')
    root = State()
    root.state = (1, 0)
    simulator = Simulator()
    print('state is {}'.format(root.state))
    actions = simulator.actions(root)
    print('available actions: {}'.format(actions))
    action = actions[-1]
    next_state, reward, cost, done = simulator.transition(root, action)
    print('take action {} and transit to {}'.format(action, next_state.state))
    print('reward is {}'.format(reward))
    print('cost is {}'.format(cost))
    
    
    print('test goal state')
    root = State()
    root.state = (3, 0)
    simulator = Simulator()
    print('state is {}'.format(root.state))
    actions = simulator.actions(root)
    print('available actions: {}'.format(actions))
    action = actions[-1]
    next_state, reward, cost, done = simulator.transition(root, action)
    print('take action {} and transit to {}'.format(action, next_state.state))
    print('reward is {}'.format(reward))
    print('cost is {}'.format(cost))
    
    print('test shortest path')
    source = (0, 0)
    target = (2, 0)
    print('source is {} target is {}'.format(source, target))
    if nx.has_path(simulator.G, source=source, target=target):
        print('path is {}'.format(nx.shortest_path(simulator.G, source=source, target=target)))  
        print('path length is {}'.format(nx.shortest_path_length(simulator.G, source=source, target=target)))
    else:
        print('shortest path doesnot exist')
    