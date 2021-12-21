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
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
# https://stackoverflow.com/questions/57512155/how-to-draw-a-tree-more-beautifully-in-networkx

class MctsSim:
    def __init__(self, lambda_, c_hat, root):
        # initialize a sample search tree
        self.tree = nx.DiGraph()
        self.state_tracking = {}
        # Prevents division by 0 in calculation of UCT
        self.EPSILON = 10e-6
        # UCB coefficient
        self.uct_k = np.sqrt(2)
        # maximum depth
        self.max_depth_roll_out = 200
        self.max_depth_simulate = 40
        # shrinking factor
        self.gamma = 1
        # lambda
        self.simulator = Simulator()
        actions = self.simulator.actions(root)
        self.node_counter = 0
        # node is indexed by a number in the tree
        self.tree.add_node(self.node_counter,
                           node_label = str(root.state),
                           state = root,
                           depth = 0,
                           N = 1, 
                           Na = dict(zip(actions, np.zeros(len(actions)))), 
                           Vc = 0, 
                           Qr = dict(zip(actions, np.zeros(len(actions)))), 
                           Qc = dict(zip(actions, np.zeros(len(actions)))),
                           )
        self.state_tracking[root.state] = {}
        self.state_tracking[root.state]['N'] = 1
        self.state_tracking[root.state]['Na'] = dict(zip(actions, np.zeros(len(actions))))
        self.state_tracking[root.state]['Vc'] = 0
        self.state_tracking[root.state]['Qr'] = dict(zip(actions, np.zeros(len(actions))))
        self.state_tracking[root.state]['Qc'] = dict(zip(actions, np.zeros(len(actions))))
        self.node_counter += 1
        
        
    def ucb(self, Ns, Nsa):
        #print('Ns is {}, Nsa is {}'.format(Ns, Nsa))
        assert (Ns >= 0) and (Nsa >= 0) 
        
        if (Ns <= 1) or (Nsa == 0):
            print('Initialize UCB')
            return sys.maxsize
        else:
            return np.sqrt(np.log(Ns)/(Nsa+self.EPSILON))
            
        
    # corresponds to the GreedyPolicy in the paper    
    def GreedyPolicy(self, node, k):
        state = self.tree.nodes[node]['state']
        depth = self.tree.nodes[node]['depth']
        # k is for exploration term
        # actions is a list of actions
        actions = self.simulator.actions(state)
        best_action = []
        best_Q = -sys.maxsize
        for action in actions:
            if self.tree.nodes[node]['Qr'][action] > best_Q:
                best_action = action
                best_Q = self.tree.nodes[node]['Qr'][action]
        print('state {}, best action is {}'.format(state.state, best_action))
        return best_action


    # expansion
    def expansion(self, node, parent_node, action, state, depth):
        
        actions = self.simulator.actions(state)
        # N is the number of times that this state has been visited
        # Na is a dictionary to track the the number of times of (s, a) 
        # Vc is the value for cost, Qr is Q-value for reward, Qc is for cost
        #if state.state in self.state_tracking:
        if False:
            self.tree.add_node(node,
                               node_label = str(state.state),
                               state = state,
                               depth = depth,
                               N = self.state_tracking[state.state]['N'], 
                               Na = self.state_tracking[state.state]['Na'], 
                               Vc = self.state_tracking[state.state]['Vc'], 
                               Qr = self.state_tracking[state.state]['Qr'], 
                               Qc = self.state_tracking[state.state]['Qc']
                               )
        else:    
            self.tree.add_node(node,
                               node_label = str(state.state),
                               state = state,
                               depth = depth,
                               N = 1, 
                               Na = dict(zip(actions, np.zeros(len(actions)))), 
                               Vc = 0, 
                               Qr = dict(zip(actions, np.zeros(len(actions)))), 
                               Qc = dict(zip(actions, np.zeros(len(actions))))
                               )
            self.state_tracking[state.state] = {}
            self.state_tracking[state.state]['N'] = 1
            self.state_tracking[state.state]['Na'] = dict(zip(actions, np.zeros(len(actions))))
            self.state_tracking[state.state]['Vc'] = 0
            self.state_tracking[state.state]['Qr'] = dict(zip(actions, np.zeros(len(actions))))
            self.state_tracking[state.state]['Qc'] = dict(zip(actions, np.zeros(len(actions))))
        self.node_counter += 1
        # for plot purpose
        # action_ = ",".join(map(str,action))
        self.tree.add_edge(parent_node, node, action=action)
        # for debug visualization
        #visulize_tree(self.tree)
        return
    
    # default policy for rollout
    def default_policy(self, state):
        # actions is a list 
        actions = self.simulator.actions(state)
        action = random.sample(actions, 1)[0]
        # for debug 
        assert action in actions
        return action
        
    def roll_out(self, state, depth):
        if depth == self.max_depth_roll_out:
            return np.array([0, 0])
        
        # if self.simulator.is_collision(state):
        #     return np.array([0, 0])
        
        # if self.simulator.is_goal(state):
        #     return np.array([0, 0])
        
        action = self.default_policy(state)
        #print('transition from state {} by taking action {} in roll_out'.format(state.state, action))
        next_state, reward, cost, done = self.simulator.transition(state, action)
        
        if done:
            return np.array([0, 0])
        # Todo: will this cause the robot to choose to go into obstacles to 
        # decrease cost
        # if self.simulator.is_collision(next_state):
        #     return np.array([-1, 1])
        
        # if self.simulator.is_goal(next_state):
        #     return np.array([0, 0])
        
        return np.array([reward, cost]) + self.gamma*self.roll_out(next_state, depth+1)
    
    
    
    # Simulate 
    def simulate(self, node, depth):
        state = self.tree.nodes[node]['state']
        #print('we are now in node {} with state {}'.format(node, state.state))
       
        # Todo add terminal state check
        if depth == self.max_depth_simulate:
            #print('reach maximum simulate depth {}'.format(self.max_depth_simulate))
            # this is different from that in the original algorithm
            return np.array([0, 0])
        
        # Todo: the following two may be not necessary
        # if self.simulator.is_collision(state):
        #     return np.array([0, 0])
        
        # if self.simulator.is_goal(state):
        #     return np.array([0, 0])
             
        action = self.GreedyPolicy(node, self.uct_k)
        # done is a flag to show whether state is already a terminal state
        #print('transition from state {} by taking action {} in simulate'.format(state.state, action))
        next_state, reward, cost, done = self.simulator.transition(state, action)
        
        if done:
            return np.array([0, 0])
        # check whether next_state is in the children node of the current node
        # indicator to represent whether next_state shows up in the child node
        # it should be <= 1
        indicator = 0
        RC = []
        for child in self.tree.successors(node):
            child_state = self.tree.nodes[child]['state']
            if next_state.state == child_state.state:
                RC = np.array([reward, cost]) + self.gamma*self.simulate(child, depth+1)
                indicator += 1
        
        assert indicator <= 1, "a state shows up more than once in the child nodes"
        # if next_state doesn't show up in the child node
        if not indicator:
            # expansion
            next_node = self.node_counter
            self.expansion(next_node, node, action, next_state, depth+1)
            RC =  np.array([reward, cost]) + self.gamma*self.roll_out(next_state, depth+1)
        
        # if the previous code on checking whether next_state is in the child node
        # not working, there will an IndexError here            
        R = RC[0]
        # if RC[1] > 1:
        #     RC[1] = 1
        C = RC[1]
        
        #assert C <= 1
        # backpropagation
        self.tree.nodes[node]['N'] += 1
        self.state_tracking[state.state]['N'] +=1
        Vc = self.tree.nodes[node]['Vc']
        self.tree.nodes[node]['Vc'] = Vc + (C-Vc)/self.tree.nodes[node]['N']
        self.state_tracking[state.state]['Vc'] = Vc + (C-Vc)/self.tree.nodes[node]['N']
        
        
        #print('Nsa is {}'.format(self.tree.nodes[node]['Na'][action]))
        self.tree.nodes[node]['Na'][action] += 1
        self.state_tracking[state.state]['Na'][action] += 1
        Qr = self.tree.nodes[node]['Qr'][action]
        Qc = self.tree.nodes[node]['Qc'][action]
        self.tree.nodes[node]['Qr'][action] = Qr + (R-Qr)/self.tree.nodes[node]['Na'][action]
        self.state_tracking[state.state]['Qr'][action] = Qr + (R-Qr)/self.tree.nodes[node]['Na'][action]
        self.tree.nodes[node]['Qc'][action] = Qc + (C-Qc)/self.tree.nodes[node]['Na'][action]
        self.state_tracking[state.state]['Qc'][action] = Qc + (C-Qc)/self.tree.nodes[node]['Na'][action]
        return RC
             
def search(state, c_hat):
    # initialize lambda
    lambda_ = 1
    # Todo: how to specify a range for lambda [0, lambda_max]
    lambda_max = 100
    # Todo: how to specify the number of iterations
    # number of times to update lambda
    iters = 20000
    # Todo: number of monte carlo simulations 
    # number of times to do monte carlo simulation
    # in author's implementation this number is 1
    Nmc = 1
    mcts = MctsSim(lambda_, c_hat, state)
    root_node = 0
    depth = 0
    for i in range(iters):
        # grow monte carlo tree
        for j in range(Nmc):
            # the second root_node is parent_node
            mcts.simulate(root_node, depth)
        action = mcts.GreedyPolicy(root_node, 0)
        if (mcts.tree.nodes[root_node]['Qc'][action] - c_hat < 0):
            # Todo: need to fine tune and check the implementation
            # author's implementation is different from his formula in the paper
            at = -1
            #print('Qc {} is less than c_hat {}'.format(mcts.tree.nodes[root_node]['Qc'][action], c_hat))
        else:
            at = 1
            #print('Qc {} is >= c_hat {}'.format(mcts.tree.nodes[root_node]['Qc'][action], c_hat))

        
        lambda_ += 1/(1+i) * at
        #lambda_ += 1/(1+i/2000) * at
        #print('new lambda is {}'.format(lambda_))
        #lambda_ += 1/(1+i/200) * at * abs((mcts.tree.nodes[0]['Qc'][action] - c_hat))
        # lambda_ += at
        #lambda_ += 1/(1+i) * at * abs((mcts.tree.nodes[0]['Qc'][action] - c_hat))
        if (lambda_ < 0):
            lambda_ = 0
        if (lambda_ > lambda_max):
            lambda_ = lambda_max
        mcts.lambda_ = lambda_
        #print('lambda is {}'.format(mcts.lambda_))
    return mcts
        
def visulize_tree(tree):
    # https://stackoverflow.com/questions/20381460/networkx-how-to-show-node-and-edge-attributes-in-a-graph-drawing
    G = nx.DiGraph()
    for edge in tree.edges:
        # G.add_edge(edge[0], edge[1], label=tree.edges[edge]['action'])
        G.add_edge(edge[0], edge[1])
    # for node in G.nodes:
    #     G.nodes[node]['label'] = tree.nodes[node]['state'].state
    
    pos = graphviz_layout(G, prog="dot", root=0)
    nx.draw(G, pos)
    node_labels = nx.get_node_attributes(tree,'node_label')
    nx.draw_networkx_labels(G, pos, labels = node_labels)
    #edge_labels = nx.get_edge_attributes(tree,'action')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
    plt.show()
   
        
        
if __name__ == "__main__":
    # tree visualization
    # https://stackoverflow.com/questions/48380550/using-networkx-to-output-a-tree-structure
    # https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209
    
    state = State()
    c_hat = 0.9
    mcts = search(state, c_hat)
    visulize_tree(mcts.tree)

    print('mcts nodes: {}'.format(mcts.tree.nodes))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
    
    
