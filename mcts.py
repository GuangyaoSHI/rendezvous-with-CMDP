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
        
        # Prevents division by 0 in calculation of UCT
        self.EPSILON = 10e-6
        # UCB coefficient
        self.uct_k = 0.5*np.sqrt(2)
        # maximum depth
        self.max_depth = 20
        # shrinking factor
        self.gamma = 1
        # lambda
        self.lambda_ = lambda_
        self.simulator = Simulator()
        self.c_hat = c_hat
        actions = self.simulator.actions(root)
        self.node_counter = 0
        # node is indexed by a number in the tree
        self.tree.add_node(self.node_counter,
                           node_label = str(root.state),
                           state = root,
                           depth = 0,
                           N = 1, 
                           Na = dict(zip(actions, np.zeros(len(actions)))), 
                           Vc=0, 
                           Qr=dict(zip(actions, np.zeros(len(actions)))), 
                           Qc=dict(zip(actions, np.zeros(len(actions)))),
                           )
        self.node_counter += 1
        
    def ucb(self, Ns, Nsa):
        #print('Ns is {}, Nsa is {}'.format(Ns, Nsa))
        assert (Ns >= 0) and (Nsa >= 0) 
        if (Ns == 0) or (Nsa == 0):
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
        #print('state is {} actions are {}'.format(state.state, actions))
        # find action star, which is a list
        action_star = []
        Qplus_star = -sys.maxsize
        for action in actions:
            #print('node is {}'.format(node))
            Qr = self.tree.nodes[node]['Qr'][action]
            Qc = self.tree.nodes[node]['Qc'][action]
            Ns = self.tree.nodes[node]['N']
            Nsa = self.tree.nodes[node]['Na'][action]
            #print('Qr is {}, Qc is {}, Ns is {}, Nsa is {}'.format(Qr, Qc, Ns, Nsa))
            # a small number is added to numerator for numerical stability
            # Todo: should I follow author's implementation on UCB?
            #print('lambda is {}, k is {}'.format(self.lambda_, k))
            #print('ucb is {}'.format(self.ucb(Ns, Nsa)))
            Qplus =  Qr - self.lambda_*Qc + k * self.ucb(Ns, Nsa)
            #print('Qplus is {}'.format(Qplus))
            if  Qplus >= Qplus_star:
                # Todo: does this part matter?
                # strictly follow author's implementation
                # it is actually a little bit probalematic 
                # only one element in the action_star set
                if Qplus > Qplus_star:
                    #print('action_star is cleared')
                    action_star = []
                Qplus_star = Qplus
                action_star.append(action)
                #print('action_star is {}'.format(action_star))
        assert len(action_star) >= 1
        # sample from the action star
        best_action = random.sample(action_star, 1)[0]
        # find minCost minCostAction
        # maxCost maxCostAction
        # this part also strictly follows the author's implementation
        biasconstant = np.exp(depth) * 0.1
        minCost = sys.maxsize
        maxCost = -sys.maxsize
        # initialize the min/max cost action
        minCostAction = best_action
        maxCostAction = best_action
        
        best_action_N = self.tree.nodes[node]['Na'][best_action]
        best_action_Qr = self.tree.nodes[node]['Qr'][best_action]
        best_action_Qc = self.tree.nodes[node]['Qc'][best_action]
        best_action_Q = best_action_Qr - self.lambda_ * best_action_Qc
        best_action_bias = biasconstant * np.log(best_action_N + 1)/(best_action_N + 1)
        
        for action in actions:
            Qr = self.tree.nodes[node]['Qr'][action]
            Qc = self.tree.nodes[node]['Qc'][action]
            Nsa = self.tree.nodes[node]['Na'][action]
            Q = Qr - self.lambda_*Qc
            action_bias = biasconstant * (np.log(Nsa + 1)/(Nsa + 1))
            # Todo: test deterministic case
            threshold = best_action_bias + action_bias
            # bestQ is Q above
            # author's implementation is a little bit problematic
            if (abs(Q-best_action_Q) <= threshold):
                if Qc < minCost:
                    minCost = Qc
                    minCostAction = action
                if Qc > maxCost:
                    maxCost = Qc
                    maxCostAction = action
        
  
        # compute policy as a probability distrbution over minCostAction
        # and maxCostAction
        assert minCost <= maxCost
        prob_minCost = 0
        #prob_maxCost = 1 - prob_minCost
        if maxCost <= self.c_hat:
            prob_minCost = 0
            #prob_maxCost = 1 - prob_minCost
        elif (minCost >= self.c_hat):
            prob_minCost = 1
            #prob_maxCost = 1 - prob_minCost
        else:
            prob_minCost = (maxCost - self.c_hat) / (maxCost - minCost)
            #prob_maxCost = 1 - prob_minCost
        
        assert 0 <= prob_minCost <= 1
        if random.random() <= prob_minCost:
            return minCostAction
            # for debug
            assert minCostAction in actions
        else:
            return maxCostAction
            # for debug
            assert maxCostAction in actions
        
    
    # expansion
    def expansion(self, node, parent_node, action, state, depth):
        
        actions = self.simulator.actions(state)
        # N is the number of times that this state has been visited
        # Na is a dictionary to track the the number of times of (s, a) 
        # Vc is the value for cost, Qr is Q-value for reward, Qc is for cost
        self.tree.add_node(node,
                           node_label = str(state.state),
                           state = state,
                           depth = depth,
                           N = 1, 
                           Na = dict(zip(actions, np.zeros(len(actions)))), 
                           Vc=0, 
                           Qr=dict(zip(actions, np.zeros(len(actions)))), 
                           Qc=dict(zip(actions, np.zeros(len(actions)))),
                           )
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
        if depth == self.max_depth:
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
        if depth == self.max_depth:
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
        C = RC[1]
        #assert C <= 1
        # backpropagation
        self.tree.nodes[node]['N'] += 1
        Vc = self.tree.nodes[node]['Vc']
        self.tree.nodes[node]['Vc'] = Vc + (C-Vc)/self.tree.nodes[node]['N']
        
        #print('Nsa is {}'.format(self.tree.nodes[node]['Na'][action]))
        self.tree.nodes[node]['Na'][action] += 1
        Qr = self.tree.nodes[node]['Qr'][action]
        Qc = self.tree.nodes[node]['Qc'][action]
        self.tree.nodes[node]['Qr'][action] = Qr + (R-Qr)/self.tree.nodes[node]['Na'][action]
        self.tree.nodes[node]['Qc'][action] = Qc + (C-Qc)/self.tree.nodes[node]['Na'][action]
        return RC
    
    def update_admissble_cost(self, action, state):
        root = self.tree.nodes[0]['state']
        assert action in self.simulator.actions(root)
        c_hat = -sys.maxsize
        for node in self.tree.successors(0):
            #print('action is {} and next_state is {}'.format(action, state.state))
            #print('edge between node {} and root is {}'.format(node, self.tree.edges[0, node]))
            #print('node state is {}'.format(self.tree.nodes[node]['state'].state))
            #print('action is {}'.format(self.tree.edges[0, node]['action']))
            if (self.tree.nodes[node]['state'].state == state.state) and (self.tree.edges[0, node]['action'] == action):
                return self.tree.nodes[node]['Vc']
            
            if self.tree.nodes[node]['Vc'] > c_hat:
                c_hat = self.tree.nodes[node]['Vc']
            
        print('state {} has not been visited before and return max Vc of children'.format(state.state))
        return c_hat
                   
            
def search(state, c_hat):
    # initialize lambda
    lambda_ = 1
    # Todo: how to specify a range for lambda [0, lambda_max]
    lambda_max = 1000
    # Todo: how to specify the number of iterations
    # number of times to update lambda
    iters = 1000
    # Todo: number of monte carlo simulations 
    # number of times to do monte carlo simulation
    # in author's implementation this number is 1
    Nmc = 1
    mcts = MctsSim(lambda_, c_hat, state)
    root_node = 0
    depth = 0
    for i in range(iters):
        # grow monte carlo tree
        for i in range(Nmc):
            # the second root_node is parent_node
            mcts.simulate(root_node, depth)
        action = mcts.GreedyPolicy(root_node, 0)
        if (mcts.tree.nodes[root_node]['Qc'][action] - c_hat < 0):
            # Todo: need to fine tune and check the implementation
            # author's implementation is different from his formula in the paper
            at = -1
        else:
            at = 1
        # lambda_ += 1/(1+i) * at * (mcts.tree.nodes[(state, 0)]['Qc'][action] - c_hat)
        lambda_ += 1/(1+i) * at
        # lambda_ += 1/(1+i) * at * abs((mcts.tree.nodes[0]['Qc'][action] - c_hat))
        # lambda_ += at
        # lambda_ += 1/(1+i) * at * abs((mcts.tree.nodes[0]['Qc'][action] - c_hat))
        if (lambda_ < 0):
            lambda_ = 0
        if (lambda_ > lambda_max):
            lambda_ = lambda_max
        mcts.lambda_ = lambda_
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
    
    