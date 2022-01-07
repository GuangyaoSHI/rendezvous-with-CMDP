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
        self.uct_k = 20*np.sqrt(2)
        # maximum depth
        self.max_depth_roll_out = 100
        self.max_depth_simulate = 40
        # shrinking factor
        self.gamma = 1
        # lambda
        self.lambda_ = lambda_
        # keep track of lambda 
        self.lambda_history = [self.lambda_]
    
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
        self.Qr_history = [copy.deepcopy(self.tree.nodes[0]['Qr'])]
        self.Qc_history = [copy.deepcopy(self.tree.nodes[0]['Qc'])]
    
    def ucb(self, Ns, Nsa):
        #print('Ns is {}, Nsa is {}'.format(Ns, Nsa))
        assert (Ns >= 0) and (Nsa >= 0) 
        if (Ns <= 1) or (Nsa == 0):
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
        #print('greedy policy for state {} of node {}'.format(state.state, node))
        for action in actions:
            #print('candidate action {}'.format(action))
            Qr = self.tree.nodes[node]['Qr'][action]
            Qc = self.tree.nodes[node]['Qc'][action]
            Ns = self.tree.nodes[node]['N']
            Nsa = self.tree.nodes[node]['Na'][action]
            #print('Qr is {}, Qc is {}, Ns is {}, Nsa is {}'.format(Qr, Qc, Ns, Nsa))
            # a small number is added to numerator for numerical stability
            # Todo: should I follow author's implementation on UCB?
            #print('lambda is {}, k is {}'.format(self.lambda_, k))
            #print('ucb is {}'.format(k*self.ucb(Ns, Nsa)))
            Qplus =  Qr - self.lambda_*Qc + k * self.ucb(Ns, Nsa)
            #print('Qplus is {} for action {}'.format(Qplus, action))
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
            
            #print('best_action_bias is {}, action_bias is {}'.format(best_action_bias, action_bias))
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
    
    # an alternative to greedy policy for root
    def root_policy(self):
        state = self.tree.nodes[0]['state']
        actions = self.simulator.actions(state)
        print('available actions {}'.format(actions))
        # find actions whose expected cost is less than threshold
        actions_c = {}
        for action in actions:
            if self.tree.nodes[0]['Qc'][action] <= self.c_hat:             
                actions_c[self.tree.nodes[0]['Qr'][action]] = action
        print('cost-constrained Qr-actions {}'.format(actions_c))
        if not actions_c:
            return 'empty'
        else:
            max_Qr = max(actions_c, key=actions_c.get)
            return actions_c[max_Qr]
                

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
        
        # done is used to decide whether the current state is the terminal state
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
        
        # Todo: the following two check the terminal 
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
        if node == 0:
            self.Qc_history.append(copy.deepcopy(self.tree.nodes[node]['Qc']))
            self.Qr_history.append(copy.deepcopy(self.tree.nodes[node]['Qr']))
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
    lambda_max = 100
    # Todo: how to specify the number of iterations
    # number of times to update lambda
    iters = 20000
    # Todo: number of monte carlo simulations 
    # number of t   imes to do monte carlo simulation
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
      
        lambda_ += 1/(1+i/15) * (mcts.tree.nodes[0]['Qc'][action] - c_hat)
        #lambda_ += 1/(1+i/40) * at
        #print('new lambda is {}'.format(lambda_))
        #lambda_ += 1/(1+i/200) * at * abs((mcts.tree.nodes[0]['Qc'][action] - c_hat))
        # lambda_ += at
        #lambda_ += 1/(1+i/1000) * at * abs((mcts.tree.nodes[0]['Qc'][action] - c_hat))
        if (lambda_ < 0):
            lambda_ = 0
        #if (lambda_ > lambda_max):
            #lambda_ = lambda_max
        mcts.lambda_ = lambda_
        mcts.lambda_history.append(mcts.lambda_)
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
    nx.draw(G, pos, node_size=10, alpha=0.8)
    node_labels = nx.get_node_attributes(tree,'node_label')
    nx.draw_networkx_labels(G, pos, labels = node_labels, font_size=5)
    edge_labels = nx.get_edge_attributes(tree,'action')
    for node in tree.nodes:
        for parent_node in tree.predecessors(node):
            action = tree.edges[parent_node, node]['action']
            edge_labels[(parent_node, node)] = tree.nodes[parent_node]['Na'][action]
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_size=4)
    plt.savefig('tree.pdf')
    plt.show()
   
def visulize_tree_Qr(tree, lambda_):
    # https://stackoverflow.com/questions/20381460/networkx-how-to-show-node-and-edge-attributes-in-a-graph-drawing
    G = nx.DiGraph()
    for edge in tree.edges:
        # G.add_edge(edge[0], edge[1], label=tree.edges[edge]['action'])
        G.add_edge(edge[0], edge[1])
    # for node in G.nodes:
    #     G.nodes[node]['label'] = tree.nodes[node]['state'].state
    
    pos = graphviz_layout(G, prog="dot", root=0)
    title = 'edge represents Qr(s,a), lambda is {}'.format(round(lambda_, 2))
    plt.title(title)
    nx.draw(G, pos, node_size=10, alpha=0.8)
    #node_labels = dict(zip(mcts.tree.nodes, mcts.tree.nodes))
    node_labels = nx.get_node_attributes(tree,'node_label')
    nx.draw_networkx_labels(G, pos, labels = node_labels, font_size=5)
    edge_labels = nx.get_edge_attributes(tree,'action')
    for node in tree.nodes:
        for parent_node in tree.predecessors(node):
            action = tree.edges[parent_node, node]['action']
            edge_labels[(parent_node, node)] = round(tree.nodes[parent_node]['Qr'][action], 1)
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_size=3)
    
    plt.savefig('tree_Qr.pdf')
    plt.show()

def visulize_tree_Qc(tree, lambda_):
    # https://stackoverflow.com/questions/20381460/networkx-how-to-show-node-and-edge-attributes-in-a-graph-drawing
    G = nx.DiGraph()
    for edge in tree.edges:
        # G.add_edge(edge[0], edge[1], label=tree.edges[edge]['action'])
        G.add_edge(edge[0], edge[1])
    # for node in G.nodes:
    #     G.nodes[node]['label'] = tree.nodes[node]['state'].state
    
    pos = graphviz_layout(G, prog="dot", root=0)
    title = 'edge represents Qc(s,a), lambda is {}'.format(round(lambda_, 2))
    plt.title(title)
    nx.draw(G, pos, node_size=10, alpha=0.8)
    #node_labels = dict(zip(mcts.tree.nodes, mcts.tree.nodes))
    node_labels = nx.get_node_attributes(tree,'node_label')
    nx.draw_networkx_labels(G, pos, labels = node_labels, font_size=5)
    edge_labels = nx.get_edge_attributes(tree,'action')
    for node in tree.nodes:
        for parent_node in tree.predecessors(node):
            action = tree.edges[parent_node, node]['action']
            edge_labels[(parent_node, node)] = round(tree.nodes[parent_node]['Qc'][action], 2)
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_size=3)
    
    plt.savefig('tree_Qc.pdf')
    plt.show()       
        
        
if __name__ == "__main__":
    # tree visualization
    # https://stackoverflow.com/questions/48380550/using-networkx-to-output-a-tree-structure
    # https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209
    state = State()
    state.state = (0, 4)
    c_hat = 0.1
    for i in range(4):
        mcts = search(state, c_hat)
        #visulize_tree(mcts.tree)
        #visulize_tree_Qr(mcts.tree, mcts.lambda_)
        #visulize_tree_Qc(mcts.tree, mcts.lambda_)
        action = mcts.GreedyPolicy(0, 0)
        #print('mcts nodes: {}'.format(mcts.tree.nodes))
        print('best action is {}'.format(action))
        print('Na is {}'.format(mcts.tree.nodes[0]['Na']))
        print('Qr is {}'.format(mcts.tree.nodes[0]['Qr']))
        print('Qc is {}'.format(mcts.tree.nodes[0]['Qc']))
        print('lambda is {}'.format(mcts.lambda_))
        Qc = {}
        for action in mcts.tree.nodes[0]['Qc']:
            Qc[action] = mcts.tree.nodes[0]['Qc'][action]
        print('Normalized Qc: {}'.format(Qc))
        
        iters = [i for i in range(len(mcts.lambda_history))]
        plt.plot(iters, mcts.lambda_history)
        plt.title('$\lambda$')
        plt.savefig('lambda_'+str(state.state[0])+'_'+str(state.state[1])+'.pdf')
        
        actions = mcts.simulator.actions(state)
        fig, axs = plt.subplots(len(actions))
        for action in actions:
            index = actions.index(action)
            Qr = [Q[action] for Q in mcts.Qr_history]
            iters = [i for i in range(len(Qr))]
            if len(actions)==1:
                axs.plot(iters, Qr)
            else:
                axs[index].plot(iters, Qr)
            title = 'Qr for action ' + str(action)
            if len(actions)==1:
                axs.set_title(title)
            else:
                axs[index].set_title(title)
        fig.tight_layout(pad=1.0)    
        fig.savefig('Qr_'+str(state.state[0])+'_'+str(state.state[1])+'.pdf')
          
        
        fig, axs = plt.subplots(len(actions))
        for action in actions:
            index = actions.index(action)
            Qc = [round(Q[action], 2) for Q in mcts.Qc_history]
            iters = [i for i in range(len(Qc))]
            if len(actions)==1:
                axs_ = axs.twinx()
                axs.plot(iters, Qc, 'g-')
                axs_.plot(iters, mcts.lambda_history, 'b-')
                axs.set_ylabel('Qc', color='g')
                axs_.set_ylabel('$\lambda$', color='b')
                title = 'Qc and $\lambda$ for action '+str(action)
                axs.set_title(title)
            else:
                axs_ = axs[index].twinx()
                axs[index].plot(iters, Qc, 'g-')
                axs_.plot(iters, mcts.lambda_history, 'b-')
                axs[index].set_ylabel('Qc', color='g')
                axs_.set_ylabel('$\lambda$', color='b')
                title = 'Qc and $\lambda$ for action '+str(action)
                axs[index].set_title(title)
              
        fig.tight_layout(pad=2.0)    
        fig.savefig('Qc_'+str(state.state[0])+'_'+str(state.state[1])+'.pdf')
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
    
    
