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
        self.uct_k = np.sqrt(2)
        # maximum depth
        self.max_depth = 20
        # shrinking factor
        self.gamma = 1
        # lambda
        self.lambda_ = lambda_
        self.simulator = Simulator()
        self.c_hat = c_hat
        actions = self.simulator.actions(root)
        self.tree.add_node((root.state, 0), 
                           N = 0, 
                           Na = dict(zip(actions, np.zeros(len(actions)))), 
                           Vc=0, 
                           Qr=dict(zip(actions, np.zeros(len(actions)))), 
                           Qc=dict(zip(actions, np.zeros(len(actions)))))
        
    def ucb(self, Ns, Nsa):
        print('Ns is {}, Nsa is {}'.format(Ns, Nsa))
        assert (Ns >= 0) and (Nsa >= 0) 
        if (Ns == 0) or (Nsa == 0):
            return sys.maxsize
        else:
            return np.sqrt(np.log(Ns)/(Nsa+self.EPSILON))
            
        
    # corresponds to the GreedyPolicy in the paper    
    def GreedyPolicy(self, state, depth, k):
        # k is for exploration term
        # actions is a list of actions
        actions = self.simulator.actions(state)
        print('state is {} actions are {}'.format(state.state, actions))
        # find action star, which is a list
        action_star = []
        Qplus_star = -sys.maxsize
        for action in actions:
            print('node is {}'.format(self.tree.nodes[(state.state, depth)]))
            Qr = self.tree.nodes[(state.state, depth)]['Qr'][action]
            Qc = self.tree.nodes[(state.state, depth)]['Qc'][action]
            Ns = self.tree.nodes[(state.state, depth)]['N']
            Nsa = self.tree.nodes[(state.state, depth)]['Na'][action]
            print('Qr is {}, Qc is {}, Ns is {}, Nsa is {}'.format(Qr, Qc, Ns, Nsa))
            # a small number is added to numerator for numerical stability
            # Todo: should I follow author's implementation on UCB?
            print('lambda is {}, k is {}'.format(self.lambda_, k))
            print('ucb is {}'.format(self.ucb(Ns, Nsa)))
            Qplus =  Qr - self.lambda_*Qc + k * self.ucb(Ns, Nsa)
            print('Qplus is {}'.format(Qplus))
            if  Qplus >= Qplus_star:
                # Todo: does this part matter?
                # strictly follow author's implementation
                # it is actually a little bit probalematic 
                # only one element in the action_star set
                if Qplus > Qplus_star:
                    print('action_star is cleared')
                    action_star = []
                Qplus_star = Qplus
                action_star.append(action)
                print('action_star is {}'.format(action_star))
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
        
        best_action_N = self.tree.nodes[(state.state, depth)]['Na'][best_action]
        best_action_Qr = self.tree.nodes[(state.state, depth)]['Qr'][best_action]
        best_action_Qc = self.tree.nodes[(state.state, depth)]['Qc'][best_action]
        best_action_Q = best_action_Qr - self.lambda_ * best_action_Qc
        best_action_bias = biasconstant * np.log(best_action_N + 1)/(best_action_N + 1)
        
        for action in actions:
            Qr = self.tree.nodes[(state.state, depth)]['Qr'][action]
            Qc = self.tree.nodes[(state.state, depth)]['Qc'][action]
            Nsa = self.tree.nodes[(state.state, depth)]['Na'][action]
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
        else:
            return maxCostAction
        
    # default policy for rollout
    def default_policy(self, state):
        # actions is a list 
        actions = self.simulator.actions(state)
        action = random.sample(actions, 1)[0]
        return action
        
    def roll_out(self, state, depth):
        if depth == self.max_depth:
            return np.array([0, 0])
        if self.simulator.is_collision(state):
            return np.array([-1, 1])
        
        if self.simulator.is_goal(state):
            return np.array([0, 0])
        
        action = self.default_policy(state)
        next_state, reward, cost = self.simulator.transition(state, action)
        return np.array([reward, cost]) + self.gamma*self.roll_out(next_state, depth+1)
    
    # Simulate 
    def simulate(self, state, depth):
        # Todo add terminal state check
        if depth == self.max_depth:
            return np.array([0, 0])
        
        if self.simulator.is_collision(state):
            return np.array([-1, 1])
        
        if self.simulator.is_goal(state):
            return np.array([0, 0])
        
        # expansion
        # replace state with state.state
        if not ((state.state, depth) in self.tree.nodes):
            # find all action:N(a, s) pairs
            actions = self.simulator.actions(state)
            
            # N is the number of times that this state has been visited
            # Na is a dictionary to track the the number of times of (s, a) 
            # Vc is the value for cost, Qr is Q-value for reward, Qc is for cost
            self.tree.add_node((state.state, depth), N = 0, 
                               Na = dict(zip(actions, np.zeros(len(actions)))), 
                               Vc=0, 
                               Qr= dict(zip(actions, np.zeros(len(actions)))), 
                               Qc= dict(zip(actions, np.zeros(len(actions)))))
            
            # rollout and use action information to update node state
            # rollout can be parallel
            action = self.default_policy(state)
            next_state, reward, cost = self.simulator.transition(state, action)
            RC =  np.array([reward, cost]) + self.gamma*self.roll_out(next_state, depth+1)
            
            R = RC[0]
            C = RC[1]
            # backpropagation
            
            self.tree.nodes[(state.state, depth)]['N'] += 1
            self.tree.nodes[(state.state, depth)]['Vc'] = C
            self.tree.nodes[(state.state, depth)]['Na'][action] += 1
            self.tree.nodes[(state.state, depth)]['Qr'][action] = R
            self.tree.nodes[(state.state, depth)]['Qc'][action] = C
            return RC
        
        action = self.GreedyPolicy(state, depth, self.uct_k)
        next_state, reward, cost = self.simulator.transition(state, action)
 
        # RC = np.array([reward, cost])
        RC = np.array([reward, cost]) + self.gamma*self.simulate(next_state, depth+1)
        self.tree.add_edge((state.state, depth), (next_state.state, depth + 1), 
                           action = action)
        R = RC[0]
        C = RC[1]
        # backpropagation
        
        self.tree.nodes[(state.state, depth)]['N'] += 1
        Vc = self.tree.nodes[(state.state, depth)]['Vc']
        self.tree.nodes[(state.state, depth)]['Vc'] = Vc + (C-Vc)/self.tree.nodes[(state.state, depth)]['N']
        
        for action in self.tree.nodes[(state.state, depth)]['Na']:
            print('Nsa is {}'.format(self.tree.nodes[(state.state, depth)]['Na'][action]))
            self.tree.nodes[(state.state, depth)]['Na'][action] += 1
            Qr = self.tree.nodes[(state.state, depth)]['Qr'][action]
            Qc = self.tree.nodes[(state.state, depth)]['Qc'][action]
            self.tree.nodes[(state.state, depth)]['Qr'][action] = Qr + (R-Qr)/self.tree.nodes[(state.state, depth)]['Na'][action]
            self.tree.nodes[(state.state, depth)]['Qc'][action] = Qc + (C-Qc)/self.tree.nodes[(state.state, depth)]['Na'][action]
        return RC

        
        
def search(state, c_hat):
    # initialize lambda
    lambda_ = 10
    # Todo: how to specify a range for lambda [0, lambda_max]
    lambda_max = 100
    # Todo: how to specify the number of iterations
    # number of times to update lambda
    iters = 10
    # Todo: number of monte carlo simulations 
    # number of times to do monte carlo simulation
    # in author's implementation this number is 1
    Nmc = 2    
    mcts = MctsSim(lambda_, c_hat, state)
    for i in range(iters):
        # grow monte carlo tree
        for i in range(Nmc):
            mcts.simulate(state, 0)
        action = mcts.GreedyPolicy(state, 0, 0)
        if (mcts.tree.nodes[(state.state, 0)]['Qc'][action] - c_hat < 0):
            # Todo: need to fine tune and check the implementation
            # author's implementation is different from his formula in the paper
            at = -1
        else:
            at = 1
        # lambda_ += 1/(1+i) * at * (mcts.tree.nodes[(state, 0)]['Qc'][action] - c_hat)
        lambda_ += 1/(1+i) * at
        if (lambda_ < 0):
            lambda_ = 0
        if (lambda_ > lambda_max):
            lambda_ = lambda_max
        mcts.lambda_ = lambda_
    return mcts
        
        
        
        
if __name__ == "__main__":
    # tree visualization
    # https://stackoverflow.com/questions/48380550/using-networkx-to-output-a-tree-structure
    # https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209
    
    state = State()
    c_hat = 0.9
    mcts = search(state, c_hat)
    pos = graphviz_layout(mcts.tree, prog="dot")
    nx.draw(mcts.tree, pos, with_labels = True)
    plt.show()
    print('mcts nodes: {}'.format(mcts.tree.nodes))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
    
    