# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:40:31 2021
​
@author: gyshi, tokekar
"""
import random
import numpy as np
import networkx as nx
import copy
import sys
from simulator import State
from simulator import Simulator
from mcts import MctsSim
from mcts import search
from scipy.io import savemat
import matplotlib.pyplot as plt
import pickle

# initial cost threshold
c_hat = 0.2
#policies
policies = {}

for x in range(0,7):
    for y in range(0,7):
        state = State(state=(x,y))
        mcts_policy = search(state, c_hat)
        print('Qc: {}'.format(mcts_policy.tree.nodes[0]['Qc']))
        policies[(x, y)] = mcts_policy

# Saving the objects:
with open('policy_map.obj', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(policies, f)

# Getting back the objects:
# with open('policy_map.obj') as f:  # Python 3: open(..., 'rb')
#     policies = pickle.load(f)

fig, ax = plt.subplots()

plt.axis([-1, 7, -1, 7])
plt.axis('equal')
for x in range(0,7):
    for y in range(0,7):
        #state = State(state=(x,y))
        mcts_policy = policies[(x, y)]
        best_action = mcts_policy.GreedyPolicy(0,0)
        print('')
        print((x, y))
        print(best_action)
        ax.plot(x, y, 'r.')
        ax.arrow(x,y,0.2*(best_action[0]-x),0.2*(best_action[1]-y),width=0.05,head_width=0.5, head_length=0.15)
        print('Vc: {}'.format(mcts_policy.tree.nodes[0]['Vc']))
        #plt.pause(0.1)
#fig.show()
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)
ax.set_aspect('equal', adjustable='box')
ax.grid()
ax.set_xticks(np.arange(-1, 8))
fig.savefig("policy_map.pdf")
plt.show()
