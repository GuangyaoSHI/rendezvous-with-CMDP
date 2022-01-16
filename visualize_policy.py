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
c_hat = 0.1

#policies
policies = {}

#grid world size
size = 5
for x in range(0, size):
    for y in range(0, size):
        state = State(state=(x,y))
        mcts_policy = search(state, c_hat)
        print('Qc: {}'.format(mcts_policy.tree.nodes[0]['Qc']))
        policies[(x, y)] = mcts_policy

# Saving the objects:
# with open('policy_map.obj', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(policies, f)

# Getting back the objects:
# with open('policy_map.obj', 'rb') as f:  # Python 3: open(..., 'rb')
#     policies = pickle.load(f)



fig, ax = plt.subplots()

plt.axis([-1, size, -1, size])
plt.axis('equal')
for x in range(0, size):
    for y in range(0, size):
        state = State(state=(x,y))
        mcts_policy = policies[(x, y)]
        best_action = mcts_policy.GreedyPolicy(0, 0)
        print('')
        print(state)
        print(best_action)
        ax.plot(x, y, 'r.')
        ax.arrow(x,y,0.2*(best_action[0]-x),0.2*(best_action[1]-y),width=0.05,head_width=0.5, head_length=0.15)
        Qr = mcts_policy.tree.nodes[0]['Qr'][best_action] 
        Qc = mcts_policy.tree.nodes[0]['Qc'][best_action]
        ax.text(x-0.3, y-0.2, str(round(Qr, 1)), size='x-small')
        ax.text(x-0.3, y+0.2, str(round(Qc, 2)), size='x-small')
        #plt.pause(0.1)
#fig.show()
ax.set_xlim(-1, size)
ax.set_ylim(-1, size)
ax.set_aspect('equal', adjustable='box')
ax.grid()
ax.set_xticks(np.arange(-1, size+1))

#plot map
G = nx.DiGraph()
G_ = nx.grid_2d_graph(size, size)
G.add_nodes_from(G_.nodes)
pos = dict(zip(G.nodes, G.nodes))
G.graph['pos'] = pos


colors = []
obstacles = [((size-1)/2,0)]
for node in G.nodes:
    if node in obstacles:
        colors.append('r')
    else:
        colors.append('#1f78b4')

nx.draw(G, pos=pos, node_size=10, alpha=0.1, node_color=colors)

fig.savefig("policy_map_"+str(size)+"by"+str(size)+"_deter3.pdf")
plt.show()





