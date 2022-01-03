# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 12:16:33 2021

@author: gyshi
"""

from mcts import MctsSim
from simulator import State
import numpy as np
import matplotlib.pyplot as plt

# lambda_ = 10
# c_hat = 0.6
# root = State()

# mcts = MctsSim(lambda_, c_hat, root)
# RC = mcts.roll_out(root, 0)
# print(RC)



# visualize rendezvous environment
scale = 1000
v_up = scale*np.array([np.math.cos(66/180*np.pi), np.math.sin(66/180*np.pi)])
v_down = scale*np.array([np.math.cos(-66/180*np.pi), np.math.sin(-66/180*np.pi)])

points = []
curr_pos = np.array([0, 0])
print('current position is {}'.format(curr_pos))
points.append(curr_pos)

for loop in range(3):
    for i in range(1, 7):
        next_pos = curr_pos + v_up
        print('next position is {}'.format(next_pos))
        points.append(next_pos)
        curr_pos = next_pos
        print('current position is {}'.format(curr_pos))

    for i in range(1, 7):
        next_pos = curr_pos + v_down
        print('next position is {}'.format(next_pos))
        points.append(next_pos)
        curr_pos = next_pos
        print('current position is {}'.format(curr_pos))
        

X = [pos[0] for pos in points]
Y = [pos[1] for pos in points]
plt.plot(X, Y, '*')