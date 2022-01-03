# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 12:16:33 2021

@author: gyshi
"""

from mcts import MctsSim
from simulator import State
lambda_ = 10
c_hat = 0.6
root = State()

mcts = MctsSim(lambda_, c_hat, root)
RC = mcts.roll_out(root, 0)
print(RC)



# visualize rendezvous environment
