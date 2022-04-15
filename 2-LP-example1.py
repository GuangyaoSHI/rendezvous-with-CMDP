# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 20:13:48 2022

@author: gyshi
"""

from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# https://math.libretexts.org/Bookshelves/Applied_Mathematics/Applied_Finite_Mathematics_(Sekhon_and_Bloom)/03%3A_Linear_Programming_-_A_Geometric_Approach/3.02%3A_Minimization_Applications

# Create a new model
m = Model()

# Create variables
x = m.addVar(name="x")
y = m.addVar(name="y")

# Set objective function
m.setObjective(15*x+25*y , GRB.MINIMIZE)

#Add constraints
m.addConstr(x >= 1, "c1")
m.addConstr(y >= 1, "c2")
m.addConstr(20*x + 30*y >= 110, "c3")


    
# Optimize model
m.optimize()

#Print values for decision variables
for v in m.getVars():
    print(v.varName, v.x)

#Print maximized profit value
print('Maximized profit:',  m.objVal)


x_currrent_step = 0
y_currrent_step = 0
u1_currrent_step, u2_currrent_step, u3_currrent_step = 0, 0, 0

x_last_step = 0
y_last_step = 0
u1_last_step, u2_last_step, u3_last_step = 0, 0, 0

ax = 0.0048
obj_traces = [15*x_currrent_step+25*y_currrent_step]
obj_mean = []
x_traces = [x_currrent_step]
x_mean = []
y_traces = [y_currrent_step]
y_mean = []

for i in range(4000):
    ax = ax/1
    ay = ax
    au = ax
    dx = 15 - u1_last_step - 20*u3_last_step
    dy = 25 - u2_last_step - 30*u3_last_step
    du1 = 1 - x_last_step
    du2 = 1 - y_last_step
    du3 = 110 - 20*x_last_step - 30*y_last_step
    
    x_currrent_step = x_last_step - ax*dx
    y_currrent_step = y_last_step - ay*dy
    u1_currrent_step = u1_last_step + au*du1
    u1_currrent_step =max(u1_currrent_step, 0)
    u2_currrent_step = u2_last_step + au*du2
    u2_currrent_step = max(u2_currrent_step, 0)
    u3_currrent_step = u3_last_step + au*du3
    u3_currrent_step = max(u3_currrent_step, 0)
    
    obj_traces.append(15*x_currrent_step+25*y_currrent_step)
    x_traces.append(x_currrent_step)
    y_traces.append(y_currrent_step)
    
    x_last_step = x_currrent_step
    y_last_step = y_currrent_step
    u1_last_step = u1_currrent_step
    u2_last_step = u2_currrent_step
    u3_last_step = u3_currrent_step

for i in range(len(obj_traces)):
    obj_mean.append(np.mean(obj_traces[0:i+1]))
    
plt.plot(obj_mean)
print("objective converges to {}".format(obj_mean[-1]))