# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 20:13:48 2022

@author: gyshi
"""

from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

#https://python.quantecon.org/lp_intro.html

# Create a new model
m = Model()

# Create variables
x1 = m.addVar(name="x1", lb=-GRB.INFINITY)
x2 = m.addVar(name="x2", lb=-GRB.INFINITY)
x3 = m.addVar(name="x3", lb=-GRB.INFINITY)
x4 = m.addVar(name="x4", lb=-GRB.INFINITY)
x5 = m.addVar(name="x5", lb=-GRB.INFINITY)


# Set objective function
m.setObjective(-(1.30*3*x1+1.06*x4+1.30*x5) , GRB.MINIMIZE)
m.Params.FeasibilityTol = 1e-9    

#Add constraints
m.addConstr(x1 + x2 == 100000, "c1")
m.addConstr(x1 - 1.06*x2 + x3 + x5 == 0, "c2")
m.addConstr(x1 - 1.06*x3 + x4 == 0, "c3")
m.addConstr(x2 >= -20000, "c4")
m.addConstr(x3 >= -20000, "c5")
m.addConstr(x4 >= -20000, "c6")
m.addConstr(x5 <= 50000, "c7")
m.addConstr(x1 >= 0, "c8")
m.addConstr(x5 >= 0, "c9")


# Optimize model
m.optimize()

#Print values for decision variables
for v in m.getVars():
    print(v.varName, v.x)

#Print maximized profit value
print('Maximized profit:',  m.objVal)


x1_current_step = 20000
x2_current_step = 70000
x3_current_step = 4000
x4_current_step = -20000
x5_current_step = 50000
u1_current_step, u2_current_step, u3_current_step = 0, 0, 0
u4_current_step, u5_current_step, u6_current_step = 0, 0, 0

v1_current_step, v2_current_step, v3_current_step = 0, 0, 0


x1_last_step = 20000
x2_last_step = 75000
x3_last_step = 4000
x4_last_step = -20000
x5_last_step = 50000
u1_last_step, u2_last_step, u3_last_step = 0, 0, 0
u4_last_step, u5_last_step, u6_last_step = 0, 0, 0

v1_last_step, v2_last_step, v3_last_step = 0, 0, 0


ax = 0.001
obj_traces = [-(1.30*3*x1_current_step+1.06*x4_current_step+1.30*x5_current_step)]
x_traces = [[x1_current_step, x2_current_step, x3_current_step, x4_current_step, x5_current_step]]
x_mean = []


for i in range(40000):
    ax = ax/1
    av = ax
    au = ax
    dx1 = -3.9 +v1_last_step+v2_last_step+v3_last_step-u5_last_step
    dx2 = v1_last_step - 1.06*v2_last_step - u1_last_step
    dx3 = v2_last_step - 1.06*v3_last_step - u2_last_step
    dx4 = -1.06 + v3_last_step - u3_last_step
    dx5 = -1.3 + v2_last_step + u4_last_step - u6_last_step
    
    du1 = -20000 - x2_last_step
    du2 = -20000 - x3_last_step
    du3 = -20000 - x4_last_step
    du4 = x5_last_step - 50000
    du5 = -x1_last_step
    du6 = -x5_last_step
    
    dv1 = x1_last_step + x2_last_step - 100000
    dv2 = x1_last_step - 1.06*x2_last_step + x3_last_step + x5_last_step
    dv3 = x1_last_step -1.06*x3_last_step + x4_last_step
    
    x1_current_step = x1_last_step - ax*dx1
    x2_current_step = x2_last_step - ax*dx2
    x3_current_step = x3_last_step - ax*dx3
    x4_current_step = x4_last_step - ax*dx4
    x5_current_step = x5_last_step - ax*dx5
    
    u1_current_step = u1_last_step + au*du1
    u1_current_step =max(u1_current_step, 0)
    u2_current_step = u2_last_step + au*du2
    u2_current_step = max(u2_current_step, 0)
    u3_current_step = u3_last_step + au*du3
    u3_current_step = max(u3_current_step, 0)
    u4_current_step = u4_last_step + au*du4
    u4_current_step = max(u4_current_step, 0)
    u5_current_step = u5_last_step + au*du5
    u5_current_step = max(u5_current_step, 0)
    u6_current_step = u6_last_step + au*du6
    u6_current_step = max(u6_current_step, 0)
    
    v1_current_step = v1_last_step + av*dv1
    v2_current_step = v2_last_step + av*dv2
    v3_current_step = v3_last_step + av*dv3
    
    
    obj_traces.append(-(3.9*x1_current_step+1.06*x4_current_step+1.3*x5_current_step))
    x_traces.append([x1_current_step, x2_current_step, x3_current_step, x4_current_step, x5_current_step])
    
    
    x1_last_step = x1_current_step
    x2_last_step = x2_current_step
    x3_last_step = x3_current_step
    x4_last_step = x4_current_step
    x5_last_step = x5_current_step
    
    u1_last_step = u1_current_step
    u2_last_step = u2_current_step
    u3_last_step = u3_current_step
    u4_last_step = u4_current_step
    u5_last_step = u5_current_step
    u6_last_step = u6_current_step
    
    v1_last_step = v1_current_step
    v2_last_step = v2_current_step
    v3_last_step = v3_current_step
    
print("iterations done!")    
obj_mean = []
for i in range(0, len(obj_traces)):
    obj_mean.append(np.mean(obj_traces[0:i+1]))
    
plt.plot(obj_mean)
print("objective converges to {}".format(obj_mean[-1]))