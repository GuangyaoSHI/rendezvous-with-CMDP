#!/usr/bin/env python3.8

from gurobipy import *

# Create a new model
m = Model()

# Create variables
x1 = m.addVar(name="x1")
x2 = m.addVar(name="x2")

# Set objective function
m.setObjective(3*x1 + 2*x2 , GRB.MAXIMIZE)

#Add constraints
m.addConstr(2.2*x1 + x2  <= 100, "c1")
m.addConstr(x2  <= 30, "c3")
m.addConstr(x1  >= 0, "c4")
m.addConstr(x2  >= 0, "c5")

# Optimize model
m.optimize()

#Print values for decision variables
for v in m.getVars():
    print(v.varName, v.x)

#Print maximized profit value
print('Maximized profit:',  m.objVal)

