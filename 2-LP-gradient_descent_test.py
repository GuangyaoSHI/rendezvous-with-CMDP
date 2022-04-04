# https://realpython.com/linear-programming-python/

from gurobipy import *
import matplotlib.pyplot as plt

# Create a new model
m = Model()

# Create variables
x = m.addVar(name="x")
y = m.addVar(name="y")

# Set objective function
m.setObjective(-(x + 2*y) , GRB.MINIMIZE)

#Add constraints
#m.addConstr(2*x + y  <= 20, "c1")
#m.addConstr(-4*x + 5*y  <= 10, "c2")
#m.addConstr(-x + 2*y >= -2, "c3")
#m.addConstr(-x + 5*y == 15, "c4")
m.addConstr(x  <= 0, "c5")
m.addConstr(y  <= 0, "c6")

# Optimize model
m.optimize()

#Print values for decision variables
for v in m.getVars():
    print(v.varName, v.x)

#Print maximized profit value
print('Maximized profit:',  -m.objVal)


x_currrent_step = 0
y_currrent_step = 0
u1_currrent_step, u2_currrent_step, u3_currrent_step, u4_currrent_step, u5_currrent_step = 0, 0, 0, 0, 0
v1_currrent_step = 0
#L_current_step = -(x_currrent_step + 2*y_currrent_step) + u1_currrent_step*(2*x_currrent_step+y_currrent_step-20)+\
#    u2_currrent_step*(-4*x_currrent_step+5*y_currrent_step-10) +\
 #       u3_currrent_step*(x_currrent_step-2*y_currrent_step-2) + \
#            u4_currrent_step*(-x_currrent_step) + \
#                u5_currrent_step*(-y_currrent_step) + \
#                  v1_currrent_step*(-x_currrent_step+5*y_currrent_step-15)
L_current_step = -(x_currrent_step + 2*y_currrent_step) + u4_currrent_step*(x_currrent_step)+u5_currrent_step*(y_currrent_step)

x_last_step = 2
y_last_step = 2
u1_last_step, u2_last_step, u3_last_step, u4_last_step, u5_last_step = 0, 0, 0, 0, 0
v1_last_step = 0
L_last_step = L_current_step - 0.1

ax = 0.1
L_traces = [L_current_step]
traces = [x_currrent_step+2*y_currrent_step]
while (abs(L_current_step - L_last_step) >0.001):
    ax=ax/1.1
    ay = ax
    au = ax
    av = ax 
    L_last_step = L_current_step
    #dx = -1 + 2*u1_last_step -4*u2_last_step + u3_last_step -u4_last_step - v1_last_step
    dx = -1 + u4_last_step
    #dy = -2 + u1_last_step +5*u2_last_step - 2*u3_last_step - u5_last_step +5*v1_last_step
    dy = -2 + u5_last_step
#    du1 = 2*x_last_step + y_last_step -20
#    du2 = -4*x_last_step + 5*y_last_step -10
#    du3 = x_last_step -2*y_last_step -2
    du4 = x_last_step
    du5 = y_last_step
#    dv1 = -x_last_step + 5*y_last_step -15
    
    x_currrent_step = x_last_step - ax*dx
    #x_currrent_step = min(x_currrent_step, 0)
    #x_currrent_step = max(x_currrent_step, 0)
    x_last_step = x_currrent_step
    
    y_currrent_step = y_last_step - ay*dy
    #y_currrent_step = max(y_currrent_step, 0)
    #y_currrent_step = min(y_currrent_step, 0)
    y_last_step = y_currrent_step
    
    #u1_currrent_step = u1_last_step + au*du1
    #u1_currrent_step = max(u1_currrent_step, 0)
    #u1_last_step = u1_currrent_step
    
    #u2_currrent_step = u2_last_step + au*du2
    #u2_currrent_step = max(u2_currrent_step, 0)
    #u2_last_step = u2_currrent_step
    
   # u3_currrent_step = u3_last_step + au*du3
   # u3_currrent_step = max(u3_currrent_step, 0)
   # u3_last_step = u3_currrent_step
    
    u4_currrent_step = u4_last_step + au*du4
    u4_currrent_step = max(u4_currrent_step, 0)
    u4_last_step = u4_currrent_step
    
    u5_currrent_step = u5_last_step + au*du5
    u5_currrent_step = max(u5_currrent_step, 0)
    u5_last_step = u5_currrent_step
    
    #v1_currrent_step = v1_last_step + av*dv1
    #v1_last_step = v1_currrent_step
    
    L_current_step = -(x_currrent_step + 2*y_currrent_step) + u4_currrent_step*(x_currrent_step)+u5_currrent_step*(y_currrent_step)
        #u2_currrent_step*(-4*x_currrent_step+5*y_currrent_step-10) +\
         #   u3_currrent_step*(x_currrent_step-2*y_currrent_step-2) + \
         #       u4_currrent_step*(-x_currrent_step) + \
          #          u5_currrent_step*(-y_currrent_step)# + \
          #              v1_currrent_step*(-x_currrent_step+5*y_currrent_step-15)
    L_traces.append(L_current_step)
    traces.append(x_currrent_step+2*y_currrent_step)

plt.plot(traces)
print('objective is {}'.format(x_currrent_step+2*y_currrent_step))
