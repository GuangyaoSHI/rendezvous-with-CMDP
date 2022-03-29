# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from utils import *
import pickle
import random
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
import copy

experiment_name = '_risk_level_example'
threshold=0.2
data_name_appendix = '-2'

#figure names
fig_name = data_name_appendix

current_directory = os.getcwd()
target_directory = os.path.join(current_directory, r'route_plots')
if not os.path.exists(target_directory):
   os.makedirs(target_directory)

UGV_fig_name =  os.path.join(target_directory, "UGV_sample_route"+ experiment_name +'_'+ str(threshold)+fig_name+".pdf")

# control flag for adding legend 
# set it to be true when generating plot different risk threshold
add_legend = False
# set it to be true when generate plot for different speed example
add_velocity_legend = True

UAV_task = generate_UAV_task()
UGV_task = generate_UGV_task()
road_network = generate_road_network()
actions = ['v_be', 'v_br', 'v_be_be', 'v_br_br']

if add_velocity_legend:
    colors = dict(zip(actions, ['b', 'b', 'r', 'r']))
else:
    colors = dict(zip(actions, ['b', 'g', 'r', 'm']))

rendezvous = Rendezvous(UAV_task, UGV_task, road_network, battery=280e3)
if experiment_name == '_velocity_comparison1':
    rendezvous.velocity_ugv = 5
    rendezvous.velocity_uav = {'v_be' : 7.5, 'v_br' : 7.5}

if experiment_name == '_velocity_comparison2':
    rendezvous.velocity_ugv = 5
    rendezvous.velocity_uav = {'v_be' : 10, 'v_br' : 10}    
    
if experiment_name == '_velocity_comparison3':
    rendezvous.velocity_ugv = 5
    rendezvous.velocity_uav = {'v_be' : 14, 'v_br' : 14} 

# set directory and file name to save the data 
current_directory = os.getcwd()
target_directory = os.path.join(current_directory, r'visualization')
data_name = os.path.join(target_directory, 'visualization_data-'+str(threshold)+experiment_name+data_name_appendix+'.obj')
with open(data_name, 'rb') as f:  # Python 3: open(..., 'wb')
    data = pickle.load(f)  


UAV_traces = data['UAV_trace']
UGV_traces = data['UGV_trace']
action_traces = data['action_trace']
UGV_task_traces = data['UGV_task_trace']
duration_traces = data['duration_trace']
    

# arrow
hw = 0.2
w = 0.04
# line
lw = 1



def hanging_line(point1, point2):
    a = (point2[1] - point1[1])/(np.cosh(point2[0]) - np.cosh(point1[0]))
    b = point1[1] - a*np.cosh(point1[0])
    x = np.linspace(point1[0], point2[0], 100)
    y = a*np.cosh(x) + b
    return (x, y)



#axs.plot([p[0] for p in UGV_traces], [p[1] for p in UGV_traces])

# plot roadnetwork
'''
add_label = True
for edge in road_network.edges:
    x = [edge[0][0]/1000, edge[1][0]/1000]
    y = [edge[0][1]/1000, edge[1][1]/1000]
    if add_label:
        axs.plot(x, y, color='b', alpha=0.4, label='road network')
        add_label = False
    else:
        axs.plot(x, y, color='b', alpha=0.4)

    
assert (17500, 1500) in UGV_traces
road_end_index = UGV_traces.index((17500, 1500))
'''

path = nx.shortest_path(road_network, source=UGV_task.nodes[0]['pos'], target=UGV_task.nodes[2]['pos'])
node_labels = {}
node_label = 0
for node in path:
    if node == (6943, 18987):
        node_labels[node] = ''
        continue
    node_labels[node] = chr(node_label+97)
    node_label += 1


print("UGV traces : \n {}".format(UGV_traces))
print("UGV task traces : \n {}".format(UGV_task_traces))
print("UAV traces : \n {}".format(UAV_traces))
if (17500, 1500) in UGV_traces:
    print("(17500, 1500) in UGV traces")
    road_end_index = UGV_traces.index((17500, 1500))
    first_segment = UAV_traces.index(road_end_index)


#road_end_index = UGV_traces.index((17837, 2963))
first_segment = UAV_traces.index(road_end_index)    
fig, axs = plt.subplots()
for p in UGV_traces:
    assert p in road_network.nodes
    
x_shift = 2
y_shift = 0.5   
# there seems no general way to present the route of UGV
# as a result the code below needs to change for different cases

# fontsize
fs = 10
# first segment
for i in range(first_segment):
    if type(UAV_traces[i]) == int:
        if type(UAV_traces[i+1]) == int:
            path = nx.shortest_path(road_network, source=UGV_traces[UAV_traces[i]], target=UGV_traces[UAV_traces[i+1]])
            x = [node[0]/1000 - x_shift for node in path]
            y = [node[1]/1000 for node in path]
            route_labels = [node_labels[node] for node in path]
            axs.plot(x, y, color='b', marker='o', linewidth=lw, markersize=1)
            for j in range(len(x)):
                axs.text(x[j], y[j], route_labels[j], fontsize = fs)   
        else:
            x1 = UGV_traces[UAV_traces[i]][0]/1000 - x_shift
            y1 = UGV_traces[UAV_traces[i]][1]/1000
            # rendezvous point
            x2 = UAV_traces[i+1][0]/1000 - x_shift*1.5
            y2 = UAV_traces[i+1][1]/1000 - y_shift
            route_labels = {(x1, y1): node_labels[UGV_traces[UAV_traces[i]]],
                            (x2, y2): node_labels[UAV_traces[i+1]]}
            # plot rendezvous point
            axs.plot(x2, y2, marker='*', color='k', markersize=8, alpha=0.8)
            for rl in route_labels:
                axs.text(rl[0], rl[1], route_labels[rl], fontsize = 'x-small')
            # plot a curve between UAV current position and rendezvous position
            x, y = hanging_line([x1, y1], [x2, y2])
            index = round(len(x)/2)
            axs.plot(x, y, color='r', linewidth=lw, linestyle='--')
            axs.plot(x1, y1, color='b', marker='o', linewidth=lw, markersize=1)         
            uv = np.array([x[index+1]-x[index], y[index+1]-y[index]])/np.linalg.norm(np.array([x[index+1]-x[index], y[index+1]-y[index]]))
            axs.arrow(x[index], y[index], 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color='r', length_includes_head=False)
    else:  # UAV is at a rendezvous node
        assert type(UAV_traces[i+1]) == int 
        # rendezvous point
        x1 = UAV_traces[i][0]/1000 - x_shift*1.5
        y1 = UAV_traces[i][1]/1000 - y_shift
        x2 = UGV_traces[UAV_traces[i+1]][0]/1000 - x_shift
        y2 = UGV_traces[UAV_traces[i+1]][1]/1000
        route_labels = {(x1, y1): node_labels[UAV_traces[i]],
                        (x2, y2): node_labels[UGV_traces[UAV_traces[i+1]]]}
        x, y = hanging_line([x1, y1], [x2, y2])
        axs.plot(x, 
                 y, color='r', linewidth=lw, linestyle='--')
        for rl in route_labels:
            axs.text(rl[0], rl[1], route_labels[rl], fontsize = fs)
        uv = np.array([x[index+1]-x[index], y[index+1]-y[index]])/np.linalg.norm(np.array([x[index+1]-x[index], y[index+1]-y[index]]))
        axs.arrow(x[index], y[index], 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color='r', length_includes_head=False)

'''
if (len(UAV_traces)-1) > first_segment:
    # connect the first segment and the second segment
    x1 = UGV_traces[UAV_traces[first_segment]][0]/1000 - x_shift
    y1 = UGV_traces[UAV_traces[first_segment]][1]/1000 
    x2 = UGV_traces[UAV_traces[first_segment]][0]/1000 + x_shift
    y2 = UGV_traces[UAV_traces[first_segment]][1]/1000 
    axs.plot([x1, x2], [y1, y2], color='b', marker='o', linewidth=lw, markersize=1, linestyle='--')
'''
    
# second segment
for i in range(first_segment, len(UAV_traces)-1):
    if type(UAV_traces[i]) == int:
        if type(UAV_traces[i+1]) == int:
            path = nx.shortest_path(road_network, source=UGV_traces[UAV_traces[i]], target=UGV_traces[UAV_traces[i+1]])
            x = [node[0]/1000 + x_shift for node in path]
            y = [node[1]/1000 for node in path]
            route_labels = [node_labels[node] for node in path]
            axs.plot(x, y, color='b', marker='o', linewidth=lw, markersize=1)
            for j in range(len(x)):
                axs.text(x[j], y[j], route_labels[j]+'\'', fontsize = 'x-small')   
        else:
            x1 = UGV_traces[UAV_traces[i]][0]/1000 + x_shift
            y1 = UGV_traces[UAV_traces[i]][1]/1000
            # rendezvous point
            x2 = UAV_traces[i+1][0]/1000 + x_shift*1.5
            y2 = UAV_traces[i+1][1]/1000 + y_shift
            route_labels = {(x1, y1): node_labels[UGV_traces[UAV_traces[i]]],
                            (x2, y2): node_labels[UAV_traces[i+1]]}
            # plot rendezvous point
            axs.plot(x2, y2, marker='*', color='k', markersize=8, alpha=0.8)
            for rl in route_labels:
                axs.text(rl[0], rl[1], route_labels[rl]+'\'', fontsize = fs)
            # plot a curve between UAV current position and rendezvous position
            x, y = hanging_line([x1, y1], [x2, y2])
            index = round(len(x)/2)
            axs.plot(x, y, color='r', linewidth=lw, linestyle='--')
            axs.plot(x1, y1, color='b', marker='o', linewidth=lw, markersize=1)         
            uv = np.array([x[index+1]-x[index], y[index+1]-y[index]])/np.linalg.norm(np.array([x[index+1]-x[index], y[index+1]-y[index]]))
            axs.arrow(x[index], y[index], 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color='r', length_includes_head=False)
    else:  # UAV is at a rendezvous node
        assert type(UAV_traces[i+1]) == int 
        # rendezvous point
        x1 = UAV_traces[i][0]/1000 + x_shift*1.5
        y1 = UAV_traces[i][1]/1000 + y_shift
        x2 = UGV_traces[UAV_traces[i+1]][0]/1000 + x_shift
        y2 = UGV_traces[UAV_traces[i+1]][1]/1000
        route_labels = {(x1, y1): node_labels[UAV_traces[i]],
                        (x2, y2): node_labels[UGV_traces[UAV_traces[i+1]]]}
        x, y = hanging_line([x1, y1], [x2, y2])
        axs.plot(x, y, color='r', linewidth=lw, linestyle='--')
        for rl in route_labels:
            axs.text(rl[0], rl[1], route_labels[rl]+'\'', fontsize = fs)
        uv = np.array([x[index+1]-x[index], y[index+1]-y[index]])/np.linalg.norm(np.array([x[index+1]-x[index], y[index+1]-y[index]]))
        axs.arrow(x[index], y[index], 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color='r', length_includes_head=False)




#axs.axis('equal') 
handles = [Line2D([0], [0], color='b', lw=4, label='Forward'), 
           Line2D([0], [0], color='r', lw=4, label='Rendezvous detour', linestyle='--'),
           Line2D([0], [0], marker='*', color='w', label='Rendezvous point',
                                 markerfacecolor='k', markersize=11)
           ]
#[mpatches.Patch(color='k', label='rendezvous point')]
axs.legend(loc = "upper right", handles=handles, fontsize = fs)
axs.set_aspect('equal', 'box') 
axs.set_xlabel('x  (km)')
axs.set_ylabel('y  (km)')
axs.set_xlim([0, 21])
axs.set_title(r'The route of UGV when UAV executes $\pi$')    
fig.savefig(UGV_fig_name , bbox_inches='tight')
print("figure is saved to {}".format(UGV_fig_name))            
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
     