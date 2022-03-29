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
threshold=0.5
data_name_appendix = '-3'

#figure names
fig_name = data_name_appendix

current_directory = os.getcwd()
target_directory = os.path.join(current_directory, r'route_plots')
if not os.path.exists(target_directory):
   os.makedirs(target_directory)
UAV_fig_name =  os.path.join(target_directory, "UAV_sample_route" + experiment_name +'_'+ str(threshold)+fig_name+".pdf") 
UGV_fig_name =  os.path.join(target_directory, "UGV_sample_route"+ experiment_name +'_'+ str(threshold)+fig_name+".pdf")

# control flag for adding legend 
# set it to be true when generating plot different risk threshold
add_legend = True
# set it to be true when generate plot for different speed example
add_velocity_legend = False

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
battery_traces = data['battery_trace']    

fig, axs = plt.subplots()
# arrow
hw = 0.2
w = 0.04
# line
lw = 1

# for better visualization
UAV_task.nodes[3]['pos'] = (4440, 12993)
UAV_task.nodes[2]['pos'] = (7350, 13480)
UAV_task.nodes[11]['pos'] = (15822, 3549)

for i in range(len(UAV_traces)-1):
    if type(UAV_traces[i]) == int:
        action = action_traces[UAV_traces[i]]
        if len(action)>4:
            assert type(UAV_traces[i+1]) == tuple, "should be rendezvous point"
            x1 = UAV_task.nodes[UAV_traces[i]]['pos'][0]/1000
            x2 = UAV_traces[i+1][0]/1000
            y1 = UAV_task.nodes[UAV_traces[i]]['pos'][1]/1000
            y2 = UAV_traces[i+1][1]/1000
            axs.plot([x1, x2], 
                     [y1, y2], color=colors[action], linewidth=lw)
            axs.plot(x2, y2, marker='*', color='k', markersize=8, alpha=0.8)
            uv = np.array([x2-x1, y2-y1])/np.linalg.norm(np.array([x2-x1, y2-y1]))
            axs.arrow((x1+x2)/2, (y1+y2)/2, 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color=colors[action], length_includes_head=False)
            axs.text(x1, y1, str(battery_traces[UAV_traces[i]])+'%', color='r', fontsize = 'x-small')
        else:
            assert type(UAV_traces[i+1]) == int, "should be a task node"
            x1 = UAV_task.nodes[UAV_traces[i]]['pos'][0]/1000
            x2 = UAV_task.nodes[UAV_traces[i+1]]['pos'][0]/1000
            y1 = UAV_task.nodes[UAV_traces[i]]['pos'][1]/1000
            y2 = UAV_task.nodes[UAV_traces[i+1]]['pos'][1]/1000
            axs.plot([x1, x2], 
                     [y1, y2], color=colors[action], linewidth=lw)
            uv = np.array([x2-x1, y2-y1])/np.linalg.norm(np.array([x2-x1, y2-y1]))

            axs.arrow((x1+x2)/2, (y1+y2)/2, 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color=colors[action], length_includes_head=False)

    else:
        assert type(UAV_traces[i-1]) == int, "previous node should be a task node"
        assert type(UAV_traces[i+1]) == int, "next node should be a task node"
        action = action_traces[UAV_traces[i-1]]
        assert len(action)>4, "should be a rendezvous action"
        x1 = UAV_traces[i][0]/1000
        x2 = UAV_task.nodes[UAV_traces[i+1]]['pos'][0]/1000
        y1 = UAV_traces[i][1]/1000
        y2 = UAV_task.nodes[UAV_traces[i+1]]['pos'][1]/1000
        
        if add_velocity_legend:
            axs.plot([x1, x2], 
                     [y1, y2], color=colors[action], linestyle='--', linewidth=lw)
            uv = np.array([x2-x1, y2-y1])/np.linalg.norm(np.array([x2-x1, y2-y1]))
            axs.arrow((x1+x2)/2, (y1+y2)/2, 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color=colors[action], length_includes_head=False)
        else:
            axs.plot([x1, x2], 
                     [y1, y2], color=colors[action], linestyle='--', linewidth=lw)
            uv = np.array([x2-x1, y2-y1])/np.linalg.norm(np.array([x2-x1, y2-y1]))
            axs.arrow((x1+x2)/2, (y1+y2)/2, 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color=colors[action], length_includes_head=False)

        
    
j=0
for node in UAV_task.nodes:
    if UAV_task.out_degree(node) == 0 and UAV_task.in_degree(node)==1 :
        continue
    axs.text(UAV_task.nodes[node]['pos'][0]/1000, UAV_task.nodes[node]['pos'][1]/1000, j)
    axs.plot(UAV_task.nodes[node]['pos'][0]/1000, UAV_task.nodes[node]['pos'][1]/1000, marker='.', color='k', markersize=5, alpha=0.8)
    j += 1

 
#axs.set_aspect('equal', 'box')
handles = []
labels = dict(zip(actions, [r'forward with $v_{be}$', r'forward with $v_{br}$', r'rendezvous with $v_{be}$', r'rendezvous with $v_{br}$']))
for action in actions:
    lengend_v = mpatches.Patch(color=colors[action], label=labels[action])
    handles.append(lengend_v)
if add_legend:
    legend1 = axs.legend(loc = "lower left", handles=handles, fontsize = 6)  
    handles = [Line2D([0], [0], marker='*', color='w', label='Rendezvous point',
                          markerfacecolor='k', markersize=11), 
               Line2D([0], [0], marker='.', color='w', label='UAV task',
                                     markerfacecolor='k', markersize=11)
               ]
    #[mpatches.Patch(color='k', label='rendezvous point')]
    legend2 = axs.legend(loc = "upper right", handles=handles, fontsize = 6)
    
    axs.add_artist(legend1)
    
if add_velocity_legend:
    #labels = {'forward': 'b', 'rendezvous': 'r', 'back to tour':'g'}
    handles = [mpatches.Patch(color='b', label='forward'), mpatches.Patch(color='r', label='rendezvous'),  
               ]
    legend1 = axs.legend(loc = "lower left", handles=handles, fontsize = 8)
    
    legend2 = axs.legend(loc = "upper right", handles=[Line2D([0], [0], marker='*', color='w', label='Rendezvous point',
                          markerfacecolor='k', markersize=11)], fontsize = 8)
    axs.add_artist(legend1)
    
    
axs.set_aspect('equal', 'box') 
#axs.axis('equal') 
axs.set_xlabel('x  (km)')
axs.set_ylabel('y  (km)')
axs.set_xlim([0, 20])
axs.set_title(r'A sample route of UAV with policy $\pi_{0.5}$')    
fig.savefig(UAV_fig_name, bbox_inches='tight')


# UAV travel time
UAV_travel_time = np.sum(np.array(duration_traces))
print("UAV travel time is {}".format(UAV_travel_time))
# UAV travel distance
X = []
for uav in UAV_traces:
    if type(uav)== int:
        X.append(UAV_task.nodes[uav]['pos'][0])
    else:
        X.append(uav[0])

Y = []
for uav in UAV_traces:
    if type(uav)== int:
        Y.append(UAV_task.nodes[uav]['pos'][1])
    else:
        Y.append(uav[1])
        
assert len(X) == len(Y)
travel_dis = 0
for i in range(len(X)-1):
    travel_dis += np.linalg.norm(np.array([X[i+1], Y[i+1]])-np.array([X[i], Y[i]]))
print("UAV travel distance is {}".format(travel_dis))


# compute the travel time of UAV if it can fly with best range speed all the time
tour_dis = 0
for i in range(len(UAV_task.nodes)-1):
    tour_dis += np.linalg.norm(np.array(UAV_task.nodes[i]['pos'])-np.array(np.array(UAV_task.nodes[i+1]['pos'])))

    
print("UAV tour distance is {}".format(tour_dis))
print("UAV travel distance overhead is {}".format((travel_dis-tour_dis)/tour_dis))
tour_time = tour_dis/rendezvous.velocity_uav['v_br']
print("UAV tour time with best range speed is {}".format(tour_time))
print("UAV travel time overhead is {}".format((UAV_travel_time-tour_time)/tour_time))




# compute number of rendezvous
num = 0
for node in UAV_traces:
    if type(node) == tuple:
        num += 1
print("Total number of rendezvous is {}".format(num))


def hanging_line(point1, point2):
    a = (point2[1] - point1[1])/(np.cosh(point2[0]) - np.cosh(point1[0]))
    b = point1[1] - a*np.cosh(point1[0])
    x = np.linspace(point1[0], point2[0], 100)
    y = a*np.cosh(x) + b
    return (x,y)



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
    node_labels[node] = str(node_label)
    node_label += 1

'''
# find UGV path
UGV_paths = []
for i in range(len(UAV_traces)-1):    
    print('\n')
    if type(UAV_traces[i]) == int:
        if type(UAV_traces[i+1]) == int:
            print("UAV is at node {} and is moving to node {} {}".format(UAV_traces[i], UAV_traces[i+1], UAV_task.nodes[UAV_traces[i+1]]['pos']))
            path = nx.shortest_path(road_network, source=UGV_traces[UAV_traces[i]], target=UGV_traces[UAV_traces[i+1]])
            print("UGV is moving from {} to {} through path {}".format(UGV_traces[UAV_traces[i]], UGV_traces[UAV_traces[i+1]], path))
            UGV_paths += [(node[0]/1000, node[1]/1000) for node in path]
            print("path added to UGV's when UAV reaches its next node is {}".format(path))
            #UGV_paths.remove(UGV_paths[-1])
        else:
            print("UAV is moving from node {} to a rendezvous point {}".format(UAV_traces[i], UAV_traces[i+1]))
            x1 = UGV_traces[UAV_traces[i]][0]/1000
            y1 = UGV_traces[UAV_traces[i]][1]/1000
            x2 = UAV_traces[i+1][0]/1000
            y2 = UAV_traces[i+1][1]/1000
            UGV_paths += [(x1, y1), (x2, y2, 'r')]
            print("path added to UGV's path when UAV reaches rendezvous point is {}".format([(x1, y1), (x2, y2, 'r')]))
    else:  
        assert type(UAV_traces[i+1]) == int
        print("UAV is moving from a rendezvous point {} to node {} {}".format(UAV_traces[i], UAV_traces[i+1], UAV_task.nodes[UAV_traces[i+1]]['pos']))
        x1 = UAV_traces[i][0]/1000
        y1 = UAV_traces[i][1]/1000
        x2 = UGV_traces[UAV_traces[i+1]][0]/1000 
        y2 = UGV_traces[UAV_traces[i+1]][1]/1000
        assert (x1, y1, 'r') in UGV_paths
        #UGV_paths += [(x2, y2)]
        #print("path added to UGV's path when UAV departs rendezvous point is {}".format([(x2, y2)]))
             
# plot UGV route


if (17.5, 1.5) in UGV_paths:
    road_end_index = UGV_traces.index((17500, 1500))
else:
    road_end_index = 10000
'''

print("UGV traces : \n {}".format(UGV_traces))
print("UGV task traces : \n {}".format(UGV_task_traces))
print("UAV traces : \n {}".format(UAV_traces))
if (17500, 1500) in UGV_traces:
    print("(17500, 1500) in UGV traces")
    road_end_index = UGV_traces.index((17500, 1500))
    first_segment = UAV_traces.index(road_end_index)


'''    
fig, axs = plt.subplots()
for p in UGV_traces:
    assert p in road_network.nodes
    
x_shift = 1
y_shift = 1    
# there seems no general way to present the route of UGV
# as a result the code below needs to change for different cases


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
                axs.text(x[j], y[j], route_labels[j], fontsize = 'x-small')   
        else:
            x1 = UGV_traces[UAV_traces[i]][0]/1000 - x_shift
            y1 = UGV_traces[UAV_traces[i]][1]/1000
            x2 = UAV_traces[i+1][0]/1000
            y2 = UAV_traces[i+1][1]/1000
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
        x1 = UAV_traces[i][0]/1000
        y1 = UAV_traces[i][1]/1000
        x2 = UGV_traces[UAV_traces[i+1]][0]/1000 - x_shift
        y2 = UGV_traces[UAV_traces[i+1]][1]/1000
        route_labels = {(x1, y1): node_labels[UAV_traces[i]],
                        (x2, y2): node_labels[UGV_traces[UAV_traces[i+1]]]}
        x, y = hanging_line([x1, y1], [x2, y2])
        axs.plot(x, 
                 y, color='r', linewidth=lw, linestyle='--')
        for rl in route_labels:
            axs.text(rl[0], rl[1], route_labels[rl], fontsize = 'x-small')
        uv = np.array([x[index+1]-x[index], y[index+1]-y[index]])/np.linalg.norm(np.array([x[index+1]-x[index], y[index+1]-y[index]]))
        axs.arrow(x[index], y[index], 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color='r', length_includes_head=False)


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
                axs.text(x[j], y[j], route_labels[j], fontsize = 'x-small')   
        else:
            x1 = UGV_traces[UAV_traces[i]][0]/1000 + x_shift
            y1 = UGV_traces[UAV_traces[i]][1]/1000
            x2 = UAV_traces[i+1][0]/1000
            y2 = UAV_traces[i+1][1]/1000
            route_labels = {(x1, y1): node_labels[UGV_traces[UAV_traces[i]]],
                            (x2, y2): node_labels[UAV_traces[i+1]]}
            # plot rendezvous point
            axs.plot(x2, y2, marker='*', color='k', markersize=8, alpha=0.8)
            for rl in route_labels:
                axs.text(rl[0], rl[1], route_labels[rl]+'\'', fontsize = 'x-small')
            # plot a curve between UAV current position and rendezvous position
            x, y = hanging_line([x1, y1], [x2, y2])
            index = round(len(x)/2)
            axs.plot(x, y, color='r', linewidth=lw, linestyle='--')
            axs.plot(x1, y1, color='b', marker='o', linewidth=lw, markersize=1)         
            uv = np.array([x[index+1]-x[index], y[index+1]-y[index]])/np.linalg.norm(np.array([x[index+1]-x[index], y[index+1]-y[index]]))
            axs.arrow(x[index], y[index], 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color='r', length_includes_head=False)
    else:  # UAV is at a rendezvous node
        assert type(UAV_traces[i+1]) == int 
        x1 = UAV_traces[i][0]/1000
        y1 = UAV_traces[i][1]/1000
        x2 = UGV_traces[UAV_traces[i+1]][0]/1000 + x_shift
        y2 = UGV_traces[UAV_traces[i+1]][1]/1000
        route_labels = {(x1, y1): node_labels[UAV_traces[i]],
                        (x2, y2): node_labels[UGV_traces[UAV_traces[i+1]]]}
        x, y = hanging_line([x1, y1], [x2, y2])
        axs.plot(x, 
                 y, color='r', linewidth=lw, linestyle='--')
        for rl in route_labels:
            axs.text(rl[0], rl[1], route_labels[rl]+'\'', fontsize = 'x-small')
        uv = np.array([x[index+1]-x[index], y[index+1]-y[index]])/np.linalg.norm(np.array([x[index+1]-x[index], y[index+1]-y[index]]))
        axs.arrow(x[index], y[index], 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color='r', length_includes_head=False)




#axs.axis('equal') 
handles = [Line2D([0], [0], color='b', lw=4, label='Forward'), 
           Line2D([0], [0], color='r', lw=4, label='Rendezvous detour', linestyle='--'),
           Line2D([0], [0], marker='*', color='w', label='Rendezvous point',
                                 markerfacecolor='k', markersize=11)
           ]
#[mpatches.Patch(color='k', label='rendezvous point')]
axs.legend(loc = "upper right", handles=handles, fontsize = 6)
axs.set_aspect('equal', 'box') 
axs.set_xlabel('x  (km)')
axs.set_ylabel('y  (km)')
axs.set_xlim([0, 21])
axs.set_title(r'The route of UGV when UAV executes $\pi$')    
fig.savefig(UGV_fig_name , bbox_inches='tight')
            
'''       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
'''
for i in range(len(UAV_traces)-1):    
    if type(UAV_traces[i]) == int:
        if type(UAV_traces[i+1]) == int:
            path = nx.shortest_path(road_network, source=UGV_traces[UAV_traces[i]], target=UGV_traces[UAV_traces[i+1]])
            if UAV_traces[i] < road_end_index-1:
                x = [node[0]/1000 - 1 for node in path]
                y = [node[1]/1000 for node in path]
                route_labels = [node_labels[node] for node in path]
            elif UAV_traces[i] == road_end_index-1:
                x = [node[0]/1000 - 1 for node in path]
                x[-1] += 1
                y = [node[1]/1000 for node in path]
                route_labels = [node_labels[node] for node in path]
            elif UAV_traces[i] == road_end_index:
                x = [node[0]/1000 + 2 for node in path]
                x[0] -= 2
                y = [node[1]/1000 for node in path]
                route_labels = [node_labels[node] for node in path]
                for j in range(1, len(path)):
                    route_labels[j] += '\'' 
            else:
                x = [node[0]/1000 + 2 for node in path]
                y = [node[1]/1000 for node in path]
                route_labels = [node_labels[node] for node in path]
                for j in range(1, len(path)):
                    route_labels[j] += '\''
                
            x1 = UGV_traces[UAV_traces[i]][0]/1000
            y1 = UGV_traces[UAV_traces[i]][1]/1000
            x2 = UGV_traces[UAV_traces[i+1]][0]/1000
            y2 = UGV_traces[UAV_traces[i+1]][1]/1000
            axs.plot(x, y, color='b', marker='o', linewidth=lw, markersize=1)
            print("x is {}".format(x))
            for j in range(len(x)):
                axs.text(x[j], y[j], route_labels[j], fontsize = 'x-small')   
            uv = np.array([x2-x1, y2-y1])/np.linalg.norm(np.array([x2-x1, y2-y1]))
            #axs.arrow((x1+x2)/2, (y1+y2)/2, 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color='b', length_includes_head=True)
        else:
            if UAV_traces[i] < road_end_index:
                x1 = UGV_traces[UAV_traces[i]][0]/1000 - 1
                y1 = UGV_traces[UAV_traces[i]][1]/1000
                x2 = UAV_traces[i+1][0]/1000
                y2 = UAV_traces[i+1][1]/1000
                route_labels = {(x1, y1): node_labels[UGV_traces[UAV_traces[i]]],
                                (x2, y2): node_labels[UAV_traces[i+1]]}
                
            elif UAV_traces[i] == road_end_index:
                x1 = UGV_traces[UAV_traces[i]][0]/1000
                y1 = UGV_traces[UAV_traces[i]][1]/1000
                x2 = UAV_traces[i+1][0]/1000 + 2
                y2 = UAV_traces[i+1][1]/1000
                route_labels = {(x1, y1): node_labels[UGV_traces[UAV_traces[i]]],
                                (x2, y2): node_labels[UAV_traces[i+1]]+'\''}
            else:
                x1 = UGV_traces[UAV_traces[i]][0]/1000 + 2
                y1 = UGV_traces[UAV_traces[i]][1]/1000
                x2 = UAV_traces[i+1][0]/1000 + 2 
                y2 = UAV_traces[i+1][1]/1000
                route_labels = {(x1, y1): node_labels[UGV_traces[UAV_traces[i]]],
                                (x2, y2): node_labels[UAV_traces[i+1]]+'\''}
            
            axs.plot(x2, y2, marker='*', color='k', markersize=8, alpha=0.8)
            for rl in route_labels:
                axs.text(rl[0], rl[1], route_labels[rl], fontsize = 'x-small')
            
            x, y = hanging_line([x1, y1], [x2, y2])
            index = round(len(x)/2)
            axs.plot(x, y, color='r', linewidth=lw, linestyle='--')
            axs.plot(x1, y1, color='b', marker='o', linewidth=lw, markersize=1)         
            uv = np.array([x[index+1]-x[index], y[index+1]-y[index]])/np.linalg.norm(np.array([x[index+1]-x[index], y[index+1]-y[index]]))
            axs.arrow(x[index], y[index], 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color='r', length_includes_head=False)
    else:  
        assert type(UAV_traces[i+1]) == int
        
        if UAV_traces[i+1] < road_end_index:
            x1 = UAV_traces[i][0]/1000
            y1 = UAV_traces[i][1]/1000
            x2 = UGV_traces[UAV_traces[i+1]][0]/1000 - 1
            y2 = UGV_traces[UAV_traces[i+1]][1]/1000
            route_labels = {(x1, y1): node_labels[UAV_traces[i]],
                            (x2, y2): node_labels[UGV_traces[UAV_traces[i+1]]]}
        elif UAV_traces[i+1] == road_end_index:
            x1 = UAV_traces[i][0]/1000
            y1 = UAV_traces[i][1]/1000
            x2 = UGV_traces[UAV_traces[i+1]][0]/1000
            y2 = UGV_traces[UAV_traces[i+1]][1]/1000
            route_labels = {(x1, y1): node_labels[UAV_traces[i]],
                            (x2, y2): node_labels[UGV_traces[UAV_traces[i+1]]]+'\''}
        else:
            x1 = UAV_traces[i][0]/1000 + 2
            y1 = UAV_traces[i][1]/1000
            x2 = UGV_traces[UAV_traces[i+1]][0]/1000 + 2
            y2 = UGV_traces[UAV_traces[i+1]][1]/1000
            route_labels = {(x1, y1): node_labels[UAV_traces[i]],
                            (x2, y2): node_labels[UGV_traces[UAV_traces[i+1]]]+'\''}
        x, y = hanging_line([x1, y1], [x2, y2])
        axs.plot(x, 
                 y, color='r', linewidth=lw, linestyle='--')
        for rl in route_labels:
            axs.text(rl[0], rl[1], route_labels[rl], fontsize = 'x-small')
        uv = np.array([x[index+1]-x[index], y[index+1]-y[index]])/np.linalg.norm(np.array([x[index+1]-x[index], y[index+1]-y[index]]))
        axs.arrow(x[index], y[index], 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color='r', length_includes_head=False)
'''




                          


