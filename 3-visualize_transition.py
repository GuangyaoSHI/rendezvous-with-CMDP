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
import os

experiment_name = '_risk_level_example'
data_name_appendix = '-3'
threshold=0.5
print("do visualization for {} with threshold {}".format(experiment_name, threshold))

# file names to get transition information 
current_directory = os.getcwd()
target_directory = os.path.join(current_directory, r'transition_information')
P_s_a_name = os.path.join(target_directory, 'P_s_a'+experiment_name+'.obj')
transition_graph_name = os.path.join(target_directory, 'state_transition_graph'+experiment_name+'.obj')
# Getting back the transition information:
with open(P_s_a_name , 'rb') as f:  # Python 3: open(..., 'rb')
    P_s_a = pickle.load(f)
with open(transition_graph_name, 'rb') as f:  # Python 3: open(..., 'rb')
    G = pickle.load(f)

        
# Getting back the policy:
target_directory = os.path.join(current_directory, r'policy')
policy_name = os.path.join(target_directory, 'policy'+str(threshold)+experiment_name+'.obj')
with open(policy_name, 'rb') as f:  # Python 3: open(..., 'rb')
    policy = pickle.load(f)
    

# set directory and file name to save the data 
target_directory = os.path.join(current_directory, r'visualization')
if not os.path.exists(target_directory):
   os.makedirs(target_directory)
data_name = os.path.join(target_directory, 'visualization_data-'+str(threshold)+experiment_name+data_name_appendix+'.obj')
fig_name = os.path.join(target_directory, "trace"+experiment_name+".pdf")
     

state_f = ('f', 'f', 'f', 'f', 'f')
state_l = ('l', 'l', 'l', 'l', 'l')    
state_init = (0, int(6.8e3), int(19.1e3), 100, 0)


# generate state transition function
UAV_task = generate_UAV_task()
UAV_goal = [x for x in UAV_task.nodes() if (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==1) or (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==0)]
UAV_goal = UAV_goal[0]
# UGV_task is a directed graph. Node name is an index
UGV_task = generate_UGV_task()
road_network = generate_road_network()
actions = ['v_be', 'v_br', 'v_be_be', 'v_br_br']
colors = dict(zip(actions, ['red', 'g', 'b', 'm']))
rendezvous = Rendezvous(UAV_task, UGV_task, road_network, battery=240e3)


if experiment_name == '_velocity_comparison1':
    rendezvous.velocity_ugv = 5
    rendezvous.velocity_uav = {'v_be' : 7.5, 'v_br' : 7.5}

if experiment_name == '_velocity_comparison2':
    rendezvous.velocity_ugv = 5
    rendezvous.velocity_uav = {'v_be' : 10, 'v_br' : 10}    
    
if experiment_name == '_velocity_comparison3':
    rendezvous.velocity_ugv = 5
    rendezvous.velocity_uav = {'v_be' : 14, 'v_br' : 14} 


state = state_init
UAV_state = state[0]
UGV_state = (state[1], state[2])
energy_state = state[3]/100*rendezvous.battery
UGV_task_node = state[4]
state_traces = [state]
action_traces = []
duration_traces = []
UAV_traces = []



i = 0

fig, axs = plt.subplots()
# plot roadnetwork
'''
add_label = True
for edge in road_network.edges:
    x = [edge[0][0], edge[1][0]]
    y = [edge[0][1], edge[1][1]]
    if add_label:
        axs.plot(x, y, color='b', alpha=0.4, label='road network')
        add_label = False
    else:
        axs.plot(x, y, color='b', alpha=0.4)
'''

add_label = True
# plot UAV task
hw = 200
w = 0.1

'''
for edge in UAV_task.edges:
    x, y = UAV_task.nodes[edge[0]]['pos'][0], UAV_task.nodes[edge[0]]['pos'][1] 
    u, v = UAV_task.nodes[edge[1]]['pos'][0]-UAV_task.nodes[edge[0]]['pos'][0], UAV_task.nodes[edge[1]]['pos'][1]-UAV_task.nodes[edge[0]]['pos'][1] 
    if add_label:
        #axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=0.3, label="UAV task")
        axs.arrow(x, y, u, v, head_width = hw, width = w, alpha=0.5, color='k', label="UAV task", linestyle=(5, (3,6)), length_includes_head=True)
        add_label = False
    else:
        #axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=0.3)
        axs.arrow(x, y, u, v, head_width = hw, width = w, alpha=0.5, color='k', linestyle=(5, (3,6)), length_includes_head=True)
'''

j=0
for node in UAV_task.nodes:
    axs.text(UAV_task.nodes[node]['pos'][0], UAV_task.nodes[node]['pos'][1], UAV_task.nodes[node]['label'])
    axs.plot(UAV_task.nodes[node]['pos'][0], UAV_task.nodes[node]['pos'][1], marker='.', color='k', markersize=2)
    j += 1

axs.axis('equal')      
#axs.set_aspect('equal', 'box')  
rendezvous_label = True
while (UAV_state != UAV_goal and state != state_f):
    UAV_traces.append(UAV_state)
    actions = []
    probs = []
    for action in P_s_a[state]:
        actions.append(action)
        s_a = state + (action,)
        probs.append(policy[s_a])
    
    print("step {}".format(i))
    print("state {} policy {}".format(state, dict(zip(actions, probs))))
    # sample 
    if np.sum(np.array(probs))<1:
        print("-------------sum of prob is less than 1 : {}--------".format(probs))
        #action = np.random.choice(actions, 1)[0]
        action = np.random.choice(actions, 1)[0]
    #print("step {}".format(i))
    #print("state {} policy {}".format(state, dict(zip(actions, probs))))
    # sample
    else:
        action = np.random.choice(actions, 1, p=probs)[0]
    print("take action {}".format(action))
    action_traces.append(action)
    
    # compute UAV position after rendezvous
    descendants = list(UAV_task.neighbors(UAV_state))
    assert len(descendants) == 1
    UAV_state_next = descendants[0]
    if len(action)>4:
        ugv_road_state = UGV_state + UGV_state
        v1 = action[0:4]
        v2 = 'v'+action[4:]
        rendezvous_state, t1, t2 = rendezvous.rendezvous_point(UAV_task.nodes[UAV_state]['pos'], 
                                                               UAV_task.nodes[UAV_state_next]['pos'],  
                                                               UGV_state, 
                                                               ugv_road_state, UGV_task_node, 
                                                               rendezvous.velocity_uav[v1], 
                                                               rendezvous.velocity_uav[v2])
        print("rendezvous at {}!!!!!!!!!!!!!".format(rendezvous_state))
        UAV_traces.append(rendezvous_state)
        duration_traces.append(t1+t2+rendezvous.charging_time)
    else:
        duration = UAV_task.edges[UAV_state, UAV_state_next]['dis'] / rendezvous.velocity_uav[action]
        duration_traces.append(duration)
    
    #UAV_traces.append(UAV_state_next)
    # Todo: 
    next_states = list(P_s_a[state][action].keys())
    '''
    assert np.sum(np.array(list(P_s_a[state][action].values()))) == 1
    if np.sum(np.array(list(P_s_a[state][action].values())))<1:
        next_state_index = np.random.choice([i for i in range(len(next_states))], 1)[0]
    else:
        next_state_index = np.random.choice([i for i in range(len(next_states))], 1, p=list(P_s_a[state][action].values()))[0]
    '''
    next_state_index = np.random.choice([i for i in range(len(next_states))], 1, p=list(P_s_a[state][action].values()))[0]
    next_state = next_states[next_state_index]
    print("transit to state {}".format(next_state))
    
    '''
    fig, axs = plt.subplots()
    # plot roadnetwork
    add_label = True
    for edge in road_network.edges:
        x = [edge[0][0], edge[1][0]]
        y = [edge[0][1], edge[1][1]]
        if add_label:
            axs.plot(x, y, marker='*', color='b', alpha=0.2, label='road network')
            add_label = False
        else:
            axs.plot(x, y, marker='*', color='b', alpha=0.2)
    
    add_label = True
    # plot UAV task
    for edge in UAV_task.edges:
        x, y = edge[0][0], edge[0][1] 
        u, v = edge[1][0]-edge[0][0], edge[1][1]-edge[0][1] 
        if add_label:
            axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=0.3, label="UAV task")
            add_label = False
        else:
            axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=0.3)

    # plot UAV UGV position now
    axs.plot(UAV_state[0], UAV_state[1], marker='o', color='r', label="UAV now")
    
    #plt.text(UAV_state[0], UAV_state[1], 'UAV now', horizontalalignment='left')
    axs.plot(UGV_state[0], UGV_state[1], marker='s', color='g', label="UGV now")
    #plt.text(UGV_state[0], UGV_state[1], 'UGV now', horizontalalignment='right')
    # plot UAV transition
    axs.legend()
    axs.set_title("policy:"+str(dict(zip(actions, probs))))
    axs.set_xlabel("battery:"+str(state[4]))
    #axs.set_aspect('equal', 'box')
    axs.axis('equal')
    fig.savefig("step"+str(i)+".pdf")
    '''
    
      
    


    # plot UAV UGV position now
    #axs.plot(UAV_state[0], UAV_state[1], marker='o', color='r', label="UAV now")
    
    #plt.text(UAV_state[0], UAV_state[1], 'UAV now', horizontalalignment='left')
    #axs.plot(UGV_state[0], UGV_state[1], marker='s', color='g', label="UGV now")
    #plt.text(UGV_state[0], UGV_state[1], 'UGV now', horizontalalignment='right')
    # plot UAV transition
    
    '''
    axs.set_title("policy:"+str(dict(zip(actions, probs))))
    axs.set_xlabel("choose action "+str(action) + "       battery:"+str(state[4]))
    #axs.set_aspect('equal', 'box')
    '''
    hw = 150
    w = 0.05
    if len(action)>4:
        x, y = UAV_task.nodes[UAV_state]['pos'][0], UAV_task.nodes[UAV_state]['pos'][1]
        u, v = rendezvous_state[0] - UAV_task.nodes[UAV_state]['pos'][0], rendezvous_state[1] - UAV_task.nodes[UAV_state]['pos'][1] 
        #axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=1, color='r')
        if rendezvous_label:
            axs.arrow(x, y, u*0.9, v*0.9, head_width = hw, width = w,  color='r', length_includes_head=True, label='rendezvous detour')
            axs.plot(rendezvous_state[0], rendezvous_state[1], marker='*', color='m', markersize=4, alpha=0.8, label='rendezvous point')
            rendezvous_label = False
        else:
            axs.plot(rendezvous_state[0], rendezvous_state[1], marker='*', color='m', markersize=4, alpha=0.8)
            axs.arrow(x, y, u*0.9, v*0.9, head_width = hw, width = w,  color='r', length_includes_head=True)

        #axs.text(rendezvous_state[0], rendezvous_state[1], 'rendezvous here')
        
        x, y = rendezvous_state[0], rendezvous_state[1]
        u, v = UAV_task.nodes[UAV_state_next]['pos'][0]-rendezvous_state[0], UAV_task.nodes[UAV_state_next]['pos'][1]-rendezvous_state[1]
        #axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=1, color='r')
        axs.arrow(x, y, u, v, head_width = hw, width = w,  color='r', length_includes_head=True)
    else:
        if next_state == state_f:
            print("\n --------------!!!!!!!!UAV fails !!!!!!---------\n")
            break
        else:
            UAV_state_next = next_state[0]
        x, y = UAV_task.nodes[UAV_state]['pos'][0], UAV_task.nodes[UAV_state]['pos'][1]
        u, v = UAV_task.nodes[UAV_state_next]['pos'][0] - UAV_task.nodes[UAV_state]['pos'][0], UAV_task.nodes[UAV_state_next]['pos'][1] - UAV_task.nodes[UAV_state]['pos'][1] 
        #axs.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=1, color='r')
        #axs.arrow(x, y, u, v, head_width = hw, width = w)
    
    
    i += 1
    state_traces.append(next_state)
    state = next_state
    UAV_state = state[0]
    UGV_state = (state[1], state[2])
    UGV_task_node = state[4]


if UAV_state == UAV_goal:
    assert UAV_state not in UAV_traces
    assert state in state_traces
    UAV_traces.append(UAV_state)




X = []
rendezvous_num = 0
for uav in UAV_traces:
    if type(uav)== int:
        X.append(UAV_task.nodes[uav]['pos'][0])
    else:
        X.append(uav[0])
        rendezvous_num += 1

Y = []
for uav in UAV_traces:
    if type(uav)== int:
        Y.append(UAV_task.nodes[uav]['pos'][1])
    else:
        Y.append(uav[1])


print("-------number of rendezvous is {}---------".format(rendezvous_num))
#axs.plot(X, Y, color='r')
axs.set_xlabel('x Position (m)')
axs.set_ylabel('y Position (m)')
axs.set_title('UAV task and rendezvous detour')
axs.legend()

#fig.savefig(fig_name, bbox_inches='tight')

print("duration {}".format(np.sum(np.array(duration_traces))))


battery_traces = []
UGV_task_traces =[]
UGV_traces = []
for i in range(len(state_traces)):
    #UAV_traces.append((state_traces[i][0], state_traces[i][1]))
    UGV_traces.append((state_traces[i][1], state_traces[i][2]))
    battery_traces.append(state_traces[i][3])
    UGV_task_traces.append(state_traces[i][4])
    #UAV_traces.append(action_traces[i])

data = {'UAV_trace':UAV_traces, 'UGV_trace':UGV_traces, 'action_trace':action_traces, 
        'UGV_task_trace':UGV_task_traces, 'duration_trace':duration_traces, 'state_trace':state_traces,
        'battery_trace':battery_traces}

#UAV_traces.append((state_traces[i+1][0], state_traces[i+1][1]))
#UGV_traces.append((state_traces[i+1][2], state_traces[i+1][3]))  
#battery_traces.append(state_traces[i+1][4])

print("UAV traces: {}".format(UAV_traces))
print("UGV traces: {}".format(UGV_traces))  
print("battery traces: {}".format(battery_traces))
    

with open(data_name, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(data, f)        
    

'''
fig, axs = plt.subplots()
hw = 150
w = 0.05
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
                     [y1, y2], color=colors[action], linestyle='--', linewidth=0.5)
            uv = np.array([x2-x1, y2-y1])/np.linalg.norm(np.array([x2-x1, y2-y1]))
            axs.arrow((x1+x2)/2, (y1+y2)/2, 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color=colors[action], length_includes_head=True)
        else:
            assert type(UAV_traces[i+1]) == int, "should be a task node"
            x1 = UAV_task.nodes[UAV_traces[i]]['pos'][0]/1000
            x2 = UAV_task.nodes[UAV_traces[i+1]]['pos'][0]/1000
            y1 = UAV_task.nodes[UAV_traces[i]]['pos'][1]/1000
            y2 = UAV_task.nodes[UAV_traces[i+1]]['pos'][1]/1000
            axs.plot([x1, x2], 
                     [y1, y2], color=colors[action], linewidth=0.5)
            uv = np.array([x2-x1, y2-y1])/np.linalg.norm(np.array([x2-x1, y2-y1]))

            axs.arrow((x1+x2)/2, (y1+y2)/2, 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color=colors[action], length_includes_head=True)

    else:
        assert type(UAV_traces[i-1]) == int, "previous node should be a task node"
        assert type(UAV_traces[i+1]) == int, "next node should be a task node"
        action = action_traces[UAV_traces[i-1]]
        assert len(action)>4, "should be a rendezvous action"
        x1 = UAV_traces[i][0]/1000
        x2 = UAV_task.nodes[UAV_traces[i+1]]['pos'][0]/1000
        y1 = UAV_traces[i][1]/1000
        y2 = UAV_task.nodes[UAV_traces[i+1]]['pos'][1]/1000
        axs.plot([x1, x2], 
                 [y1, y2], color=colors[action], linestyle='--', linewidth=0.5)
        uv = np.array([x2-x1, y2-y1])/np.linalg.norm(np.array([x2-x1, y2-y1]))
        axs.arrow((x1+x2)/2, (y1+y2)/2, 0.01*uv[0], 0.01*uv[1], head_width = hw, width = w, color=colors[action], length_includes_head=True)

        
    
j=0
for node in UAV_task.nodes:
    axs.text(UAV_task.nodes[node]['pos'][0]/1000, UAV_task.nodes[node]['pos'][1]/1000, UAV_task.nodes[node]['label'])
    axs.plot(UAV_task.nodes[node]['pos'][0]/1000, UAV_task.nodes[node]['pos'][1]/1000, marker='.', color='k', markersize=2)
    j += 1
axs.axis('equal')  

handles = []
labels = dict(zip(actions, [r'move forward with $v_{be}$', r'move forward with $v_{br}$', r'rendezvous with $v_{be}$', r'rendezvous with $v_{br}$']))
for action in actions:
    lengend_v = mpatches.Patch(color=colors[action], label=labels[action])
    handles.append(lengend_v)
axs.legend(handles=handles)    
axs.set_aspect('equal', 'box') 
axs.set_xlabel('x  (km)')
axs.set_ylabel('y  (km)')
axs.set_title(r'A sample route of UAV with policy $\pi$')    
fig.savefig("sample_route"+experiment_name+".pdf", bbox_inches='tight')
'''