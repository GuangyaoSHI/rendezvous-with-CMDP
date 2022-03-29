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
import re

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import os


def simulate_rendezvous(P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous):
    state = state_init
    UAV_state = state[0]
    UGV_state = (state[1], state[2])
    #energy_state = state[3]/100*rendezvous.battery
    UGV_task_node = state[4]
    state_traces = [state]
    action_traces = []
    duration_traces = []
    
    UAV_traces = []
    
    i = 0
    while (UAV_state != UAV_goal and state != state_f):
        UAV_traces.append(UAV_state)
        actions = []
        probs = []
        for action in P_s_a[state]:
            actions.append(action)
            s_a = state + (action,)
            if not (type(policy[s_a]) == float or type(policy[s_a]) == int):
                print("s_a : {} policy : {}".format(s_a, policy[s_a]))
            probs.append(policy[s_a])
        
        #probs = list(np.array(probs)/np.sum(np.array(probs)))
        if np.sum(np.array(probs))<1:
            #print("sum of prob is less than 1 : {}".format(probs))
            action = np.random.choice(actions, 1)[0]
        #print("step {}".format(i))
        #print("state {} policy {}".format(state, dict(zip(actions, probs))))
        # sample
        else:
            action = np.random.choice(actions, 1, p=probs)[0]
        #print("take action {}".format(action))
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
            #print("rendezvous at {}!!!!!!!!!!!!!".format(rendezvous_state))
            UAV_traces.append(rendezvous_state)
            duration_traces.append(t1+t2+rendezvous.charging_time)
        else:
            duration = UAV_task.edges[UAV_state, UAV_state_next]['dis'] / rendezvous.velocity_uav[action]
            duration_traces.append(duration)
        
        #UAV_traces.append(UAV_state_next)
        # Todo: 
        next_states = list(P_s_a[state][action].keys())
        next_state_index = np.random.choice([i for i in range(len(next_states))], 1, p=list(P_s_a[state][action].values()))[0]
        next_state = next_states[next_state_index]
        #print("transit to state {}".format(next_state))
        
        i += 1
        state_traces.append(next_state)
        state = next_state
        UAV_state = state[0]
        UGV_state = (state[1], state[2])
        UGV_task_node = state[4]
    
    if UAV_state == UAV_goal:
        assert UAV_state not in UAV_traces
        UAV_traces.append(UAV_state)
    
    
    #print("UAV traces: {}".format(UAV_traces))
    #print("UGV traces: {}".format(UGV_traces))  
    #print("battery traces: {}".format(battery_traces))
    
    return  state_traces, action_traces, duration_traces, UAV_traces
        

if __name__ == "__main__":
    # '_risk_level_comparison'     '_velocity_comparison3'  '_risk_level_example'
    experiment_name = '_velocity_comparison3'
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
     
    state_f = ( 'f', 'f', 'f', 'f', 'f')
    state_l = ( 'l', 'l', 'l', 'l', 'l')    
    state_init = (0, int(6.8e3), int(19.1e3), 100, 0)
    
    # generate state transition function
    # remember to change this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    UAV_task = generate_UAV_task(option='long')
    UAV_goal = [x for x in UAV_task.nodes() if (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==1) or (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==0)]
    UAV_goal = UAV_goal[0]
    print("UAV task is {} and goal is {}".format(UAV_task.nodes, UAV_goal))

    # UGV_task is a directed graph. Node name is an index
    UGV_task = generate_UGV_task()
    road_network = generate_road_network()
    actions = ['v_be', 'v_br', 'v_be_be', 'v_br_br']
    rendezvous = Rendezvous(UAV_task, UGV_task, road_network, battery=240e3)
    #state_traces, action_traces, duration_traces = simulate_rendezvous(P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous)
    #state_traces, action_traces, duration_traces = simulate_rendezvous_baseline(P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous)
    if experiment_name == '_velocity_comparison1':
        rendezvous.velocity_ugv = 5
        rendezvous.velocity_uav = {'v_be' : 7.5, 'v_br' : 7.5}
    
    if experiment_name == '_velocity_comparison2':
        rendezvous.velocity_ugv = 5
        rendezvous.velocity_uav = {'v_be' : 10, 'v_br' : 10}    
        
    if experiment_name == '_velocity_comparison3':
        rendezvous.velocity_ugv = 5
        rendezvous.velocity_uav = {'v_be' : 14, 'v_br' : 14} 

    
    # increase this to a higher value for the final results 
    mc = 2000
    thresholds =  [0.1]
    successes = dict(zip(thresholds, [0]*len(thresholds)))
    durations = dict(zip(thresholds, []*len(thresholds)))
    durations_success = dict(zip(thresholds, []*len(thresholds)))
    UAV_travel_distance = dict(zip(thresholds, []*len(thresholds)))
    UAV_travel_distance_success = dict(zip(thresholds, []*len(thresholds)))
    
    UAV_rendezvous_times = dict(zip(thresholds, []*len(thresholds)))
    UAV_rendezvous_times_success = dict(zip(thresholds, []*len(thresholds)))

    for threshold in thresholds:
        target_directory = os.path.join(current_directory, r'policy')
        policy_name = os.path.join(target_directory, 'policy'+str(threshold)+experiment_name+'.obj')
        with open(policy_name, 'rb') as f:  # Python 3: open(..., 'rb')
            policy = pickle.load(f)
        
        durations[threshold] = []
        durations_success[threshold] = []
        UAV_travel_distance[threshold] = []
        UAV_travel_distance_success[threshold] = [] 
        UAV_rendezvous_times[threshold] = [] 
        UAV_rendezvous_times_success[threshold] = []
        
        for i in range(mc):
            #print("Round {} MC".format(i))
            state_traces, action_traces, duration_traces, UAV_traces= simulate_rendezvous(P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous)
            durations[threshold].append(np.sum(np.array(duration_traces)))
            
            # UAV travel distance
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
            
                    
            assert len(X) == len(Y)
            travel_dis = 0
            for i in range(len(X)-1):
                travel_dis += np.linalg.norm(np.array([X[i+1], Y[i+1]])-np.array([X[i], Y[i]]))
            UAV_travel_distance[threshold].append(travel_dis)   
            UAV_rendezvous_times[threshold].append(rendezvous_num)
            
            UAV_state = state_traces[-1][0]
            if UAV_state == UAV_goal and state_traces[-1] != state_f:
                successes[threshold] += 1
                durations_success[threshold].append(np.sum(np.array(duration_traces)))
                UAV_travel_distance_success[threshold].append(travel_dis)
                UAV_rendezvous_times_success[threshold].append(rendezvous_num)

                
            
    # put the following information in a table
    prob_success = dict(zip(thresholds, [successes[threshold]/mc for threshold in successes]))
    print("empirical prob of success: {}".format(prob_success))
     
    # CMDP statistics
    durations_mean = [np.mean(durations[threshold]) for threshold in thresholds][0]
    durations_std = [np.std(durations[threshold]) for threshold in thresholds]
    success_durations_mean = [np.mean(durations_success[threshold]) for threshold in thresholds][0]
    success_durations_std = [np.std(durations_success[threshold]) for threshold in thresholds]
    UAV_travel_distance_mean = [np.mean(UAV_travel_distance[threshold]) for threshold in thresholds][0]
    UAV_travel_distance_success_mean = [np.mean(UAV_travel_distance_success[threshold]) for threshold in thresholds][0]
    UAV_rendezvous_times_mean = [np.mean(UAV_rendezvous_times[threshold]) for threshold in thresholds][0]
    UAV_rendezvous_times_success_mean = [np.mean(UAV_rendezvous_times_success[threshold]) for threshold in thresholds][0]
    
    
    # compute the travel time of UAV if it can fly with best range speed all the time
    tour_dis = 0
    for i in range(len(UAV_task.nodes)-1):
        tour_dis += np.linalg.norm(np.array(UAV_task.nodes[i]['pos'])-np.array(np.array(UAV_task.nodes[i+1]['pos'])))
    
    print("average task duration for {} is  {}".format(experiment_name, durations_mean))
    tour_time = tour_dis/rendezvous.velocity_uav['v_br']
    print("UAV tour time with best range speed is {}".format(tour_time))
    print("UAV travel time overhead is {}".format((durations_mean-tour_time)/tour_time))
    print("average travel distance for {} is {}".format(experiment_name, UAV_travel_distance_mean))
    print("UAV tour distance is {}".format(tour_dis))
    print("UAV travel distance overhead is {}".format((UAV_travel_distance_mean-tour_dis)/tour_dis))
    print("average number of rendezvous times is {}".format(UAV_rendezvous_times_mean))
    print("\n")
    
    
    print("average success task duration for {} is  {}".format(experiment_name, success_durations_mean))
    print("UAV success travel time overhead is {}".format((success_durations_mean-tour_time)/tour_time))
    print("average success travel distance for {} is {}".format(experiment_name, UAV_travel_distance_success_mean))
    print("UAV tour distance is {}".format(tour_dis))
    print("UAV success travel distance overhead is {}".format((UAV_travel_distance_success_mean-tour_dis)/tour_dis))
    print("\n")
    print("average number of rendezvous times (success) is {}".format(UAV_rendezvous_times_success_mean))
    

    
   
    
    
    