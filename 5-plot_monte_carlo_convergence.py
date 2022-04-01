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
    
    i = 0
    while (UAV_state != UAV_goal and state != state_f):
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
            index = probs.index(max(probs))
            action = actions[index]
            #action = np.random.choice(actions, 1, p=probs)[0]
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
            duration_traces.append(t1+t2+rendezvous.charging_time)
        else:
            duration = UAV_task.edges[UAV_state, UAV_state_next]['dis'] / rendezvous.velocity_uav[action]
            duration_traces.append(duration)
        
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
    
    
    
    UAV_traces = []
    UGV_traces = []
    battery_traces = []
    for i in range(len(action_traces)):
        UAV_traces.append(state_traces[i][0])
        UGV_traces.append((state_traces[i][1], state_traces[i][2]))
        battery_traces.append(state_traces[i][3])
        UAV_traces.append(action_traces[i])
    
    UAV_traces.append(state_traces[i+1][0])
    UGV_traces.append((state_traces[i+1][1], state_traces[i+1][2]))  
    battery_traces.append(state_traces[i+1][3])
    
    #print("UAV traces: {}".format(UAV_traces))
    #print("UGV traces: {}".format(UGV_traces))  
    #print("battery traces: {}".format(battery_traces))
    
    return  state_traces, action_traces, duration_traces
        

def simulate_rendezvous_baseline(greedy_threshold, P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous):
    state = state_init
    UAV_state = state[0]
    UGV_state = (state[1], state[2])
    #energy_state = state[3]/100*rendezvous.battery
    UGV_task_node = state[4]
    state_traces = [state]
    action_traces = []
    duration_traces = []
    
    i = 0
    while (UAV_state != UAV_goal and state != state_f):
        actions = []
        probs = []
        for action in P_s_a[state]:
            actions.append(action)
            s_a = state + (action,)
            probs.append(policy[s_a])
        
        #print("step {}".format(i))
        #print("state {} policy {}".format(state, dict(zip(actions, probs))))
        # sample 
        #action = np.random.choice(actions, 1, p=probs)[0]
        #print("take action {}".format(action))
        
        if len(actions) == 1 and ('l' in actions):
            action = 'l'
            print("state is {}".format(state))
        else:
            
            if state[3] < greedy_threshold:
                #action = np.random.choice(['v_br_br', 'v_be_be'], 1)[0]
                action = 'v_br_br'
            else:
                action = 'v_br'
            
            '''
            action = 'v_br'
            for nstate in P_s_a[state][action]:
                if (nstate==state_f) and (P_s_a[state][action][nstate] > 0.1):
                    action = np.random.choice(['v_br_br'], 1)[0]
                    break
                if (nstate==state_l) and (P_s_a[state][action][nstate] > 0.5):
                    break
                #print(nstate[3])
                #print(P_s_a[state][action][nstate])
                if (nstate[3] < 70) and (P_s_a[state][action][nstate] > 0.5):
                    if state_f in P_s_a[nstate]['v_br_br']:
                        if P_s_a[nstate]['v_br_br'][state_f]>0.2:
                            action = np.random.choice(['v_br_br'], 1)[0]
                    break
            '''
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
            duration_traces.append(t1+t2+rendezvous.charging_time)
        else:
            duration = UAV_task.edges[UAV_state, UAV_state_next]['dis'] / rendezvous.velocity_uav[action]
            duration_traces.append(duration)
        
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
    
    
    
    UAV_traces = []
    UGV_traces = []
    battery_traces = []
    for i in range(len(action_traces)):
        UAV_traces.append(state_traces[i][0])
        UGV_traces.append((state_traces[i][1], state_traces[i][2]))
        battery_traces.append(state_traces[i][3])
        UAV_traces.append(action_traces[i])
    
    UAV_traces.append(state_traces[i+1][0])
    UGV_traces.append((state_traces[i+1][1], state_traces[i+1][2]))  
    battery_traces.append(state_traces[i+1][3])
    
    #print("UAV traces: {}".format(UAV_traces))
    #print("UGV traces: {}".format(UGV_traces))  
    #print("battery traces: {}".format(battery_traces))
    
    return  state_traces, action_traces, duration_traces
                


if __name__ == "__main__":
    experiment_name = '_risk_tolerance'
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
    rendezvous = Rendezvous(UAV_task, UGV_task, road_network, battery=280e3)
    #state_traces, action_traces, duration_traces = simulate_rendezvous(P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous)
    #state_traces, action_traces, duration_traces = simulate_rendezvous_baseline(P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous)


    
    # number of Monte Carlo 
    #MC = [100, 500, 1000, 5000]
    '''
    MC = [500, 1000, 3000, 5000, 8000, 10000]
    successes = dict(zip(MC, [0]*len(MC)))
    threshold = 0.1
    target_directory = os.path.join(current_directory, r'policy')
    policy_name = os.path.join(target_directory, 'policy'+str(threshold)+experiment_name+'.obj')
    with open(policy_name, 'rb') as f:  # Python 3: open(..., 'rb')
        policy = pickle.load(f)
    for mc in MC:
        for i in range(mc):
            state_traces, action_traces, duration_traces = simulate_rendezvous(P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous)
            UAV_state = state_traces[-1][0]
            if UAV_state == UAV_goal and state_traces[-1] != state_f:
                successes[mc] += 1
    print(successes)
    PF = [(mc-successes[mc])/mc for mc in successes]
    print("failure probability is {}".format(PF))
    
    # use this result to create a form
    PF_threshold = 0.1
    Error = [abs((mc-successes[mc])/mc - PF_threshold)/PF_threshold for mc in successes]
    PF = [(mc-successes[mc])/mc for mc in successes]
    KL = [np.log(pf/PF_threshold)*pf + np.log((1-pf)/(1-PF_threshold))*(1-pf) for pf in PF]
    Prob_success = [successes[mc]/mc for mc in successes]
    print(Prob_success)
    '''
    
    
    # increase this to a higher value for the final results 
    mc = 2000
    #thresholds =  [0.01, 0.02, 0.03]+[i/100 for i in range(5, 50, 5)]
    thresholds = [i/100 for i in range(5, 30, 5)]
    successes = dict(zip(thresholds, [0]*len(thresholds)))
    durations = dict(zip(thresholds, []*len(thresholds)))
    durations_success = dict(zip(thresholds, []*len(thresholds)))
    
    for threshold in thresholds:
        target_directory = os.path.join(current_directory, r'policy')
        policy_name = os.path.join(target_directory, 'policy'+str(threshold)+experiment_name+'.obj')
        with open(policy_name, 'rb') as f:  # Python 3: open(..., 'rb')
            policy = pickle.load(f)
            
            durations[threshold] = []
            durations_success[threshold] = []
            
            for i in range(mc):
                state_traces, action_traces, duration_traces = simulate_rendezvous(P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous)
                durations[threshold].append(np.sum(np.array(duration_traces)))
                
                UAV_state = state_traces[-1][0]
                if UAV_state == UAV_goal and state_traces[-1] != state_f:
                    successes[threshold] += 1
                    durations_success[threshold].append(np.sum(np.array(duration_traces)))
                    
    
    # put the following information in a table
    prob_success = dict(zip(thresholds, [successes[threshold]/mc for threshold in successes]))
    
     
    # CMDP statistics
    durations_mean = [np.mean(durations[threshold]) for threshold in thresholds]
    durations_std = [np.std(durations[threshold]) for threshold in thresholds]
    success_durations_mean = [np.mean(durations_success[threshold]) for threshold in thresholds]
    success_durations_std = [np.std(durations_success[threshold]) for threshold in thresholds]
   
    
    greedy_thresholds = [40, 50, 60, 70]
    durations_baseline_mean = {}
    durations_baseline_std = {}
    success_baseline_mean = {}
    success_baseline_std = {}
    success_baseline_prob = {}
    for greedy_threshold in greedy_thresholds:
        # coompute baseline
        success_baseline = 0
        durations_baseline = []
        durations_success_baseline = []
        
        for i in range(mc):
            state_traces, action_traces, duration_traces = simulate_rendezvous_baseline(greedy_threshold, P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous)
            durations_baseline.append(np.sum(np.array(duration_traces)))
            
            UAV_state = state_traces[-1][0]
            if UAV_state == UAV_goal and state_traces[-1] != state_f:
                #print("UAV reaches the goal using baseline policy")
                success_baseline += 1
                durations_success_baseline.append(np.sum(np.array(duration_traces)))
          
        # baseline statistics
        durations_baseline_mean[greedy_threshold] = np.mean(durations_baseline)
        durations_baseline_std[greedy_threshold] = np.std(durations_baseline)
        success_baseline_mean[greedy_threshold] = np.mean(durations_success_baseline)
        success_baseline_std[greedy_threshold] = np.std(durations_success_baseline)
        success_baseline_prob[greedy_threshold] = success_baseline/mc
    
    
    target_directory = os.path.join(current_directory, r'comparison_results')
    if not os.path.exists(target_directory):
       os.makedirs(target_directory)   
    # compare with baseline
    fig, axs = plt.subplots()
    axs1 = axs.twinx()
    width = 0.2
    threshold = 0.1
    x = np.array([0, 1.5, ])
    y_CMDP = [np.mean(durations[threshold]), np.mean(durations_success[threshold])]
    axs.bar(x-width-width, y_CMDP, width, color='r')
    y_baseline = [durations_baseline_mean[40], success_baseline_mean[40]]
    axs.bar(x-width, y_baseline, width, color='g')
    y_baseline = [durations_baseline_mean[50], success_baseline_mean[50]]
    axs.bar(x, y_baseline, width, color='b')
    y_baseline = [durations_baseline_mean[60], success_baseline_mean[60]]
    axs.bar(x+width, y_baseline, width, color='m')
    y_baseline = [durations_baseline_mean[70], success_baseline_mean[70]]
    axs.bar(x+width+width, y_baseline, width, color='orange')
    
    axs.set_xticks(x, ['task duration', 'success \n task duration'])
   
    axs.set_ylabel('Time (second)')
    axs.set_ylim(2000, 10000)
    
    
    x=3
    axs1.bar(x-width-width, prob_success[threshold], width, color='green')
    axs1.bar(x-width, success_baseline_prob[40], width)
    axs1.bar(x, success_baseline_prob[50], width)
    axs1.bar(x+width, success_baseline_prob[60], width)
    axs1.bar(x+width+width, success_baseline_prob[70], width)
    
    axs1.set_xticks([0, 1.5, 3], ['task duration', 'success \n task duration', "success probability"])
    axs1.set_ylabel("Prob of reaching the goal")
    axs1.set_ylim(0, 1)
    axs1.legend(["CMDP", "Greedy-40", "Greedy-50", "Greedy-60", "Greedy-70"])
    axs1.set_title(r"Results comparison for $\delta=0.1$")
    fig_name = os.path.join(target_directory, "comparison"+str(threshold)+experiment_name+".pdf")
    fig.savefig(fig_name, bbox_inches='tight')
    
    
    # plot task duration and conditional task duration
    fig, axs = plt.subplots()
    axs.plot(thresholds, durations_mean, '-', color='g', label='Avg task duration')
    #axs.plot(thresholds, success_durations_mean, '-', color='r', label='task duration|success')
    axs.fill_between(thresholds, np.array(durations_mean)-0.5*np.array(durations_std), np.array(durations_mean)+0.5*np.array(durations_std), color='g', alpha=0.2)
    #axs.fill_between(thresholds, np.array(success_durations_mean)-0.5*np.array(success_durations_std), np.array(success_durations_mean)+0.5*np.array(success_durations_std), color='r', alpha=0.2)
    axs.set_title("Risk tolerance vs task duration Pareto Curve")
    axs.set_xlabel(r'Risk tolerance $\delta$')
    axs.set_ylabel('Task duration (seconds)')
    axs.legend()
    fig_name = os.path.join(target_directory, "task_duration"+experiment_name+".pdf")
    fig.savefig(fig_name, bbox_inches='tight')
    
    
    fig, axs = plt.subplots()
    axs.plot(thresholds, durations_mean, '-', color='g', label='task duration')
    axs.fill_between(thresholds, np.array(durations_mean)-0.5*np.array(durations_std), np.array(durations_mean)+0.5*np.array(durations_std), color='g', alpha=0.2)
    axs.set_title("UAV flight duration")
    axs.set_xlabel("cost threshold")
    axs.set_ylabel("flight duration in seconds")
    fig_name = os.path.join(target_directory, "task_duration_unconditional"+str(threshold)+experiment_name+".pdf")
    fig.savefig(fig_name, bbox_inches='tight')
    
    
    
    
    
    
    
    
    
    
    
    
    # statistical data for objective gap 
    mc = 2000
    thresholds =  [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    successes = dict(zip(thresholds, [0]*len(thresholds)))
    durations = dict(zip(thresholds, []*len(thresholds)))
    durations_success = dict(zip(thresholds, []*len(thresholds)))
    
    for threshold in thresholds:
        target_directory = os.path.join(current_directory, r'policy')
        policy_name = os.path.join(target_directory, 'policy'+str(threshold)+experiment_name+'.obj')
        with open(policy_name, 'rb') as f:  # Python 3: open(..., 'rb')
            policy = pickle.load(f)
            
            durations[threshold] = []
            durations_success[threshold] = []
            
            for i in range(mc):
                state_traces, action_traces, duration_traces = simulate_rendezvous(P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous)
                durations[threshold].append(np.sum(np.array(duration_traces)))
                
                UAV_state = state_traces[-1][0]
                if UAV_state == UAV_goal and state_traces[-1] != state_f:
                    successes[threshold] += 1
                    durations_success[threshold].append(np.sum(np.array(duration_traces)))
                    
    
    # put the following information in a table
    prob_success = dict(zip(thresholds, [successes[threshold]/mc for threshold in successes]))
    
    # CMDP statistics
    durations_mean = [np.mean(durations[threshold]) for threshold in thresholds]
    durations_std = [np.std(durations[threshold]) for threshold in thresholds]
    success_durations_mean = [np.mean(durations_success[threshold]) for threshold in thresholds]
    success_durations_std = [np.std(durations_success[threshold]) for threshold in thresholds]
    
    
    target_directory = os.path.join(current_directory, r'comparison_results')
    if not os.path.exists(target_directory):
       os.makedirs(target_directory)   
    fig, axs = plt.subplots()
    gap = [abs(success_durations_mean[i]-durations_mean[i])/success_durations_mean[i] for i in range(len(success_durations_mean))]
    dict(zip(thresholds, gap))
    gap = [round(g, 2)*100 for g in gap]
    axs.plot(thresholds[0:7], gap[0:7], '-', color='g', label='task duration')
    axs.set_title(r'Objective gap $\frac{ \overline{T}_{success} -  \overline{T}_{all}}{ \overline{T}_{success}}$')
    axs.set_xlabel("cost threshold")
    axs.set_ylabel("Percentage")
    fig_name = os.path.join(target_directory, "flight_duration_gap"+experiment_name+".pdf")
    fig.savefig(fig_name, bbox_inches='tight')
    
    
    # plot task duration and conditional task duration
    fig, axs = plt.subplots()
    axs.plot(thresholds, durations_mean, '-', color='g', label='Avg task duration')
    #axs.plot(thresholds, success_durations_mean, '-', color='r', label='task duration|success')
    axs.fill_between(thresholds, np.array(durations_mean)-0.3*np.array(durations_std), np.array(durations_mean)+0.5*np.array(durations_std), color='g', alpha=0.2)
    #axs.fill_between(thresholds, np.array(success_durations_mean)-0.5*np.array(success_durations_std), np.array(success_durations_mean)+0.5*np.array(success_durations_std), color='r', alpha=0.2)
    axs.set_title("Risk tolerance vs task duration Pareto Curve")
    axs.set_xlabel(r'Risk tolerance $\delta$')
    axs.set_ylabel('Task duration (seconds)')
    axs.set_ylim(7000, 11600)
    #axs.set_aspect('equal') 
    axs.legend()
    fig_name = os.path.join(target_directory, "task_duration"+experiment_name+".pdf")
    fig.savefig(fig_name, bbox_inches='tight')
    
    
    
    
    
    
    
    
    
    
    