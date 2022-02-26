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
            duration_traces.append(t1+t2)
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
        

def simulate_rendezvous_baseline(P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous):
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
            if state[4] < 50:
                action = 'v_br_br'
            else:
                action = 'v_br'
            
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
            duration_traces.append(t1+t2)
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
    
    PF_threshold = 0.1
    
    experiment_name = '0.1'
    # Getting back the objects:
    with open('P_s_a'+experiment_name+'.obj', 'rb') as f:  # Python 3: open(..., 'rb')
        P_s_a = pickle.load(f)
        
    # Getting back the objects:
    with open('policy'+experiment_name+'.obj', 'rb') as f:  # Python 3: open(..., 'rb')
        policy = pickle.load(f)

    # Getting back the objects:
    with open('state_transition_graph'+experiment_name+'.obj', 'rb') as f:  # Python 3: open(..., 'rb')
        G = pickle.load(f)
        
    state_f = ( 'f', 'f', 'f', 'f', 'f')
    state_l = ( 'l', 'l', 'l', 'l', 'l')    
    state_init = (0, int(6.8e3), int(19.1e3), 100, 0)
    
    # generate state transition function
    UAV_task = generate_UAV_task()
    UAV_goal = [x for x in UAV_task.nodes() if (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==1) or (UAV_task.out_degree(x)==0 and UAV_task.in_degree(x)==0)]
    UAV_goal = UAV_goal[0]
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
    for mc in MC:
        for i in range(mc):
            state_traces, action_traces, duration_traces = simulate_rendezvous(P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous)
            UAV_state = (state_traces[-1][0], state_traces[-1][1])
            if UAV_state == UAV_goal and state_traces[-1] != state_f:
                successes[mc] += 1
    print(successes)
    
    
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
    thresholds = [i/100 for i in range(5, 50, 5)]
    successes = dict(zip(thresholds, [0]*len(thresholds)))
    durations = dict(zip(thresholds, []*len(thresholds)))
    durations_success = dict(zip(thresholds, []*len(thresholds)))
    
    for threshold in thresholds:
        with open('policy'+str(threshold)+'.obj', 'rb') as f:  # Python 3: open(..., 'rb')
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
    
    
    # coompute baseline
    success_baseline = 0
    durations_baseline = []
    durations_success_baseline = []
    
    for i in range(mc):
        state_traces, action_traces, duration_traces = simulate_rendezvous_baseline(P_s_a, policy, G, state_l, state_f, state_init, UAV_task, UAV_goal, UGV_task, road_network, actions, rendezvous)
        durations_baseline.append(np.sum(np.array(duration_traces)))
        
        UAV_state = state_traces[-1][0]
        if UAV_state == UAV_goal and state_traces[-1] != state_f:
            success_baseline += 1
            durations_success_baseline.append(np.sum(np.array(duration_traces)))
      
    # baseline statistics
    durations_baseline_mean = np.mean(durations_baseline)
    durations_baseline_std = np.std(durations_baseline)
    success_baseline_mean = np.mean(durations_success_baseline)
    success_baseline_std = np.std(durations_success_baseline)
    
    
    # CMDP statistics
    durations_mean = [np.mean(durations[threshold]) for threshold in thresholds]
    durations_std = [np.std(durations[threshold]) for threshold in thresholds]
    success_durations_mean = [np.mean(durations_success[threshold]) for threshold in thresholds]
    success_durations_std = [np.std(durations_success[threshold]) for threshold in thresholds]
    
    # compare with baseline
    fig, axs = plt.subplots()
    axs1 = axs.twinx()
    # axs.plot(thresholds, np.ones([len(thresholds), ])*durations_baseline_mean, '-', color='k', label='baseline')
    # axs.plot(thresholds, durations_mean, '-', color='g', label='CMDP')
    # axs.plot(thresholds, np.ones([len(thresholds),])*success_baseline_mean, '-', color='m', label='baseline-success')
    # axs.plot(thresholds, success_durations_mean, '-', color='r', label='CMDP-success')
    
    
    # axs.fill_between(thresholds,  durations_mean-0.5*np.ones([len(thresholds), ])*durations_baseline_std,
    #                  durations_mean+0.5*np.ones([len(thresholds), ])*durations_baseline_std, color='k', alpha=0.2)

    # axs.fill_between(thresholds, success_baseline_mean-0.5*np.ones([len(thresholds), ])*success_baseline_std,
    #                  success_baseline_mean+0.5*np.ones([len(thresholds), ])*success_baseline_std, color='m', alpha=0.2)
      
    # axs.fill_between(thresholds, np.array(durations_mean)-0.5*np.array(durations_std), np.array(durations_mean)+0.5*np.array(durations_std), color='g', alpha=0.2)
    
    # axs.fill_between(thresholds, np.array(success_durations_mean)-0.5*np.array(success_durations_std), np.array(success_durations_mean)+0.5*np.array(success_durations_std), color='r', alpha=0.2)
    # axs.set_title("UAV flight duration")
    # axs.legend()
    # https://www.geeksforgeeks.org/create-a-grouped-bar-plot-in-matplotlib/
    width = 0.4
    threshold = 0.1
    x = np.arange(2)
    y_baseline = [durations_baseline_mean, success_baseline_mean]
    y_CMDP = [np.mean(durations[threshold]), np.mean(durations_success[threshold])]
    axs.bar(x-width/2, y_baseline, width, color='orange')
    axs.bar(x+width/2, y_CMDP, width, color='green')
    axs.set_xticks(x, ['task duration', 'success \n task duration'])
   
    axs.set_ylabel('Flight duration')
    axs.set_ylim(1500, 4000)
    
    
    x=2
    axs1.bar(x-width/2, success_baseline/mc, width, color='orange')
    axs1.bar(x+width/2, prob_success[threshold], width, color='green')
    axs1.set_xticks([0, 1, 2], ['task duration', 'success \n task duration', "success probability"])
    axs1.set_ylabel("Prob of reaching the goal")
    axs1.set_ylim(0.4, 1)
    axs1.legend(["Greedy", "CMDP"])
    axs1.set_title("Results comparison for c=0.1")
    fig.savefig("comparison"+experiment_name+".pdf", bbox_inches='tight')
    
    
    # plot task duration and conditional task duration
    fig, axs = plt.subplots()
    axs.plot(thresholds, durations_mean, '-', color='g', label='task duration')
    axs.plot(thresholds, success_durations_mean, '-', color='r', label='task duration|success')
    axs.fill_between(thresholds, np.array(durations_mean)-0.5*np.array(durations_std), np.array(durations_mean)+0.5*np.array(durations_std), color='g', alpha=0.2)
    axs.fill_between(thresholds, np.array(success_durations_mean)-0.5*np.array(success_durations_std), np.array(success_durations_mean)+0.5*np.array(success_durations_std), color='r', alpha=0.2)
    axs.set_title("UAV flight duration")
    axs.legend()
    fig.savefig("task_duration"+experiment_name+".pdf", bbox_inches='tight')
    
    
    fig, axs = plt.subplots()
    axs.plot(thresholds, durations_mean, '-', color='g', label='task duration')
    axs.fill_between(thresholds, np.array(durations_mean)-0.5*np.array(durations_std), np.array(durations_mean)+0.5*np.array(durations_std), color='g', alpha=0.2)
    axs.set_title("UAV flight duration")
    axs.set_xlabel("cost threshold")
    axs.set_ylabel("flight duration in seconds")
    fig.savefig("task_duration_unconditional"+experiment_name+".pdf", bbox_inches='tight')
    
    
    
    fig, axs = plt.subplots()
    gap = [abs(success_durations_mean[i]-durations_mean[i])/success_durations_mean[i] for i in range(len(success_durations_mean))]
    gap = [round(g*100) for g in gap]
    axs.plot(thresholds, gap, '-', color='g', label='task duration')
    axs.set_title("UAV flight duration gap")
    axs.set_xlabel("cost threshold")
    axs.set_ylabel("Percentage")
    fig.savefig("flight_duration_gap"+experiment_name+".pdf", bbox_inches='tight')
    
    
    
    
    
    
    
    
    
    
    