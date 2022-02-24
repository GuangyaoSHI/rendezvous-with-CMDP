# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:59:38 2022

@author: gyshi
"""

import numpy as np 
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import random
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import copy
import sys
import csv
import pickle

class Rendezvous():
    def __init__(self, UAV_task, UGV_task, road_network, battery):
        # a sequence of nodes in 2D plane, represented as a directed graph
        self.UAV_task = UAV_task 
        
        # a sequence of nodes in road network, represented as a directed graph
        # rendezvous action may change the task, some nodes may be inserted into tha task 
        self.UGV_task = UGV_task 
        self.UGV_goal = UGV_task.graph['UGV_goal']
        #self.power_measure = UAV_task.graph['dis_per_energy']
        
        # road network can be viewed as a fine discretization of 2D continuous road network
        self.road_network = road_network 
        self.road_network_ = nx.Graph(road_network)
        # uav velocity
        self.velocity_uav = {'v_be' : 9.8, 'v_br' : 14}
        
        # ugv velocity
        self.velocity_ugv = 4.5

        self.check_UGV_task()
        
        # time for changing battery
        # Todo: choose a proper value 
        self.charging_time = 300
        
        # total battery
        # Todo:choose a proper value 
        self.battery = battery
        # power consumption
        # heading = wind heading angle m/s, randomly sampled unless provided by user
        self.wind_heading = np.random.rand()*2*np.pi
        # meanW = average weight in kg
        self.meanW = 2.3
        # stdW = standard deviation of weight in kg
        self.stdW = 0.05
        # aG = characteristic velocity for
        self.aG = 1.5
        # bG = shape function of veloicty distribution, from https://wind-data.ch/tools/weibull.php
        self.bG = 3
        # coefficient
        self.b = [-88.7661477109457,3.53178719017177,-0.420567520590965,0.0427521866683907,107.473389967445,-2.73619087492112]
        # display transition
        self.display = True
        
        # normal distribution parameters
        # mean
        self.mean = 0
        # variance
        self.sigma = 0.1

                
    def check_UGV_task(self):
        # check whether each task point is in road network
        for node in self.UGV_task.nodes:
            state = self.UGV_task.nodes[node]['pos']
            assert state in self.road_network.nodes, "UGV task node not in road network"
        
    
        
    def transit(self, state, action, UGV_road_state, UGV_task_node):
        # return probability distribution P(s' | s, a)
        # state = (xa, ya, xg, yg, SoC)
        # action: {'v_be', 'v_br', 'v_be_be', 'v_be_br', 'v_br_be', 'v_br_br'}
        # UGV_state_road (x1, y1, x2, y2): UGV is transiting from (x1, y1) to (x2, y2) on road network
        # UGV_task_node: the task node in UGV_task that UGV is moving towards
        UAV_state, UGV_state, battery_state = self.get_states(state)
        UAV_state_next = []
        UGV_state_next = []
        UGV_road_state_next = []
        # the task node that UGV should move towards after transition
        UGV_task_node_next = []
        battery_state_next = []
        
        if action in ['v_be', 'v_br']:
            # UAV choose to go to next task node with best endurance velocity
            descendants = list(self.UAV_task.neighbors(UAV_state))
            assert len(descendants) == 1
            UAV_state_next = descendants[0]
            # compute next state for UGV
            duration = self.UAV_task.edges[UAV_state, UAV_state_next]['dis']/self.velocity_uav[action]
            UGV_state_next, UGV_road_state_next, UGV_task_node_next = self.UGV_transit(UGV_state, UGV_road_state, UGV_task_node, duration)
            # Todo should return power distribution
            #power_consumed = self.power_consumption(self.velocity_uav[action], duration)
            power_consumed = self.get_power_consumption_sample(self.UAV_task.edges[UAV_state, UAV_state_next]['dis'])
            battery_state_next = battery_state - power_consumed
            if self.display:
                self.display_task_transition(UAV_state, UAV_state_next, UGV_state, UGV_state_next, int(battery_state/self.battery*100), int(battery_state_next/self.battery*100))
            # if battery_state_next < 0:
            #    print("out of battery")
            #    UAV_state_next = ('f', 'f')
            #    UGV_state_next = ('f', 'f')
            #    UGV_road_state_next = ('f', 'f', 'f', 'f')
            #    UGV_task_node_next = 'f'
            #    battery_state_next = 'f'
            #    return UAV_state_next, UGV_state_next, UGV_road_state_next, UGV_task_node_next, battery_state_next
        
            
        if action in ['v_be_be', 'v_be_br', 'v_br_be', 'v_br_br']:
            v1 = action[0:4]
            v2 = 'v'+action[4:]
            # UAV choose to go to next task node with best endurance velocity
            descendants = list(self.UAV_task.neighbors(UAV_state))
            assert len(descendants) == 1
            UAV_state_next = descendants[0]
            # compute rendezvous point and time
            rendezvous_state, t1, t2 = self.rendezvous_point(UAV_state, UAV_state_next, 
                                               UGV_state, UGV_road_state, UGV_task_node, self.velocity_uav[v1], self.velocity_uav[v2])
            
            #Todo: compute the state of UGV after t2 and return
            #print("rendezvous state is {}".format(rendezvous_state))
            assert rendezvous_state in self.road_network.nodes
            # since rendezvous_state is on road network, the following code should be fine
            rendezvous_road_state = rendezvous_state + rendezvous_state 
            UGV_state_next, UGV_road_state_next, UGV_task_node_next = self.UGV_transit(rendezvous_state, rendezvous_road_state, UGV_task_node, t2)
            
            
            # power consumed for rendezvous 
            dis1 = np.linalg.norm(np.array(UAV_state)-np.array(rendezvous_state))
            power_consumed1 = self.get_power_consumption_sample(dis1)
            dis2 = np.linalg.norm(np.array(UAV_state_next)-np.array(rendezvous_state))
            power_consumed2 = self.get_power_consumption_sample(dis2)
            battery_state_next = battery_state - power_consumed1
            if self.display:
                self.display_rendezvous(rendezvous_state, UAV_state, UAV_state_next, 
                                    UGV_state, UGV_state_next, battery_state, 
                                   int((self.battery - power_consumed2)/self.battery*100), 
                                   int((battery_state - power_consumed1)/self.battery*100))
            # UAV cannot rendezvous
            if battery_state_next < 0:
                UAV_state_next = ('f', 'f')
                UGV_state_next = ('f', 'f')
                UGV_road_state_next = ('f', 'f', 'f', 'f')
                UGV_task_node_next = 'f'
                battery_state_next = 'f'
                return UAV_state_next, UGV_state_next, UGV_road_state_next, UGV_task_node_next, battery_state_next

            
            # UAV cannot go back to next task node
            battery_state_next = self.battery - power_consumed2
            if battery_state_next < 0:
                UAV_state_next = ('f', 'f')
                UGV_state_next = ('f', 'f')
                UGV_road_state_next = ('f', 'f', 'f', 'f')
                UGV_task_node_next = 'f'
                battery_state_next = 'f'
                return UAV_state_next, UGV_state_next, UGV_road_state_next, UGV_task_node_next, battery_state_next
            
        return UAV_state_next, UGV_state_next, UGV_road_state_next, UGV_task_node_next, battery_state_next
            
    
    def UGV_transit(self, UGV_state, UGV_road_state, UGV_task_node, duration):
        # UGV_road_state = (x1, y1, x2, y2)
        # this information helps to locate UGV in the road network
        # Todo: check UGV_state is indeed between two task nodes
        
        # temporarily insert UGV_state into road network
        change_road_network = False
        previous_road_state = (UGV_road_state[0], UGV_road_state[1])
        assert previous_road_state in self.road_network.nodes
        next_road_state = (UGV_road_state[2], UGV_road_state[3])
        assert next_road_state in self.road_network.nodes
        if UGV_state not in [previous_road_state, next_road_state]:
            self.road_network.remove_edge(previous_road_state, next_road_state)
            dis = np.linalg.norm(np.array(previous_road_state)-np.array(UGV_state))
            self.road_network.add_edge(previous_road_state, UGV_state, dis=dis)
            dis = np.linalg.norm(np.array(next_road_state)-np.array(UGV_state))
            self.road_network.add_edge(UGV_state, next_road_state, dis=dis)
            change_road_network = True
        assert UGV_state in self.road_network.nodes    
        
        # UGV will move duration * velocity distance along the task path
        total_dis = self.velocity_ugv * duration
        
        # task node after moving t=duration
        node_before_stop = UGV_task_node
        state_before_stop = self.UGV_task.nodes[node_before_stop]['pos']
        dis = nx.shortest_path_length(self.road_network, source=UGV_state, target=state_before_stop, weight='dis')
        
        # identify the task node before which UGV will stop
        while (dis < total_dis):
            #print("node before stop: {}".format(node_before_stop))
            descendant_task = list(self.UGV_task.neighbors(node_before_stop))[0]
            descendant_task_state = self.UGV_task.nodes[descendant_task]['pos']
            next_two_task_dis = nx.shortest_path_length(self.road_network, source=state_before_stop, target=descendant_task_state, weight='dis')
            dis += next_two_task_dis
            node_before_stop = descendant_task
            state_before_stop = self.UGV_task.nodes[node_before_stop]['pos']
        
        previous_node = list(self.UGV_task.predecessors(node_before_stop))
        if previous_node:
            previous_node_state = self.UGV_task.nodes[previous_node[0]]['pos'] 
        else:
            previous_node_state = self.UGV_task.nodes[node_before_stop]['pos']
        
        path = nx.shortest_path(self.road_network, source=previous_node_state, target=state_before_stop)
        # update task node for UGV
        UGV_task_node_next = node_before_stop 
        # back tracking to identify UGV's position
        L = len(path)
        node_index = L-1
        sp_dis = 0
        # from last to first in the path list
        for i in range(0, L):
            node_index = (L-1) - i;
            # shortest path length from source to this node
            # Todo: this is very inefficient 
            sp_dis = nx.shortest_path_length(self.road_network, source=path[node_index], target=state_before_stop, weight='dis')
            dis_ = dis - sp_dis
            if dis_ <= total_dis:
                break
            
        # recovering road network
        if change_road_network:
            self.road_network = nx.Graph(self.road_network_)
        
        previous_road_state = path[node_index]
        if node_index < len(path)-1:
            next_road_state = path[node_index+1]
            vector =  np.array(next_road_state) - np.array(previous_road_state)
            vector = vector/np.linalg.norm(vector)
            #assert dis>= total_dis
            UGV_state_next = tuple(np.array(previous_road_state)+(total_dis-dis_)*vector)
        else:
            next_road_state = path[node_index]
            UGV_state_next = path[node_index]
        UGV_state_road_next = previous_road_state + next_road_state
        return UGV_state_next, UGV_state_road_next, UGV_task_node_next
    
    def get_power_consumption_sample(self, distance):
        stochastic_consumption = np.random.normal(self.mean, self.sigma)
        if stochastic_consumption < -0.25:
            stochastic_consumption = -0.25
        if stochastic_consumption > 0.25:
            stochastic_consumption = 0.25
        return distance/self.power_measure + stochastic_consumption 
        
    def power_consumption(self, tgtV, duration):
        # return power distribution after taking action with SoC
        # sample weight using random normal distribution of weight
        W = self.stdW*np.random.randn() + self.meanW
        
        # solve for true airspeed by adding in weibull wind distribution with
        # equally random heading direction of wind
        # simplifying assumption is that only wind tangential to UAS heading affects power
        disturbance = weibull_min.rvs(c=self.aG, loc=0, scale=self.bG)
        V = abs(tgtV + disturbance*np.math.cos(-self.wind_heading))
        P = self.b[0] + self.b[1]*V + self.b[2]*V**2 + self.b[3]*V**3 + self.b[4]*W + self.b[5]*V*W
        return P*duration
    
    def get_power_consumption_distribution(self, tgtV):
        # return power distribution when the UAV flies at a target velocity
        stats = {}
        num_of_samples = 0
        samples = 100000
        for i in range(100, 300, 10):
            stats[(i, i+10)] = 0
        for i in range(samples):
            # sample weight using random normal distribution of weight
            W = self.stdW*np.random.randn() + self.meanW
            # solve for true airspeed by adding in weibull wind distribution with
            # equally random heading direction of wind
            # simplifying assumption is that only wind tangential to UAS heading affects power
            disturbance = weibull_min.rvs(c=self.aG, loc=0, scale=self.bG)
            V = abs(tgtV + disturbance*np.math.cos(-self.wind_heading))
            P = self.b[0] + self.b[1]*V + self.b[2]*V**2 + self.b[3]*V**3 + self.b[4]*W + self.b[5]*V*W
            for interval in stats:
                if interval[0] <= P < interval[1]:
                    stats[interval] += 1
                    num_of_samples += 1
                    break
        if num_of_samples != samples:
            print("some samples are out of range")
        
        for interval in stats:
            stats[interval] = stats[interval] / num_of_samples
        # https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        keys = list(stats.keys())
        keys = [str(key) for key in keys]
        ax.barh(keys, list(stats.values()))
        #plt.xticks(rotation='vertical')
        ax.set_title("power consumption distribution for velocity "+str(tgtV))
        
        # Saving the state-transition graph:
        with open('P_'+str(tgtV)+'.obj', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(stats, f)
        return stats
    
    
    def rendezvous_point(self, UAV_state, UAV_state_next, UGV_state, UGV_road_state, UGV_task_node, vel_rdv, vel_sep):
        # return rendezvous point
        # temporarily insert UGV_state into road network
        previous_road_state = (UGV_road_state[0], UGV_road_state[1])
        assert previous_road_state in self.road_network.nodes
        next_road_state = (UGV_road_state[2], UGV_road_state[3])
        assert next_road_state in self.road_network.nodes
        
        # all candidate nodes for rendezvous
        candidate_nodes = list(self.road_network_.nodes)
        if UGV_state not in [previous_road_state, next_road_state]:
            self.road_network.remove_edge(previous_road_state, next_road_state)
            dis = np.linalg.norm(np.array(previous_road_state)-np.array(UGV_state))
            self.road_network.add_edge(previous_road_state, UGV_state, dis=dis)
            dis = np.linalg.norm(np.array(next_road_state)-np.array(UGV_state))
            self.road_network.add_edge(UGV_state, next_road_state, dis=dis)
        assert UGV_state in self.road_network.nodes
        
        rendezvous_state = UGV_state
        dis1 = np.linalg.norm(np.array(UAV_state) - np.array(rendezvous_state))
        dis2 = np.linalg.norm(np.array(UAV_state_next) - np.array(rendezvous_state))
        # time taken to rendezvous
        rendezvous_time1 = max(dis1/vel_rdv, 
                              nx.shortest_path_length(self.road_network, source=UGV_state, 
                                                      target=rendezvous_state, weight='dis')/self.velocity_ugv)
        # time taken to go back to task
        rendezvous_time2 = dis2/vel_sep
        # Todo: add charging time
        rendezvous_time = sys.maxsize
        
        # Todo: make this more efficient
        for node in candidate_nodes:
            dis1 = np.linalg.norm(np.array(UAV_state) - np.array(node))
            dis2 = np.linalg.norm(np.array(UAV_state_next) - np.array(node))
            time1 = max(dis1/vel_rdv, 
                                  nx.shortest_path_length(self.road_network, source=UGV_state, 
                                                          target=node, weight='dis')/self.velocity_ugv)
            time2 = dis2/vel_sep
            time = time1 + time2 + self.charging_time
            if time < rendezvous_time:
                rendezvous_state = node
                rendezvous_time = time
                rendezvous_time1 = time1
                rendezvous_time2 = time2
        
        # recovering road network
        self.road_network = nx.Graph(self.road_network_)
        # update UGV task 
        # Todo: consider the influence of rendezvous to UGV's task
        return rendezvous_state, rendezvous_time1, rendezvous_time2
    
    def get_states(self, state):
        # state = (xa, ya, xg, yg, SoC)
        UAV_state = (state[0], state[1])
        assert UAV_state in self.UAV_task, "UAV state is not in task"
        UGV_state = (state[2], state[3])
        battery_state = state[4]
        return UAV_state, UGV_state, battery_state
    
    
    def display_task_transition(self, UAV_state_last, UAV_state_next, UGV_state_last, UGV_state_next, battery_state_last, battery_state_next):
        # plot road network
        fig, ax = plt.subplots()
        for edge in self.road_network_.edges:
            node1 = edge[0]
            node2 = edge[1]
            x = [node1[0], node2[0]]
            y = [node1[1], node2[1]]
            line_road, = ax.plot(x, y, marker='o',color='k', alpha=0.2)
        line_road.set_label('road network')
        
        # plot UAV task
        for edge in self.UAV_task.edges:
            node1 = edge[0]
            node2 = edge[1]
            x, y = node1[0], node1[1]
            dx, dy = node2[0]-node1[0], node2[1]-node1[1]
            line_UAV = ax.quiver(x, y, dx, dy, scale_units='xy', angles='xy', scale=1, alpha=0.2, color='g')
        # plot UAV transition
        x, y = UAV_state_last[0], UAV_state_last[1]
        dx, dy = UAV_state_next[0]-UAV_state_last[0],  UAV_state_next[1]-UAV_state_last[1]
        ax.quiver(x, y, dx, dy, scale_units='xy', angles='xy', scale=1, alpha=1, color='r')
        ax.text(x+0.5*dx, y+0.5*dy, 'UAV from '+str((int(UAV_state_last[0]), int(UAV_state_last[1])))
                +' to '+str((int(UAV_state_next[0]), int(UAV_state_next[1]))))
        # battery state
        ax.set_title("battery state from " + str(battery_state_last)+" to " + str(battery_state_next))
        
        # plot UGV transition
        x, y = UGV_state_last[0], UGV_state_last[1]
        dx, dy = UGV_state_next[0]-UGV_state_last[0],  UGV_state_next[1]-UGV_state_last[1]
        ax.quiver(x, y, dx, dy, scale_units='xy', angles='xy', scale=1, alpha=1, color='b')
        ax.text(x, y, 'UGV from '+ str((int(UGV_state_last[0]), int(UGV_state_last[1]))) 
                + ' to '+ str((int(UGV_state_next[0]), int(UGV_state_next[1]))))
        ax.legend()
        #fig.savefig("task_transition.pdf")
        
    def display_rendezvous(self, rendezvous_node, UAV_state_last, UAV_state_next, 
                           UGV_state_last, UGV_state_next, battery_state_last, 
                           battery_state_next, battery_rendezvous):
        # plot road network
        fig, ax = plt.subplots()
        for edge in self.road_network_.edges:
            node1 = edge[0]
            node2 = edge[1]
            x = [node1[0], node2[0]]
            y = [node1[1], node2[1]]
            line_road, = ax.plot(x, y, marker='o',color='k', alpha=0.2)
        
        line_road.set_label('road network')
        # plot UAV task
        for edge in self.UAV_task.edges:
            node1 = edge[0]
            node2 = edge[1]
            x, y = node1[0], node1[1]
            dx, dy = node2[0]-node1[0], node2[1]-node1[1]
            line_UAV = ax.quiver(x, y, dx, dy, scale_units='xy', angles='xy', scale=1, alpha=0.2, color='g')
        # plot UAV transition to rendezvous node 
        x, y = UAV_state_last[0], UAV_state_last[1]
        dx, dy = rendezvous_node[0]-UAV_state_last[0],  rendezvous_node[1]-UAV_state_last[1]
        ax.quiver(x, y, dx, dy, scale_units='xy', angles='xy', scale=1, alpha=1, color='r')
        ax.text(x+0.5*dx, y+0.5*dy, 'UAV rendezvous:'+str(UAV_state_last)+' to '+str(rendezvous_node))
        
        x, y = rendezvous_node[0], rendezvous_node[1]
        dx, dy = UAV_state_next[0]-rendezvous_node[0],  UAV_state_next[1]-rendezvous_node[1]
        ax.quiver(x, y, dx, dy, scale_units='xy', angles='xy', scale=1, alpha=1, color='r')
        ax.text(x+0.5*dx, y+0.5*dy, 'UAV from '+str(rendezvous_node)+' to '+str(UAV_state_next))
        # battery state
        ax.set_title("Battery:"+str(int(battery_state_last))+" to "+ str(int(battery_rendezvous))+" to "+str(int(battery_state_next)))
        
        
        # plot UGV transition to rendezvous node 
        x, y = UGV_state_last[0], UGV_state_last[1]
        dx, dy = rendezvous_node[0]-UGV_state_last[0],  rendezvous_node[1]-UGV_state_last[1]
        ax.quiver(x, y, dx, dy, scale_units='xy', angles='xy', scale=1, alpha=1, color='b')
        #ax.text(x+0.5*dx, y+100, 'UGV rendezvous:'+str(UGV_state_last)+' to '+str(rendezvous_node))
        
        x, y = rendezvous_node[0], rendezvous_node[1]
        dx, dy = UGV_state_next[0]-rendezvous_node[0],  UGV_state_next[1]-rendezvous_node[1]
        ax.quiver(x, y, dx, dy, scale_units='xy', angles='xy', scale=1, alpha=1, color='b')
        #ax.text(x+0.5*dx, y-100, 'UGV from '+str(rendezvous_node)+' to '+str(UGV_state_next))
        ax.set_xlabel("UGV:"+str((int(UGV_state_last[0]), int(UGV_state_last[1])))+" to "+
                      str((int(rendezvous_node[0]), int(rendezvous_node[1])))+
                      " to "+str((int(UGV_state_next[0]), int(UGV_state_next[1]))))
        #fig.savefig("rendezvous.pdf")
    
    

def generate_road_network():
    # it shouldn't be a directed graph
    #G = nx.Graph()
    # # a simple straight line network
    # for i in range(1, 20):
    #     dis = np.linalg.norm(np.array(((i-1)*5+3731, 0))-np.array((i*5+3731, 0)))
    #     G.add_edge(((i-1)*5+3731, 0), (i*5+3731, 0), dis=dis)  
 
    # nodes = [(3,3),(3,2),(3,1),(3,0),(3,-1),(4,-1),(5,-1),(6,-1),(7,-1),(8,-1),(9, -1),(10,-1)]
    # for i in range(len(nodes)-1):
    #     G.add_edge(nodes[i], nodes[i+1], dis=1)
    # pos = dict(zip(G.nodes, G.nodes))
    # nx.draw(G, pos=pos)
    file = open("Coords.csv")
    csvreader = csv.reader(file)
    header = next(csvreader)
    print(header)

    rows = []
    for row in csvreader:
        rows.append(row)
    #print(rows)
    
    road_network = nx.Graph()
    downsample = 6
    for i in range(7, len(rows)-1):
        node1 = (int(float(rows[i][1])*1e3), int(float(rows[i][2])*1e3))
        node2 = (int(float(rows[i+1][1])*1e3), int(float(rows[i+1][2])*1e3))
        dis = np.linalg.norm(np.array(node1)-np.array(node2))
        road_network.add_edge(node1, node2, dis=dis)
    
    if ((int(6.29*1e3), int(11.14*1e3)), (int(1.0*1e3), int(13.4*1e3))) in road_network.edges:
        road_network.remove_edge((int(6.29*1e3), int(11.14*1e3)), (int(1.0*1e3), int(13.4*1e3)))
    
    if ((int(6.29*1e3), int(11.14*1e3)), (int(17.5*1e3), int(1.5*1e3))) in road_network.edges:
        road_network.remove_edge((int(6.29*1e3), int(11.14*1e3)), (int(17.5*1e3), int(1.5*1e3)))
    
    # down sample the roadnetwork
    G = nx.Graph()
    sp = nx.shortest_path(road_network, (6290, 11140), (17500, 1500))
    paths = []
    for i in range(0, len(sp), downsample):
        paths.append(sp[i])
    if (17500, 1500) not in paths:
        paths.append((17500, 1500))
    for i in range(len(paths)-1):
        node1 = paths[i]
        node2 = paths[i+1]
        dis = np.linalg.norm(np.array(node1)-np.array(node2))
        G.add_edge(node1, node2, dis=dis)
    
    sp = nx.shortest_path(road_network, (6290, 11140), (1000, 13400))
    paths = []
    for i in range(0, len(sp), downsample):
        paths.append(sp[i])
    if (1000, 13400) not in paths:
        paths.append((1000, 13400))
    for i in range(len(paths)-1):
        node1 = paths[i]
        node2 = paths[i+1]
        dis = np.linalg.norm(np.array(node1)-np.array(node2))
        G.add_edge(node1, node2, dis=dis)
    
    sp = nx.shortest_path(road_network, (6290, 11140), (6800, 19100))
    paths = []
    for i in range(0, len(sp), downsample):
        paths.append(sp[i])
    if (6800, 19100) not in paths:
        paths.append((6800, 19100))
    for i in range(len(paths)-1):
        node1 = paths[i]
        node2 = paths[i+1]
        dis = np.linalg.norm(np.array(node1)-np.array(node2))
        G.add_edge(node1, node2, dis=dis)
    
    
    
    pos = dict(zip(road_network.nodes, road_network.nodes))
    nx.draw(G, pos=pos, alpha=1, node_color='r', node_size=2)
    return G


def generate_road_network_random():
    road_network = nx.grid_2d_graph(11, 11)
    mapping = {}
    for node in road_network.nodes:
        mapping[node] = (node[0]*2000, node[1]*2000)
    road_network = nx.relabel_nodes(road_network, mapping)
    
    # downsample the graph
    for i in range(70):
        has_removed = False
        while not has_removed:
            G = nx.Graph(road_network)
            node = random.sample(list(G.nodes), 1)[0]
            if node == (0, 0):
                continue
            G.remove_node(node)
            if nx.is_connected(G):
                has_removed = True
                road_network.remove_node(node)
    
    assert (0, 0) in road_network.nodes
    for edge in road_network.edges:
        road_network.edges[edge]['dis'] = np.linalg.norm(np.array(edge[0])-np.array(edge[1]))
    
    pos = dict(zip(road_network.nodes, road_network.nodes))
    nx.draw(road_network, pos=pos, alpha=1, node_color='r', node_size=2)
    return road_network


def generate_UAV_task():
    # angle = 70 / 180 * np.pi
    # length = 13*60*20 / 2
    # height = 0.5*np.math.sin(angle)*(length)
    # segments = 5
    # vector_plus = np.array([np.math.cos(angle), np.math.sin(angle)]) * length/segments
    # vector_minus = np.array([np.math.cos(-angle), np.math.sin(-angle)]) * length/segments
    # G = nx.DiGraph()
    # G.graph['dis_per_energy'] = np.linalg.norm(vector_plus)
    # G.add_node((0, int(height)))
    # for i in range(2):
    #     leaf = [x for x in G.nodes() if (G.out_degree(x)==0 and G.in_degree(x)==1) or (G.out_degree(x)==0 and G.in_degree(x)==0)]
    #     assert len(leaf) == 1
    #     curr_node = leaf[0]
    #     for t in range(1, segments+1):
    #         next_node = (int(curr_node[0]+vector_minus[0]), int(curr_node[1]+vector_minus[1]))
    #         dis = np.linalg.norm(np.array(curr_node) - np.array(next_node))
    #         G.add_edge(curr_node, next_node, dis=dis)
    #         # watch out deep copy
    #         curr_node = next_node
            
    #     for t in range(1, segments+1):
    #         next_node = (int(curr_node[0]+vector_plus[0]), int(curr_node[1]+vector_plus[1]))
    #         dis = np.linalg.norm(np.array(curr_node) - np.array(next_node))
    #         G.add_edge(curr_node, next_node, dis=dis)
    #         # watch out deep copy
    #         curr_node = next_node
    
    G = nx.DiGraph() 
    #nodes = [(i, 0) for i in range(8)]   
    nodes = [(6.8*1e3, 19.1*1e3), (5.83*1e3, 16.495*1e3), (7.35*1e3, 14.48*1e3), (4.44*1e3, 13.993*1e3), (1*1e3, 13.4*1e3),
             (2.69*1e3, 10.62*1e3), (5.7*1e3, 11.45*1e3), (10*1e3, 12*1e3), (9.72*1e3, 9.2*1e3), (11.243*1e3, 7.518*1e3),
             (12.507*1e3, 6.27*1e3), (14.076*1e3, 4.845*1e3), (13.61*1e3, 1.23*1e3), (16.322*1e3, 3.549*1e3),
             (17.5*1e3, 1.5*1e3), (10e3, 17e3), (9.5e3, 15e3), (7e3, 8.9e3), (12e3, 4e3)]
    
    loop = [1,2,3,4,5,6,7,18,19,13,15,14,12,11,10,9,8,17,16,1,2,3,4,5,6,7,18,19,13,15,14,12,11,10,9,8,17,16,1]
    loop = [i-1 for i in loop]
    for i in range(len(loop)-1):
        dis = np.linalg.norm(np.array(nodes[loop[i]])-np.array(nodes[loop[i+1]]))
        G.add_node(i, pos=(int(nodes[loop[i]][0]), int(nodes[loop[i]][1])), label=loop[i])
        G.add_node(i+1, pos=(int(nodes[loop[i+1]][0]), int(nodes[loop[i+1]][1])), label=loop[i+1])
        G.add_edge(i, i+1, dis=dis)
    
    #pos = dict(zip(G.nodes, G.nodes))
    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos=pos, alpha=0.5, node_color='b', node_size=8, labels=labels)
    G.graph['dis_per_energy'] = 1
    return G


def generate_UAV_task_random():
    road_network = nx.grid_2d_graph(21, 21)
    mapping = {}
    for node in road_network.nodes:
        mapping[node] = (node[0]*1000, node[1]*1000)
    road_network = nx.relabel_nodes(road_network, mapping)
    
    # downsample the graph
    for i in range(250):
        has_removed = False
        while not has_removed:
            G = nx.Graph(road_network)
            node = random.sample(list(G.nodes), 1)[0]
            if node == (0, 0):
                continue
            G.remove_node(node)
            if nx.is_connected(G):
                has_removed = True
                road_network.remove_node(node)
    
    assert (0, 0) in road_network.nodes
    for edge in road_network.edges:
        road_network.edges[edge]['dis'] = np.linalg.norm(np.array(edge[0])-np.array(edge[1]))
    
    goal_flag = True
    while goal_flag:
        goal = random.sample(road_network.nodes, 1)[0]
        nodes = nx.shortest_path(road_network, source=(0, 0), target=goal)
        if len(nodes)<30:
            continue
        else:
            nodes = [nodes[i] for i in range(len(nodes)) if i % 3 == 0]
            goal_flag = False
            
    G = nx.DiGraph()
    for i in range(len(nodes)-1):
        dis = np.linalg.norm(np.array(nodes[i])-np.array(nodes[i+1]))
        G.add_edge((int(nodes[i][0]), int(nodes[i][1])), (int(nodes[i+1][0]), int(nodes[i+1][1])), dis=dis)
   
    
    pos = dict(zip(G.nodes, G.nodes))
    nx.draw(G, pos=pos, alpha=0.5, node_color='b', node_size=8)
    return G



    
def generate_UGV_task():
    G = nx.DiGraph()
    # a simple straight line network
    goal = []
    node_index = 0
    # for i in range(1):
    #     dis = np.linalg.norm(np.array(((1-1)*5*60+3731, 0))-np.array((19*5+3731, 0)))
    #     G.add_edge(node_index, node_index+1, dis=dis)
    #     G.add_edge(node_index+1, node_index, dis=dis)
    #     G.nodes[node_index]['pos'] = ((1-1)*5*60+3731, 0)
    #     G.nodes[node_index+1]['pos'] = (19*5+3731, 0)
    #     node_index += 1
        # dis = np.linalg.norm(np.array((19*5+3731, 0))-np.array(((1-1)*5*60+3731, 0)))
        # G.add_edge(node_index, node_index+1, dis=dis)
        # G.nodes[node_index]['pos'] = (19*5+3731, 0)
        # G.nodes[node_index+1]['pos'] = ((1-1)*5*60+3731, 0)
        # node_index += 1
    #nodes = [(3,3),(3,2),(3,1),(3,0),(3,-1),(4,-1),(5,-1),(6,-1),(7,-1),(8,-1),(9, -1),(10,-1)]
    # nodes = [(6.8*1e3, 19.1*1e3), (5.46*1e3, 15.32*1e3), (4.04*1e3, 13.13*1e3), (6.29*1e3, 11.14*1e3), (10.4*1e3, 8.35*1e3),
    #          (14.523*1e3, 4.53*1e3), (17.5*1e3, 1.5*1e3)] 
    nodes = [(6.8*1e3, 19.1*1e3), (6.29*1e3, 11.14*1e3), (17.5*1e3, 1.5*1e3)]
    for node_index in range(len(nodes)-1):
        G.add_edge(node_index, node_index+1, dis=1)
        G.nodes[node_index]['pos'] = (int(nodes[node_index][0]), int(nodes[node_index][1]))
        G.nodes[node_index+1]['pos'] = (int(nodes[node_index+1][0]), int(nodes[node_index+1][1]))
    G.add_edge(node_index+1, 0, dis=7)    
    G.graph['UGV_goal'] = node_index+1
    pos = nx.get_node_attributes(G,'pos')
    #nx.draw(G, pos=pos,alpha=0.5, node_color='r', node_size=8)
    return G



def generate_UGV_task_random(road_network):
    G = nx.DiGraph()
    # a simple straight line network
    goal_flag = True
    while goal_flag:
        goal = random.sample(road_network.nodes, 1)[0]
        nodes = nx.shortest_path(road_network, source=(0, 0), target=goal)
        if len(nodes) < 20:
            continue
        else:
            goal_flag = False
    
    node_index = 0
    
    nodes = [(0, 0), goal]
    for node_index in range(len(nodes)-1):
        G.add_edge(node_index, node_index+1)
        G.nodes[node_index]['pos'] = (int(nodes[node_index][0]), int(nodes[node_index][1]))
        G.nodes[node_index+1]['pos'] = (int(nodes[node_index+1][0]), int(nodes[node_index+1][1]))
        G.edges[(node_index, node_index+1)]['dis'] = np.linalg.norm(np.array(G.nodes[node_index]['pos'])-np.array(G.nodes[node_index+1]['pos']))
    
    dis = np.linalg.norm(np.array(G.nodes[0]['pos'])-np.array(G.nodes[node_index+1]['pos']))
    G.add_edge(node_index+1, 0, dis=dis)    
    G.graph['UGV_goal'] = node_index+1
    pos = nx.get_node_attributes(G,'pos')
    nx.draw(G, pos=pos, alpha=0.5, node_color='r', node_size=8)
    return G



def plot_state(road_network, UAV_task, UGV_task, UAV_state, UGV_state, battery_state, ):
    # plot road network
    fig, ax = plt.subplots()
    line_road, = ax.plot([0, 30*60*5*5], [0, 0], color='k')
 
    line_road.set_label('road network')
    
    # plot UAV task
    for edge in UAV_task.edges:
        node1 = edge[0]
        node2 = edge[1]
        x, y = node1[0], node1[1]
        dx, dy = node2[0]-node1[0], node2[1]-node1[1]
        line_UAV = ax.quiver(x, y, dx, dy, scale_units='xy', angles='xy', scale=1, alpha=0.8, color='g') 
    
    # plot UGV task
    # for edge in UGV_task.edges:
    #     node1 = edge[0]
    #     node2 = edge[1]
    #     x, y = node1[0], node1[1]
    #     dx, dy = node2[0]-node1[0], node2[1]-node1[1]
    #     line_UGV = ax.arrow(x, y, dx, dy, alpha=0.8, color='k') 
    # line_UGV.set_label('UGV task')
    
    # plot UAV
    ax.plot(UAV_state[0], UAV_state[1], color='r', marker='s')
    ax.text(UAV_state[0], UAV_state[1], 'UAV')    
    
    # plot UGV
    ax.plot(UGV_state[0], UGV_state[1], color='r', marker='s')
    ax.text(UGV_state[0], UGV_state[1], 'UGV')
    
    # battery state
    ax.set_title("battery state is: " + str(battery_state))
    ax.legend()

    
    

if __name__ == "__main__" :
    # print("hello world")
    UAV_task = generate_UAV_task()
    # # UGV_task is a directed graph. Node name is an index
    UGV_task = generate_UGV_task()
    road_network = generate_road_network()
    # UAV_state = [x for x in UAV_task.nodes if (UAV_task.out_degree(x)==1 and UAV_task.in_degree(x)==0)][0]
    # UGV_state = UGV_task.nodes[0]['pos']
    # battery_state = 20
    rendezvous = Rendezvous(UAV_task, UGV_task, road_network, battery=280e3)
    # #plot_state(road_network, UAV_task, UGV_task, UAV_state, UGV_state, battery_state)
    # print("test UAV task action")
    # print("UAV state is {}, UGV state is {}, battery state is {}".format(UAV_state, UGV_state, battery_state))
    # action = 'v_be'
    # print("UAV take action {} to transit to the next task node".format(action))
    # state = (UAV_state[0], UAV_state[1], UGV_state[0], UGV_state[1], battery_state)

    # UGV_road_state = UGV_state + UGV_state
    # UGV_task_node = 1
    # UAV_state, UGV_state, UGV_road_state, UGV_task_node, battery_state = rendezvous.transit(state, action, UGV_road_state, UGV_task_node)
    # #plot_state(road_network, UAV_task, UGV_task, UAV_state, UGV_state, battery_state)
    
    
    # # test rendezvous action
    # print("test UAV rendezvous action")
    # print("UAV state is {}, UGV state is {}, battery state is {}".format(UAV_state, (int(UGV_state[0]), int(UGV_state[1])), battery_state))
    # action = 'v_be_be'
    # state = (UAV_state[0], UAV_state[1], UGV_state[0], UGV_state[1], battery_state)
    # UAV_state, UGV_state, UGV_road_state, UGV_task_node, battery_state = rendezvous.transit(state, action, UGV_road_state, UGV_task_node)
    # #plot_state(road_network, UAV_task, UGV_task, UAV_state, UGV_state, battery_state)

    
    print("test UGV transition")
    #state = (5830, 16495, 6775, 18131, 88, 1)
    state = (int(6.8e3), int(19.1e3), int(6.8e3), int(19.1e3), 100, 0)
    UAV_state = (state[0], state[1])
    UGV_state = (state[2], state[3])
    energy_state = state[4]/100*rendezvous.battery
    UGV_task_node = state[5]
    UGV_road_state = UGV_state + UGV_state
    descendants = list(UAV_task.neighbors(UAV_state))
    assert len(descendants) == 1
    UAV_state_next = descendants[0]
    duration = UAV_task.edges[UAV_state, UAV_state_next]['dis'] / rendezvous.velocity_uav['v_be']
    
    UGV_state_next, UGV_road_state_next, UGV_task_node_next = rendezvous.UGV_transit(UGV_state, UGV_road_state, UGV_task_node, duration)                        
    
    action = 'v_be_be'
    v1 = action[0:4]
    v2 = 'v'+action[4:]
    # UAV choose to go to next task node with best endurance velocity
    descendants = list(UAV_task.neighbors(UAV_state))
    assert len(descendants) == 1
    UAV_state_next = descendants[0]
    # compute rendezvous point and time
    rendezvous_state, t1, t2 = rendezvous.rendezvous_point(UAV_state, UAV_state_next, 
                                       UGV_state, UGV_road_state, UGV_task_node, 
                                       rendezvous.velocity_uav[v1], rendezvous.velocity_uav[v2])
    
    #Todo: compute the state of UGV after t2 and return
    #print("rendezvous state is {}".format(rendezvous_state))
    assert rendezvous_state in road_network.nodes
    

    
     
    
    
    