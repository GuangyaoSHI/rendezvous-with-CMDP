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


class rendezvous():
    def __init__(self, UAV_task, UGV_task, road_network):
        # a sequence of nodes in 2D plane, represented as a directed graph
        self.UAV_task = UAV_task 
        
        # a sequence of nodes in road network, represented as a directed graph
        # rendezvous action may change the task, some nodes may be inserted into tha task 
        self.UGV_task = UGV_task 
        self.UGV_goal = UGV_task.graph['UGV_goal']
        
        # road network can be viewed as a fine discretization of 2D continuous road network
        self.road_network = road_network 
        
        # uav velocity
        self.velocity_uav = {'v_be' : 9.8, 'v_br' : 14}
        
        # ugv velocity
        self.velocity_ugv = 4.5

        self.check_UGV_task()
        
        # time for changing battery
        # Todo: choose a proper value 
        self.chargin_time = 0
        
        # total battery
        # Todo:choose a proper value 
        self.battery = 10000
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

                
    def check_UGV_task(self):
        # check whether each task point is in road network
        for node in self.UGV_task.nodes:
            assert node in self.road_network.nodes, "UGV task node not in road network"
        
        
    def transit(self, state, action, UGV_task_state):
        # return probability distribution P(s' | s, a)
        # state = (xa, ya, xg, yg, SoC)
        # action: {'v_be', 'v_br', 'v_be_be', 'v_be_br', 'v_br_be', 'v_br_br'}
        # UGV_task_state: UGV is transiting from which node to which
        # (x, y, x1, y1)
        UAV_state, UGV_state, battery_state = self.get_states(state)
        UAV_state_next = []
        UGV_state_next = []
        battery_state_next = []
        
        if action in ['v_be', 'v_br']:
            # UAV choose to go to next task node with best endurance velocity
            descendants = list(self.UGV_task.neighbors[UAV_state])
            assert len(descendants) == 1
            UAV_state_next = descendants[0]
            # compute next state for UGV
            duration = self.UAV_task.edges[UAV_state, UAV_state_next]['dis']/self.velocity_uav[action]
            UGV_state_next = self.UGV_transit(UGV_state, (UGV_task_state[2], UGV_task_state[3]), duration)
            power_consumed = self.power_consumption(self.velocity_uav[action], duration)
            battery_state_next = battery_state - power_consumed
            if battery_state_next < 0:
               UAV_state_next = ('f', 'f')
               UGV_state_next = ('f', 'f')
               battery_state_next = ('empty')
               return UAV_state_next, UGV_state_next, battery_state_next
        
            
        if action in ['v_be_be', 'v_be_br', 'v_br_be', 'v_br_br']:
            v1 = action[0:4]
            v2 = 'v'+action[4:]
            # UAV choose to go to next task node with best endurance velocity
            descendants = list(self.UGV_task.neighbors[UAV_state])
            assert len(descendants) == 1
            UAV_state_next = descendants[0]
            # compute rendezvous point and time
            rendezvous_node, t1, t2 = self.rendezvous_point(UAV_state, UAV_state_next, 
                                               UGV_state, UGV_task_state, self.velocity_uav[v1], self.velocity_uav[v2])
            
            # power consumed for rendezvous 
            power_consumed = self.power_consumption(self.velocity_uav['v_be'], t1)
            battery_state_next = battery_state - power_consumed
            # UAV cannot rendezvous
            if battery_state_next < 0:
                UAV_state_next = ('f', 'f')
                UGV_state_next = ('f', 'f')
                battery_state_next = ('empty')
                return UAV_state_next, UGV_state_next, battery_state_next
            
            # UAV cannot go back to next task node
            power_consumed = self.power_consumption(self.velocity_uav['v_be'], t2)
            battery_state_next = self.battery - power_consumed
            if battery_state_next < 0:
                UAV_state_next = ('f', 'f')
                UGV_state_next = ('f', 'f')
                battery_state_next = ('empty')
                return UAV_state_next, UGV_state_next, battery_state_next
            
            #Todo: in the rendezvous function the task of UGV will change
            #Todo: compute the state of UGV after t2 and return
            UGV_next_task = self.UGV_task.neighbors(rendezvous_node)[0]
            UGV_state_next = self.UGV_transit(rendezvous_node, UGV_next_task, duration)
        
        
        return UAV_state_next, UGV_state_next, battery_state_next
            
            
        
    
    def UGV_transit(self, UGV_state, UGV_next_task, duration):
        #last_task_state = (UGV_task_state[0], UGV_task_state[1])
        # Todo: check UGV_state is indeed between two task nodes
        next_task_state = UGV_next_task
        
        # UGV will move duration * velocity distance along the task path
        total_dis = self.velocity_ugv * duration
        
        state_before_stop = next_task_state
        dis = np.linalg.norm(np.array(UGV_state)-np.array(state_before_stop))
        
        while (dis < total_dis):
            descendants = list(self.UGV_task.neighbors(state_before_stop))[0]
            dis += np.linalg.norm(np.array(descendants)-np.array(state_before_stop))
            state_before_stop = descendants
            
        previous_state = self.UGV_task.predecessors(state_before_stop)
        vector = np.array(previous_state) - np.array(state_before_stop)
        vector = vector/np.linalg.norm(vector)
        assert dis>= total_dis
        UGV_state_next = tuple(np.array(state_before_stop)-(dis-total_dis)*vector)
        return UGV_state_next
        
    def power_consumption(self, tgtV, duration):
        # return power distribution after taking action with SoC
        # sample weight using random normal distribution of weight
        W = self.stdW*np.random.randn() + self.meanW
        
        # solve for true airspeed by adding in weibull wind distribution with
        # equally random heading direction of wind
        # simplifying assumption is that only wind tangential to UAS heading affects power
        disturbance = weibull_min.rvs(c=self.aG, loc=0, scale=self.bG)
        V = abs(tgtV + disturbance*np.math.cos(-self.heading))
        P = self.b[0] + self.b[1]*V + self.b[2]*V**2 + self.b[3]*V**3 + self.b[4]*W + self.b[5]*V*W
        return P*duration
    
    def rendezvous_point(self, UAV_state, UAV_state_next, UGV_state, UGV_task_state, vel_rdv, vel_sep):
        # return rendezvous point
        UGV_task_before = (UGV_task_state[0], UGV_task_state[1])
        UGV_task_next = (UGV_task_state[2], UGV_task_state[3])
        G_road = copy.deepcopy(self.road_network)
        G_road.remove_edge(UGV_task_before, UGV_task_next)
        assert UGV_state != UGV_task_before
        assert UGV_state != UGV_task_next
        dis = np.linalg.norm(np.array(UGV_task_before) - np.array(UGV_state))
        G_road.add_edge(UGV_task_before, UGV_state, dis=dis)
        dis = np.linalg.norm(np.array(UGV_state) - np.array(UGV_task_next))
        G_road.add_edge(UGV_state, UGV_task_next, dis=dis)
        
        rendezvous_node = UGV_state
        dis1 = np.linalg.norm(np.array(UAV_state) - np.array(rendezvous_node))
        dis2 = np.linalg.norm(np.array(UAV_state_next) - np.array(rendezvous_node))
        # time taken to rendezvous
        rendezvous_time1 = max(dis1/vel_rdv, 
                              nx.shortest_path_length(G_road, source=UGV_state, 
                                                      target=rendezvous_node, weight='dis'))
        # time taken to go back to task
        rendezvous_time2 = dis2/vel_sep
        rendezvous_time = rendezvous_time1 + rendezvous_time2
        
        for node in G_road:
            dis1 = np.linalg.norm(np.array(UAV_state) - np.array(node))
            dis2 = np.linalg.norm(np.array(UAV_state_next) - np.array(node))
            time1 = max(dis1/vel_rdv, 
                                  nx.shortest_path_length(G_road, source=UGV_state, 
                                                          target=rendezvous_node, weight='dis'))
            time2 = dis2/vel_sep
            time = time1 + time2
            if time < rendezvous_time:
                rendezvous_node = node
                rendezvous_time = time
                rendezvous_time1 = time1
                rendezvous_time2 = time2
        
        # update UGV task 
        # Todo: 
        rendezvous_node2task = nx.shortest_path(self.road_network, source=rendezvous_node, target=UGV_task_next)
        task2goal = nx.shortest_path(self.UGV_task, source=UGV_task_next, target=self.UGV_goal)
        new_task = nx.DiGraph()
        new_task.add_node(rendezvous_node)
        for i in range(len(rendezvous_node2task)-1):
            new_task.add_edge(rendezvous_node2task[i], rendezvous_node2task[i+1])
        for i in range(len(task2goal)-1):
            new_task.add_edge(task2goal[i], task2goal[i+1])    
        self.UGV_task = copy.deepcopy(new_task)
        return rendezvous_node, rendezvous_time1, rendezvous_time2
    
    def get_states(self, state):
        # state = (xa, ya, xg, yg, SoC)
        UAV_state = (state[0], state[1])
        assert UAV_state in self.UAV_task, "UAV state is not in task"
        UGV_state = (state[2], state[3])
        battery_state = state[-1]
        return UAV_state, UGV_state, battery_state
        

def generate_road_network():
    # it shouldn't be a directed graph
    G = nx.Graph()
    # a simple straight line network
    for i in range(1, 30*60*5):
        dis = np.linalg.norm(np.array((0, (i-1)*5))-np.array((0, i*5)))
        G.add_edge((0, (i-1)*5), (0, i*5), dis=dis)  
    return G

def generate_UAV_task():
    angle = 70 / 180 * np.pi
    length = 13*60*20 / 2
    height = 0.5*np.math.sin(angle)*(length)
    segments = 5
    vector_plus = np.array([np.math.cos(angle), np.math.sin(angle)]) * length/segments
    vector_minus = np.array([np.math.cos(-angle), np.math.sin(-angle)]) * length/segments
    G = nx.DiGraph()
    G.add_node((0, int(height)))
    for i in range(4):
        leaf = [x for x in G.nodes() if (G.out_degree(x)==0 and G.in_degree(x)==1) or (G.out_degree(x)==0 and G.in_degree(x)==0)]
        assert len(leaf) == 1
        curr_node = leaf[0]
        for t in range(1, segments+1):
            next_node = (int(curr_node[0]+vector_minus[0]), int(curr_node[1]+vector_minus[1]))
            dis = np.linalg.norm(np.array(curr_node) - np.array(next_node))
            G.add_edge(curr_node, next_node, dis=dis)
            # watch out deep copy
            curr_node = next_node
            
        for t in range(1, segments+1):
            next_node = (int(curr_node[0]+vector_plus[0]), int(curr_node[1]+vector_plus[1]))
            dis = np.linalg.norm(np.array(curr_node) - np.array(next_node))
            G.add_edge(curr_node, next_node, dis=dis)
            # watch out deep copy
            curr_node = next_node
            
    pos = dict(zip(G.nodes, G.nodes))
    nx.draw(G, pos=pos)
    return G

    
def generate_UGV_task():
    G = nx.DiGraph()
    # a simple straight line network
    goal = []
    for i in range(1, 30*60*4):
        dis = np.linalg.norm(np.array((0, (i-1)*5))-np.array((0, i*5)))
        G.add_edge((0, (i-1)*5), (0, i*5), dis=dis)
        goal = (0, i*5)
    G.graph['UGV_goal'] = goal
    return G
    
    
     
    
    
    