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


class Rendezvous():
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
        self.battery = 319.7e3
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
            descendants = list(self.UAV_task.neighbors(UAV_state))
            assert len(descendants) == 1
            UAV_state_next = descendants[0]
            # compute next state for UGV
            duration = self.UAV_task.edges[UAV_state, UAV_state_next]['dis']/self.velocity_uav[action]
            UGV_state_next, UGV_task_state_ = self.UGV_transit(UGV_state, (UGV_task_state[2], UGV_task_state[3]), duration)
            power_consumed = self.power_consumption(self.velocity_uav[action], duration)
            battery_state_next = battery_state - power_consumed
            if battery_state_next < 0:
               UAV_state_next = ('f', 'f')
               UGV_state_next = ('f', 'f')
               battery_state_next = ('empty')
               return UAV_state_next, UGV_state_next, UGV_task_state_, battery_state_next
        
            
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
            UGV_state_next, UGV_task_state_ = self.UGV_transit(rendezvous_node, UGV_next_task, duration)
        
        
        return UAV_state_next, UGV_state_next, UGV_task_state_, battery_state_next
            
            
        
    
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
            
        previous_state = list(self.UGV_task.predecessors(state_before_stop))
        assert len(previous_state) == 1
        previous_state = previous_state[0]
        vector = np.array(previous_state) - np.array(state_before_stop)
        vector = vector/np.linalg.norm(vector)
        assert dis>= total_dis
        UGV_state_next = tuple(np.array(state_before_stop)-(dis-total_dis)*vector)
        UGV_task_state = (previous_state[0], previous_state[1], state_before_stop[0], state_before_stop[1])
        return UGV_state_next, UGV_task_state
        
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
    
    def display_task_transition(self, ):
        # plot road network
        fig, ax = plt.subplots()
        line_road, = ax.plot([0, 30*60*5*5], [0, 0], color='k')
        line_road.set_label('road network')
        
          # plot UAV task
        for edge in self.UAV_task.edges:
            node1 = edge[0]
            node2 = edge[1]
            x, y = node1[0], node1[1]
            dx, dy = node2[0]-node1[0], node2[1]-node1[1]
            line_UAV = ax.quiver(x, y, dx, dy, scale_units='xy', angles='xy', scale=1, alpha=0.5, color='g')
        return
        
    def display_rendezvous(self,):
        return
        

def generate_road_network():
    # it shouldn't be a directed graph
    G = nx.Graph()
    # a simple straight line network
    for i in range(1, 30*60*5):
        dis = np.linalg.norm(np.array((0, (i-1)*5))-np.array((0, i*5)))
        G.add_edge(((i-1)*5, 0), (i*5, 0), dis=dis)  
    pos = dict(zip(G.nodes, G.nodes))
    #nx.draw(G, pos=pos)
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
    #nx.draw(G, pos=pos)
    return G

    
def generate_UGV_task():
    G = nx.DiGraph()
    # a simple straight line network
    goal = []
    for i in range(1, 30*60*4):
        dis = np.linalg.norm(np.array((0, (i-1)*5))-np.array((0, i*5)))
        G.add_edge(((i-1)*5, 0), (i*5, 0), dis=dis)
        goal = (0, i*5)
    G.graph['UGV_goal'] = goal
    pos = dict(zip(G.nodes, G.nodes))
    #nx.draw(G, pos=pos)
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
    print("hello world")
    UAV_task = generate_UAV_task()
    UGV_task = generate_UGV_task()
    road_network = generate_road_network()
    UAV_state = [x for x in UAV_task.nodes if (UAV_task.out_degree(x)==1 and UAV_task.in_degree(x)==0)][0]
    UGV_state = (0, 0)
    battery_state = 319.7e3
    rendezvous = Rendezvous(UAV_task, UGV_task, road_network)
    plot_state(road_network, UAV_task, UGV_task, UAV_state, UGV_state, battery_state)
    print("test UAV task action")
    print("UAV state is {}, UGV state is {}, battery state is {}".format(UAV_state, UGV_state, battery_state))
    action = 'v_be'
    print("UAV take action {} to transit to the next task node".format(action))
    state = (UAV_state[0], UAV_state[1], UGV_state[0], UGV_state[1], battery_state)
    UGV_task_next = list(UGV_task.neighbors(UGV_state))[0]
    UGV_task_state = (UGV_state[0], UGV_state[1], UGV_task_next[0], UGV_task_next[1])
    UAV_state, UGV_state, UGV_task_state, battery_state = rendezvous.transit(state, action, UGV_task_state)
    plot_state(road_network, UAV_task, UGV_task, UAV_state, UGV_state, battery_state)
    
    # test rendezvous action
    print("test UAV rendezvous action")
    print("UAV state is {}, UGV state is {}, battery state is {}".format(UAV_state, UGV_state, battery_state))
    action = 'v_be_be'
    state = (UAV_state[0], UAV_state[1], UGV_state[0], UGV_state[1], battery_state)
    UAV_state, UGV_state, UGV_task_state, battery_state = rendezvous.transit(state, action, UGV_task_state)
    plot_state(road_network, UAV_task, UGV_task, UAV_state, UGV_state, battery_state)

    
    

    
     
    
    
    