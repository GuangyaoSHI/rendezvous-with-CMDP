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
            state = self.UGV_task.nodes[node]['pos']
            assert state in self.road_network.nodes, "UGV task node not in road network"
        
    
        
    def transit(self, state, action, UGV_task_state):
        # return probability distribution P(s' | s, a)
        # state = (xa, ya, xg, yg, SoC)
        # action: {'v_be', 'v_br', 'v_be_be', 'v_be_br', 'v_br_be', 'v_br_br'}
        # UGV_task_state: UGV is transiting from which node to which
        # (node1, node2)
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
            UGV_state_next, UGV_task_state_ = self.UGV_transit(UGV_state, UGV_task_state[1], duration)
            power_consumed = self.power_consumption(self.velocity_uav[action], duration)
            battery_state_next = battery_state - power_consumed
            if self.display:
                self.display_task_transition(UAV_state, UAV_state_next, UGV_state, UGV_state_next, battery_state, battery_state_next)
            if battery_state_next < 0:
               print("out of battery")
               UAV_state_next = ('f', 'f')
               UGV_state_next = ('f', 'f')
               battery_state_next = ('empty')
               return UAV_state_next, UGV_state_next, UGV_task_state_, battery_state_next
        
            
        if action in ['v_be_be', 'v_be_br', 'v_br_be', 'v_br_br']:
            v1 = action[0:4]
            v2 = 'v'+action[4:]
            # UAV choose to go to next task node with best endurance velocity
            descendants = list(self.UAV_task.neighbors(UAV_state))
            assert len(descendants) == 1
            UAV_state_next = descendants[0]
            # compute rendezvous point and time
            rendezvous_node, t1, t2 = self.rendezvous_point(UAV_state, UAV_state_next, 
                                               UGV_state, UGV_task_state, self.velocity_uav[v1], self.velocity_uav[v2])
            
            #Todo: in the rendezvous function the task of UGV will change
            #Todo: compute the state of UGV after t2 and return
            UGV_next_node = 1
            UGV_state_next, UGV_task_state_ = self.UGV_transit(rendezvous_node, UGV_next_node, t2)
            
            
            # power consumed for rendezvous 
            power_consumed1 = self.power_consumption(self.velocity_uav[v1], t1)
            power_consumed2 = self.power_consumption(self.velocity_uav[v2], t2)
            battery_state_next = battery_state - power_consumed1
            if self.display:
                self.display_rendezvous(rendezvous_node, UAV_state, UAV_state_next, 
                                    UGV_state, UGV_state_next, battery_state, 
                                    self.battery - power_consumed2, battery_state - power_consumed1)
            # UAV cannot rendezvous
            if battery_state_next < 0:
                UAV_state_next = ('f', 'f')
                UGV_state_next = ('f', 'f')
                battery_state_next = ('empty')
                return UAV_state_next, UGV_state_next, battery_state_next
            
            # UAV cannot go back to next task node
            battery_state_next = self.battery - power_consumed2
            if battery_state_next < 0:
                UAV_state_next = ('f', 'f')
                UGV_state_next = ('f', 'f')
                battery_state_next = ('empty')
                return UAV_state_next, UGV_state_next, battery_state_next
            
        return UAV_state_next, UGV_state_next, UGV_task_state_, battery_state_next
            
            
        
    
    def UGV_transit(self, UGV_state, UGV_state_road, UGV_next_task_node, duration):
        # UGV_state_road = (x1, y1, x2, y2)
        # this information helps to locate UGV in the road network
        # Todo: check UGV_state is indeed between two task nodes
        
        # temporarily insert UGV_state into road network
        previous_road_state = (UGV_state_road[0], UGV_state_road[1])
        assert previous_road_state in self.road_network.nodes
        next_road_state = (UGV_state_road[2], UGV_state_road[3])
        assert next_road_state in self.road_network.nodes
        if UGV_state not in [previous_road_state, next_road_state]:
            self.road_network.remove_edge(previous_road_state, next_road_state)
            dis = np.linalg.norm(np.array(previous_road_state)-np.array(UGV_state))
            self.road_network.add_edge(previous_road_state, UGV_state, dis=dis)
            dis = np.linalg.norm(np.array(next_road_state)-np.array(UGV_state))
            self.road_network.add_edge(UGV_state, next_road_state, dis=dis)
            
          
 
            
        
        # UGV will move duration * velocity distance along the task path
        total_dis = self.velocity_ugv * duration
        
        node_before_stop = UGV_next_task_node
        state_before_stop = self.UGV_task.nodes[node_before_stop]['pos']
        path = nx.shortest_path(self.road_network, source=UGV_state, target=state_before_stop)
        dis = np.linalg.norm(np.array(UGV_state)-np.array(state_before_stop))
        
        while (dis < total_dis):
            descendant = list(self.UGV_task.neighbors(node_before_stop))[0]
            descendant_state = self.UGV_task.nodes[descendant]['pos']
            dis += np.linalg.norm(np.array(descendant_state)-np.array(state_before_stop))
            node_before_stop = descendant
            state_before_stop = self.UGV_task.nodes[node_before_stop]['pos']
            
        previous_node = list(self.UGV_task.predecessors(node_before_stop))
        assert len(previous_node) == 1
        previous_state = self.UGV_task.nodes[previous_node[0]]['pos'] 
        vector = np.array(previous_state) - np.array(state_before_stop)
        vector = vector/np.linalg.norm(vector)
        assert dis>= total_dis
        UGV_state_next = tuple(np.array(state_before_stop)+(dis-total_dis)*vector)
        UGV_task_state = (previous_node[0], node_before_stop)
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
        UGV_task_before = self.UGV_task.nodes[UGV_task_state[0]]['pos']
        UGV_task_next = self.UGV_task.nodes[UGV_task_state[1]]['pos']
        G_road = copy.deepcopy(self.road_network)
        
        assert UGV_task_before in G_road.nodes
        assert UGV_task_next in G_road.nodes
        
        # Todo: if UGV_state is exactly on one task node, there may be an error
        if not (UGV_task_before == UGV_state or UGV_task_next == UGV_state):
            G_road.remove_edge(UGV_task_before, UGV_task_next)
            dis = np.linalg.norm(np.array(UGV_task_before) - np.array(UGV_state))
            G_road.add_edge(UGV_task_before, UGV_state, dis=dis)
            dis = np.linalg.norm(np.array(UGV_state) - np.array(UGV_task_next))
            G_road.add_edge(UGV_state, UGV_task_next, dis=dis)
        
        # make sure UGV_state is in G_road
        assert UGV_state in G_road.nodes
        
        rendezvous_node = UGV_state
        dis1 = np.linalg.norm(np.array(UAV_state) - np.array(rendezvous_node))
        dis2 = np.linalg.norm(np.array(UAV_state_next) - np.array(rendezvous_node))
        # time taken to rendezvous
        rendezvous_time1 = max(dis1/vel_rdv, 
                              nx.shortest_path_length(G_road, source=UGV_state, 
                                                      target=rendezvous_node, weight='dis')/self.velocity_ugv)
        # time taken to go back to task
        rendezvous_time2 = dis2/vel_sep
        rendezvous_time = rendezvous_time1 + rendezvous_time2
        
        for node in G_road:
            dis1 = np.linalg.norm(np.array(UAV_state) - np.array(node))
            dis2 = np.linalg.norm(np.array(UAV_state_next) - np.array(node))
            time1 = max(dis1/vel_rdv, 
                                  nx.shortest_path_length(G_road, source=UGV_state, 
                                                          target=rendezvous_node, weight='dis')/self.velocity_ugv)
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
        task2goal = nx.shortest_path(self.UGV_task, source=UGV_task_state[1], target=self.UGV_goal)
        new_task = nx.DiGraph()
        node_index = 0
        for i in range(len(rendezvous_node2task)-1):
            new_task.add_edge(node_index, node_index+1)
            new_task.nodes[node_index]['pos'] = rendezvous_node2task[i]
            new_task.nodes[node_index+1]['pos'] = rendezvous_node2task[i+1]
            node_index += 1
        for i in range(len(task2goal)-1):
            new_task.add_edge(node_index, node_index+1)
            new_task.nodes[node_index]['pos'] = self.UGV_task.nodes[task2goal[i]]['pos']
            new_task.nodes[node_index+1]['pos'] = self.UGV_task.nodes[task2goal[i+1]]['pos']
            node_index += 1
        self.UGV_task = new_task
        return rendezvous_node, rendezvous_time1, rendezvous_time2
    
    def get_states(self, state):
        # state = (xa, ya, xg, yg, SoC)
        UAV_state = (state[0], state[1])
        assert UAV_state in self.UAV_task, "UAV state is not in task"
        UGV_state = (state[2], state[3])
        battery_state = state[-1]
        return UAV_state, UGV_state, battery_state
    
    def display_task_transition(self, UAV_state_last, UAV_state_next, UGV_state_last, UGV_state_next, battery_state_last, battery_state_next):
        # plot road network
        fig, ax = plt.subplots()
        line_road, = ax.plot([0, 30*60*5*2.5], [0, 0], color='k', alpha=0.2)
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
        fig.savefig("task_transition.pdf")
        
    
    
    def display_rendezvous(self, rendezvous_node, UAV_state_last, UAV_state_next, 
                           UGV_state_last, UGV_state_next, battery_state_last, 
                           battery_state_next, battery_rendezvous):
        # plot road network
        fig, ax = plt.subplots()
        line_road, = ax.plot([0, 30*60*5*2.5], [0, 0], color='k', alpha=0.2)
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
        fig.savefig("rendezvous.pdf")
        

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
    node_index = 0
    for i in range(1, 30*4):
        dis = np.linalg.norm(np.array((0, (i-1)*5*60))-np.array((0, i*5*60)))
        G.add_edge(node_index, node_index+1, dis=dis)
        G.nodes[node_index]['pos'] = ((i-1)*5*60, 0)
        G.nodes[node_index+1]['pos'] = (i*5*60, 0)
        node_index += 1
    G.graph['UGV_goal'] = node_index
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
    # UGV_task is a directed graph. Node name is an index
    UGV_task = generate_UGV_task()
    road_network = generate_road_network()
    UAV_state = [x for x in UAV_task.nodes if (UAV_task.out_degree(x)==1 and UAV_task.in_degree(x)==0)][0]
    UGV_state = (0, 0)
    battery_state = 319.7e3
    rendezvous = Rendezvous(UAV_task, UGV_task, road_network)
    #plot_state(road_network, UAV_task, UGV_task, UAV_state, UGV_state, battery_state)
    print("test UAV task action")
    print("UAV state is {}, UGV state is {}, battery state is {}".format(UAV_state, UGV_state, battery_state))
    action = 'v_be'
    print("UAV take action {} to transit to the next task node".format(action))
    state = (UAV_state[0], UAV_state[1], UGV_state[0], UGV_state[1], battery_state)

    UGV_task_state = (0, 1)
    UAV_state, UGV_state, UGV_task_state, battery_state = rendezvous.transit(state, action, UGV_task_state)
    #plot_state(road_network, UAV_task, UGV_task, UAV_state, UGV_state, battery_state)
    
    # test rendezvous action
    print("test UAV rendezvous action")
    print("UAV state is {}, UGV state is {}, battery state is {}".format(UAV_state, UGV_state, battery_state))
    action = 'v_be_be'
    state = (UAV_state[0], UAV_state[1], UGV_state[0], UGV_state[1], battery_state)
    UAV_state, UGV_state, UGV_task_state, battery_state = rendezvous.transit(state, action, UGV_task_state)
    #plot_state(road_network, UAV_task, UGV_task, UAV_state, UGV_state, battery_state)

    
    

    
     
    
    
    