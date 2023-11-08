import random
from classes import *
from statistics import mean
class ConstructiveHeuristic:
    def __init__(self, nodes,capacity, blackbox, tsEnvironment, weight=0.7, real_alpha=1, count=0, dict_of_types=None):
        self.nodes = nodes
        self.capacity = capacity
        self.currentcapacity = capacity
        self.firstEdge = 0
        self.blackbox = blackbox
        self.tsEnvironment = tsEnvironment
        self.weight = weight
        self.max_capacity = -1
        self.max_min_dist = -1
        self.real_alpha = real_alpha

        # Dynamic enviroment
        #random.seed(123)
        self.weather = random.randint(0, 1)
        self.congestion = {i: random.randint(0, 1) for i in range(len(nodes))}

        if dict_of_types:
            self.dict_of_types = dict_of_types
        else:
            self.dict_of_types = {i: 1 for i in range(len(nodes))}

    def test_build_constructive_heuristic_deterministic(self, iter):
        of_list = []

        for _ in range(iter):
            self.currentcapacity = self.capacity
            nodes = self.nodes.copy()
            previousNode = nodes[0]
            of = 0
            while len(nodes) > 2:
                cl = {}
                for i in range(1,len(nodes)-1):
                    distance_to_end = edge( nodes[i], nodes[len(nodes)-1]).distance
                    if i != 0 and i != len(nodes)-1 and distance_to_end < self.currentcapacity:
                        edge_i = edge(previousNode, nodes[i])
                        distance = edge_i.distance
                        if distance < self.currentcapacity:
                            nodeId = nodes[i].id
                            nodeType = nodeId % 5
                            reward = nodes[i].reward
                            eval = -self.weight * distance/self.capacity + (1-self.weight) * reward/15
                            cl[i] = [eval, distance, i, nodeId, nodeType, reward]
                if len(cl) > 0:
                    cl = dict(sorted(cl.items(), key=lambda item: -item[1][0]))
                    arg_candidate = max(cl, key=lambda key: cl[key][0])
                    candidate = cl[arg_candidate]

                    sim = self.blackbox.simulate(node_type=candidate[4],
                                                 weather=self.weather,
                                                 congestion=self.congestion[candidate[3]],
                                                 battery = self.currentcapacity/self.capacity - candidate[1] /self.capacity )
                    of += candidate[5] * sim
                    self.currentcapacity -= candidate[1]
                    previousNode = nodes[candidate[2]]
                    nodes.pop(arg_candidate)

                else:
                    break
            of_list.append(of)
        return mean(of_list)

    def test_build_constructive_heuristic_ts(self,iter):
        of_list = []
        ts = self.tsEnvironment
        for _ in range(iter):
            nodes = self.nodes.copy()
            self.currentcapacity = self.capacity
            previousNode = nodes[0]
            of = 0
            while len(nodes) >2:
                cl = {}
                for i in range(1, len(nodes)-1):
                    distance_to_end = edge( nodes[i], nodes[len(nodes)-1]).distance
                    if i != 0 and i != len(nodes)-1 and distance_to_end < self.currentcapacity:
                        edge_i = edge(previousNode, nodes[i])
                        distance = edge_i.distance
                        if distance < self.currentcapacity:
                            nodeId = nodes[i].id
                            nodeType = nodeId % 5
                            tsSim = ts.simulate((nodeType, self.weather, self.congestion[nodeId], 1))
                            reward = nodes[i].reward * tsSim
                            eval = -self.weight * distance/self.capacity + (1-self.weight) * reward/15
                            cl[i] = [eval, distance, i, nodeId, nodeType, reward]
                if len(cl) > 0:
                    cl = dict(sorted(cl.items(), key=lambda item: -item[1][0]))
                    arg_candidate = max(cl, key=lambda key: cl[key][0])
                    candidate = cl[arg_candidate]

                    sim = self.blackbox.simulate(node_type=candidate[4],
                                                 weather=self.weather,
                                                 congestion=self.congestion[candidate[3]],
                                                 battery = self.currentcapacity/self.capacity - candidate[1] /self.capacity )
                    of += candidate[5] * sim

                    if sim == 1:
                        ts.update_dict((candidate[4], self.weather, self.congestion[candidate[3]], 1), 1)
                    else:
                        ts.update_dict((candidate[4], self.weather, self.congestion[candidate[3]], 1), 0)

                    self.currentcapacity -= candidate[1]
                    previousNode = nodes[candidate[2]]
                    nodes.pop(arg_candidate)

                else:
                    break
            of_list.append(of)
        return mean(of_list)
