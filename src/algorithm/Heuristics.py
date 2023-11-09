from src.utils.classes import *
from statistics import mean


class ConstructiveHeuristic:
    def __init__(self, nodes, capacity, blackbox, ts_environment, weight=0.7, real_alpha=1, dict_of_types=None):
        self.nodes = nodes
        self.capacity = capacity
        self.current_capacity = capacity
        self.firstEdge = 0
        self.blackbox = blackbox
        self.ts_environment = ts_environment
        self.weight = weight
        self.max_capacity = -1
        self.max_min_dist = -1
        self.real_alpha = real_alpha

        self.weather = random.randint(0, 1)
        self.congestion = {i: random.randint(0, 1) for i in range(len(nodes))}

        if dict_of_types:
            self.dict_of_types = dict_of_types
        else:
            self.dict_of_types = {i: 1 for i in range(len(nodes))}

    def test_build_constructive_heuristic_deterministic(self, iterations):
        of_list = []

        for _ in range(iterations):
            self.current_capacity = self.capacity
            nodes = self.nodes.copy()
            previous_node = nodes[0]
            of = 0
            while len(nodes) > 2:
                cl = {}
                for i in range(1, len(nodes) - 1):
                    distance_to_end = edge(nodes[i], nodes[len(nodes) - 1]).distance
                    if i != 0 and i != len(nodes) - 1 and distance_to_end < self.current_capacity:
                        edge_i = edge(previous_node, nodes[i])
                        distance = edge_i.distance
                        if distance < self.current_capacity:
                            node_id = nodes[i].id
                            node_type = node_id % 5
                            reward = nodes[i].reward
                            eval = -self.weight * distance / self.capacity + (1 - self.weight) * reward / 15
                            cl[i] = [eval, distance, i, node_id, node_type, reward]
                if len(cl) > 0:
                    cl = dict(sorted(cl.items(), key=lambda item: -item[1][0]))
                    arg_candidate = max(cl, key=lambda key: cl[key][0])
                    candidate = cl[arg_candidate]

                    sim = self.blackbox.simulate(node_type=candidate[4],
                                                 weather=self.weather,
                                                 congestion=self.congestion[candidate[3]],
                                                 battery=self.current_capacity / self.capacity - candidate[
                                                     1] / self.capacity)
                    of += candidate[5] * sim
                    self.current_capacity -= candidate[1]
                    previous_node = nodes[candidate[2]]
                    nodes.pop(arg_candidate)

                else:
                    break
            of_list.append(of)
        return mean(of_list)

    def test_build_constructive_heuristic_ts(self, iterations):
        of_list = []
        ts = self.ts_environment
        for _ in range(iterations):
            nodes = self.nodes.copy()
            self.current_capacity = self.capacity
            previous_node = nodes[0]
            of = 0
            while len(nodes) > 2:
                cl = {}
                for i in range(1, len(nodes) - 1):
                    distance_to_end = edge(nodes[i], nodes[len(nodes) - 1]).distance
                    if i != 0 and i != len(nodes) - 1 and distance_to_end < self.current_capacity:
                        edge_i = edge(previous_node, nodes[i])
                        distance = edge_i.distance
                        if distance < self.current_capacity:
                            node_id = nodes[i].id
                            node_type = node_id % 5
                            ts_sim = ts.simulate((node_type, self.weather, self.congestion[node_id], 1))
                            reward = nodes[i].reward * ts_sim
                            eval = -self.weight * distance / self.capacity + (1 - self.weight) * reward / 15
                            cl[i] = [eval, distance, i, node_id, node_type, reward]
                if len(cl) > 0:
                    cl = dict(sorted(cl.items(), key=lambda item: -item[1][0]))
                    arg_candidate = max(cl, key=lambda key: cl[key][0])
                    candidate = cl[arg_candidate]

                    sim = self.blackbox.simulate(node_type=candidate[4],
                                                 weather=self.weather,
                                                 congestion=self.congestion[candidate[3]],
                                                 battery=self.current_capacity / self.capacity - candidate[1] /
                                                         self.capacity)
                    of += candidate[5] * sim

                    if sim == 1:
                        ts.update_dict((candidate[4], self.weather, self.congestion[candidate[3]], 1), 1)
                    else:
                        ts.update_dict((candidate[4], self.weather, self.congestion[candidate[3]], 1), 0)

                    self.current_capacity -= candidate[1]
                    previous_node = nodes[candidate[2]]
                    nodes.pop(arg_candidate)

                else:
                    break
            of_list.append(of)
        return mean(of_list)
