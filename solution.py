import copy

from classes import *
import random as rnd
from sim import simheuristic
import numpy as np
import time
import sys
import pandas as pd
class solution:

    def __init__(self, nodes, max_dist, max_vehicles = 1,alpha = 0.7):
        self.routes = []
        self.of = 0
        #self.stochastic_of = []
        #self.reliability = 0
        self.nodes = nodes
        self.savings = []
        self.alpha = alpha
        self.max_dist = max_dist
        self.max_vehicles = max_vehicles

    def reset(self):
        self.routes = []
        self.stochastic_of = []
        self.savings = []

    def dummy_solution(self):
        for i in range(len(self.nodes)-2):
            edges = [edge(self.nodes[0], self.nodes[i+1]), edge(self.nodes[i+1], self.nodes[-1])]
            self.routes.append(route(i, edges, sum([i.distance for i in edges])))
            self.routes[i].reward = self.nodes[i+1].reward
            self.nodes[i+1].route = self.routes[i]

    def create_saving_list(self):
        for i in range(len(self.nodes)-2):
            for j in range(len(self.nodes)-2):
                if i == j:
                    continue
                edge_a_b = edge(self.nodes[i+1], self.nodes[j+1])
                edge_a_end = edge(self.nodes[i + 1], self.nodes[-1])
                edge_depot_b = edge(self.nodes[0], self.nodes[j + 1])
                self.savings.append(saving(self.nodes[j + 1],
                                           self.nodes[i + 1],
                                           self.alpha * (edge_a_end.distance + edge_depot_b.distance - edge_a_b.distance) + (1 - self.alpha) * (self.nodes[i+1].reward + self.nodes[j+1].reward),
                                           edge_a_b.distance))

        self.savings.sort(key = lambda x: x.distance)

    def select_saving(self, random = 2):
        return self.savings.pop(rnd.randint(0, min([random, len(self.savings)-1])))

    def merge_routes(self, saving):
        route_a = saving.start.route
        route_b = saving.end.route
        distance = route_a.distance + route_b.distance + saving.a_to_b - route_a.edges[-1].distance - route_b.edges[0].distance
        if route_a.id != route_b.id and route_a.edges[0].end.id == saving.start.id and route_b.edges[0].end.id == saving.end.id and distance <= self.max_dist:

            route_a.edges.pop()
            route_a.edges = route_a.edges + [edge(saving.start, saving.end)] + route_b.edges[1:]
            route_a.distance = distance
            route_a.reward += route_b.reward
            for i in route_b.edges[:-1]:
                i.end.route = route_a

            self.routes.pop(self.routes.index(route_b))

    def local_search_same_route(self):
        """
        En el determinista no tiene sentido, pero en el simheurístico sí.
        :return:
        """
        for route in self.routes:
            for edge in route:
                pass

    def determinstic_algorithm(self):
        self.dummy_solution()
        self.create_saving_list()
        while len(self.savings) != 0:
            self.merge_routes(self.select_saving())

        self.routes.sort(key=lambda x: x.reward, reverse=True)
        self.of = sum([self.routes[i].reward for i in range(self.max_vehicles)])

        #print(self.of)
        #for i in self.routes:
        #    print(i.__str__())

        return self.routes, self.of

    def determinstic_algorithm_test(self):
        self.dummy_solution()
        self.create_saving_list()
        while len(self.savings) != 0:
            self.merge_routes(self.select_saving())

        self.routes.sort(key=lambda x: x.reward, reverse=True)

        simulations_test = []
        for i in range(self.max_vehicles):
            if np.random.uniform() < 0.5:
                simulations_test.append(1)
            else:
                simulations_test.append(0)

        self.of = sum([self.routes[i].reward * simulations_test[i] for i in range(self.max_vehicles)])

        real_of = 0
        a = self.routes[0].edges
        for i in self.routes[0].edges:
            node = i.end
            if np.random.uniform() < 0.5:
                real_of += node.reward
            else:
                real_of += 0
        print(real_of)

        #print(self.of)
        #for i in self.routes:
        #    print(i.__str__())

        return self.routes, self.of


    def deterministic_multi_start(self, max_time):
        start = time.time()
        best_route, best_of = self.determinstic_algorithm()
        while time.time()-start <= max_time:
            self.reset()
            new_route, new_sol = self.determinstic_algorithm()
            if best_of < new_sol:
                best_of = new_sol
                best_route = copy.deepcopy(new_route)

        self.routes = best_route
        print(best_of)
        for i in self.routes:
            print(i.__str__())


    def test_dynamic_algo(self):
        self.dummy_solution()
        self.create_saving_list()
        while len(self.savings) != 0:
            self.merge_routes(self.select_saving())

        self.routes.sort(key=lambda x: x.reward, reverse=True)

        simulations_test = []
        for i in range(self.max_vehicles):
            if np.random.uniform() < 0.5:
                simulations_test.append(1)
            else:
                simulations_test.append(0)

        self.of = sum([self.routes[i].reward * simulations_test[i] for i in range(self.max_vehicles)])

        real_of = 0
        a = self.routes[0].edges
        for i in self.routes[0].edges:
            node = i.end
            if np.random.uniform() < 0.5:
                real_of += node.reward
            else:
                real_of += 0
        print(real_of)

        #print(self.of)
        #for i in self.routes:
        #    print(i.__str__())

        return self.routes, self.of



    def create_saving_list_dyn(self):
        for i in range(len(self.nodes)-2):
            for j in range(len(self.nodes)-2):
                if i == j:
                    continue
                edge_a_b = edge(self.nodes[i+1], self.nodes[j+1])
                edge_a_end = edge(self.nodes[i + 1], self.nodes[-1])
                edge_depot_b = edge(self.nodes[0], self.nodes[j + 1])
                self.savings.append(saving(self.nodes[j + 1],
                                           self.nodes[i + 1],
                                           self.alpha * (edge_a_end.distance + edge_depot_b.distance - edge_a_b.distance) + (1 - self.alpha) * (self.nodes[i+1].reward + self.nodes[j+1].reward),
                                           edge_a_b.distance))

        self.savings.sort(key = lambda x: x.distance)