import copy
from src.utils.classes import *
import random as rnd
import numpy as np
import time
from src.algorithm.static_algorithm import *


class Solution:

    def __init__(self, nodes, max_dist, seed=0, max_vehicles=1, alpha=0.7, neighbour_limit=-1, bb=None,
                 dict_of_types=None,
                 max_iter_dynamic=100, max_iter_random=100):
        self.routes = []
        self.of = 0

        self.max_iter_random = max_iter_random
        self.seed = seed
        self.nodes = nodes
        self.savings = []
        self.alpha = alpha
        self.max_dist = max_dist
        self.max_vehicles = max_vehicles
        self.neighbour_limit = neighbour_limit

        random.seed = self.seed
        self.weather = random.randint(0, 1)
        self.congestion = {i: random.randint(0, 1) for i in range(len(nodes))}
        self.bb = bb
        self.max_iter_dynamic = max_iter_dynamic

        if dict_of_types:
            self.dict_of_types = dict_of_types
        else:
            self.dict_of_types = {i: 1 for i in range(len(nodes))}

    def reset(self):
        self.routes = []
        self.savings = []

    def select_saving(self, random_selection=2):
        raise NotImplementedError("La subclase debe implementar este método abstracto")

    def local_search_same_route(self):
        raise NotImplementedError("La subclase debe implementar este método abstracto")

    def run_static(self, select_saving_function):
        s = Static(self.nodes, self.max_dist, self.seed, self.max_vehicles, self.alpha, self.neighbour_limit,
                   self.bb, self.dict_of_types, self.max_iter_dynamic, select_saving_function=select_saving_function)

        return s.run_multi_start_static(self.max_iter_random)

    def run_dynamic(self):
        raise NotImplementedError("La subclase debe implementar este método abstracto")
