from src.algorithm.static_algorithm import *
from src.utils.Constants import Algorithm, Key
from src.utils.contextual_TS import OnlineLogisticRegression
from src.algorithm.dynamic import *
from src.algorithm.dynamic_constructive import DynamicConstructive
from src.algorithm.dynamic_with_bb import DynamicConstructiveBB
from src.algorithm.static_constructive import StaticConstructive


class Solution:

    def __init__(self, nodes, max_dist, seed=0, max_vehicles=1, alpha=0.7, neighbour_limit=-1, bb=None,
                 dict_of_types=None,
                 max_iter_dynamic=100, max_iter_random=100, beta_bias=0.8, standard=True):
        self.routes = []
        self.of = 0

        self.beta_bias = beta_bias

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

        self.standard = standard

    def reset(self):
        self.routes = []
        self.savings = []

    def local_search_same_route(self):
        raise NotImplementedError("La subclase debe implementar este método abstracto")

    def run(self, algo, select_saving_function, instance):

        if algo == Algorithm.STATIC:
            s = Static(self.nodes, self.max_dist, self.seed, self.max_vehicles, self.alpha, self.neighbour_limit,
                       self.bb, self.dict_of_types, self.max_iter_dynamic,
                       select_saving_function=select_saving_function)

            return s.run_multi_scenarios_static(self.max_iter_random)

        if algo == Algorithm.DYNAMIC:
            ts = {k: OnlineLogisticRegression(0.5, 2, 3) for k in range(instance[Key.N_TYPE_NODES])}
            s = Dynamic(self.nodes, self.max_dist, self.seed, self.max_vehicles, self.alpha, self.neighbour_limit,
                        self.bb, ts, self.dict_of_types, self.max_iter_dynamic,
                        select_saving_function=select_saving_function)

            return s.run_multi_start_dynamic(10, self.max_iter_random)

        if algo == Algorithm.CONSTRUCTIVE_DYNAMIC:
            ts = {k: OnlineLogisticRegression(0.5, 1, 3) for k in range(instance[Key.N_TYPE_NODES])}
            s = DynamicConstructive(self.nodes, self.max_dist, self.seed, self.max_vehicles, self.alpha,
                                    self.neighbour_limit, self.bb, ts, self.dict_of_types, instance[Key.N_TYPE_NODES],
                                    self.max_iter_dynamic, beta=self.beta_bias, standard=self.standard)

            return s.run_dynamic()

        if algo == Algorithm.CONSTRUCTIVE_STATIC:
            s = StaticConstructive(self.nodes, self.max_dist, self.seed, self.max_vehicles, self.alpha,
                                   self.neighbour_limit, self.bb, self.dict_of_types, instance[Key.N_TYPE_NODES],
                                   self.max_iter_dynamic, beta=self.beta_bias,
                                   select_saving_function=select_saving_function, standard=self.standard)

            return s.run_static_constructive(instance["max_iter_random"])

        if algo == Algorithm.CONSTRUCTIVE_DYNAMIC_BB:
            ts = {k: OnlineLogisticRegression(0.5, 1, 3) for k in range(instance[Key.N_TYPE_NODES])}
            s = DynamicConstructiveBB(self.nodes, self.max_dist, self.seed, self.max_vehicles, self.alpha,
                                      self.neighbour_limit, self.bb, ts, self.dict_of_types, instance[Key.N_TYPE_NODES],
                                      self.max_iter_dynamic, self.standard)

            return s.run_dynamic_bb()

    def run_dynamic(self):
        raise NotImplementedError("La subclase debe implementar este método abstracto")
