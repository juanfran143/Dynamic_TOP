import copy
from src.utils.classes import *
import random as rnd
import numpy as np
from src.utils.Constants import Algorithm
import time


class Dynamic:
    """
    Clase que ejecutará el algoritmo statico, es decir, el que no tiene en cuenta
    el dinamismo del problema. Habrá una función statica que es la que se tendrá
    que ejecutar: "static_algorithm(nodes, max_dist, max_vehicles, alpha, bb, dict_of_types, neighbour_limit=inf)".

    Esta función tiene como input los conceptos básicos:
        - nodes: Nodos de la instancia
        - max_distance: Máxima distancia que puede hacer los drones
        - max_vehicles: Número de drones
        - alpha: Parámetro que regula los savings
        - bb: Black-box de nuestro problema
        - dict_of_types: Los diferetes tipos de nodos que tenemos

    Opcional:
        - neighbour_limit: Marcará el límite en distancia que puede haber entre 2 nodos para marcarlos como vecinos.
                           Por defecto será infinito.

    Devuelve:
        - Función objetivo media de los escenarios estáticos
        - Función objetivo media de los escenarios dinámica
        - Ruta para cada instancia

    """

    def __init__(self, nodes, max_dist, seed=0, max_vehicles=1, alpha=0.7, neighbour_limit=-1, bb=None, ts=None,
                 dict_of_types=None, max_iter_dynamic=100, select_saving_function=None, random_selection=2):
        self.routes = []
        self.of = 0

        self.seed = seed
        random.seed = seed
        np.seed = seed

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
        self.ts = ts # OnlineLogisticRegression(0.5, 1, 3)
        self.X = {k: [] for k in range(len(ts.keys()))}
        self.y = {k: [] for k in range(len(ts.keys()))}
        self.max_iter_dynamic = max_iter_dynamic

        if dict_of_types:
            self.dict_of_types = dict_of_types
        else:
            self.dict_of_types = {i: 1 for i in range(len(nodes))}

        self.random_selection = random_selection
        self.select_saving_function = self.saving_function(select_saving_function)

    def saving_function(self, select_saving_function):
        if select_saving_function == Algorithm.SELECT_SAVING_GREEDY:
            return self.select_saving_greedy
        if select_saving_function == Algorithm.SELECT_SAVING_GRASP:
            return self.select_saving_grasp
        if select_saving_function is None:
            return self.select_saving_grasp

    def reset(self):
        self.routes = []
        self.savings = []

    def dummy_solution(self):
        for i in range(len(self.nodes) - 2):
            edges = [Edge(self.nodes[0], self.nodes[i + 1]), Edge(self.nodes[i + 1], self.nodes[-1])]
            self.routes.append(Route(i, edges, sum([i.distance for i in edges])))
            self.routes[i].reward = self.nodes[i + 1].reward
            self.nodes[i + 1].route = self.routes[i]

    def is_neighbour(self, edge: Edge):
        if self.neighbour_limit == -1:
            return True
        if edge.distance > self.neighbour_limit:
            return False
        return True

    def create_saving_list(self):
        for i in range(len(self.nodes) - 2):
            for j in range(len(self.nodes) - 2):
                if i == j:
                    continue
                edge_a_b = Edge(self.nodes[i + 1], self.nodes[j + 1])
                if not self.is_neighbour(edge_a_b):
                    continue

                edge_a_end = Edge(self.nodes[i + 1], self.nodes[-1])
                edge_depot_b = Edge(self.nodes[0], self.nodes[j + 1])
                
                node_type_a = self.dict_of_types[self.nodes[i + 1].id]
                node_type_b = self.dict_of_types[self.nodes[j + 1].id]

                array_a = np.array((self.weather, self.congestion[self.nodes[i + 1].id], 1))
                array_b = np.array((self.weather, self.congestion[self.nodes[i + 1].id], 1))
                ts_sim_a = self.ts[node_type_a].predict_proba(array_a, 'sample')
                ts_sim_b = self.ts[node_type_b].predict_proba(array_b, 'sample')
                
                saving_distance = self.alpha * (edge_a_end.distance + edge_depot_b.distance - edge_a_b.distance)
                saving_reward = (1 - self.alpha) * (self.nodes[i + 1].reward * ts_sim_a[0] +
                                                    self.nodes[j + 1].reward * ts_sim_b[0])

                self.savings.append(Saving(self.nodes[j + 1], self.nodes[i + 1], saving_distance + saving_reward,
                                           edge_a_b.distance))

        self.savings.sort(key=lambda x: x.saving)

    def select_saving_grasp(self):
        return self.savings.pop(rnd.randint(0, min([self.random_selection, len(self.savings) - 1])))

    def select_saving_greedy(self):
        return self.savings.pop(0)

    def merge_routes(self, saving):
        route_a = saving.start.route
        route_b = saving.end.route
        distance = route_a.distance + route_b.distance + saving.a_to_b - route_a.edges[-1].distance - route_b.edges[
            0].distance
        if route_a.id != route_b.id and route_a.edges[-1].start.id == saving.start.id and route_b.edges[
            0].end.id == saving.end.id and distance <= self.max_dist:

            route_a.edges.pop()
            route_a.edges = route_a.edges + [Edge(saving.start, saving.end)] + route_b.edges[1:]
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
            for Edge in route:
                pass

    def dynamic_algorithm(self):
        self.dummy_solution()
        self.create_saving_list()
        while len(self.savings) != 0:
            self.merge_routes(self.select_saving_function())

        self.routes.sort(key=lambda x: x.reward, reverse=True)
        self.of = sum([self.routes[i].reward for i in range(self.max_vehicles)])

        return self.routes, self.of

    def dynamic_algorithm_real_time(self):
        self.dummy_solution()
        self.create_saving_list()
        while len(self.savings) != 0:
            self.merge_routes_real_time(self.select_saving_function(self))

        self.routes.sort(key=lambda x: x.reward, reverse=True)
        self.of = sum([self.routes[i].reward for i in range(self.max_vehicles)])

        return self.routes, self.of

    def change_environment(self):
        random.seed = self.seed
        self.seed += 1
        self.weather = random.randint(0, 1)
        self.congestion = {i: random.randint(0, 1) for i in range(len(self.nodes))}

    def dynamic_of(self):
        of_list = []
        for _ in range(self.max_iter_dynamic):
            of = 0
            for r in self.routes[:self.max_vehicles]:
                distance = 0
                for e in r.edges[:-1]:
                    distance += e.distance
                    node_type = self.dict_of_types[e.end.id]
                    has_reward = self.bb.simulate(node_type, distance / self.max_dist, self.weather,
                                                  self.congestion[e.end.id])

                    self.X[node_type] = np.append(self.X[node_type], np.array([self.weather, self.congestion[e.end.id],
                                                  distance / self.max_dist]))
                    self.y[node_type] = np.append(self.y[node_type], has_reward)

                    of += e.end.reward * has_reward
                    self.change_environment()

            of_list.append(of)

        return np.mean(of_list)

    def run_dynamic(self):
        self.dynamic_algorithm()
        dynamic_of = self.dynamic_of()
        return self.routes[:self.max_vehicles], self.of, dynamic_of

    def dynamic_multi_start_iter(self, max_iter: int):
        best_route, _, best_of = self.run_dynamic()
        for _ in range(max_iter):
            self.reset()
            new_route, _, new_sol = self.run_dynamic()
            if best_of < new_sol:
                best_of = new_sol
                best_route = copy.deepcopy(new_route)

        self.routes = best_route

    def run_multi_start_dynamic(self, iter_problem, max_iter):
        for _ in range(iter_problem):
            self.dynamic_multi_start_iter(max_iter)
            for k in self.ts.keys():
                if len(self.y[k]) > 1:
                    self.ts[k].fit(self.X[k], self.y[k])
        return self.routes[:self.max_vehicles], self.of
