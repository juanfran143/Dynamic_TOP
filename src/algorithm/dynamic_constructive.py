import copy
from src.utils.classes import *
import random as rnd
import numpy as np
import time
from src.utils.Constants import Algorithm
from statistics import mean

class DynamicConstructive:
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

    def __init__(self, nodes, max_dist, seed=0, max_vehicles=1, alpha=0.7, neighbour_limit=-1, bb=None, wb=None,
                 dict_of_types=None, n_types_nodes=2, max_iter_dynamic=100, select_saving_function=None, random_selection=2):
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
        self.n_types_nodes = n_types_nodes

        random.seed = self.seed
        self.weather = random.choice([-1, 1])
        self.congestion = {i: random.choice([-1, 1]) for i in range(len(nodes))}
        self.bb = bb
        self.ts = wb # OnlineLogisticRegression(0.5, 1, 3)
        self.new_data = {i: [] for i in range(n_types_nodes)}
        #self.X = np.array([])
        #self.y = np.array([])
        self.max_iter_dynamic = max_iter_dynamic

        if dict_of_types:
            self.dict_of_types = dict_of_types
        else:
            self.dict_of_types = {i: 1 for i in range(len(nodes))}

        self.random_selection = random_selection
        self.select_saving_function = self.saving_function(select_saving_function)

    def saving_function(self, select_saving_function):
        if select_saving_function == Algorithm.SELECT_SAVING_GREEDY:
            return None
        if select_saving_function == Algorithm.SELECT_SAVING_GRASP:
            return None
        if select_saving_function is None:
            return None

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

    """
    not needed this code right?

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
                
                node_type_a = self.dict_of_types[self.nodes[i + 1]]
                node_type_b = self.dict_of_types[self.nodes[j + 1]]
                ts_sim_a = ts.simulate((node_type_a, self.weather, self.congestion[self.nodes[i + 1]], 1))
                ts_sim_b = ts.simulate((node_type_b, self.weather, self.congestion[self.nodes[i + 1]], 1))
                
                saving_distance = self.alpha * (edge_a_end.distance + edge_depot_b.distance - edge_a_b.distance)
                saving_reward = (1 - self.alpha) * (self.nodes[i + 1].reward * ts_sim_a + 
                                                    self.nodes[j + 1].reward * ts_sim_b) 

                self.savings.append(Saving(self.nodes[j + 1], self.nodes[i + 1], saving_distance + saving_reward,
                                           edge_a_b.distance))

        self.savings.sort(key=lambda x: x.saving)

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
"""

    def constructive_dynamic_solution(self):
        self.routes = {}
        self.new_data = {i: [] for i in range(self.n_types_nodes)}
        nodes = copy.deepcopy(self.nodes)

        self.routes = {k: [nodes[0]] for k in range(self.max_vehicles)}
        end = {k: False for k in range(self.max_vehicles)}
        dist = {k: 0 for k in range(self.max_vehicles)}
        reward = {k: 0 for k in range(self.max_vehicles)}
        dynamic_reward = {k: 0 for k in range(self.max_vehicles)}
        nodes_used = []
        while sum(end.values()) != len(end):
            for v in range(self.max_vehicles):
                if end[v]:
                    continue
                start_node = self.routes[v][-1]
                self.savings = []
                for j in range(len(self.nodes) - 2):
                    edge_a_b = Edge(start_node, self.nodes[j + 1])

                    if not self.is_neighbour(edge_a_b) or self.nodes[j+1].id in nodes_used:
                        continue

                    edge_depot_b = Edge(self.nodes[j + 1], self.nodes[-1])

                    if dist[v] + edge_depot_b.distance + edge_a_b.distance > self.max_dist:
                        continue

                    node_type_b = self.dict_of_types[self.nodes[j + 1].id]
                    array_b = np.array((self.weather, self.congestion[self.nodes[j + 1].id],
                                        (((1-(dist[v] + edge_a_b.distance)/self.max_dist) - 0.5)*2)))
                    ts_sim_b = self.ts[node_type_b].predict_proba(array_b, 'sample')

                    saving_distance = self.alpha * edge_a_b.distance
                    saving_reward = (1 - self.alpha) * (self.nodes[j + 1].reward * ts_sim_b[0])

                    self.savings.append(Saving(start_node, self.nodes[j + 1], saving_distance + saving_reward,
                                               edge_a_b.distance))

                if len(self.savings) == 0:
                    end[v] = True
                else:
                    self.savings.sort(key=lambda x: x.saving)

                if len(self.savings) != 0:
                    self.routes[v].append(self.savings[0].end)
                    nodes_used.append(self.savings[0].end.id)

                    reward[v] += self.savings[0].end.reward
                    dist[v] += self.savings[0].a_to_b
                    node_id = self.savings[0].end.id
                    node_type = self.dict_of_types[node_id]

                    weather = self.weather
                    congestion = self.congestion[node_id]
                    battery = ((1-dist[v]/self.max_dist) - 0.5)*2 #transformed into (-1,1)
                    #print(weather,congestion,battery)
                    has_reward = self.bb.simulate(node_type= node_type,weather= weather,
                                                  congestion=congestion, battery = battery, verbose=False)
                    new_row = np.array([weather, congestion, battery, has_reward])
                    self.new_data[node_type].append(new_row)
                    dynamic_reward[v] += self.savings[0].end.reward * has_reward
                else:
                    last = Edge(self.routes[v][-1], self.nodes[-1])
                    self.routes[v].append(self.nodes[-1])
                    dist[v] += last.distance
                self.change_environment()

        routes = []
        for k, v in self.routes.items():
            edges = []
            for n in range(len(v)-1):
                edges.append(Edge(v[n], v[n+1]))
            routes.append(Route(k, edges, dist[k]))
            routes[k].reward = reward[k]
        self.routes = routes
        return sum(dynamic_reward[v] for v in range(self.max_vehicles))

    def constructive_dynamic_algorithm(self):
        of_list = []
        for _ in range(0, 100):
            dynamic_of = self.constructive_dynamic_solution()
            of_list.append(dynamic_of)
            for i in range(0, 2):
                data = self.new_data[i]
                X = np.array(data)[:, :3].reshape(-3, 3)
                Y = np.array(data)[:, 3:]
                self.ts[i].fit(X,Y)
        print((mean(of_list)))
        self.of = sum([self.routes[i].reward for i in range(self.max_vehicles)])

        return self.routes, self.of, dynamic_of

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
        self.weather = random.choice([-1, 1])
        self.congestion = {i: random.choice([-1, 1])for i in range(len(self.nodes))}

    def dynamic_of(self):
        of_list = []
        for _ in range(self.max_iter_dynamic):
            of = 0
            for r in self.routes[:self.max_vehicles]:
                distance = 0
                for e in r.edges[:-1]:
                    self.change_environment()
                    distance += e.distance
                    has_reward = self.bb.simulate(self.dict_of_types[e.end.id], self.weather,
                                                  self.congestion[e.end.id], ((distance / self.max_dist)-0.5)/2)
                    of += e.end.reward * has_reward

            of_list.append(of)

        return np.mean(of_list)

    def run_dynamic(self):
        self.dynamic_algorithm()
        dynamic_of = self.dynamic_of()
        return self.routes[:self.max_vehicles], self.of, dynamic_of

    def static_multi_start_iter(self, max_iter: int):
        best_route, best_of = self.static_algorithm()
        for _ in range(max_iter):
            self.reset()
            new_route, new_sol = self.static_algorithm()
            if best_of < new_sol:
                best_of = new_sol
                best_route = copy.deepcopy(new_route)

        self.routes = best_route

    def run_multi_start_static(self, max_iter):
        self.static_multi_start_iter(max_iter)
        dynamic_of = self.dynamic_of()
        return self.routes[:self.max_vehicles], self.of, dynamic_of
