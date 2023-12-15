import copy
from src.utils.classes import *
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, roc_auc_score, roc_curve
from math import log


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
                 dict_of_types=None, n_types_nodes=2, max_iter_dynamic=100, beta=0.7, standard=True):
        self.routes = []
        self.of = 0
        self.beta = beta
        self.mseList = []

        self.standard = standard
        self.seed_route = seed
        self.seed = seed
        np.random.seed(seed)

        self.nodes = nodes
        self.savings = []
        self.alpha = alpha
        self.max_dist = max_dist
        self.max_vehicles = max_vehicles
        self.neighbour_limit = neighbour_limit

        self.n_types_nodes = n_types_nodes

        self.weather = np.random.choice([-1, 1])
        self.congestion = {i: np.random.choice([-1, 1]) for i in range(len(nodes))}
        self.bb = bb
        self.ts = wb
        self.new_data = {i: [] for i in range(n_types_nodes)}
        self.max_iter_dynamic = max_iter_dynamic

        if dict_of_types:
            self.dict_of_types = dict_of_types
        else:
            self.dict_of_types = {i: 1 for i in range(len(nodes))}

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

        max_distance = 1
        max_reward = 1
        if self.standard:
            max_distance = max([i.distance(j) for i in self.nodes for j in self.nodes])
            max_reward = max([i.reward for i in self.nodes])

        while sum(end.values()) != len(end):
            for v in range(self.max_vehicles):
                if end[v]:
                    continue
                start_node = self.routes[v][-1]
                self.savings = []
                for j in range(len(self.nodes) - 2):
                    edge_a_b = Edge(start_node, self.nodes[j + 1])

                    if not self.is_neighbour(edge_a_b) or self.nodes[j + 1].id in nodes_used:
                        continue

                    edge_depot_b = Edge(self.nodes[j + 1], self.nodes[-1])

                    if dist[v] + edge_depot_b.distance + edge_a_b.distance > self.max_dist:
                        continue

                    node_type_b = self.dict_of_types[self.nodes[j + 1].id]
                    array_b = np.array((self.weather, self.congestion[self.nodes[j + 1].id],
                                        (((1 - (dist[v] + edge_a_b.distance) / self.max_dist) - 0.5) * 2)))

                    ts_sim_b = self.ts[node_type_b].predict_proba(array_b, 'sample')
                    prob_node = ts_sim_b[1]
                    """
                    bb_sim_b = self.bb.get_value(node_type_b,  self.weather, self.congestion[self.nodes[j + 1].id], 
                                                 (((1 - (dist[v] + edge_a_b.distance) / self.max_dist) - 0.5) * 2))
                    saving_reward = (1 - self.alpha) * (self.nodes[j + 1].reward/max_reward * bb_sim_b)
                    """
                    saving_distance = self.alpha * (1-edge_a_b.distance/max_distance)

                    saving_reward = (1 - self.alpha) * (self.nodes[j + 1].reward/max_reward * prob_node)

                   #if (prob_node < 0.25):
                   #  saving_reward -= 1

                    self.savings.append(Saving(start_node, self.nodes[j + 1], saving_distance + saving_reward,
                                               edge_a_b.distance))

                if len(self.savings) == 0:
                    end[v] = True
                else:
                    self.savings.sort(key=lambda x: x.saving, reverse=True)

                if len(self.savings) != 0:
                    # Changed to BIAS
                    index = int((log(np.random.random()) / log(1 - self.beta))) % len(self.savings)
                    self.routes[v].append(self.savings[index].end)
                    nodes_used.append(self.savings[index].end.id)

                    reward[v] += self.savings[0].end.reward
                    dist[v] += self.savings[0].a_to_b
                    node_id = self.savings[0].end.id
                    node_type = self.dict_of_types[node_id]

                    weather = self.weather
                    congestion = self.congestion[node_id]
                    # transformed into (-1,1)
                    battery = ((1 - dist[v] / self.max_dist) - 0.5) * 2
                    # print(weather,congestion,battery)

                    has_reward = self.bb.simulate(node_type=node_type, weather=weather,
                                                  congestion=congestion, battery=battery, verbose=False)

                    new_row = np.array([weather, congestion, battery, has_reward])
                    self.new_data[node_type].append(new_row)
                    dynamic_reward[v] += self.savings[0].end.reward * has_reward
                    self.change_environment()

                else:
                    last = Edge(self.routes[v][-1], self.nodes[-1])
                    self.routes[v].append(self.nodes[-1])
                    dist[v] += last.distance


        routes = []
        for k, v in self.routes.items():
            edges = []
            for n in range(len(v) - 1):
                edges.append(Edge(v[n], v[n + 1]))
            routes.append(Route(k, edges, dist[k]))
            routes[k].reward = reward[k]
        self.routes = routes
        return sum(dynamic_reward[v] for v in range(self.max_vehicles))

    def change_environment(self):
        self.seed_route += 1
        np.random.seed(self.seed_route)
        self.weather = np.random.choice([-1, 1])
        self.congestion = {i: np.random.choice([-1, 1]) for i in range(len(self.nodes))}

    def fit_wb(self):
        for i in range(self.n_types_nodes):
            data = self.new_data[i]
            if len(data) > 0:
                x = np.array(data)[:, :-1].reshape(-3, 3)
                y = np.array(data)[:, -1]
                self.ts[i].fit(x, y)

    def change_seed(self):
        self.seed += 10000
        self.seed_route = self.seed + 1
        np.random.seed(self.seed)

    def check_wb(self):
        ts = []
        bb = []
        dic = {}
        for j in range(self.n_types_nodes):
            for weather in [-1, 1]:
                for congestion in [-1, 1]:
                    for battery in range(-10, 10):
                        node_type_b = j
                        array_b = np.array((weather, congestion, battery / 10))
                        ts_value = round(self.ts[node_type_b].predict_proba(array_b, 'sample')[1], 2)

                        bb_value = self.bb.get_value(node_type=node_type_b, weather=weather,
                                                    congestion=congestion, battery=battery/10)
                        ts.append(ts_value)
                        bb.append(bb_value)
                        dic[(j, weather, congestion, battery / 10)] = (ts_value, bb_value)

        # Cálculo de MSE, RMSE, MAE y Log-Loss
        mse = mean_squared_error(ts, bb)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(ts, bb)

        self.mseList.append(mse)

        #print(dic)

        #print(f"MSE: {mse}; RMSE: {rmse}; MAE: {mae}")

    def run_dynamic(self):
        of_dynamic_list, of_list, route_list = [], [], []
        for _ in range(self.max_iter_dynamic):
            of_dynamic_list.append(self.constructive_dynamic_solution())
            of_list.append(sum([self.routes[i].reward for i in range(self.max_vehicles)]))
            route_list.append(copy.deepcopy(self.routes))
            self.fit_wb()
            self.change_seed()
            self.check_wb()
        print(self.mseList)
        import matplotlib.pyplot as plt

        plt.plot(self.mseList)
        plt.ylabel('MSE')
        plt.xlabel('batches of data')
        plt.show()

        return route_list, of_list, of_dynamic_list
