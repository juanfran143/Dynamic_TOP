from src.utils.classes import Node
import os


def go_up_to_dynamic_top(start_path):
    current_path = start_path
    while os.path.basename(current_path) != "Dynamic_TOP":
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):  # This means we're at the root and didn't find "Dynamic_TOP"
            return None
    return current_path


def find_instances_directory(start_path):
    for root, dirs, files in os.walk(start_path):
        if "Instances" in dirs:
            return os.path.join(root, "Instances")
    return None


def read(name="tsiligirides_problem_1_budget_40.txt"):
    instances_directory = find_instances_directory(go_up_to_dynamic_top(os.getcwd()))
    nodes = []
    f = open(instances_directory + "\\" + name, "r")
    f1 = open(instances_directory + "\\" + name, "r")
    a = f1.read()

    capacity, vehicles = f.readline()[:-1].split("\t")
    aux = f.readline()[:-1].split("\t")
    nodes.append(Node(0, 0, float(aux[0]), float(aux[1])))
    aux = f.readline()[:-1].split("\t")
    nodes.append(Node(a.count("\n"), 0, float(aux[0]), float(aux[1])))

    aux = f.readline()[:-1].split("\t")
    i = 1
    while len(aux) != 1:
        nodes.append(Node(i, float(aux[2]), float(aux[0]), float(aux[1])))
        i += 1
        aux = f.readline()[:-1].split("\t")

    nodes.append(nodes[1])
    nodes.pop(1)

    return nodes, int(capacity), int(vehicles)
