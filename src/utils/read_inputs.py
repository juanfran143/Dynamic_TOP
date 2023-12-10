from src.utils.classes import Node
import os
from src.utils.Constants import Key


def go_up_to_dynamic_top(start_path):
    current_path = start_path
    while os.path.basename(current_path) != Key.ROOT_FOLDER:
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            return None
    return current_path


def find_instances_directory(start_path):
    for root, dirs, files in os.walk(start_path):
        if Key.INSTANCE_FOLDER in dirs:
            return os.path.join(root, Key.INSTANCE_FOLDER)
    return None


def find_directory(start_path, folder):
    for root, dirs, files in os.walk(start_path):
        if folder in dirs:
            return os.path.join(root, folder)
    return None


def read_run(name="run.txt"):
    instances_directory = find_directory(go_up_to_dynamic_top(os.getcwd()), Key.FOLDER_RUN)
    with open(instances_directory + "\\" + name, 'r') as file:
        lines = file.readlines()

    filtered_lines = [line for line in lines if not line.startswith("#")]

    data_dict = []
    for line in filtered_lines:
        parts = line.strip().split(';')
        if len(parts) > 1:
            data_dict.append({
                Key.INSTANCE: str(parts[0]),
                Key.SEED: int(parts[1]),
                Key.MAX_ITER_DYNAMIC: int(parts[2]),
                Key.MAX_TIME: int(parts[3]),
                Key.ALGORITHM: parts[4].upper(),
                Key.SELECTED_NODE_FUNCTION: parts[5].upper(),
                Key.BETA_BLACKBOX: parts[6].upper(),
                Key.MAX_ITER_RANDOM: int(parts[7]),
                Key.ALPHA: float(parts[8]),
                Key.PERCENTAGE: float(parts[9]),
                Key.N_TYPE_NODES: int(parts[10]),
                Key.BETA_BIAS: float(parts[11]),
                Key.STANDARD: bool(parts[12])
            })

    return data_dict


def read(name):
    instances_directory = find_directory(go_up_to_dynamic_top(os.getcwd()), Key.INSTANCE_FOLDER)
    nodes = []
    f = open(instances_directory + "\\" + name, "r")
    f1 = open(instances_directory + "\\" + name, "r")
    a = f1.read()
    f.readline()
    vehicles = f.readline()[:-1].split(" ")[1]
    capacity = f.readline()[:-1].split(" ")[1]
    aux = f.readline()[:-1].split("\t")
    nodes.append(Node(0, 0, float(aux[0]), float(aux[1])))

    """
    aux = f.readline()[:-1].split("\t")
    nodes.append(Node(a.count("\n"), 0, float(aux[0]), float(aux[1])))
    """

    aux = f.readline()[:-1].split("\t")
    i = 1
    while len(aux) != 1:
        nodes.append(Node(i, float(aux[2]), float(aux[0]), float(aux[1])))
        i += 1
        aux = f.readline()[:-1].split("\t")
    """
    nodes.append(nodes[1])
    nodes.pop(1)
    """
    # aux = f.readline()[:-1].split("\t")
    # nodes.append(Node(a.count("\n"), 0, float(aux[0]), float(aux[1])))

    return nodes, float(capacity), int(vehicles)
