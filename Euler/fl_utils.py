import copy
import torch
from torch import nn
import numpy as np
import networkx as nx
import time
from scipy import stats
import torch.nn.functional as F

def calculate_model_memory_cost(model, input_data):
    return
    # model = model
    dummy_input = input_data
    # torch.randn(*input_size, device=device)

    # Parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

    # Forward pass to calculate activation memory
    activations = []
    hooks = []

    def forward_hook(module, input, output):
        # Hook to capture activation memory
        if isinstance(output, torch.Tensor):
            activations.append(output.numel() * output.element_size())
        elif isinstance(output, (tuple, list)):
            activations.extend(o.numel() * o.element_size() for o in output if isinstance(o, torch.Tensor))

    # Register hooks
    for module in model.modules():
        hooks.append(module.register_forward_hook(forward_hook))

    # Perform a forward pass
    with torch.no_grad():
        model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Activation memory
    activation_memory = sum(activations)

    # Total memory
    total_memory = param_memory + activation_memory

    # Convert to MB
    memory_details = {
        'param_memory_mb': param_memory / (1024 ** 2),
        'activation_memory_mb': activation_memory / (1024 ** 2),
        'total_memory_mb': total_memory / (1024 ** 2)
    }
    print(memory_details)

    return memory_details

def FedAvg_back(w):
    w_avg = copy.deepcopy(w[0])
    weights = torch.tensor([0.8, 0.2], dtype=torch.float64).cuda()
    weights = torch.mul(weights, len(w))
    # print(*w_avg)
    for k in w_avg.keys():
        w_avg[k] = torch.mul(w_avg[k], weights[0])
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] *weights[i]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg(w):
    #i: number of clients, k: different params
    w_avg = copy.deepcopy(w[0])
    # print(*w_avg)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            if i == 3:
                w_avg[k] += w[i][k] #* 100
            else:
                w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def calculate_client_weights(cm):
    # Simple Example: Weighting by the number of samples per client
    weights = [len(cm[c]) for c in cm]
    total_weight = sum(weights)
    weights = [weight / total_weight for weight in weights]
    #ZL: raw weights, good on LANL
    return weights
    #ZL: softmax weights to smooth the differences, not as good as raw weights on LANL
    #return np.exp(weights) / np.sum(np.exp(weights))
    #sqrt weights
    #return np.sqrt(weights) / np.sum(np.sqrt(weights))


def reverse_machine_cluster(mc):
    #reverse machine_clusters and count
    cm = {}
    for m in mc.keys():
        if not mc[m] in cm:
            cm[mc[m]] = []
        cm[mc[m]].append(m)
    return cm

def WeightedFedAvg(w, client_list, mc):
    #ZL: compute weights from mc and client_list
    cm = reverse_machine_cluster(mc)
    weights = calculate_client_weights(cm)
    print('Clients ' + str(client_list) + ' weights are ' + str(weights))
    w_avg = copy.deepcopy(w[0])
    # print(*w_avg)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += weights[i] * w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def build_random_graph(mc):
    #build a random graph from the node list
    #Number of nodes (n) and edges (m) to attach from a new node to existing nodes
    #ZL: I'm not sure about m, just a random number for now

    nodes = mc.keys()
    n = len(nodes)
    m = 5
    # print(m, "!!!!!!!!!")
    tmp = time.time()
    # Generate a random graph using the Barabási-Albert model
    global_G = nx.barabasi_albert_graph(n, m)
    tmp_time = time.time() - tmp
    print("random graph time: ", tmp_time)

    # Relabel the nodes according to your specific set
    mapping = {i: node for i, node in enumerate(nodes)}
    global_G = nx.relabel_nodes(global_G, mapping)

    # Get some statistics
    print("Random graph generated")
    print(f"Number of nodes: {global_G.number_of_nodes()}")
    print(f"Number of edges: {global_G.number_of_edges()}")
    print(f"Average clustering: {nx.average_clustering(global_G)}")
    #print(f"Diameter: {nx.diameter(global_G)}")
    #print(f"Degree centrality: {nx.degree_centrality(global_G)}")

    return global_G

def calculate_client_weights2(similarity, w):
    total_weight = sum(similarity)
    weights = []
    for i, weight in enumerate(similarity):
        if w[i] != None:
            weights.append(weight / total_weight)

    # weights = [weight / total_weight for weight in similarity]
    return weights


def WeightedFedAvg2(w, client_list, similarity, defense=False):
    #ZL: weights based on graph similarity
    #print('Clients similarities to the random graph is ' + str(similarity))

    if defense == True:
        weight_sum = []
        for i in range(len(w)):
            total = 0
            for value in w[i].values():
                # print(np.absolute(np.sum(value.detach().tolist())))
                total += np.absolute(np.sum(value.detach().tolist()))
            weight_sum.append(total)
        z_scores = np.abs(stats.zscore(weight_sum))
        print(z_scores)
        threshold = 1.5
        anomaly_indices = np.where(z_scores > threshold)[0]
        # Create a list of anomalies and their corresponding values
        anomalies = [(weight_sum[i], z_scores[i]) for i in anomaly_indices]
        # Sort anomalies by the magnitude of their Z-scores (higher Z-score means more anomalous)
        sorted_anomalies = sorted(anomalies, key=lambda x: x[1], reverse=True)
        if len(sorted_anomalies) == 0:
            print("no attack detected.")
        else:
            print("malicious client:", anomaly_indices)
        weights = calculate_client_weights2(similarity)
        print('Clients ' + str(client_list) + ' weights are ' + str(weights))
        w_avg = copy.deepcopy(w[0])
        # print(client_index, POISON)
        for k in w_avg.keys():
            for i in range(1, len(w)):
                if i in anomaly_indices:
                    continue
                else:
                    w_avg[k] += weights[i] * w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w)-len(anomaly_indices))
    else:
        weights = calculate_client_weights2(similarity)
        print('Clients ' + str(client_list) + ' weights are ' + str(weights))
        w_avg = copy.deepcopy(w[0])
        # print(client_index, POISON)
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += weights[i] * w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def calculate_contribution_metrics(w_local, w_fed):
    mass = []
    velocity_list = []
    for k in w_fed.keys():
        # Calculate the cosine of the angle using dot product and norm
        dot_product = torch.dot(w_local[k].flatten(), w_fed[k].flatten())
        norm_local = torch.linalg.norm(w_local[k])
        norm_fed = torch.linalg.norm(w_fed[k])
        cos_angle = dot_product / (norm_local * norm_fed)
        # print(dot_product, norm_local * norm_fed)
        single_mass = torch.nan_to_num(torch.cos(cos_angle)) # Some varibales are 0
        mass.append(single_mass)

        # Calculate the norm of the difference (Euclidean distance)
        velocity_list.append(torch.linalg.norm(w_local[k] - w_fed[k]))
        # single_mass = torch.nan_to_num(torch.cos(cos_angle)) # Some varibales are 0

    # Sum up the mass and velocity
    mass = torch.sum(torch.stack(mass))
    # velocity = torch.sum(torch.stack(velocity_list))


     # Normalized velocity:
    # Calculate the L2 norm of delta_w
    velocity_list = torch.stack(velocity_list)
    norm_velocity = torch.norm(velocity_list, p=2)
    M = 5
    denominator = max(1, norm_velocity / M)
    velocity_updated = norm_velocity / denominator
    velocity = torch.sum(velocity_updated)
    print("mass, norm_velocity, velocity: ", mass.item(), norm_velocity.item(), velocity.item())
    return mass, velocity


def Momentum(w, w_fed, client_list, similarity, defense=False):
    #ZL: to update 08/16/24, add norm bounding
    if defense == True:
        weight_sum = []
        for i in range(len(w)):
            total = 0
            for value in w[i].values():
                # print(np.absolute(np.sum(value.detach().tolist())))
                total += np.absolute(np.sum(value.detach().tolist()))
            weight_sum.append(total)
        z_scores = np.abs(stats.zscore(weight_sum))
        print(z_scores)
        threshold = 1.5
        anomaly_indices = np.where(z_scores > threshold)[0]
        # Create a list of anomalies and their corresponding values
        anomalies = [(weight_sum[i], z_scores[i]) for i in anomaly_indices]
        # Sort anomalies by the magnitude of their Z-scores (higher Z-score means more anomalous)
        sorted_anomalies = sorted(anomalies, key=lambda x: x[1], reverse=True)
        if len(sorted_anomalies) == 0:
            print("no attack detected.")
        else:
            print("malicious client:", anomaly_indices)
        weights = calculate_client_weights2(similarity)
        momentums = []
        for i, w_local in enumerate(w):
            mass, velocity = calculate_contribution_metrics(w_local, w_fed)
            momentums.append(mass * velocity)

        # weights = calculate_client_weights2(momentum)
        # print(momentums)
        momentums = torch.tensor(momentums)
        weights = torch.tensor(weights)

        momentums = F.softmax(momentums, dim=0)
        # momentums = 0.9 * weights + 0.1 * momentums
        print('Weights: ', weights, momentums)

        weights = 0.8 * weights + 0.2 * momentums



        w_avg = copy.deepcopy(w[0])
        # print(client_index, POISON)
        for k in w_avg.keys():
            for i in range(1, len(w)):
                if i in anomaly_indices:
                    continue
                else:
                    w_avg[k] += weights[i] * w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w)-len(anomaly_indices))

    else:
        client_weights = calculate_client_weights2(similarity, w)
        momentums = []
        for i, w_local in enumerate(w):
            if w_local != None:
                mass, velocity = calculate_contribution_metrics(w_local, w_fed)
                momentums.append(mass * velocity)

        # weights = calculate_client_weights2(momentum)
        # print(momentums)
        momentums = torch.tensor(momentums)
        client_weights = torch.tensor(client_weights)

        momentums = F.softmax(momentums, dim=0)
        # momentums = 0.9 * weights + 0.1 * momentums
        # print('Weights: ', weights, momentums)

        weights = 0.8 * client_weights + 0.2 * momentums
        print('Weights: ', weights)
        # output_file = "weight_training.txt"
        # with open(output_file, 'a') as f:
        #     f.write(str(weights.tolist())+'\n')

        # bounded W
        M = 5
        for i in range(len(w)):
            if w[i] != None:
                w[i] = clip_update(w[i], M, w_fed)


        # from random import randrange
        # number = randrange(4)
        # print("block number: ", number)
        # w_avg = copy.deepcopy(w[0])
        # for k in w_avg.keys():
        #     if number == 0:
        #         w_avg = copy.deepcopy(w[1])
        #         for i in range(2, len(w)):
        #             w_avg[k] = weights[i] * w[i][k]
        #     else:
        #         w_avg = copy.deepcopy(w[0])
        #         for i in range(1, len(w)):
        #             if number != i:
        #                 w_avg[k] = weights[i] * w[i][k]
        #     w_avg[k] = torch.div(w_avg[k], len(w)-1)

        # w_avg = copy.deepcopy(w[0])
        # for k in w_avg.keys():
        #     for i in range(1, len(w)):
        #         w_avg[k] += weights[i] * w[i][k]
        #     w_avg[k] = torch.div(w_avg[k], len(w))

        non_none_indices = [i for i, params in enumerate(w) if params is not None]
        if not non_none_indices:
            raise ValueError("All model parameters are None. Averaging cannot be performed.")

        w_avg = copy.deepcopy(w[non_none_indices[0]])

        # Perform weighted averaging, skipping None entries
        for k in w_avg.keys():
            total_weight = 0.0
            w_avg[k] *= weights[non_none_indices[0]]  # Scale the first non-None parameters

            for i in range(len(w)):
                if w[i] is not None:
                    w_avg[k] += weights[i] * w[i][k]
                    total_weight += weights[i]

            # Divide by the total weight
            w_avg[k] = torch.div(w_avg[k], total_weight)

    return w_avg



# def Momentum(w, w_fed, client_list, similarity, defense=False, POISON=True, client_index=3):
#     if defense == True:
#         weight_sum = []
#         for i in range(len(w)):
#             total = 0
#             for value in w[i].values():
#                 # print(np.absolute(np.sum(value.detach().tolist())))
#                 total += np.absolute(np.sum(value.detach().tolist()))
#             weight_sum.append(total)
#         z_scores = np.abs(stats.zscore(weight_sum))
#         print(z_scores)
#         threshold = 1.5
#         anomaly_indices = np.where(z_scores > threshold)[0]
#         # Create a list of anomalies and their corresponding values
#         anomalies = [(weight_sum[i], z_scores[i]) for i in anomaly_indices]
#         # Sort anomalies by the magnitude of their Z-scores (higher Z-score means more anomalous)
#         sorted_anomalies = sorted(anomalies, key=lambda x: x[1], reverse=True)
#         if len(sorted_anomalies) == 0:
#             print("no attack detected.")
#         else:
#             print("malicious client:", anomaly_indices)
#         weights = calculate_client_weights2(similarity)
#         print('Clients ' + str(client_list) + ' weights are ' + str(weights))
#         w_avg = copy.deepcopy(w[0])
#         # print(client_index, POISON)
#         for k in w_avg.keys():
#             for i in range(1, len(w)):
#                 if i in anomaly_indices:
#                     continue
#                 else:
#                     w_avg[k] += weights[i] * w[i][k]
#             w_avg[k] = torch.div(w_avg[k], len(w)-len(anomaly_indices))
#     else:
#         weights = calculate_client_weights2(similarity)
#         momentums = []
#         for i, w_local in enumerate(w):
#             mass, velocity = calculate_contribution_metrics(w_local, w_fed)
#             momentums.append(mass * velocity)

#         # weights = calculate_client_weights2(momentum)
#         # print(momentums)
#         momentums = torch.tensor(momentums)
#         weights = torch.tensor(weights)

#         momentums = F.softmax(momentums, dim=0)
#         # momentums = 0.9 * weights + 0.1 * momentums
#         print('Weights: ', weights, momentums)

#         weights = 0.8 * weights + 0.2 * momentums

#         w_avg = copy.deepcopy(w[0])
#         for k in w_avg.keys():
#             # for i in range(1, len(w)):
#             #     w_avg[k] += weights[i] * w[i][k]
#             for i in range(1, len(w)):
#                 if (str(i) == client_index) & (POISON == True):
#                     # print(i, "this is i!")
#                     w_avg[k] += weights[i] * w[i][k] * 100
#                 else:
#                     w_avg[k] += weights[i] * w[i][k]

#             w_avg[k] = torch.div(w_avg[k], len(w))
#     return w_avg



def WeightedFedAvg2_back(w, client_list, similarity, POISON=False, client_index=1):
    #ZL: weights based on graph similarity
    #print('Clients similarities to the random graph is ' + str(similarity))
    weights = calculate_client_weights2(similarity)
    print('Clients ' + str(client_list) + ' weights are ' + str(weights))
    w_avg = copy.deepcopy(w[0])
    # print(client_index, POISON)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            # if (str(i) == client_index) & (POISON == True):
            #     # print(i, "this is i!")
            #     w_avg[k] += weights[i] * w[i][k] * 100
            # else:
            w_avg[k] += weights[i] * w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedProx(w):
    #ZL FedProx server side is the same as FedAvg.
    return FedAvg(w)

def FedOpt(w):
    #ZL FedProx server side is the same as FedAvg.
    return FedAvg(w)


def clip_update(w, M, fed_model):
    norm = 0.0
    delta_dict = {key: w[key] - fed_model[key] for key in w}

    for i in delta_dict.keys():
        norm += torch.norm(delta_dict[i].float())  # Changed to torch.norm and ensuring it's float for gradient updates
    scaling_factor = 1.0 if norm <= M else M / norm
    print("scaling_factor: ", scaling_factor)
    for i in fed_model.keys():
        w[i] = w[i] * scaling_factor
    return w

# def clip_update(delta_w, M, fed_model):
#     norm = 0.0
#     delta_dict = {key: delta_w[key] - fed_model[key] for key in delta_w}

#     for i in delta_dict.keys():
#         norm += torch.norm(delta_dict[i].float())  # Changed to torch.norm and ensuring it's float for gradient updates
#     scaling_factor = 1.0 if norm <= M else M / norm
#     for i in delta_w.keys():
#         delta_w[i] = delta_w[i] * scaling_factor
#     return delta_w


# Differential privacy
def add_noise(delta_w, sigma, device):
    for i in delta_w.keys():
        noise = torch.normal(0, sigma, delta_w[i].size())#.to(device)  # Changed to torch.normal
        delta_w[i] = delta_w[i] + noise
    return delta_w


class MomentAccountant:
    def __init__(self, total_participants):
        self.epsilon = 1.0
        self.total_participants = total_participants
        self.privacy_spent = 0

    def get_privacy_spent(self):
        return self.privacy_spent

    def accumulate_spent_privacy(self, z):
        self.privacy_spent += z
        print(self.privacy_spent)


def central_DP_FL(w_locals, fed_model, N, T, q, bound, noise_scale, accountant, device, similarity):
    # global_model = SimpleModel(model_size)
    # accountant = MomentAccountant(epsilon, N)
    weights = calculate_client_weights2(similarity)

    # round_num = 1
    # while True:
    if True:
        # Randomly select participants with probability q
        participating_clients = np.random.choice(N, int(N * q))
        print(participating_clients)
        # Check if the privacy budget is spent
        if accountant.get_privacy_spent() > T:
            return fed_model

        global_updates = []
        for k in participating_clients:
            # delta = []
            delta_dict = {key: w_locals[k][key] - fed_model[key] for key in w_locals[k]}
            norm = 0.0
            for i in delta_dict.keys():
                norm += torch.norm(delta_dict[i].float())
            for i in delta_dict.keys():
                delta_dict[i] = bound * delta_dict[i] / max(1, norm)
            global_updates.append(delta_dict)

        n = len(global_updates)
        sum_dict = {}

        # Iterate through each dictionary and accumulate the values
        for d in range(len(global_updates)):
            for key, value in global_updates[d].items():
                sum_dict[key] = sum_dict.get(key, 0) + value * weights[participating_clients[d]]

        # Calculate the average for each key based on the number of dictionaries
        fed_model = {key: value / n for key, value in sum_dict.items()}
        # print(fed_model)
        z = noise_scale  # Some function or value that determines the noise scale
        sigma = z * bound / q
        for i in fed_model.keys():
            noise = torch.normal(0, sigma, fed_model[i].size())#.to(device)  # Changed to torch.normal
            fed_model[i] = fed_model[i] + noise



        # noise = torch.normal(0.0, sigma, global_model.weights.size())

        # aggregated_update = torch.stack(global_updates).mean(0) + noise
        # global_model.update(aggregated_update)

        accountant.accumulate_spent_privacy(z)
        # round_num += 1

    return fed_model

