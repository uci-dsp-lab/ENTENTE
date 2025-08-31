import copy
import torch
from torch import nn
import numpy as np
import networkx as nx
import time
import json
import pandas as pd
import hashlib
from collections import defaultdict, Counter
from scipy import stats
import torch.nn.functional as F



def fed_combine(client_models, args, fed_model):
    #ZL: it seems w_locals have all weights for Encoder/Decoder/RNN
    #Pytorch state_dict[1] weights
    w_glob = fed_model.save_states()[1]
    w_locals = [client_models[i].save_states()[1] for i in range(args.client_number)]
    # if args.dp_defense == True:
    #     M = 5.0
    #     sigma = 1.0
    #     fed_model = fed_model.save_states()[1]

    #     for i in range(len(args.client_list)):

    #         clipped_update = clip_update(w_locals[i], M, fed_model)
    #         w_locals[i] = add_noise(clipped_update, sigma, args.device)

    # if args.nb_defense == True:
    #     M = 5.0
    #     fed_model = fed_model.save_states()[1]

    #     for i in range(len(args.client_list)):
    #         w_locals[i] = clip_update(w_locals[i], M, fed_model)

    # if args.cdp_defense == True:
    #     N = args.client_number  # Total number of participants
    #     T = 2  # Privacy budget threshold
    #     q = 0.5  # Fraction of participants in each round
    #     bound = 1.0  # Clipping bound
    #     noise_scale = 0.1  # Some predefined noise scale value for DP
    #     fed_model = fed_model.save_states()[1]
    #     w_glob = central_DP_FL(w_locals, fed_model, N, T, q, bound, noise_scale, accountant, args.device, args.similarity)
    #     for i in range(len(args.client_list)):
    #         client_models['model-client-'+str(args.client_list[i])].load_states(*(client_models['model-client-'+str(args.client_list[i])].save_states()[0], w_glob))
    #     # else:

    #     #ZL: initial version uses client-0 model for fed_model
    #     #print(client_models['model-client-'+str(args.client_list[0])])
    #     return client_models['model-client-'+str(args.client_list[0])], client_models

    # if args.poison == True:
    #     for pi in args.poison_client:
    #         for i in w_locals[pi].keys():
    #             w_locals[pi][i] *= 5

    #ZL TODO: implementing weighted version of FedAvg
    if args.fed == 'FedAvg':
        print('Generate global model with FedAvg over clients')
        w_glob = FedAvg(w_locals)
    elif args.fed == 'WeightedFedAvg':
        print('Generate global model with weighted FedAvg over clients')
        w_glob = WeightedFedAvg(w_locals, args.client_number, args.mc)
    elif args.fed == 'WeightedFedAvg2':
        print('Weighted FedAvg but the weights come from graph distance')
        for i in range(len(w_locals)):
            total = 0
            for value in w_locals[i].values():
                # print(np.absolute(np.sum(value.detach().tolist())))
                total += np.absolute(np.sum(value.detach().tolist()))
            print(i, total)
        w_glob = WeightedFedAvg2(w_locals, args.client_number, args.similarity, None)
    elif args.fed == 'Momentum':
        for i in range(len(w_locals)):
            total = 0
            for value in w_locals[i].values():
                total += np.absolute(np.sum(value.detach().tolist()))
            print(i, total)
        for value in w_glob.values():
            total += np.absolute(np.sum(value.detach().tolist()))
        print("fed:", total)
        # w_glob = WeightedFedAvg2(w_locals, args.client_list, args.similarity, args.poison, args.poison_client)
        w_glob = Momentum(w_locals, w_glob, args.client_number, args.similarity)
    elif args.fed == 'FedOpt':
        w_glob = FedOpt(w_locals)
    elif args.fed == 'FedProx':
        w_glob = FedProx(w_locals)
    # elif args.fed == 'None':
    #     states = pickle.load(open('./Exps/model_save_'+args.dataset+'.pkl', 'rb'))
    #     model.load_state(*states['states'])
    #     h0 = states['h0']
    # if args.personalization == False:
    # print(*(client_models[0].save_states()[0], w_glob))
    for i in range(args.client_number):
        # client_models[i].load_states(*(client_models[i].save_states()[0], w_glob))
        client_models[i].load_states(w_glob)
    fed_model.load_states(w_glob)

    return fed_model, client_models

def compute_graph_weights(client_train_data, base_graph):
    G = nx.Graph()
    # for t in range(self.T):
    #     ei = self.ei_masked(self.TRAIN, t)
    for j in range(client_train_data.n_interactions):
            #if self.nmap is not None:
            #    src, dst = self.nmap[ei[0,j]], self.nmap[ei[1,j]]
            #else:
        src, dst = client_train_data.sources[j], client_train_data.destinations[j]
        # src, dst = ei[0,j].item(), ei[1,j].item()
        G.add_edge(src, dst)
            #debugging
            #if cnt < 10:
                #print((src, dst))
                #cnt = cnt + 1
    #number of iterations set to 3 in usual cases
    histograms_G1 = weisfeiler_lehman_histogram(G, 3)
    histograms_G2 = weisfeiler_lehman_histogram(base_graph, 3)
    KL = False
    if KL == True:
        values_list_1 = []
        values_list_2 = []
        # for i in range(3):
        values_list_1 = [value for dictionary in histograms_G1 for value in dictionary.values()]
        values_list_2 = [value for dictionary in histograms_G2 for value in dictionary.values()]
        from scipy.stats import entropy
        p = np.array(values_list_1, dtype=float).flatten()
        q = np.array(values_list_2, dtype=float).flatten()
        print(p.shape,q.shape)
        p /= p.sum()
        q /= q.sum()
        min_length = min(len(p), len(q))
        print(min_length)
        # Calculate the KL divergence
        kl_divergence = entropy(p[:min_length], q[:min_length])

        print("KL Divergence:", kl_divergence)

    #debugging
    # pickle.dump(histograms_G1, open(OUTPATH+'/optc/4/client_hist'+str(client)+'.dat', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # Compute similarity for each iteration's histogram
    similarities = [jaccard_similarity(histograms_G1[i], histograms_G2[i]) for i in range(3)]
    # print(similarities)
    return similarities


def separate_cluster(mc, dataset_name, CLIENT_NUMBER):
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  graph_df = graph_df.head(n)
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name)) # (49341299, 5)
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name)) # (15611, 10)




def load_machine_clusters(cluster_fname):
    clusters = json.loads(open(cluster_fname).read())

    machine_clusters = {}
    count = 0
    max_cluster_ind = 0
    for cluster_ind, machines in enumerate(clusters['bottom']):
        # print(cluster_ind, machines)
        for m in machines.keys():
            machine_clusters[m] = cluster_ind
            #machine_clusters[m] = cluster_ind
        if cluster_ind > max_cluster_ind:
            max_cluster_ind = cluster_ind
    #print(str(count) + ' machines are not in the clusters')
    #Cluster 0-11, regular clusters
    #put the 70 outlier machines into a new cluster, ID: 12
    for m in range(15610):
        if not str(m) in machine_clusters:
            machine_clusters[str(m)] = max_cluster_ind + 1
    # print(machine_clusters)
    for i in range(12):
        count = sum(1 for v in machine_clusters.values() if v == i)
        print(i, count)
    # print(machine_clusters)
    # exit()
    return machine_clusters


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
        # print(k)
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

def WeightedFedAvg(w, client_number, mc):
    #ZL: compute weights from mc and client_list
    cm = reverse_machine_cluster(mc)
    weights = calculate_client_weights(cm)
    print('Clients ' + str(client_number) + ' weights are ' + str(weights))
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

def calculate_client_weights2(similarity):
    total_weight = sum(similarity)
    weights = [weight / total_weight for weight in similarity]
    return weights


def WeightedFedAvg2(w, client_number, similarity, defense=False):
    #ZL: weights based on graph similarity
    #print('Clients similarities to the random graph is ' + str(similarity))

    if defense == True:
        from scipy import stats
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
        print('Clients ' + str(client_number) + ' weights are ' + str(weights))
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
        print('Clients ' + str(client_number) + ' weights are ' + str(weights))
        w_avg = copy.deepcopy(w[0])
        # print(client_index, POISON)
        # print(w_avg)
        # for k in w_avg.keys():
        #     print(k)
        #     for i in range(0, len(w)):
        #         print(i, w_avg[k].shape, w[i][k].shape)
        # exit()
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

    # Sum up the mass and velocity
    mass = torch.sum(torch.stack(mass))
    # velocity = torch.sum(torch.stack(velocity_list))

    velocity_list = torch.stack(velocity_list)
    norm_velocity = torch.norm(velocity_list, p=2)
    M = 50
    denominator = max(1, norm_velocity / M)
    velocity_updated = norm_velocity / denominator
    velocity = torch.sum(velocity_updated)
    print("mass, norm_velocity, velocity: ", mass.item(), norm_velocity.item(), velocity.item())

    return mass, velocity



def Momentum(w, w_fed, client_list, similarity, defense=False):
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
        print('weights are ' + str(weights))
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


        # M = 50
        # for i in range(len(w)):
        #     w[i] = clip_update(w[i], M, w_fed)

        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += weights[i] * w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg



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


# Differential privacy
def add_noise(delta_w, sigma, device):
    for i in delta_w.keys():
        noise = torch.normal(0, sigma, delta_w[i].size()).to(device)  # Changed to torch.normal
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
            noise = torch.normal(0, sigma, fed_model[i].size()).to(device)  # Changed to torch.normal
            fed_model[i] = fed_model[i] + noise



        # noise = torch.normal(0.0, sigma, global_model.weights.size())

        # aggregated_update = torch.stack(global_updates).mean(0) + noise
        # global_model.update(aggregated_update)

        accountant.accumulate_spent_privacy(z)
        # round_num += 1

    return fed_model


def jaccard_similarity(hist1, hist2):
    # Labels present in either histogram
    all_labels = set(hist1.keys()).union(hist2.keys())

    intersection_sum = sum([min(hist1.get(label, 0), hist2.get(label, 0)) for label in all_labels])
    union_sum = sum([max(hist1.get(label, 0), hist2.get(label, 0)) for label in all_labels])

    return intersection_sum / union_sum


def weisfeiler_lehman_histogram(G, num_iterations):
    # Initial labeling based on node degree
    labels = {node: str(G.degree(node)) for node in G.nodes()}

    # This will store the histogram at each iteration
    all_histograms = []

    for iteration in range(num_iterations):
        # Build label-to-hash mapping
        label_map = {}
        new_labels = {}

        for node in G.nodes():
            # Form the aggregated label (old label + sorted neighboring labels)
            aggregated_label = labels[node] + ''.join(sorted([labels[neighbor] for neighbor in G.neighbors(node)]))

            # Hash the aggregated label and store it as new label
            new_label = hashlib.sha1(aggregated_label.encode()).hexdigest()

            if new_label not in label_map:
                label_map[new_label] = str(len(label_map))

            # Convert hashed labels to their integer representation
            new_labels[node] = label_map[new_label]

        labels = new_labels

        # Update histogram for this iteration
        histogram = Counter(labels.values())
        # print(histogram)
        all_histograms.append(histogram)

    return all_histograms

