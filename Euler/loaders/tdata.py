import numpy as np
import pandas as pd
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.utils import structured_negative_sampling
import networkx as nx
from collections import defaultdict, Counter
import hashlib

OUTPATH = './Exps/result/'


'''
Special data object that the dist_framework uses
'''
class TData(Data):
    # Enum like for masked function used by worker processes
    TRAIN = 0
    VAL = 1
    TEST = 2
    ALL = 2

    #eas for edge attributes
    def __init__(self, slices, eis, xs, ys, masks, ews=None, eas=None, use_flows=False, nmap=None, **kwargs):
        super(TData, self).__init__(**kwargs)

        # Required fields for models to use this
        self.slices = slices
        self.eis = eis
        self.T = len(eis)
        self.xs = xs
        self.masks = masks
        self.dynamic_feats = isinstance(xs, list)
        self.ews = ews
        self.eas = eas
        self.ys = ys
        self.is_test = not isinstance(ys, None.__class__)
        self.nmap = nmap

        # Makes finding sizes of positive samples a little easier
        self.ei_sizes = [
            (
                self.ei_masked(self.TRAIN, t).size(1),
                self.ei_masked(self.VAL, t).size(1),
                self.eis[t].size(1)
            )
            for t in range(self.T)
        ]

        if self.dynamic_feats:
            self.num_nodes = max([x.size(0) for x in xs])
            self.x_dim = xs[0].size(1)
        else:
            self.num_nodes = xs.size(0)
            self.x_dim = xs.size(1)

        #number of edge features
        if isinstance(eas, None.__class__):
            #Without flows, 3 features from auth, otherwise, adding 7 features
            if use_flows:
                self.ea_dim = 5
            else:
                self.ea_dim = 0
        else:
            self.ea_dim = 5
            # self.ea_dim = self.eas[0].size(0)

    '''
    Returns masked ei/ew/ea at timestep t
    Assumes it will only be called on tr or val data
    (i.e. test data is the entirity of certain time steps)
    '''
    def ei_masked(self, enum, t):
        if enum == self.TEST or self.is_test:
            return self.eis[t]
        if enum == self.TRAIN:
            return self.eis[t][:, self.masks[t]]
        else:
            return self.eis[t][:, ~self.masks[t]]

    def ew_masked(self, enum, t):
        if isinstance(self.ews, None.__class__):
            return None

        if enum == self.TEST or self.is_test:
            return self.ews[t]

        return self.ews[t][self.masks[t]] if enum == self.TRAIN \
            else self.ews[t][~self.masks[t]]

    def ea_masked(self, enum, t):
        if isinstance(self.eas, None.__class__):
            return None

        if enum == self.TEST or self.is_test:
            return self.eas[t]

        # print("eas size:", self.eas[t].shape, self.ews[t].shape)
        # print(self.eis[t].shape)
        # print(self.masks[t].shape)
        # exit()

        #To implement, eas is edge attr and have different dimensions
        # print(self.eas[t].shape)
        return self.eas[t][:, self.masks[t]] if enum == self.TRAIN \
            else self.eas[t][:, ~self.masks[t]]


    def get_negative_edges(self, enum, nratio=1, start=0):
        negs = []
        size = []
        for t in range(start, self.T):
            if enum == self.TRAIN:
                pos = self.ei_masked(enum, t)
            else:
                pos = self.eis[t]

            num_pos = self.ei_sizes[t][enum]
            #ZL, use another sampling method
            # src, dst, neg_dst = structured_negative_sampling(pos, self.num_nodes)
            negs.append(
                fast_negative_sampling(pos, int(num_pos*nratio),self.num_nodes)
                # [src, neg_dst]
            )
            size.append(negs[-1].size(1))
        size = sum(size)
        # print("get_negative_edges", size)
        # exit()
        # print('negative sampling ends')
        return negs




    def get_val_repr(self, scores, delta=1):
        pairs = []
        for i in range(len(scores)):
            score = scores[i]
            ei = self.eis[i]

            for j in range(ei.size(1)):
                if self.nmap is not None:
                    src, dst = self.nmap[ei[0,j]], self.nmap[ei[1,j]]
                else:
                    src, dst = ei[0,j], ei[1,j]
                if self.hr:
                    ts = self.hr[i]
                else:
                    ts = '%d-%d' % (i*delta, (i+1)*delta)

                s = '%s\t%s\t%s' % (src, dst, ts)
                pairs.append((score[j], s))

        pairs.sort(key=lambda x : x[0])
        return pairs

    def compute_graph_weights(self, client, base_graph):
        #base_graph is a networkx graph
        #ZL TODO: compute the weights based on the graph distance between client's subgraph with a random graph, assuming the nodes are public information
        G = nx.Graph()
        #cnt = 0
        for t in range(self.T):
            ei = self.ei_masked(self.TRAIN, t)
            for j in range(ei.size(1)):
                #if self.nmap is not None:
                #    src, dst = self.nmap[ei[0,j]], self.nmap[ei[1,j]]
                #else:
                src, dst = ei[0,j].item(), ei[1,j].item()
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
        pickle.dump(histograms_G1, open(OUTPATH+'client_hist'+str(client)+'.dat', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        #pickle.dump(G, open(OUTPATH+'client_g'+str(client)+'.dat', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        #pickle.dump(histograms_G2, open(OUTPATH+'random_hist.dat', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        #pickle.dump(base_graph, open(OUTPATH+'random_g.dat', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


        # Compute similarity for each iteration's histogram
        similarities = [jaccard_similarity(histograms_G1[i], histograms_G2[i]) for i in range(3)]
        # print(similarities)
        return similarities




'''
Uses Kipf-Welling pull #25 to quickly find negative edges
(For some reason, this works a touch better than the builtin
torch geo method)
ZL: A very simple sampler that generates 2d matrix of random src and dest
'''
def fast_negative_sampling(edge_list, batch_size, num_nodes, oversample=1.25):
    # For faster membership checking
    el_hash = lambda x : x[0,:] + x[1,:]*num_nodes

    el1d = el_hash(edge_list).cpu().numpy()
    neg = np.array([[],[]])

    while(neg.shape[1] < batch_size):
        maybe_neg = np.random.randint(0,num_nodes, (2, int(batch_size*oversample))) #generates a 2d matrix
        neg_hash = el_hash(maybe_neg)

        neg = np.concatenate(
            [neg, maybe_neg[:, ~np.in1d(neg_hash, el1d)]],
            axis=1
        )

    # May have gotten some extras
    neg = neg[:, :batch_size]
    # print(neg.shape)
    # print(neg)
    # exit()
    return torch.tensor(neg).long()

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


def jaccard_similarity(hist1, hist2):
    # Labels present in either histogram
    all_labels = set(hist1.keys()).union(hist2.keys())

    intersection_sum = sum([min(hist1.get(label, 0), hist2.get(label, 0)) for label in all_labels])
    union_sum = sum([max(hist1.get(label, 0), hist2.get(label, 0)) for label in all_labels])

    return intersection_sum / union_sum
