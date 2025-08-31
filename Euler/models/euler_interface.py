from copy import deepcopy

import torch
from torch import nn
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from torch_geometric.utils import to_dense_adj
import random
# from libauc_back.losses import APLoss

#ZL: decoder adopted by Euler
class DecoderSM(nn.Module):
    def __init__(self, num_nodes, sample_num):
        #input_size, output size are number of nodes
        self.sample_num = sample_num
        self.out = nn.Sequential(nn.Linear(num_nodes, num_nodes), nn.Softmax())

    def inner_forward(self, z, s):
        return self.out(1 / float(self.sample_num) * s * z)

    def forward(self, ei, z, num_nodes, no_grad):
        #get adjacency matrix A
        a = to_dense_adj(ei)
        if no_grad:
            self.eval()
            with torch.no_grad():
                #testing: normalized adjacency matrix
                s = a
        else:
            #training:
            s = torch.eye(num_nodes)
            #sample with replacement
            for i in s.size(0):
                r = torch.nonzero(a[i,:], as_tuple=True)
                idx = random.choices(r[r!=i], self.sample_num)
                rs = tensor([0] * s.size(1))
                rs[idx] = 1
                s[i] = s[i] + rs
        return self.inner_forward(z, s)


class Euler_Embed_Unit(nn.Module):
    '''
    Wrapper class to ensure calls to Embedders are formatted properly
    '''

    def inner_forward(self, mask_enum):
        '''
        The meat of the forward method. Runs data acquired from the worker
        through the mask_enum through whatever model it holds

        mask_enum : int
            enum representing train, validation, test sent to workers
        '''
        raise NotImplementedError

    def forward(self, mask_enum, no_grad):
        '''
        Forward method called by default. This ensures models can still use torch.no_grad()
        with minimal extra hacking

        mask_enum : int
            enum representing train, validation, test sent to workers
        no_grad : bool
            If true, tensor is returned without gradients for faster forward passes during eval
        '''
        if no_grad:
            self.eval()
            with torch.no_grad():
                return self.inner_forward(mask_enum)

        return self.inner_forward(mask_enum)


    def decode(self, src, dst, z):
        return torch.sigmoid(
            (z[src] * z[dst]).sum(dim=1)
        )

    def decode_embed(self, e, z, no_grad):
        src,dst = e
        return z[src] * z[dst]


    #LoF to get the anomaly scores, clf is the LoF class object with initialized params
    def lof(self, src, dst, z, clf):
        edge_z = z[src] * z[dst]
        lof_labels = clf.fit_predict(edge_z)
        lof_scores = clf.negative_outlier_factor_
        return lof_labels,lof_scores

    #Adjust the edge prediction score based on other edges sharing the same src (excluding the edge)
    def get_src_score(self, src, dst, preds):
        src_dict = {}
        for i in range(0, len(src)):
            k = int(src[i])
            if not k in src_dict:
                src_dict[k] = [float(preds[i])]
            else:
                src_dict[k].append(float(preds[i]))
        preds_src = []
        #weights for edge score and neighborhood score
        lambda1 = 0.5
        lambda2 = 0.5
        for i in range(0, len(src)):
            k = int(src[i])
            preds_src.append(lambda1 * float(preds[i]) + lambda2 * np.mean(src_dict[k]))
        return torch.tensor(preds_src)

class Euler_Encoder(Euler_Embed_Unit):
    '''
    Wrapper class for the DDP class that holds the data this module will operate on
    as well as a clone of the module itself

    Requirements: module must have a field called data containing all time slices it will
    operate on
    '''

    def __init__(self, module: Euler_Embed_Unit, **kwargs):
        '''
        Constructor for distributed encoder

        module : Euler_Embed_Unit
            The model to encode temporal data. module.forward must accept an enum
            reprsenting train/val/test and nothing else. See embedders.py for acceptable
            modules
        kwargs : dict
            any args for the DDP constructor
        '''
        # super().__init__(module, **kwargs)
        super().__init__(**kwargs)
        self.module = module
        self.sample_num = 5
        self.num_nodes = 15611 #optc: 815/ lanl: 15611
        self.zdim = 16
        self.out = nn.Sequential(nn.Linear(self.zdim, self.zdim), nn.Softmax(dim=1))
        # self.ap_loss = APLoss(pos_len=1238452, margin=0.8, gamma=0.1, surrogate_loss='squared') # optc

        # self.ap_loss = APLoss(pos_len=283573, margin=0.8, gamma=0.01, surrogate_loss='squared') # squared, squared_hinge, logistic lanl: 283573
        # 20: 429655,
        # 10: 387936
        # 5: 299422 self.ap_loss =  APLoss(pos_len=299422, margin=0.8, gamma=0.1, surrogate_loss='squared')
        # default: 283573 self.ap_loss = APLoss(pos_len=283573, margin=0.8, gamma=0.01, surrogate_loss='squared')


    def train(self, mode=True):
        '''
        This method is inacceessable in the DDP wrapped model by default
        '''

        self.module.train()


    def load_new_data(self, loader, kwargs):
        '''
        Put different data on worker. Must be called before work can be done

        loader : callable[..., loaders.TGraph]
            callable method that returns a loaders.TGraph object
        kwargs : dict
            kwargs for loader with a field for "jobs", the number of threads
            to load the TGraph with
        '''
        self.module.data = loader(**kwargs)

        return True

    def get_data_field(self, field):
        '''
        Return some field from this worker's data object
        '''
        return self.module.data.__getattribute__(field)


    def get_data(self):
        return self.module.data


    def run_arbitrary_fn(self, fn, *args, **kwargs):
        '''
        Run an arbitrary function using this machine
        '''
        return fn(*args, **kwargs)


    def decode(self, e, z, no_grad):
        '''
        Given a single edge list and embeddings, return the dot product
        likelihood of each edge. Uses inner product by default
        e : torch.Tensor
            A 2xE list of edges where e[0,:] are the source nodes and e[1,:] are the dst nodes
        z : torch.Tensor
            A dxN list of node embeddings generated to represent nodes at this snapshot
        '''
        src,dst = e
        return self.module.decode(
            src,dst,z
        )


    def inner_forward_sm(self, z, s):
        # print(s.shape, z.shape) # (15611,15611) * (15611,16)
        temp = torch.matmul(s, z)
        # print(temp.shape)
        return self.out(1 / float(self.sample_num) * temp)

    def decode_back(self, ei, z, no_grad):
        #get adjacency matrix A
        a = to_dense_adj(ei, max_num_nodes=self.num_nodes)[0]
        # print('a: ', a.shape, ei.shape)
        if no_grad:
            self.eval()
            with torch.no_grad():
                #testing: normalized adjacency matrix
                s = a.detach().clone()
        else:
            #training:
            s = torch.eye(self.num_nodes)
            #sample with replacement
            # print(torch.count_nonzero(a))
            for i in range(s.size(0)):
                # print(torch.count_nonzero(a[i]))
                if torch.count_nonzero(a[i]).item() >= 2: #self.sample_num:
                    r = a[i].nonzero(as_tuple=True)[0]
                    # print(i, r)
                    idx = np.random.choice(range(len(r)), self.sample_num)

                    # idx = random.sample(range(len(r)), self.sample_num)
                    # print(idx)
                    rs = torch.tensor([0] * s.size(1))
                    # print(rs.shape)
                    # print(rs)
                    rs[r[idx]] = 1
                    s[i] = s[i] + rs
            # print('s: ', s.shape)
        z = self.inner_forward_sm(z, s)
        src,dst = ei
        return self.module.decode(
            src,dst,z
        )
        # return


    def lof(self, e,z,clf):
        src,dst = e
        return self.module.lof(
            src,dst,z,clf
        )

    def get_src_score(self, e, preds):
        src,dst = e
        return self.module.get_src_score(
            src,dst,preds
        )

    def bce(self, t_scores, f_scores):
        '''
        Computes binary cross entropy loss

        t_scores : torch.Tensor
            a 1-dimensional tensor of likelihood scores given to edges that exist
        f_scores : torch.Tensor
            a 1-dimensional tensor of likelihood scores given to edges that do not exist
        '''
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return (pos_loss + neg_loss) * 0.5


    def calc_loss(self, z, partition, nratio):
        '''
        Rather than sending edge index to master, calculate loss
        on workers all at once. Must be implimented by the user

        z : torch.Tensor
            A T x d x N tensor of node embeddings generated by the models,
            it is safe to assume z[n] are the embeddings for nodes in the
            snapshot held by this model's TGraph at timestep n
        partition : int
            An enum representing if this is training/validation/testing for
            generating negative edges
        nratio : float
            The model samples nratio * |E| negative edges for calculating loss
        '''
        raise NotImplementedError


    def decode_all(self, zs, unsqueeze=False):
        '''
        Given node embeddings, return edge likelihoods for all edges in snapshots held by this model.
        Implimented differently for predictor and detector models

        zs : torch.Tensor
            A T x d x N tensor of node embeddings generated by the models,
            it is safe to assume z[n] are the embeddings for nodes in the
            snapshot held by this model's TGraph at timestep n
        '''
        raise NotImplementedError


    def score_edges(self, z, partition, nratio):
        '''
        Scores all known edges and randomly sampled non-edges. The same as calc_loss but
        does not return BCE, instead returns the actual scores given to edges

        z : torch.Tensor
            A T x d x N tensor of node embeddings generated by the models,
            it is safe to assume z[n] are the embeddings for nodes in the
            snapshot held by this model's TGraph at timestep n
        partition : int
            An enum representing if this is training/validation/testing for
            generating negative edges
        nratio : float
            The model samples nratio * |E| negative edges for calculating loss
        '''
        raise NotImplementedError



class Euler_Recurrent(nn.Module):
    '''
    Abstract class for master module that holds all workers
    and calculates loss
    '''
    def __init__(self, rnn: nn.Module, encoder: nn.Module, device):
        '''
        Constructor for Recurrent layer of the Euler framework

        Parameters
        ------------
        rnn : torch.nn.Module
            An RNN-like module that accepts 3D tensors as input, and returns
            T x d x N tensors of node embeddings for each snapshot
        remote_rrefs: list[torch.distributed.rpc.RRef]
            a list of RRefs to Euler_Workers

        Fields
        ------------
        gcns : list[torch.distributed.rpc.RRef]
            List of RRefs to workers
        rnn : torch.nn.Module
            The module used to process topological embeddings
        num_workers : int
            The number of remote workers
        len_from_each : list[int]
            The number of snapshots held by each worker
        cutoff : float
            The threshold for anomalousness used by this model during classifiation
            Can be updated later, but defaults to 0.5
        '''
        super(Euler_Recurrent, self).__init__()

        self.gcns = encoder
        self.rnn = rnn
        self.device = device

        # self.num_workers = len(self.gcns)
        self.len_from_each = []

        # Used for LR when classifying anomalies
        self.cutoff = 0.5


    def forward(self, mask_enum, include_h=False, h0=None, no_grad=False):
        '''
        First have each worker encode their data, then run the embeddings through the RNN

        mask_enum : int
            enum representing train, validation, test sent to workers
        include_h : boolean
            if true, returns hidden state of RNN as well as embeddings
        h0 : torch.Tensor
            initial hidden state of RNN. Defaults to zero-vector if None
        no_grad : boolean
            if true, tells all workers to execute without calculating gradients.
            Used for speedy evaluation
        '''

        futs = self.encode(mask_enum, no_grad)
        if futs == None:
            return None
        self.rnn = self.rnn.to(self.device)
        # print(futs.device.index)
        # print(next(self.rnn.parameters()).device.index)
        # print(self.device)
        # print(self.rnn.device.index)
        # print(futs.shape)
        # Run through RNN as embeddings come in
        # Also prevents sequences that are super long from being encoded
        # all at once. (This is another reason to put extra tasks on the
        # workers with higher pids)
        zs = []
        # print(no_grad, mask_enum, futs[0].shape)
        # print(futs)
        # print("futs: ", futs.grad_fn)
        # print(self.rnn)
        # print(xs.requires_grad, xs.grad_fn)

        # for f in futs:
            # print(f.shape)
        # zs, h0 = self.rnn(futs.to(self.device), h0, include_h=True)
        zs, h0 = self.rnn(futs, h0, include_h=True)

        # zs.append(z)

        # May as well do this every time, not super expensive
        self.len_from_each = [embed.size(0) for embed in zs]
        # zs = torch.cat(zs, dim=0)
        self.z_dim = zs.size(-1)
        # print("zs", zs.grad_fn)
        #zs, h0 = self.rnn(torch.cat(zs, dim=0), h0, include_h=True)
        # exit()
        if include_h:
            return zs, h0
        else:
            return zs


    def encode(self, mask_enum, no_grad):
        '''
        Tell each remote worker to encode their data. Data lives on workers to minimize net traffic

        mask_enum : int
            enum representing train, validation, test sent to workers
        no_grad : boolean
            if true, tells all workers to execute without calculating gradients.
            Used for speedy evaluation
        '''
        return self.gcns.module(mask_enum, no_grad)

    def save_states(self):
        '''
        Makes a copy of the current state dict as well as
        the distributed GCN state dict (just worker 0)
        '''
        # jason replace
        # gcn = _remote_method(
        #     DDP.state_dict, self.gcns[0]
        # )
        gcn = self.gcns
        return gcn, deepcopy(self.state_dict())


    def load_states(self, gcn_state_dict, rnn_state_dict):
        '''
        Given the state dict for one GCN and the RNN load them
        into the dist and local models

        gcn_state_dict : dict
            Parameter dict for remote worker
        rnn_state_dict : dict
            Parameter dict for local RNN
        '''
        self.load_state_dict(rnn_state_dict)

        # jobs = []
        # jason replace
        # for rref in self.gcns:
        #     jobs.append(
        #         _remote_method_async(
        #             DDP.load_state_dict, rref,
        #             gcn_state_dict
        #         )
        #     )

        # [j.wait() for j in jobs]


    def train(self, mode=True):
        '''
        Propogate training mode to all workers
        '''
        super(Euler_Recurrent, self).train()
        # jason replace
        # [_remote_method(
        #     Euler_Encoder.train,
        #     self.gcns[i],
        #     mode=mode
        # ) for i in range(self.num_workers)]


    def eval(self):
        '''
        Propogate training mode to all workers
        '''
        super(Euler_Recurrent, self).train(False)
        # jason replace
        # [_remote_method(
        #     Euler_Encoder.train,
        #     self.gcns[i],
        #     mode=False
        # ) for i in range(self.num_workers)]


    def score_all(self, zs, unsqueeze=False):
        '''
        Has the distributed models score and label all of their edges
        Need to change which zs are given to workers depending on if
        predictor or detector

        zs : torch.Tensor
            A T x d x N tensor of node embeddings generated by each graph snapshot
            Need to offset according to how far in the future embeddings are supposed
            to represent.
        '''
        raise NotImplementedError


    def loss_fn(self, zs, partition, nratio=1.0):
        '''
        Runs NLL on each worker machine given the generated embeds
        Need to change which zs are given to workers depending on if
        predictor or detector

        zs : torch.Tensor
            A T x d x N tensor of node embeddings generated by each graph snapshot
            Need to offset according to how far in the future embeddings are supposed
            to represent.
        partition : int
            enum representing train, validation, test sent to workers
        nratio : float
            The workers sample nratio * |E| negative edges for calculating loss
        '''
        raise NotImplementedError


    def score_edges(self, zs, partition, nratio=1):
        '''
        Gets edge scores from dist modules, and negative edges. Similar to
        loss_fn but returns actual scores instead of BCE loss

        zs : torch.Tensor
            A T x d x N tensor of node embeddings generated by each graph snapshot
            Need to offset according to how far in the future embeddings are supposed
            to represent.
        partition : int
            enum representing train, validation, test sent to workers
        nratio : float
            The workers sample nratio * |E| negative edges for calculating loss
        '''
        raise NotImplementedError
