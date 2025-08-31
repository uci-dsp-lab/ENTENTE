import torch
from torch import nn
from torch.nn import functional as F
# from torch.distributed import rpc
from torch_geometric.nn import GCNConv, GATConv, NNConv, SAGEConv
from torch_geometric.nn.conv.message_passing import MessagePassing

# from .evo_gcn import LP_EGCN_h, LP_EGCN_o

# from .serial_models import VGRNN


from .euler_interface import Euler_Embed_Unit
from .euler_detector import DetectorEncoder

class DropEdge(nn.Module):
    '''
    Implimenting DropEdge https://openreview.net/forum?id=Hkx1qkrKPr
    '''
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, ei, ew=None, ea=None):
        if self.training and self.p > 0:
            mask = torch.rand(ei.size(1))
            if ew is None:
                return ei[:, mask > self.p]
            else:
                return ei[:, mask > self.p], ew[mask > self.p], ea[mask > self.p]

        if ew is None:
            return ei
        else:
            return ei, ew, ea


# class DecoderSM(nn.Module):
#     def __init__(self, num_nodes, sample_num, zdim):
#         #input_size, output size are number of nodes
#         # self.sample_num = sample_num
#         # self.out = nn.Sequential(nn.Linear(num_nodes, num_nodes), nn.Softmax())
#         self.sample_num = sample_num #5
#         self.num_nodes = num_nodes #815
#         self.zdim = zdim #16
#         self.out = nn.Sequential(nn.Linear(self.zdim, self.zdim), nn.Softmax(dim=1))

#     def inner_forward_sm(self, z, s):
#         # print(s.shape, z.shape) # (15611,15611) * (15611,16)
#         temp = torch.matmul(s, z)
#         # print(temp.shape)
#         return self.out(1 / float(self.sample_num) * temp)

#     def decode(self, ei, z, no_grad):
#         #get adjacency matrix A
#         a = to_dense_adj(ei, max_num_nodes=self.num_nodes)[0]
#         # print('a: ', a.shape, ei.shape)
#         if no_grad:
#             self.eval()
#             with torch.no_grad():
#                 #testing: normalized adjacency matrix
#                 s = a.detach().clone()
#         else:
#             #training:
#             s = torch.eye(self.num_nodes)
#             #sample with replacement
#             for i in range(s.size(0)):
#                 if torch.count_nonzero(a[i]).item() >= self.sample_num:
#                     r = a[i].nonzero(as_tuple=True)[0]
#                     idx = random.sample(range(len(r)), self.sample_num)
#                     rs = torch.tensor([0] * s.size(1))
#                     rs[r[idx]] = 1
#                     s[i] = s[i] + rs
#             # print('s: ', s.shape)
#         z = self.inner_forward_sm(z, s)
#         return self.inner_forward(z, s)

#         # src,dst = ei
#         # return self.module.decode(
#         #     src,dst,z
#         # )
#         # return



class DropEdge(nn.Module):
    '''
    Implimenting DropEdge https://openreview.net/forum?id=Hkx1qkrKPr
    '''
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, ei, ewa=None):
        #either edge weights or edge attributes (ewa can be either ew or ea)
        if self.training and self.p > 0:
            mask = torch.rand(ei.size(1))
            if ewa is None:
                return ei[:, mask > self.p]
            else:
                #accommodate edge weights and features
                if len(ewa.size()) == 1:
                    return ei[:, mask > self.p], ewa[mask > self.p]
                else:
                    return ei[:, mask > self.p], ewa[:, mask > self.p]

        if ewa is None:
            return ei
        else:
            return ei, ewa


class GCN(Euler_Embed_Unit):
    '''
    2-layer GCN implimenting the Euler Embed Unit interface
    '''

    def __init__(self, data_load, data_kws, h_dim, z_dim, device):
        '''
        Constructor for the model

        parameters
        ----------
        data_load : callable[..., loaders.TGraph]
            Function to load data onto this worker. Must return a loaders.TGraph object
        data_kws : dict
            Dictionary of keyword args for the loader function
        h_dim : int
            The dimension of the hidden layer
        z_dim : int
            The dimension of the output layer

        attributes
        ----------
        data : loaders.TGraph
            A TGraph object holding data for all snapshots loaded into this model
        '''
        super(GCN, self).__init__()
        self.device = device
        self.data = data_load(**data_kws).to(self.device) # load data

        # Params
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.tanh = nn.Tanh()

        # self.de = DropEdge(0.5)


    def inner_forward(self, mask_enum):
        '''
        Override parent's abstract inner_forward method

        mask_enum : int
            enum representing train, validation, test used to mask which
            edges are sent into the model
        '''
        zs = []
        # print("self.data.T: ", self.data.T)
        # edge_number = 0
        # for i in self.data.eis:
        # # fed_model.gcns.module.data.eis:
        #     edge_number += len(i[1])
        # print("inner forward edge number: ", edge_number)

        for i in range(self.data.T):
            # Small optimization. Running each loop step as its own thread
            # is a tiny bit faster.
            # print(i, self.data.T)
            zs.append(
                #torch.jit._fork(self.forward_once, mask_enum, i)
                self.forward_once(mask_enum, i)#.cpu().detach()
            )
        #return torch.stack([torch.jit._wait(z) for z in zs])
        # zs =
        # print(zs.grad_fn, zs.shape)
        # print(len(zs))
        if len(zs) == 0:
            return None
        else:
            return torch.stack(zs)


    def forward_once(self, mask_enum, i):
        '''
        Helper function to make inner_forward a little more readable
        Just passes each time step through a 2-layer GCN with final tanh activation

        mask_enum : int
            enum representing train, validation, test
            used to mask edges passed into model
        i : int
            The index of the snapshot being processed
        '''
        # print(next(self.rnn.parameters()).device.index)
        # print(self.device)

        if self.data.dynamic_feats:
            x = self.data.xs[i].to(self.device)
        else:
            x = self.data.xs.to(self.device)
        ei = self.data.ei_masked(mask_enum, i).to(self.device)
        ew = self.data.ew_masked(mask_enum, i).to(self.device)
        # print(ei.shape)

        # ei, ew = self.de(ei, ewa=ew)
        # Simple 2-layer GCN. Tweak if desired
        # print(x.is_cuda, ei.is_cuda)
        # x = self.c1(x, ei)
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        # x = self.c2(x, ei)

        x = self.c2(x, ei, edge_weight=ew)

        # Experiments have shown this is the best activation for GCN+GRU
        # return self.tanh(x) # LANL
        return self.relu(x) # OPTC


# Added dummy **kws param so we can use the same constructor for predictor
def detector_gcn_rref(loader, kwargs, h_dim, z_dim, **kws):
    '''
    Returns a rref to a GCN wrapped in a DetectorEncoder DDP

    loader : callable[..., loaders.TGraph]
        Function to load data onto this worker. Must return a loaders.TGraph object
    kwargs : dict
        Dictionary of keyword args for the loader function (must include a field for 'jobs')
    h_dim : int
        The dimension of the hidden layer
    z_dim : int
        The dimension of the output layer
    kws : dummy value for matching method signatures
    '''
    device = kwargs.pop('device')
    return DetectorEncoder(
        GCN(loader, kwargs, h_dim, z_dim, device)
    )

class GAT(GCN):
    '''
    2-layer GAT implimenting the Euler Embed Unit interface. Inherits GCN
    as the only difference is the forward method, and which submodules are used
    '''

    def __init__(self, data_load, data_kws, h_dim, z_dim, device, heads=3):
        super().__init__(data_load, data_kws, h_dim, z_dim, device)

        # Concat=False seems to work best
        self.c1 = GATConv(self.data.x_dim, h_dim, heads=heads, concat=False)
        self.c2 = GATConv(h_dim, z_dim, heads=heads, concat=False)

    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i]
        else:
            x = self.data.xs

        ei = self.data.ei_masked(mask_enum, i)
        ei = self.de(ei)

        # Only difference is GATs can't handle edge weights
        x = self.c1(x, ei)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei)

        # Experiments have shown this is the best activation for GCN+GRU
        return self.tanh(x)


def detector_gat_rref(loader, kwargs, h_dim, z_dim, **kws):
    return DetectorEncoder(
        GAT(loader, kwargs, h_dim, z_dim, device)
    )


class PoolSAGEConv(MessagePassing):
    '''
    The official PyTorch Geometric package does not actually follow the paper
    This is problematic from both a performance standpoint, and an accuracy one.
    I have taken it upon myself to build a more correct Maxpool GraphSAGE implientation
    '''
    def __init__(self, in_channels, out_channels, device):
        super().__init__(aggr='max')
        self.device = device
        self.aggr_n = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )
        self.e_lin = nn.Linear(out_channels, out_channels)
        self.r_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, ei):
        x = x.to(self.device)
        ei = ei.to(self.device)
        x_e = self.aggr_n(x)
        x_e = self.propagate(ei, x=x_e, size=None)
        x_e = self.e_lin(x_e)

        x_r = self.r_lin(x)

        x = x_r + x_e
        x = F.normalize(x, p=2., dim=-1)
        return x


class SAGE(GAT):
    '''
    2-layer GraphSAGE implimenting the Euler Embed Unit interface. Inherits GAT
    as the only difference is which submodules are used
    '''
    def __init__(self, data_load, data_kws, h_dim, z_dim, device):
        super().__init__(data_load, data_kws, h_dim, z_dim, device)

        self.c1 = PoolSAGEConv(self.data.x_dim, h_dim, device)
        self.c2 = PoolSAGEConv(h_dim, z_dim, device)


def detector_sage_rref(loader, kwargs, h_dim, z_dim, **kws):
    device = kwargs.pop('device')
    return DetectorEncoder(
        SAGE(loader, kwargs, h_dim, z_dim, device)
    )

class Argus_OPTC_back(GCN):
    def __init__(self, data_load, data_kws, h_dim, z_dim, device):
        super().__init__(data_load, data_kws, h_dim, z_dim, device)
        self.data.x_dim = self.data.x_dim
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, h_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.1)
        self.ac = nn.Tanh()
        # self.fc1 = nn.Linear(h_dim, z_dim) # comment it the ap will decrease
        # self.de = DropEdge(0.2)
        self.c3 = GCNConv(h_dim, h_dim, add_self_loops=True)
        self.c4 = GCNConv(h_dim, z_dim, add_self_loops=True)



    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i].to(self.device)
        else:
            x = self.data.xs.to(self.device)

        ei = self.data.ei_masked(mask_enum, i).to(self.device)
        # ea = self.data.ea_masked(mask_enum, i).to(self.device)
        ew = self.data.ew_masked(mask_enum, i).to(self.device)
        x1 = self.c1(x, ei, edge_weight=ew)

        # x = self.relu(x)
        # x = self.drop(x)
        x = self.c2(x1, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c3(x, ei, edge_weight=ew)
        # print(x.shape, ei.shape, ea.shape)
        # x = self.c3(x, ei, edge_attr=ea)

        # x = self.relu(x)
        # x = self.drop(x)
        # x = self.c4(x, ei, edge_weight=ew)
        return self.ac(x)


class Argus_OPTC(GCN):
    def __init__(self, data_load, data_kws, h_dim, z_dim, device):
        super().__init__(data_load, data_kws, h_dim, z_dim, device)
        self.de = DropEdge(0.5)
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.c2 = GCNConv(h_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.c3 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.c4 = GCNConv(h_dim, z_dim, add_self_loops=True)

        # nn4 = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, h_dim * z_dim))
        # self.c4 = NNConv(h_dim, z_dim, nn4, aggr='mean')

        # self.tanh = nn.Tanh()
        self.ac = nn.Tanh()
        # self.ac = nn.Softmax(dim=1)
        # self.ac = nn.ReLU()
        # self.ac = nn.Sigmoid()

    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i].to(self.device)
        else:
            x = self.data.xs.to(self.device)

        ei = self.data.ei_masked(mask_enum, i).to(self.device)
        ew = self.data.ew_masked(mask_enum, i).to(self.device)

        # ew_nnconv = ew.clone()
        # ew_nnconv = ew_nnconv[None, :]
        # ew_nnconv = ew_nnconv.transpose(1, 0)

        ei, ew = self.de(ei, ewa=ew) # increase 2%

        x = self.c1(x, ei, edge_weight=ew)
        # x = self.c1(x, ei, edge_attr=ew_nnconv)
        # x = self.relu(x)
        # x = self.drop(x)
        x = self.c2(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)

        # # print(x.shape, ei.shape, ea.shape)
        x = self.c3(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        # x = self.c4(x, ei, edge_attr=ew_nnconv)
        x = self.c4(x, ei, edge_weight=ew)
        # x = self.relu(x)

        return self.ac(x)


def detector_argusoptc_rref(loader, kwargs, h_dim, z_dim, **kws):
    device = kwargs.pop('device')
    return DetectorEncoder(
        Argus_OPTC(loader, kwargs, h_dim, z_dim, device)
    )


class Argus_LANL(GCN):
    '''
    2-layer NNConv implimenting the Euler Embed Unit interface. Inherits GCN
    as the only difference is the forward method, and which submodules are used
    edge_attr is used, not a single edge_weight
    '''

    def __init__(self, data_load, data_kws, h_dim, z_dim, device):
        super().__init__(data_load, data_kws, h_dim, z_dim, device)

        # TODO: figure out the numbers
        # following this example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_nn_conv.py
        # nn.Linear(input_features, output_features)
        # self.data.x_dim = self.data.x_dim + 1
        # nn1 = nn.Sequential(nn.Linear(self.data.ea_dim, 64), nn.ReLU(),
        #                     nn.Linear(64, self.data.x_dim * h_dim))
        # self.c1 = NNConv(self.data.x_dim, h_dim, nn1, aggr='mean')

        self.data.x_dim = self.data.x_dim
        # self.c3 = SAGEConv(h_dim, z_dim, aggr='mean', normalize=True, bias=False)
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, h_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.1)
        # self.tanh = nn.Tanh()
        self.ac = nn.Tanh()
        # self.ac = nn.Softmax(dim=1)
        # self.ac = nn.ReLU()
        # self.ac = nn.Sigmoid()



        # self.fc1 = nn.Linear(h_dim, z_dim) # comment it the ap will decrease
        # self.de = DropEdge(0.2)
        self.c3 = GCNConv(h_dim, z_dim, add_self_loops=True)
        # nn3 = nn.Sequential(nn.Linear(10, 8), nn.ReLU(),
        #                     nn.Linear(8, h_dim * h_dim))
        # self.c3 = NNConv(h_dim, z_dim, nn3, aggr='mean')

        nn4 = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), # lanl: 3 or 10; optc: 5
                            nn.Linear(8, h_dim * z_dim))
        self.c4 = NNConv(h_dim, z_dim, nn4, aggr='mean')


    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i].to(self.device)
        else:
            x = self.data.xs.to(self.device)

        ei = self.data.ei_masked(mask_enum, i).to(self.device)
        ea = self.data.ea_masked(mask_enum, i).to(self.device)
        ew = self.data.ew_masked(mask_enum, i).to(self.device)
        # nd = self.data.nds[i].transpose(1,0)
        # print(x.shape, nd.shape)
        # x = torch.concat((nd,x),1)#.transpose(1,0)
        # print(x.shape)

        # ei, ew = self.de(ei, ewa=ew)
        # #ea should be transposed num_edges * num_features
        ea = torch.transpose(ea, 0, 1)#.float()
        # print(ea.shape)
        x1 = self.c1(x, ei, edge_weight=ew)

        # x = self.relu(x)
        # x = self.drop(x)
        x = self.c2(x1, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c3(x, ei, edge_weight=ew)
        # print(x.shape, ei.shape, ea.shape)
        # x = self.c3(x, ei, edge_attr=ea)

        # comment if optc
        x = self.relu(x)
        x = self.drop(x)
        x = self.c4(x, ei, edge_attr=ea)

        # Experiments have shown this is the best activation for GCN+GRU
        # print(x.grad_fn)
        return self.ac(x)

def detector_arguslanl_rref(loader, kwargs, h_dim, z_dim, **kws):
    device = kwargs.pop('device')
    return DetectorEncoder(
        Argus_LANL(loader, kwargs, h_dim, z_dim, device)
    )

# def detector_vgrnn_rref(loader, kwargs, h_dim, z_dim, **kws):
#     device = kwargs.pop('device')
#     return DetectorEncoder(
#         VGRNN(loader, kwargs, h_dim, z_dim, device)
#     )
# def detector_egcno_rref(loader, kwargs, h_dim, z_dim, **kws):
#     device = kwargs.pop('device')
#     return DetectorEncoder(
#         LP_EGCN_o(loader, kwargs, h_dim, z_dim, device)
#     )
# def detector_egcnh_rref(loader, kwargs, h_dim, z_dim, **kws):
#     device = kwargs.pop('device')
#     return DetectorEncoder(
#         LP_EGCN_h(loader, kwargs, h_dim, z_dim, device)
#     )

# tn, fp, fn, tp:  2493536 31790 40 399
# DetectorRecurrent
# Learned Cutoff 0.5575
# TPR: 0.9089, FPR: 0.0126
# TP: 399  FP: 31790
# F1: 0.02445752
# AUC: 0.9897  AP: 0.2740

class NNC_10(GCN): # 20 AP: 0.10; real AP: 0.1805;lr=0.005, patience=10
    '''
    2-layer NNConv implimenting the Euler Embed Unit interface. Inherits GCN
    as the only difference is the forward method, and which submodules are used
    edge_attr is used, not a single edge_weight
    '''

    def __init__(self, data_load, data_kws, h_dim, z_dim, device):
        super().__init__(data_load, data_kws, h_dim, z_dim, device)

        # TODO: figure out the numbers
        # following this example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_nn_conv.py
        # nn.Linear(input_features, output_features)
        # self.data.x_dim = self.data.x_dim + 1
        # nn1 = nn.Sequential(nn.Linear(self.data.ea_dim, 64), nn.ReLU(),
        #                     nn.Linear(64, self.data.x_dim * h_dim))
        # self.c1 = NNConv(self.data.x_dim, h_dim, nn1, aggr='mean')

        self.data.x_dim = self.data.x_dim
        # self.c3 = SAGEConv(h_dim, z_dim, aggr='mean', normalize=True, bias=False)
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.1)
        # self.tanh = nn.Tanh()
        self.ac = nn.Tanh()
        # nn.Softmax(dim=1)
        # nn.ReLU()
        # nn.Sigmoid()


        self.fc1 = nn.Linear(h_dim, z_dim) # comment it the ap will decrease
        # self.de = DropEdge(0.2)
        self.c3 = GCNConv(h_dim, z_dim, add_self_loops=True)
        # nn3 = nn.Sequential(nn.Linear(self.data.ea_dim, 8), nn.ReLU(),
        #                     nn.Linear(8, h_dim * h_dim))
        # self.c3 = NNConv(h_dim, z_dim, nn3, aggr='mean')

        # nn4 = nn.Sequential(nn.Linear(3, 8), nn.ReLU(),
        #                     nn.Linear(8, h_dim * z_dim))
        # self.c4 = NNConv(h_dim, z_dim, nn4, aggr='mean')


    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i].to(self.device)
        else:
            x = self.data.xs.to(self.device)

        ei = self.data.ei_masked(mask_enum, i).to(self.device)
        # ea = self.data.ea_masked(mask_enum, i).to(self.device)
        ew = self.data.ew_masked(mask_enum, i).to(self.device)
        # nd = self.data.nds[i].transpose(1,0)
        # print(x.shape, nd.shape)
        # x = torch.concat((nd,x),1)#.transpose(1,0)
        # print(x.shape)

        # ei, ew = self.de(ei, ewa=ew)
        # #ea should be transposed num_edges * num_features
        # ea = torch.transpose(ea, 0, 1)
        # ei, ew = self.de(ei, ew)
        # print(x.shape, ei.shape, ew.shape, ea.shape)
        # exit()
        x1 = self.c1(x, ei, edge_weight=ew)

        # x = self.relu(x)
        # x = self.drop(x)
        x = self.c2(x1, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c3(x, ei, edge_weight=ew)
        # print(x.shape, ei.shape, ea.shape)
        # x = self.c3(x, ei, edge_attr=ea)

        # x = self.relu(x)
        # x = self.drop(x)
        # x = self.c4(x, ei, edge_attr=ea)

        # Experiments have shown this is the best activation for GCN+GRU

        return self.ac(x)

# tn, fp, fn, tp:  2488256 37070 27 412
# DetectorRecurrent
# Learned Cutoff 0.5260
# TPR: 0.9385, FPR: 0.0147
# TP: 412  FP: 37070
# F1: 0.02172938
# AUC: 0.9936  AP: 0.1805

# FwdTime 19.096827745437622
# test_time:  130.7915871143341
# [{'Model': 'DetectorRecurrent', 'TPR': 0.9384965831435079, 'FPR': 0.014679292891294035, 'TP': 412, 'FP': 37070, 'F1': 0.02172938477360829, 'AUC': 0.9935659395152189, 'AP': 0.18046542456527323, 'FwdTime': 19.096827745437622, 'tn': 2488256, 'fp': 37070, 'fn': 27, 'tp': 412, 'TPE': 0.5995400060306896, 'tr_time': 19.236570358276367}]


class NNC_14_8(GCN): # 20 AP: 0.08; real AP: 0.1452
    def __init__(self, data_load, data_kws, h_dim, z_dim, device):
        super().__init__(data_load, data_kws, h_dim, z_dim, device)

        self.data.x_dim = self.data.x_dim
        # self.c3 = SAGEConv(h_dim, z_dim, aggr='mean', normalize=True, bias=False)
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.1)
        # self.tanh = nn.Tanh()
        self.ac = nn.Tanh()

        self.fc1 = torch.nn.Linear(h_dim, z_dim)
        # self.de = DropEdge(0.99)
        self.c3 = GCNConv(h_dim, z_dim, add_self_loops=True)

    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i].to(self.device)
        else:
            x = self.data.xs.to(self.device)

        ei = self.data.ei_masked(mask_enum, i).to(self.device)
        # ea = self.data.ea_masked(mask_enum, i).to(self.device)
        ew = self.data.ew_masked(mask_enum, i).to(self.device)
        x1 = self.c1(x, ei, edge_weight=ew)
        x2 = self.c2(x1, ei, edge_weight=ew)
        x = self.relu(x2)
        x = self.drop(x)
        x = self.c3(x, ei, edge_weight=ew)
        # Experiments have shown this is the best activation for GCN+GRU
        return self.ac(x)

def detector_nnc_rref(loader, kwargs, h_dim, z_dim, **kws):
    device = kwargs.pop('device')
    return DetectorEncoder(
        NNC(loader, kwargs, h_dim, z_dim, device)
    )





class NNC_back(GCN):
    '''
    2-layer NNConv implimenting the Euler Embed Unit interface. Inherits GCN
    as the only difference is the forward method, and which submodules are used
    edge_attr is used, not a single edge_weight
    '''

    def __init__(self, data_load, data_kws, h_dim, z_dim, device):
        super().__init__(data_load, data_kws, h_dim, z_dim, device)

        # TODO: figure out the numbers
        # following this example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_nn_conv.py
        # nn.Linear(input_features, output_features)
        # self.data.x_dim = self.data.x_dim + 1
        # nn1 = nn.Sequential(nn.Linear(self.data.ea_dim, 64), nn.ReLU(),
        #                     nn.Linear(64, self.data.x_dim * h_dim))
        # self.c1 = NNConv(self.data.x_dim, h_dim, nn1, aggr='mean')

        self.data.x_dim = self.data.x_dim
        # self.c3 = SAGEConv(h_dim, z_dim, aggr='mean', normalize=True, bias=False)
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.c2 = GCNConv(h_dim, h_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.ac = nn.Tanh()
        # nn.Softmax(dim=1)
        # nn.ReLU()
        # nn.Sigmoid()


        # self.fc1 = nn.Linear(h_dim, z_dim) # comment it the ap will decrease
        # self.de = DropEdge(0.2)
        self.c3 = GCNConv(h_dim, z_dim, add_self_loops=True)
        # torch.nn.init.xavier_uniform(self.c1.weight)
        # torch.nn.init.xavier_uniform(self.c2.weight)
        # torch.nn.init.xavier_uniform(self.c3.weight)

        # nn3 = nn.Sequential(nn.Linear(self.data.ea_dim, 8), nn.ReLU(),
        #                     nn.Linear(8, h_dim * h_dim))
        # self.c3 = NNConv(h_dim, z_dim, nn3, aggr='mean')

        nn4 = nn.Sequential(nn.Linear(3, 8), nn.ReLU(),
                            nn.Linear(8, h_dim * z_dim))
        self.c4 = NNConv(h_dim, z_dim, nn4, aggr='mean')


    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i].to(self.device)
        else:
            x = self.data.xs.to(self.device)

        ei = self.data.ei_masked(mask_enum, i).to(self.device)
        ea = self.data.ea_masked(mask_enum, i).to(self.device)
        ew = self.data.ew_masked(mask_enum, i).to(self.device)
        ea = torch.transpose(ea, 0, 1)
        # nd = self.data.nds[i].transpose(1,0)
        # print(x.shape, nd.shape)
        # x = torch.concat((nd,x),1)#.transpose(1,0)
        # print(x.shape)

        # ei, ew = self.de(ei, ewa=ew)
        # #ea should be transposed num_edges * num_features
        # ei, ew = self.de(ei, ew)
        # print(x.shape, ei.shape, ew.shape, ea.shape)
        # exit()
        x1 = self.c1(x, ei, edge_weight=ew)

        # x = self.relu(x)
        # x = self.drop(x)
        x = self.c2(x1, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c3(x, ei, edge_weight=ew)
        # print(x.shape, ei.shape, ea.shape)
        # x = self.c3(x, ei, edge_attr=ea)

        x = self.relu(x)
        x = self.drop(x)
        x = self.c4(x, ei, edge_attr=ea)

        # Experiments have shown this is the best activation for GCN+GRU
        # print(x.grad_fn, x.shape)
        return self.ac(x)



#Add NNConv to accommodate edge_attr 1x3: #U login, #C login, and #A login
class NNC_back(GCN):
    '''
    2-layer NNConv implimenting the Euler Embed Unit interface. Inherits GCN
    as the only difference is the forward method, and which submodules are used
    edge_attr is used, not a single edge_weight
    '''

    def __init__(self, data_load, data_kws, h_dim, z_dim, device):
        super().__init__(data_load, data_kws, h_dim, z_dim, device)

        # TODO: figure out the numbers
        # following this example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_nn_conv.py
        # nn.Linear(input_features, output_features)

        # pid 1675
        # ppid 1661
        # dest_port 12
        # l4protocol 2
        # img_path 127
        # self.pid_edim = 11
        # self.ppid_edim = 11
        # self.dest_edim = 4
        # self.l4_edim = 1
        # self.img_edim = 8
        # self.use_flows = use_flows

        # if self.use_flows == True:
        # self.embedding_pid = nn.Embedding(1675, self.pid_edim, max_norm=True)
        # self.embedding_ppid = nn.Embedding(1661, self.ppid_edim, max_norm=True)
        # self.embedding_dest = nn.Embedding(12, self.dest_edim, max_norm=True)
        # self.embedding_l4 = nn.Embedding(2, self.l4_edim, max_norm=True)
        # self.embedding_img = nn.Embedding(127, self.img_edim, max_norm=True)

        # nn1 = nn.Sequential(nn.Linear(self.pid_edim+self.ppid_edim+self.dest_edim+self.img_edim+self.l4_edim, 64), nn.ReLU(), nn.Linear(64, self.data.x_dim * h_dim))
        # nn2 = nn.Sequential(nn.Linear(self.pid_edim+self.ppid_edim+self.dest_edim+self.img_edim+self.l4_edim, 32), nn.ReLU(), nn.Linear(32, h_dim * h_dim))

        # nn1 = nn.Sequential(nn.Linear(self.l4_edim, 64), nn.ReLU(), nn.Linear(64, self.data.x_dim * h_dim))
        # nn2 = nn.Sequential(nn.Linear(self.l4_edim, 32), nn.ReLU(), nn.Linear(32,   h_dim * h_dim))

        # print(self.data.ea_dim, self.data.x_dim)
        # nn1 = nn.Sequential(nn.Linear(self.data.ea_dim, 64), nn.ReLU(), nn.Linear(64, self.data.x_dim * h_dim))
        # self.c1 = NNConv(self.data.x_dim, h_dim, nn1, aggr='mean')

        # nn2 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(),
        #                     nn.Linear(32, self.data.x_dim * h_dim))
        # self.c1 = NNConv(self.data.x_dim, h_dim, nn2, aggr='mean')
        self.de = DropEdge(0.5)
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        # self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        # self.c2 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.c2 = GCNConv(h_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)
        self.c3 = GCNConv(h_dim, z_dim, add_self_loops=True)
        # self.c4 = GCNConv(h_dim, z_dim, add_self_loops=True)

        nn4 = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, h_dim * z_dim))
        self.c4 = NNConv(h_dim, z_dim, nn4, aggr='mean')

        self.tanh = nn.Tanh()

        # self.c3 = SAGEConv(h_dim, z_dim, aggr='mean', normalize=True, bias=False)



    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i].to(self.device)
        else:
            x = self.data.xs.to(self.device)
        # x = self.data.xs

        ei = self.data.ei_masked(mask_enum, i).to(self.device)
        ew = self.data.ew_masked(mask_enum, i).to(self.device)
        # ea = self.data.ea_masked(mask_enum, i)

        ew_nnconv = ew.clone()
        ew_nnconv = ew_nnconv[None, :]
        ew_nnconv = ew_nnconv.transpose(1, 0)
        # print(ew.shape, ew_nnconv.shape)

        # ei, ew = self.de(ei, ewa=ew)

        # print(ea.shape)
        # ea_pid = self.embedding_pid(ea[0].int())
        # ea_ppid = self.embedding_ppid(ea[1].int())
        # ea_dest = self.embedding_dest(ea[2].int())
        # ea_l4 = self.embedding_l4(ea[3].int())
        # ea_img = self.embedding_img(ea[4].int())
        # print(ea_pid.shape, ea_ppid.shape, ea_l4.shape, ea_dest.shape, ea_img.shape)
        # ea = torch.concat((ea_pid, ea_ppid, ea_l4, ea_dest, ea_img), dim=1).transpose(1,0)


        # # Simple 2-layer GCN. Tweak if desired
        # x = self.c1(x, ei, edge_weight=ew)
        # x = self.relu(x)
        # x = self.drop(x)
        # x = self.c2(x, ei, edge_weight=ew)
        # return self.tanh(x)


        # print(ea.shape)
        # ei= self.de(ei)
        # print(ei.shape, ew.shape)
        # ei, ew = self.de(ei, ewa=ew)
        # print(ei.shape, ew.shape)

        #ea should be transposed num_edges * num_features
        # ea = torch.transpose(ea_l4, 0, 1)
        # print(ea)
        # exit()
        # print("c1", x.shape, ei.shape, ea.shape)
        # x = self.c1(x, ei)
        # print(x.shape, ei.shape, ea.shape)
        # x = self.c1(x, ei, edge_weight=ew)

        # x = self.c1(x, ei, edge_attr=ew)
        # x = self.relu(x)
        # x = self.drop(x)
        # print("c2", x.shape, ei.shape, ea.shape)
        # x = self.c2(x, ei)
        # ea_3 = ea[3].clone().detach().float()
        # print(type(ew), type(ea))
        x = self.c1(x, ei, edge_weight=ew)
        # x = self.c1(x, ei, edge_attr=ew_nnconv)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei, edge_weight=ew)

        # .        x = self.c2(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        # print(x.shape, ei.shape, ea.shape)
        x = self.c3(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        # print(ea_3.shape)
        # print(ea_3, ew)
        x = self.c4(x, ei, edge_attr=ew_nnconv)

        # x = self.c3(x, ei)
        # x = F.softmax(x,dim=1)
        # Experiments have shown this is the best activation for GCN+GRU
        return self.tanh(x)
