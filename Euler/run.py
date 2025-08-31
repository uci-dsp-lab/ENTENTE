from argparse import ArgumentParser
import os, datetime
import pandas as pd
import torch

import loaders.load_optc as optc
from models.recurrent import GRU, LSTM, EmptyModel
from models.embedders import detector_gcn_rref
from spinup import run_all
from simple import simple_run_all
from loaders.split_lanl import reverse_load_map
import json

# Reproducibility
import numpy as np
import random
seed = 0
random.seed(seed) # python random generator
np.random.seed(seed) # numpy random generator

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


AUX_FOLDER = './Euler/'
LANL_FOLDER = './data/fl_lanl/'

DEFAULT_TR = {
    # 'anom_lr': 0.05,
    'epochs': 30,
    'min': 1,
    'nratio': 1,
    'val_nratio': 1,
    'fedT': 10,
    'fedMu': 0.05
}


def load_machine_clusters(nmap_fname, cluster_fname, dataset):
    if dataset.startswith('L'):
        reverse_nm = reverse_load_map(nmap_fname)
    elif dataset.startswith('O'):
        with open(nmap_fname, 'r') as f:
            reverse_nm = json.load(f)
    clusters = json.loads(open(cluster_fname).read())
    machine_clusters = {}
    count = 0
    max_cluster_ind = 0
    for cluster_ind, machines in enumerate(clusters['bottom']):
        for m in machines.keys():
            #convert the machine name into ID with reversed node map
            if not m in reverse_nm:
                #print(m + ' is not in node map')
                #count = count + 1
                continue
            machine_clusters[reverse_nm[m]] = cluster_ind
        if cluster_ind > max_cluster_ind:
            max_cluster_ind = cluster_ind
    #Cluster 0-11, regular clusters
    #put the 70 outlier machines into a new cluster, ID: 12
    for m in reverse_nm.keys():
        if not reverse_nm[m] in machine_clusters:
            machine_clusters[reverse_nm[m]] = max_cluster_ind + 1
    for i in range(12):
        count = sum(1 for v in machine_clusters.values() if v == i)
        print(i, count)

    return machine_clusters


def get_args():
    global DEFAULT_TR
    ap = ArgumentParser()
    ap.add_argument('-d', '--delta', type=float, default=0.1)
    ap.add_argument('-e', '--encoder_name', choices=['GCN', 'GAT', 'SAGE', 'ARGUS' , 'VGRNN', 'LOF', 'IF'], type=str.upper, default="GCN") # VGRNN may has some issues
    ap.add_argument('-r', '--rnn', choices=['GRU', 'LSTM', 'NONE'], type=str.upper, default="GRU")
    ap.add_argument('-H', '--hidden', type=int, default=32)
    ap.add_argument('-z', '--zdim', type=int, default=16)
    ap.add_argument('-n', '--ngrus', type=int, default=1)
    ap.add_argument('-t', '--tests', type=int, default=1, help="repeated testing number")
    ap.add_argument('-l', '--load', action='store_true')
    ap.add_argument('--gpu', action='store_false')
    ap.add_argument('-te', '--te_end', choices=['20', '100', '500', 'all', 'real'], type=str.lower, default="all")
    ap.add_argument('--fpweight', type=float, default=0.6)
    ap.add_argument('--nowrite', default=False)
    ap.add_argument('--impl', '-i', type=str.upper, choices=['DETECT', 'D'], default="DETECT")
    ap.add_argument('--dataset', default='OPTC', type=str.upper, choices=['OPTC', 'LANL'])
    ap.add_argument('--lr', default=0.005, type=float) #lanl: 0.01; optc: 0.005
    ap.add_argument('--patience', default=100, type=int) # lanl 3; otpc 8
    ap.add_argument('--flows', action='store_true')
    ap.add_argument('--fed', default='Momentum', choices=['None', 'WeightedFedAvg', 'WeightedFedAvg2', 'FedAvg', 'FedOpt', 'FedProx', 'Momentum'])
    ap.add_argument('--fedopt_epoch', default=5)
    ap.add_argument('--cluster_fname', default='optc_2_2.json', help='optc_5_5.json, optc_2_2.json')
    ap.add_argument('--client_number', default=3)
    ap.add_argument('--personalization', action='store_true')
    ap.add_argument('--poison', action='store_true')
    ap.add_argument('--poison_defense', action='store_true')
    ap.add_argument('--nb_defense', action='store_true')
    ap.add_argument('--dp_defense', action='store_true')
    ap.add_argument('--cdp_defense', action='store_true')
    ap.add_argument('--poison_client', default=[2], choices=['3', '1', '0'])
    ap.add_argument('--poison_lambda', default=100)
    ap.add_argument('--poison_p', default=0.5)
    ap.add_argument('--epochs', default=None)
    ap.add_argument('--equal_weights', action='store_true')


    if ap.parse_args().dataset == 'LANL':
        default_all_clients = [str(x) for x in range(0,int(ap.parse_args().client_number))] # LANL: 5/12, OPTC 3
    elif ap.parse_args().dataset == 'OPTC':
        default_all_clients = [str(x) for x in range(0,int(ap.parse_args().client_number))]


    ap.add_argument('--client_list', nargs='+', default=default_all_clients)
    ap.add_argument('--client', default='0', choices=default_all_clients + ['All', 'Fed']) # 0
    ap.add_argument('--test_client', default='All', choices=['DOM1', 'Others', 'All']) # for init_single function

    args = ap.parse_args()
    if args.fed == 'None':
        args.client = 'All'
    if args.epochs is not None:
        DEFAULT_TR['fedT'] = int(args.epochs)
        print('Using epochs:', DEFAULT_TR['fedT'])
    assert args.fpweight >= 0 and args.fpweight <=1, '--fpweight must be a value between 0 and 1 (inclusive)'
    readable = str(args)
    print(readable)
    model_str = '%s -> %s (%s)' % (args.encoder_name , args.rnn, args.impl)
    print(model_str)
    args.dataset = args.dataset+'_'+args.encoder_name

    # Parse dataset info
    if args.dataset.startswith('O'):
        args.loader = optc.load_optc_dist
        args.tr_start = 0
        args.tr_end = optc.DATE_OF_EVIL_LANL
        args.val_times = None # Computed later
        if args.te_end == 'real':
            args.te_end = 'all'
        args.te_times = [(args.tr_end, optc.TIMES[args.te_end])]
        args.delta = int(args.delta * (60**2))
    elif args.dataset.startswith('L'):
        args.loader = lanl.load_lanl_dist
        args.tr_start = 0
        args.tr_end = lanl.DATE_OF_EVIL_LANL
        args.val_times = None # Computed later
        args.te_times = [(args.tr_end, lanl.TIMES[args.te_end])]
        args.delta = int(args.delta * (60**2))
    else:
        raise NotImplementedError('Only OpTC and LANL data sets are supported in this release')

    # Convert from str to function pointer
    if args.encoder_name == 'GCN':
        args.encoder = detector_gcn_rref
    elif args.encoder_name == 'GAT':
        args.encoder = detector_gat_rref
    elif args.encoder_name == 'SAGE':
        args.encoder = detector_sage_rref
    elif (args.encoder_name == 'ARGUS') and (args.dataset.startswith('L')):
        args.encoder = detector_arguslanl_rref
    elif (args.encoder_name == 'ARGUS') and (args.dataset.startswith('O')):
        args.encoder = detector_argusoptc_rref
    elif args.encoder_name == 'VGRNN':
        args.encoder = detector_vgrnn_rref
    elif args.encoder_name == 'EGCNO':
        args.encoder = detector_egcno_rref
    elif args.encoder_name == 'EGCNH':
        args.encoder = detector_egcnh_rref
    elif args.encoder_name in ['LOF','IF']:
        pass
    else:
        raise NotImplementedError("wrong encoder", args.encoder_name, args.dataset)

    if args.rnn == 'GRU':
        args.rnn = GRU
    elif args.rnn == 'LSTM':
        args.rnn = LSTM
    else:
        args.rnn = EmptyModel
    if args.dataset.startswith('L'):
        mc = load_machine_clusters(LANL_FOLDER + 'nmap.pkl', AUX_FOLDER+args.cluster_fname, args.dataset)
    elif args.dataset.startswith('O'):
        mc = load_machine_clusters(LANL_FOLDER + 'ip_dict.json', AUX_FOLDER+args.cluster_fname, args.dataset)
    args.mc = mc
    return args, readable, model_str

if __name__ == '__main__':
    args, argstr, modelstr = get_args()
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fed_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        device = torch.device('cpu')

    DEFAULT_TR['lr'] = args.lr
    DEFAULT_TR['patience'] = args.patience
    OUTPATH = './Exps/result/'
    if not os.path.exists(OUTPATH): os.makedirs(OUTPATH)
    if args.rnn != EmptyModel:
        worker_args = [args.hidden, args.hidden]
        rnn_args = [args.hidden, args.hidden, args.zdim]
    else:
        worker_args = [args.hidden, args.zdim]
        rnn_args = [None, None, None]

    if args.encoder_name in ['LOF', 'IF']:
        stats = [simple_run_all(args, rnn_args, worker_args, DEFAULT_TR, OUTPATH, device) for _ in range(args.tests)]
        exit()
    else:
        stats = [run_all(args, rnn_args, worker_args, DEFAULT_TR, OUTPATH, device, fed_device) for _ in range(args.tests)]

    if args.nowrite:
        exit()

    f = open(OUTPATH+'results.txt', 'a')
    f.write("======New Exp======\n")
    f.write(str(argstr) + '\n')
    f.write(modelstr + ', LR: ' + str(args.lr) + '\n')

    dfs = [pd.DataFrame(s) for s in list(zip(*stats))]
    dfs = pd.concat(dfs, axis=0)

    for m in dfs['Model'].unique():
        df = dfs[dfs['Model'] == m]
        compressed = pd.DataFrame(
            [df.mean(), df.sem()],
            index=['mean', 'stderr']
        ).to_csv().replace(',', '\t') # For easier copying into Excel
        full = df.to_csv(index=False, header=False)
        full = full.replace(',', ', ')
        f.write(m + '\n')
        f.write(str(compressed) + '\n')
        f.write(full + '\n')
    f.close()
