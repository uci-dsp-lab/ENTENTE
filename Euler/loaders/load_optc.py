from copy import deepcopy
import os
import pickle
from joblib import Parallel, delayed

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized, std_edge_a

import numpy as np

DATE_OF_EVIL_LANL = 573290 #original 573290
FILE_DELTA = 10000

# Input where LANL data cleaned with .clean_lanl.py is stored
LANL_FOLDER = './data/optc_euler_features/'
assert LANL_FOLDER, 'Please fill in the LANL_FOLDER variable:\n line 17'

COMP = 0
USR = 1
SPEC = 2
X_DIM = 17688

TIMES = {
    '20'      : 573383, # First 20 anoms 1.55%
    '100'     : 573751, # First 100 anoms 11.7%
    '500'     : 575885, # First 500 anoms 18.73%
    'all'  : 745983,  # Full 21784
    'test' :  10000
}

torch.set_num_threads(1)

def empty_lanl(use_flows=False):
    return make_data_obj(None,[],None,None,None,use_flows=use_flows)

def load_optc_dist(start=0, end=635015, delta=8640, is_test=False, use_flows=False, ew_fn=std_edge_w, ea_fn=std_edge_a, client='All', mc=None):

    if start == None or end == None:
        return empty_lanl(use_flows)

    num_slices = ((end - start) // delta)
    remainder = (end-start) % delta
    num_slices = num_slices + 1 if remainder else num_slices

    # Can't distribute the job if not enough workers
    return load_partial_lanl(start, end, delta, is_test, use_flows, ew_fn, ea_fn, client, mc)


# wrapper bc its annoying to send kwargs with Parallel
def load_partial_lanl_job(pid, args):
    data = load_partial_lanl(**args)
    return data


def make_data_obj(cur_slice, eis, ys, ew_fn, ea_fn, ews=None, eas=None, use_flows=False, **kwargs):
    if 'node_map' in kwargs:
        nm = kwargs['node_map']
    else:
        nm = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))

    cl_cnt = len(nm)
    x = torch.eye(cl_cnt+1)

    # Build time-partitioned edge lists
    eis_t = []
    masks = []

    for i in range(len(eis)):
        ei = torch.tensor(eis[i])
        eis_t.append(ei)

        # This is training data if no ys present
        if isinstance(ys, None.__class__):
            masks.append(edge_tv_split(ei)[0])

    # Balance the edge weights if they exist
    if not isinstance(ews, None.__class__):
        cnt = deepcopy(ews)
        ews = ew_fn(ews)
    else:
        cnt = None
    lengths = [len(sublist[0]) for sublist in eis_t]
    print("lengths: ", sum(lengths))
    return TData(
        cur_slice, eis_t, x, ys, masks, ews=ews, eas=eas, use_flows=use_flows, cnt=cnt, node_map=nm
    )

'''
Read a file in flows and return the edge features

'''
def load_flows(fname, start, end):
    eas_flows = {}
    temp_flows = {}
    if not os.path.exists(fname):
        return eas_flows
    in_f = open(fname)
    line = in_f.readline()

    fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[6]), int(x[7]), int(x[8]))

    while line:
        l = line.split(',')
        ts = int(l[0])
        if ts < start:
            line = in_f.readline()
            continue
        if ts > end:
            break
        ts, src, dst, duration, pck_cnt, byte_cnt = fmt_line(l)
        et = (src,dst)
        if et in temp_flows:
            temp_flows[et][0].append(duration)
            temp_flows[et][1].append(pck_cnt)
            temp_flows[et][2].append(byte_cnt)
        else:
            temp_flows[et] = [[duration], [pck_cnt], [byte_cnt]]
        line = in_f.readline()
    in_f.close()
    #computes features, # of flows, mean & std of duration, pck_cnt and byte_cnt
    for et in temp_flows.keys():
        eas_flows[et] = [len(temp_flows[et][0]), np.mean(temp_flows[et][0]), np.std(temp_flows[et][0]), \
        np.mean(temp_flows[et][1]), np.std(temp_flows[et][1]), np.mean(temp_flows[et][2]), np.std(temp_flows[et][2])]
    return eas_flows

'''
Equivilant to load_cyber.load_lanl but uses the sliced LANL files
for faster scanning to the correct lines
'''
def load_partial_lanl(start=140000, end=156659, delta=8640, is_test=False, use_flows=False, ew_fn=standardized, ea_fn=std_edge_a, client='Others', mc=None):
    print('client ' + client + 'start:' + str(start) + ', end:' + str(end))
    cur_slice = int(start - (start % FILE_DELTA))
    start_f = str(cur_slice) + '.txt'
    in_f = open(LANL_FOLDER + start_f, 'r')

    edges = []
    ews = []
    edges_t = {}
    ys = []
    slices = []
    eas = []

    # Predefined for easier loading so everyone agrees on NIDs
    node_map = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))
    fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]), int(x[6]), int(x[7]),int(x[8][:-1]))
    def add_edge(et, ea, is_anom=0):
        if et in edges_t:
            val = edges_t[et]
            edges_t[et] = (max(is_anom, val[0]), val[1]+1, ea)
        else:
            edges_t[et] = (is_anom, 1, ea)


    scan_prog = tqdm(desc='Finding start', total=start-cur_slice-1)
    prog = tqdm(desc='Seconds read', total=end-start-1)

    anom_marked = False
    keep_reading = True
    next_split = start+delta

    line = in_f.readline()
    curtime = fmt_line(line.split(','))[0]
    old_ts = curtime

    while keep_reading:
        while line:
            l = line.split(',')

            # Scan to the correct part of the file
            ts = int(l[0])
            if ts < start:
                line = in_f.readline()
                scan_prog.update(ts-old_ts)
                old_ts = ts
                curtime = ts

            ts, src, dst, pid, ppid, dest_port, l4protocol, img_path, label = fmt_line(l)
            # Federated learning
            if not client in ['All', 'Fed']:
                #if All or Fed, no filtering
                src_cluster = mc[src]
                dst_cluster = mc[dst]
                # 1 hop
                if (str(src_cluster) != client) and (str(dst_cluster) != str(client)):
                    line = in_f.readline()
                    continue



            ea = (int(pid), int(ppid), int(dest_port), int(l4protocol), int(img_path))

            #Take the first char of src_u -> C, U or A, and the frequency of each type is the edge feature (3 features)
            et = (src,dst)

            # Not totally necessary but I like the loading bar
            prog.update(ts-old_ts)
            old_ts = ts

            # Split edge list if delta is hit
            if ts >= next_split:
                if len(edges_t):
                    ei = list(zip(*edges_t.keys()))
                    edges.append(ei)

                    #uc, us, ua: user C+, user U+, user Anonymous
                    y,ew, ea = list(zip(*edges_t.values()))
                    ews.append(torch.tensor(ew))

                    if use_flows:
                        ea = [ list(elem) if len(elem) ==5 else list(elem[0]) for elem in ea ]
                        ea = np.array(ea)
                        if ea.ndim == 3:
                            ea = ea[0]
                        eas.append(torch.tensor(ea).transpose(1,0))
                    if is_test:
                        ys.append(torch.tensor(y))

                    slices.append(str(ts))

                    edges_t = {}

                # If the list was empty, just keep going if you can
                curtime = next_split
                next_split += delta

                # Break out of loop after saving if hit final timestep
                if ts >= end:
                    keep_reading = False
                    break

            # Skip self-loops
            if et[0] == et[1]:
                line = in_f.readline()
                continue
            add_edge(et, ea, is_anom=label)
            # add_edge(et, src_u, is_anom=label)
            line = in_f.readline()

        in_f.close()
        cur_slice += FILE_DELTA

        if os.path.exists(LANL_FOLDER + str(cur_slice) + '.txt'):
            in_f = open(LANL_FOLDER + str(cur_slice) + '.txt', 'r')
            line = in_f.readline()
        else:
            keep_reading=False
            break

    ys = ys if is_test else None

    scan_prog.close()
    prog.close()
    edge_number = 0
    min_edge = 10000000
    max_edge = 0
    for i in edges:
    # fed_model.gcns.module.data.eis:
        edge_number += len(i[1])
        if max_edge < len(i[1]):
            max_edge = len(i[1])
        if min_edge > len(i[1]):
            min_edge = len(i[1])

    print("edges: ", len(edges), edge_number/len(edges))
    print("max_edge: ", max_edge)
    print("min_edge: ", min_edge)

    return make_data_obj(
        slices, edges, ys, ew_fn, ea_fn,
        ews=ews, eas=eas, use_flows=use_flows, node_map=node_map
    )


if __name__ == '__main__':


    test_data = load_optc_dist(start=573290, end=745983, delta=360, use_flows=False, is_test=True)
    test_eis = []
    test_ys = []
    for i in range(test_data.T):

        test_eis.append(test_data.eis[i].transpose(1,0).cpu().detach().numpy())
        test_ys.append(test_data.ys[i].cpu().detach().numpy())

    exit()
