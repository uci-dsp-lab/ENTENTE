import math
import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score as auc_score, \
    f1_score, average_precision_score as ap_score, \
    precision_recall_curve
from sklearn.metrics import confusion_matrix, precision_score, roc_auc_score, f1_score
import copy
import torch
from torch.optim import Adam, Adadelta, AdamW
from loaders.tdata import TData
from loaders.load_optc import load_optc_dist
from models.euler_detector import DetectorEncoder, DetectorRecurrent
from utils import get_score, get_optimal_cutoff
from fl_utils import *
from copy import deepcopy

TMP_FILE = 'tmp.dat'
SCORE_FILE = 'scores.txt'
LOAD_FN = None

NORM_MIN = math.inf
NORM_MAX = 0



def init_single(args, rnn_args, worker_args, times, tr_args, OUTPATH, device):
    global LOAD_FN
    LOAD_FN = args.loader

    # Evaluating a pre-trained model, so no need to train
    if args.load:
        kwargs = {
            'start': None,
            'end': None,
            'use_flows': args.flows,
            'device': device,
            'client': args.client,
            'mc': args.mc
        }
        rrefs = args.encoder(LOAD_FN, kwargs, *worker_args)
        rnn = args.rnn(*rnn_args)
        model = DetectorRecurrent(rnn, rrefs, device)
        if args.client =='All':
            states = pickle.load(open('./Exps/FL/All_LANL_'+args.encoder_name+'.pkl', 'rb'))
            model.load_states(*states['states'])
        elif args.client =='DOM1':
            states = pickle.load(open('./Exps/FL/DOM1_LANL_'+args.encoder_name+'.pkl', 'rb'))
            model.load_states(*states['states'])
        elif args.client == 'Others':
            states = pickle.load(open('./Exps/FL/Others_LANL_'+args.encoder_name+'.pkl', 'rb'))
            model.load_states(*states['states'])
        h0 = states['h0']
        tpe = 0
        tr_time = 0
    # Building and training a fresh model
    else:
        kwargs = {
                'start': times['tr_start'],
                'end': times['tr_end'],
                'delta': times['delta'],
                'is_test': False,
                'use_flows': args.flows,
                'device': device,
                'client': args.client, # set training client
                'mc': args.mc
                }
        rrefs = args.encoder(LOAD_FN, kwargs, *worker_args)
        tmp = time.time()
        model, h0, tpe = train(rrefs, tr_args, args, rnn_args, device)
        tr_time = time.time() - tmp
        print("tr_time: ", tr_time)
    model = model.to(device)
    h0, zs = get_cutoff(model, h0, times, tr_args, args.fpweight, args.flows, device, args.client, args.mc)
    stats = []
    for te_start,te_end in times['te_times']:
        test_times = {
            'te_start': te_start,
            'te_end': te_end,
            'delta': times['delta']
        }

        st = test(model, h0, test_times, args.flows, OUTPATH, device, args, args.test_client, args.mc) # local test client

        for s in st:
            s['TPE'] = tpe
        stats += st

    pickle.dump(stats, open(OUTPATH+TMP_FILE, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)

    if True:
        for i in range(len(args.client_list)):
            print('Test model-client-'+str(args.client_list[i]+' on its local data'))
            h0, zs = get_cutoff(model, h0, times, tr_args, args.fpweight, args.flows, device, args.client_list[i], args.mc)
            stats = []
            for te_start,te_end in times['te_times']:
                test_times = {
                'te_start': te_start,
                'te_end': te_end,
                'delta': times['delta']}
                st = test(model, h0, test_times, args.flows, OUTPATH, device, args, args.client_list[i], args.mc)
                stats += st
            pickle.dump(stats, open(OUTPATH+TMP_FILE, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)

    return


def train(rrefs, kwargs, args, rnn_args, device):
    rnn_constructor = args.rnn
    impl = args.impl
    dataset = args.dataset
    rnn = rnn_constructor(*rnn_args)
    model = DetectorRecurrent(rnn, rrefs, device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=kwargs['lr'])
    times = []
    best = (model.save_states(), 0)
    no_progress = 0
    if args.poison == True:
        with open('./malicious_edges_cluster_3.pkl', "rb") as file:
            malicious_edges = pickle.load(file)
        malicious_edges = torch.tensor(malicious_edges).transpose(1,0).to(device)
        for i in range(40):
            malicious_edges = torch.tensor(all_malicious_edges[i]).to(device)
            model.gcns.module.data.eis[i] = torch.cat((model.gcns.module.data.eis[i], malicious_edges), dim=1)
            temp_mask = torch.zeros_like(malicious_edges[0]).bool()
            model.gcns.module.data.masks[i] = torch.cat((model.gcns.module.data.masks[i], temp_mask), dim=0)
            original_tensor = torch.full((malicious_edges[0].shape), 0.5).to(device)
            temp_weights = torch.rand_like(original_tensor)
            model.gcns.module.data.ews[i] = torch.cat((model.gcns.module.data.ews[i], temp_weights), dim=0)

    for e in range(kwargs['epochs']):
        # Get loss and send backward
        model.train()
        # with dist_autograd.context() as context_id:
        print("forward")
        st = time.time()

        zs = model.forward(TData.TRAIN)
        loss = model.loss_fn(zs, TData.TRAIN, nratio=kwargs['nratio'], device=device, encoder_name=args.encoder_name)
        print("backward")
        loss.backward()
        opt.step()
        elapsed = time.time()-st
        times.append(elapsed)
        l = loss.sum()
        print('[%d] Loss %0.8f  %0.2fs' % (e, l.item(), elapsed))

        # Get validation info to prevent overfitting
        model.eval()
        with torch.no_grad():
            zs = model.forward(TData.TRAIN, no_grad=True)
            p,n = model.score_edges(zs, TData.VAL)

            auc,ap = get_score(p,n)
            print("\tValidation: AP: %0.4f  AUC: %0.4f" % (ap, auc), end='')

            # Either incriment or update early stopping criteria
            tot = auc+ap
            if tot > best[1]:
                print('*\n')
                best = (model.save_states(), tot)
                no_progress = 0
            else:
                print('\n')
                if e >= kwargs['min']:
                    no_progress += 1
            if no_progress == kwargs['patience']:
                print("Early stopping!")
                break
    model.load_states(*best[0])
    # Get the best possible h0 to eval with
    zs, h0 = model(TData.TEST, include_h=True)
    states = {'states': best[0], 'h0': h0}
    f = open('./Exps/FL/'+args.client+'_'+dataset+'.pkl', 'wb+')

    pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)
    tpe = sum(times)/len(times)
    print("Exiting train loop")
    print("Avg TPE: %0.4fs" % tpe)
    return model, h0, tpe



def fed_combine(client_models, args, fed_model, accountant, device, fed_device):
    global NORM_MAX
    global NORM_MIN
    w_glob = fed_model.cpu().save_states()[1]
    w_locals = []
    for i in range(len(args.client_list)):
        if client_models['model-client-' + str(args.client_list[i])] != None:
            w_local = client_models['model-client-' + str(args.client_list[i])].cpu().save_states()[1]
            w_locals.append(w_local)
        else:
            w_locals.append(None)




    if args.dp_defense == True:
        M = 5.0
        sigma = 0.01
        fed_model_weight = fed_model.save_states()[1]

        for i in range(len(args.client_list)):
            clipped_update = clip_update(w_locals[i], M, fed_model_weight)
            w_locals[i] = add_noise(clipped_update, sigma, fed_device)

    if args.nb_defense == True:
        M = 5.0
        fed_model_weight = fed_model.save_states()[1]

        for i in range(len(args.client_list)):
            w_locals[i] = clip_update(w_locals[i], M, fed_model_weight)



    if args.cdp_defense == True:
        N = args.client_number  # Total eparticipants
        T = 2  # Privacy budget threshold
        q = 0.5  # Fraction of participants in each round
        bound = 1.0  # Clipping bound
        # noise_scale = 0.1  # Some predefined noise scale value for DP
        noise_scale = 2.5  # Some predefined noise scale value for DP

        fed_model_weight = fed_model.save_states()[1]
        w_glob = central_DP_FL(w_locals, fed_model_weight, N, T, q, bound, noise_scale, accountant, fed_device, args.similarity)
        for i in range(len(args.client_list)):
            client_models['model-client-'+str(args.client_list[i])].load_states(*(client_models['model-client-'+str(args.client_list[i])].save_states()[0], w_glob))
            client_models['model-client-'+str(args.client_list[i])] = client_models['model-client-'+str(args.client_list[i])].to(device)
        fed_model.load_states(*(fed_model.save_states()[0], w_glob))
        fed_model = fed_model.to(fed_device)
        return fed_model, client_models

    if args.poison == True:
        for pi in args.poison_client:
            for i in w_locals[pi].keys():
                w_locals[pi][i] *= float(args.poison_lambda) # lambda


    for i in range(len(w_locals)):
        total = 0
        total_size = 0
        try:
            for value in w_locals[i].values():
                total += np.absolute(np.sum(value.detach().tolist()))
                total_size += value.numel() * value.element_size()
            print("model parameter norm: ",i, total)
            print(f"Model weights size: {total_size / 1024**2:.2f} MB")
        except:
            pass
        if total>NORM_MAX:
            NORM_MAX = total
        elif total<NORM_MIN:
            NORM_MIN = total
    total_size = 0
    for value in w_glob.values():
        total += np.absolute(np.sum(value.detach().tolist()))
        total_size += value.numel() * value.element_size()
    print("fed:", total)
    print(f"Fed Model weights size: {total_size / 1024**2:.2f} MB")

    if args.fed == 'FedAvg':
        print('Generate global model with FedAvg over clients')
        w_glob = FedAvg(w_locals)
    elif args.fed == 'WeightedFedAvg':
        print('Generate global model with weighted FedAvg over clients')
        w_glob = WeightedFedAvg(w_locals, args.client_list, args.mc)
    elif args.fed == 'WeightedFedAvg2':
        print('Weighted FedAvg but the weights come from graph distance')
        w_glob = WeightedFedAvg2(w_locals, args.client_list, args.similarity, args.poison_defense)
    elif args.fed == 'Momentum':
        w_glob = Momentum(w_locals, w_glob, args.client_list, args.similarity)
    elif args.fed == 'FedOpt':
        w_glob = FedOpt(w_locals)
    elif args.fed == 'FedProx':
        w_glob = FedProx(w_locals)
    elif args.fed == 'None':
        states = pickle.load(open('./Exps/model_save_'+args.dataset+'.pkl', 'rb'))
        model.load_state(*states['states'])
        h0 = states['h0']
    for i in range(len(args.client_list)):
        client_models['model-client-'+str(args.client_list[i])].load_states(*(client_models['model-client-'+str(args.client_list[i])].save_states()[0], w_glob))
        client_models['model-client-'+str(args.client_list[i])] = client_models['model-client-'+str(args.client_list[i])].to(device)
    fed_model.load_states(*(fed_model.save_states()[0], w_glob))
    fed_model = fed_model.to(fed_device)

    return fed_model, client_models


def init_fed(args, rnn_args, worker_args, times, tr_args, OUTPATH, device, fed_device):
    print("init fed!")
    global LOAD_FN
    LOAD_FN = args.loader
    global_G = None
    if (args.fed == 'WeightedFedAvg2') or (args.fed == 'Momentum'):
        global_G = build_random_graph(args.mc)

    #load models that have been trained ahead
    if args.load:
        kwargs = {
            'start': None,
            'end': None,
            'use_flows': args.flows,
            'device': device,
            'client': args.client,
            'mc': args.mc
        }
        rrefs = args.encoder(LOAD_FN, kwargs, *worker_args)
        rnn = args.rnn(*rnn_args)
        fed_model = DetectorRecurrent(rnn, rrefs, fed_device)
        if args.client =='All':
            states = pickle.load(open('./Exps/FL/All_LANL_'+args.encoder_name+'.pkl', 'rb'))
        elif args.client =='DOM1':
            states = pickle.load(open('./Exps/FL/DOM1_LANL_'+args.encoder_name+'.pkl', 'rb'))
        elif args.client == 'Others':
            states = pickle.load(open('./Exps/FL/Others_LANL_'+args.encoder_name+'.pkl', 'rb'))
        elif args.client == 'Fed':
            states = pickle.load(open('./Exps/FL/Fed_LANL_'+args.encoder_name+'.pkl', 'rb'))
        fed_model.load_states(*states['states'])
        fed_model = fed_model.to(fed_device)
        h0_ori = states['h0'].to(fed_device)
        tpe = 0
        tr_time = 0
    # Building and training a fresh model
    else:
        tmp = time.time()
        print("strat training.")
        rrefs_dict = dict()
        for i in range(len(args.client_list)):
            kwargs = {
                'start': times['tr_start'],
                'end': times['tr_end'],
                'delta': times['delta'],
                'is_test': False,
                'use_flows': args.flows,
                'device': device,
                'client': args.client_list[i],
                'mc': args.mc}
            rrefs_name="rrefs-client-" + str(args.client_list[i])
            rrefs = args.encoder(LOAD_FN, kwargs, *worker_args) # load data
            rrefs_dict.update({rrefs_name: rrefs})
            if args.fed == 'WeightedFedAvg2':
                #get the GCN data to compute the similarity
                wl_tmp = time.time()
                similarities = rrefs.module.data.compute_graph_weights(args.client_list[i], global_G)
                wl_tmp_time = time.time() - wl_tmp
                print("wl_tmp: ", wl_tmp_time)
                #we use 3-hop WL kernel, all the similarities turn out to be the same for the 3 iterations
                if not hasattr(args, 'similarity'):
                    args.similarity = []
                args.similarity.append(similarities[2])
            if args.fed == 'Momentum':
                wl_tmp = time.time()
                similarities = rrefs.module.data.compute_graph_weights(args.client_list[i], global_G)
                wl_tmp_time = time.time() - wl_tmp
                print("wl_tmp: ", wl_tmp_time)
                if not hasattr(args, 'similarity'):
                    args.similarity = []
                args.similarity.append(similarities[2])
        if args.equal_weights == True:
            args.similarity = [1.0/len(args.client_list) for _ in range(len(args.client_list))]


        fed_kwargs = {
                'start': times['tr_start'],
                'end': times['tr_end'],
                'delta': times['delta'],
                'is_test': False,
                'use_flows': args.flows,
                'device': fed_device,
                'client': "All",
                'mc': args.mc}
        fed_rrefs = args.encoder(LOAD_FN, fed_kwargs, *worker_args) # load data
        rrefs_dict.update({'fed': fed_rrefs})
        print("fed_rrefs",fed_rrefs)
        fed_model, h0_ori, tpe, client_models, client_h0  = train_fed(rrefs_dict, tr_args, args, rnn_args, device, fed_device, times, OUTPATH)
        tr_time = time.time() - tmp
        print("tr_time: ", tr_time)
    # calculate the cutoff point for the whole testing dataset
    if 1 == 1:
        print('Test the learnt global model on all data')
        fed_model.eval()
        h0, zs = get_cutoff(fed_model, h0_ori, times, tr_args, args.fpweight, args.flows, fed_device, 'All', args.mc)
        stats = []
        for te_start,te_end in times['te_times']:
            test_times = {
            'te_start': te_start,
            'te_end': te_end,
            'delta': times['delta']}
            st = test(fed_model, h0, test_times, args.flows, OUTPATH, fed_device, args, 'All', args.mc)
            for s in st:
                s['TPE'] = tpe
                s['tr_time'] = tr_time
            stats += st
        pickle.dump(stats, open(OUTPATH+TMP_FILE, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)
    # check model on each local dataset
    if 1 == 0:
        for i in range(len(args.client_list)):
            print('Test Fed Model on Client '+str(i)+' local data')
            fed_model.eval()
            h0, zs = get_cutoff(fed_model, h0_ori, times, tr_args, args.fpweight, args.flows, fed_device, args.client_list[i], args.mc)
            stats = []
            for te_start,te_end in times['te_times']:
                test_times = {
                'te_start': te_start,
                'te_end': te_end,
                'delta': times['delta']}
                st = test(fed_model, h0, test_times, args.flows, OUTPATH, fed_device, args, args.client_list[i], args.mc)

                stats += st
            pickle.dump(stats, open(OUTPATH+TMP_FILE, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)
    if args.personalization:
        for i in range(len(args.client_list)):
            model_name = 'model-client-'+str(args.client_list[i])
            client_models[model_name].eval()
            print('Test model-client-'+str(args.client_list[i]+' on its local data'))
            h0, zs = get_cutoff(client_models[model_name], client_h0[i], times, tr_args, args.fpweight, args.flows, device, args.client_list[i], args.mc)
            stats = []
            for te_start,te_end in times['te_times']:
                test_times = {
                'te_start': te_start,
                'te_end': te_end,
                'delta': times['delta']}
                st = test(client_models[model_name], h0, test_times, args.flows, OUTPATH, device, args, args.client_list[i], args.mc)
                stats += st
            pickle.dump(stats, open(OUTPATH+TMP_FILE, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)

    return

def get_model_size(model):
    """Returns the model size in megabytes (MB)."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = (param_size + buffer_size)
    return total_size

def train_fed(rrefs_dict, kwargs, args, rnn_args, device, fed_device,times_temp, OUTPATH):
    rnn_constructor = args.rnn
    impl = args.impl
    dataset = args.dataset
    rnn = rnn_constructor(*rnn_args)
    client_models = dict()
    client_opts = dict()
    client_bests = dict()
    print(device, fed_device)
    count = 0
    print("rrefs_dict: ", rrefs_dict)
    
    for i in range(len(args.client_list)):
        
        # set client
        model_name = 'model-client-'+str(count)
        temp_model = DetectorRecurrent(rnn, rrefs_dict['rrefs-client-'+str(args.client_list[i])], device)
        temp_model = temp_model.to(device)

        if len(temp_model.gcns.module.data.eis) != 0:
            client_models.update({model_name: temp_model})
            count += 1
        else:
            args.similarity[i] = None


    args.client_list = [str(x) for x in range(0,len(client_models))]
    args.similarity = [sim for sim in args.similarity if sim is not None]
    print("args.client_list: ", args.client_list, args.similarity)

    for i in range(len(args.client_list)):
        opt_name = 'opt-client-'+str(args.client_list[i])
        opt = torch.optim.Adam(client_models['model-client-'+str(args.client_list[i])].parameters(), lr=kwargs['lr'])
        client_opts.update({opt_name: opt})
    times = []

    fed_model = DetectorRecurrent(rnn, rrefs_dict['fed'], fed_device)
    fed_model = fed_model.to(fed_device)

    fed_model_best = (fed_model.save_states(), 0)

    no_progress = 0
    client_loss = dict()
    if args.cdp_defense == True:
        accountant = MomentAccountant(args.client_number)
    else:
        accountant = None

    edge = 0
    if args.poison == True:
        for pi in args.poison_client:
            client_index = str(pi)
            if args.dataset.startswith('L'):
                with open('./client4_malicious_edges_cluster_'+str(pi)+'.pkl', "rb") as file:
                    all_malicious_edges = pickle.load(file)
                    snapshot_number=80
                print(len(all_malicious_edges))
                print("all_malicious_edges")
            elif args.dataset.startswith('O'):
                with open('./optc_temp_malicious.pkl', "rb") as file:
                    all_malicious_edges = pickle.load(file)
                print(len(all_malicious_edges))
                snapshot_number = 1513
            for i in range(snapshot_number):
                if np.random.uniform() <= float(args.poison_p):
                    edge += len(all_malicious_edges)
                    malicious_edges = torch.tensor(all_malicious_edges).transpose(1,0).to(device)

                    client_models['model-client-'+client_index].gcns.module.data.eis[i] = torch.cat((client_models['model-client-'+client_index].gcns.module.data.eis[i], malicious_edges), dim=1)
                    temp_mask = torch.zeros_like(malicious_edges[0]).bool()
                    client_models['model-client-'+client_index].gcns.module.data.masks[i] = torch.cat((client_models['model-client-'+client_index].gcns.module.data.masks[i], temp_mask), dim=0)
                    original_tensor = torch.full((malicious_edges[0].shape), 0.0).to(device)
                    temp_weights = torch.rand_like(original_tensor)
                    client_models['model-client-'+client_index].gcns.module.data.ews[i] = torch.cat((client_models['model-client-'+client_index].gcns.module.data.ews[i], temp_weights), dim=0)
        print("edge per malicious: ", edge/len(all_malicious_edges))

    for t in range(kwargs['fedT']):
        
        for i in range(len(args.client_list)):
            model_name = 'model-client-'+str(args.client_list[i])
            client_models[model_name].train()
            if args.fed != 'FedOpt':
                st = time.time()
                zs = client_models[model_name].forward(TData.TRAIN)
                if zs == None:
                    client_models[model_name] = None
                else:
                    loss = client_models[model_name].loss_fn(zs, TData.TRAIN, nratio=kwargs['nratio'], device=device, encoder_name=args.encoder_name)
                    if args.fed == 'FedProx' and t > 0:
                        client_params = client_models[model_name].save_states()[1]
                        global_params = fed_model.save_states()[1]
                        proximal_term = 0.0
                        for key in client_params.keys():
                            proximal_term += torch.sum(torch.pow(client_params[key] - global_params[key], 2))
                        proximal_term = kwargs['fedMu'] / 2 * proximal_term
                        print('FedProx local loss %0.4f; proximal_term %f' % (loss, proximal_term))
                        loss += proximal_term
                        for param in client_models[model_name].parameters():
                            param.grad = None
                    loss.backward()
                    client_opts['opt-client-'+str(args.client_list[i])].step()
                    elapsed = time.time()-st
                    times.append(elapsed)
                print("calculate_model_memory_cost")
                calculate_model_memory_cost(client_models[model_name], TData.TRAIN)

            else:
                st = time.time()
                for e in range(args.fedopt_epoch):
                    zs = client_models[model_name].forward(TData.TRAIN)
                    loss = client_models[model_name].loss_fn(zs, TData.TRAIN, nratio=kwargs['nratio'], device=device, encoder_name=args.encoder_name)
                    loss.backward()
                    client_opts['opt-client-'+str(args.client_list[i])].step()
                elapsed = time.time()-st
                times.append(elapsed)
                calculate_model_memory_cost(client_models[model_name], TData.TRAIN)


            l = loss.sum()
            print('[%d] [%d] Loss %0.8f  %0.2fs' % (i, t, l.item(), elapsed))
            print("elapsed max/min time: ", max(times), min(times) )


        print('epoch: ', t)

        fed_model, client_models = fed_combine(client_models, args, fed_model, accountant, device, fed_device)
        zs = fed_model.forward(TData.TRAIN)
        try:
            with torch.no_grad():
                zs = fed_model.forward(TData.TRAIN, no_grad=True)
                p,n = fed_model.score_edges(zs, TData.VAL)
                auc,ap = get_score(p,n)
                print("\tValidation: AP: %0.4f  AUC: %0.4f" % (ap, auc), end='')
                tot = auc+ap
                if tot > fed_model_best[1]:
                    print('*\n')
                    best_name = 'best-client-'+str(args.client_list[i])
                    fed_model_best = (fed_model.save_states(), tot)
                    no_progress = 0
                else:
                    print('\n')
                    if t >= kwargs['min']:
                        no_progress += 1
                if no_progress == kwargs['patience']:
                    print("Early stopping!")
                    break
        except:
            break

    fed_model.load_states(*fed_model_best[0])
    zs, fed_h0 = fed_model(TData.TEST, include_h=True)
    states = {'states': fed_model_best[0], 'h0': fed_h0}
    os.makedirs('./Exps/FL/', exist_ok=True)
    f = open('./Exps/FL/Fed_'+dataset+'.pkl', 'wb+')
    pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)
    client_h0 = []
    if args.personalization:
        for i in range(len(args.client_list)):
            model_name = 'model-client-'+str(args.client_list[i])
            temp_model_best = (client_models[model_name].save_states(), 0)
            client_models[model_name].train()
            print('model-client-'+str(args.client_list[i]) + ' forward')
            for e in range(100):
                zs = client_models[model_name].forward(TData.TRAIN)
                loss = client_models[model_name].loss_fn(zs, TData.TRAIN, nratio=kwargs['nratio'], device=device, encoder_name=args.encoder_name)
                loss.backward()
                client_opts['opt-client-'+str(args.client_list[i])].step()
                l = loss.sum()
                print('[%d] [%d] Loss %0.4f  %0.2fs' % (i, e, l.item(), elapsed))

                with torch.no_grad():
                    zs = client_models[model_name].forward(TData.TRAIN, no_grad=True)
                    p,n = client_models[model_name].score_edges(zs, TData.VAL)
                    auc,ap = get_score(p,n)
                    print("\tValidation: AP: %0.4f  AUC: %0.4f" % (ap, auc), end='')
                    tot = auc+ap
                    if tot > temp_model_best[1]:
                        print('*\n')
                        best_name = 'best-client-'+str(args.client_list[i])
                        temp_model_best = (client_models[model_name].save_states(), tot)
                        no_progress = 0
                    else:
                        print('\n')
                        if t >= kwargs['min']:
                            no_progress += 1
                    if no_progress == kwargs['patience']:
                        print("Early stopping!")
                        break

            client_models[model_name].load_states(*temp_model_best[0])
            zs, h0 = client_models[model_name](TData.TEST, include_h=True)
            client_h0.append(h0)
        tpe = sum(times)/len(times)
        print("Exiting train loop")
        print("Avg TPE: %0.4fs" % tpe)
        return fed_model, fed_h0, tpe, client_models, client_h0

    tpe = sum(times)/len(times)
    print("Exiting train loop")
    print("Avg TPE: %0.4fs" % tpe)
    return fed_model, fed_h0, tpe, client_models, client_h0


def get_cutoff(model, h0, times, kwargs, lambda_param, use_flows, device, client, mc):
    Encoder = DetectorEncoder
    cutoff_args = {
            'start': times['val_start'],
            'end': times['val_end'],
            'delta': times['delta'],
            'is_test': False,
            'use_flows': use_flows,
            'client': client}
    print("cutoff: ", cutoff_args)
    cutoff_args['mc'] = mc

    Encoder.load_new_data(model.gcns, LOAD_FN, cutoff_args)
    # Then generate GCN embeds
    model.eval()

    zs = Encoder.forward(model.gcns.module, TData.ALL, True).to(device)
    h0 = h0.to(device)
    # Finally, generate actual embeds
    with torch.no_grad():
        zs, h0 = model.rnn(zs, h0, include_h=True)

    # Then score them
    p, n = Encoder.score_edges(model.gcns, zs, TData.ALL, kwargs['val_nratio'])
    # Finally, figure out the optimal cutoff score
    p = p.cpu()
    n = n.cpu()
    model.cutoff = get_optimal_cutoff(p,n,fw=lambda_param)
    return h0, zs[-1]

def test(model, h0, times, use_flows, OUTPATH, device, args, client, mc, POISON=False):
    Encoder = DetectorEncoder
    test_args = {'start': times['te_start'],
                'end': times['te_end'],
                'delta': times['delta'],
                'is_test': True,
                'use_flows': use_flows,
                'client': client,
                'mc': mc}
    print("Loading test data: ", client)
    edge_number = 0
    for i in model.gcns.module.data.eis:
        edge_number += len(i[1])
    print("edge number: ", edge_number)
    Encoder.load_new_data(model.gcns, LOAD_FN, test_args)
    edge_number = 0
    for i in model.gcns.module.data.eis:
        edge_number += len(i[1])
    print("edge number: ", edge_number)

    stats = []
    print("Embedding Test Data...")
    test_tmp = time.time()
    with torch.no_grad():
        model.eval()
        calculate_model_memory_cost(model, TData.TEST)

        s = time.time()
        zs = model.forward(TData.TEST, h0=h0, no_grad=True)
        ctime = time.time()-s

    # Scores all edges and matches them with name/timestamp
    print("Scoring", len(zs))
    scores, labels, weights = model.score_all(zs, OUTPATH)
    test_time = time.time() - test_tmp
    print("test_time: ", test_time)
    print("data: ", model.gcns.module.data.eis[0].shape) # it can access to the model's data

    print("model.cutoff: ", model.cutoff)
    all_edges = []
    for i in model.gcns.module.data.eis:
        all_edges.append(i.transpose(1,0).detach().cpu().numpy())
    all_edges = np.concatenate(all_edges, axis=0, dtype=int)

    stats.append(score_stats(args, scores, labels, weights, model.cutoff, ctime, all_edges))

    return stats

def score_stats(args, scores, labels, weights, cutoff, ctime, all_edges):
    # Cat scores from timesteps together bc separation
    # is no longer necessary
    # cutoff = 0.5012609 # LANL
    # cutoff = 0.5043199 # Optc
    # cutoff = 0.9766 # CDP
    # cutoff = 0.5115241
    # cutoff = 0.9943 # DP
    scores = np.concatenate(scores, axis=0)#detach().cpu()
    labels = np.concatenate(labels, axis=0).clip(max=1)
    weights = np.concatenate(weights, axis=0)

    combined_data = list(zip(scores, labels))
    sorted_data = sorted(combined_data, key=lambda x: x[0], reverse=True)

    # Classify using cutoff from earlier
    classified = np.zeros(labels.shape)
    classified[scores <= cutoff] = 1

    if args.poison == True:
        for pi in args.poison_client:
            malicious_result = []
            all_malicious_edges = []
            if args.dataset.startswith('L'):
                with open('./client4_malicious_edges_cluster_'+str(pi)+'.pkl', "rb") as file:
                    malicious_edges = pickle.load(file)
                    # all_malicious_edges = [np.array(t, dtype=int) for t in malicious_edges]
                    all_malicious_edges = np.array(malicious_edges)

            elif args.dataset.startswith('O'):
                with open('./optc_temp_malicious.pkl', "rb") as file:
                    malicious_edges = pickle.load(file)
                    all_malicious_edges = np.array(malicious_edges)

            for i in range(len(labels)):
                if labels[i] == 1:
                    if all_edges[i] in all_malicious_edges:
                        malicious_result.append(classified[i])
            print(pi, " success rate: ", 1- np.mean(malicious_result), len(malicious_result))

    # Calculate TPR
    p = classified[labels==1]
    tpr = p.mean()
    tp = p.sum()
    del p

    # Calculate FPR
    f = classified[labels==0]
    fp = f.sum()
    fpr = f.mean()
    del f

    cm = confusion_matrix(labels, classified, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    print("tn, fp, fn, tp: ", tn, fp, fn, tp)


    # Because a low score correlates to a 1 lable, sub from 1 to get
    # accurate AUC/AP scores
    scores = 1-scores


    # Get metrics
    try:
        auc = auc_score(labels, scores)
    except ValueError:
        auc = 0
        print('Only one class label in the dataset')

    try:
        ap = ap_score(labels, scores)
    except ValueError:
        ap = 0

    try:
        f1 = f1_score(labels, classified)
    except ValueError:
        f1 = 0

    timestr = time.strftime("%Y%m%d-%H%M%S")
    PATH = './Exps/result/scores/Real/'+str(args.encoder_name)+'_'+str(timestr)+'.npz'
    os.makedirs('./Exps/result/scores/', exist_ok=True)
    os.makedirs('./Exps/result/scores/Real/', exist_ok=True)
    np.savez(PATH, labels=labels, scores=scores)



    print("Learned Cutoff %0.4f" % cutoff)
    print("TPR: %0.4f, FPR: %0.4f" % (tpr, fpr))
    print("TP: %d  FP: %d" % (tp, fp))
    print("F1: %0.8f" % f1)
    print("AUC: %0.4f  AP: %0.4f\n" % (auc,ap))
    print("FwdTime", ctime, )
    title = "test"
    return {
        'Model': title,
        'TPR':tpr.item(),
        'FPR':fpr.item(),
        'TP':tp.item(),
        'FP':fp.item(),
        'F1':f1,
        'AUC':auc,

        'AP': ap,
        'FwdTime':ctime,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }

def run_all(args, rnn_args, worker_args, tr_args, OUTPATH, device, fed_device):
    # Need at least 2 deltas; default to 5% of tr data if that's enough
    if args.val_times is None:
        val = max((args.tr_end - args.tr_start) // 20, args.delta*2)
        args.val_start = args.tr_end-val
        args.val_end = args.tr_end
        args.tr_end = args.val_start
    else:
        args.val_start = args.val_times[0]
        args.val_end = args.val_times[1]

    times = {
        'tr_start': args.tr_start,
        'tr_end': args.tr_end,
        'val_start': args.val_start,
        'val_end': args.val_end,
        'te_times': args.te_times,
        'delta': args.delta
    }

    print(times)
    if args.fed == 'None':
        init_single(args, rnn_args, worker_args, times, tr_args, OUTPATH, device)
    else:
        init_fed(args, rnn_args, worker_args, times, tr_args, OUTPATH, device, fed_device)

    # Retrieve stats, and cleanup temp file
    stats = pickle.load(open(OUTPATH+TMP_FILE, 'rb'))

    print(stats)
    return stats
