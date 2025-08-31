import math
import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score as auc_score, f1_score, average_precision_score as ap_score, precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

import torch
from torch.optim import Adam, Adadelta, AdamW
from loaders.tdata import TData
from loaders.load_optc import load_optc_dist
from models.euler_detector import DetectorEncoder, DetectorRecurrent
from utils import get_score, get_optimal_cutoff

TMP_FILE = 'tmp.dat'
SCORE_FILE = 'scores.txt'
LOAD_FN = None

def init(args, rnn_args, worker_args, times, tr_args, OUTPATH, device):
    global LOAD_FN
    LOAD_FN = args.loader

    # Evaluating a pre-trained model, so no need to train
    if args.load:
        kwargs = {
            'start': None,
            'end': None,
            'use_flows': args.flows,
            'device': device
        }
        with open('./Exps/'+args.encoder+'_save_'+args.dataset+'.pkl', 'rb') as f:
            model = pickle.load(f)
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
                'device': device}
        # rrefs = args.encoder(LOAD_FN, kwargs, *worker_args)
        train_dataset = args.loader(kwargs['start'], kwargs['end'], kwargs['end']-kwargs['start'], use_flows=kwargs['use_flows'])
        snapshot_length = len(train_dataset.cnt)
        print(train_dataset)
        if args.dataset.startswith('O'):
            train_x = train_dataset.eas[0].transpose(1,0).numpy() #eas flow
            print(train_x.shape)

        elif args.dataset.startswith('L'):
            train_x = torch.concat((train_dataset.ews[0].unsqueeze(0), train_dataset.eas[0]), 0).transpose(1,0).numpy() #eas flow
            print(train_x.shape, train_dataset.ews[0].shape, train_dataset.eas[0].shape)
            exit()
        else:
            print("wrong dataset")
            exit()
        if args.encoder_name == 'LOF':
            model = LocalOutlierFactor(n_neighbors=2, contamination=0.05)#, novelty=True)
        elif args.encoder_name == 'IF':
            model = IsolationForest(random_state=0)
        tmp = time.time()
        print("train_x", train_x.shape)
        model.fit(train_x)

        tr_time = time.time() - tmp
    stats = []

    # for te_start,te_end in times['te_times']:
    test_times = {
        'te_start': times['te_times'][0][0],
        'te_end': times['te_times'][0][1],
        'delta': times['delta']
    }
    print(test_times)
    test_tmp = time.time()
    test_dataset = args.loader(test_times['te_start'], test_times['te_end'], test_times['delta'], use_flows=kwargs['use_flows'], is_test=True)
    print(test_dataset)
    exit()
    print(test_dataset)

    snapshot_length = len(test_dataset.cnt)
    test_preds = []
    labels = []
    classifieds = []
    stats = []
    temp = []


    for i in range(snapshot_length):
        if args.dataset.startswith('O'):
            test_x = test_dataset.eas[i].transpose(1,0).numpy() #eas flow
        elif args.dataset.startswith('L'):
            test_x = torch.concat((test_dataset.ews[i].unsqueeze(0), test_dataset.eas[i]), 0).transpose(1,0).numpy() #eas flow
        else:
            print("wrong dataset")
            exit()
        if (test_x.shape[0] < 10):# and (args.encoder == 'LOF'):
            continue
        if test_x.shape[0] != test_dataset.ys[i].shape[0]:
            print(i, test_x.shape, test_dataset.ys[i].shape)
        temp.append(test_x)
        classified = model.fit_predict(test_x)
        classified[classified == 1] = 0
        classified[classified == -1] = 1
        if args.encoder_name == 'LOF':
            test_pred = - model.negative_outlier_factor_
        elif args.encoder_name == 'IF':
            test_pred = - model.decision_function(test_x)

        test_preds.append(test_pred)
        classifieds.append(classified)
        labels.append(test_dataset.ys[i])

    temp = np.concatenate(temp, axis=0)
    print(temp.shape)
    stats.append(score_stats(args, test_preds, labels, classifieds))

    test_time = time.time() - test_tmp
    print("test_time: ", test_time)

    pickle.dump(stats, open(OUTPATH+TMP_FILE, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)
    return

def score_stats(args, scores, labels, classified):
    scores = np.concatenate(scores, axis=0)#detach().cpu()
    labels = np.concatenate(labels, axis=0).clip(max=1)
    classified = np.concatenate(classified, axis=0)
    print(scores.shape, labels.shape,classified.shape)

    #try to get p, r, thresholds from the testing data, though NNC has higher AP,
    #it can't make any alarm because the learnt cutoff is too low, while all events in testing have higher scores
    #p, r, th = precision_recall_curve(labels, scores)
    #with open('./prc.pkl', 'wb') as f:
        #pickle.dump([p, r, th], f)

    # Classify using cutoff from earlier
    # classified = np.zeros(labels.shape)
    # classified[scores <= cutoff] = 1

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

    from sklearn.metrics import confusion_matrix, precision_score, roc_auc_score, f1_score
    cm = confusion_matrix(labels, classified, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    print("tn, fp, fn, tp: ", tn, fp, fn, tp)


    # Because a low score correlates to a 1 lable, sub from 1 to get
    # accurate AUC/AP scores
    scores = 1-scores

    # scores = scores.detach().cpu()
    # labels = labels.detach().cpu()
    # weights = weights.detach().cpu()

    # Get metrics
    auc = auc_score(labels, scores)
    ap = ap_score(labels, scores)
    f1 = f1_score(labels, classified)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    PATH = './Exps/result/scores/Real/'+str(args.encoder_name)+'_'+str(timestr)+'.npz'
    np.savez(PATH, labels=labels, scores=scores)


    print("TPR: %0.4f, FPR: %0.4f" % (tpr, fpr))
    print("TP: %d  FP: %d" % (tp, fp))
    print("F1: %0.8f" % f1)
    print("AUC: %0.4f  AP: %0.4f\n" % (auc,ap))

    return {
        'TPR':tpr.item(),
        'FPR':fpr.item(),
        'TP':tp.item(),
        'FP':fp.item(),
        'F1':f1,
        'AUC':auc,
        'AP': ap,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }

def simple_run_all(args, rnn_args, worker_args, tr_args, OUTPATH, device):
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

    init(args, rnn_args, worker_args, times, tr_args, OUTPATH, device)

    stats = pickle.load(open(OUTPATH+TMP_FILE, 'rb'))

    print(stats)
    return stats
