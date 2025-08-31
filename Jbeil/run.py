import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics, get_data_fl
from utils.fl_utils import *
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, recall_score, precision_score, roc_curve


torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('Jbeil self-supervised training')
parser.add_argument('--induct', type=float, default=0.3)
parser.add_argument('--n', type=int, default=1209600) # 1000000 # 5011199 # n is timestamp instead of edge number. # 14 days
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. auth or pivoting)',
                    default='auth')
parser.add_argument('--bs', type=int, default=2000, help='Batch_size')
parser.add_argument('--prefix', type=str, default='jbeil', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=5, help='Number of neighbors to sample') # 10
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=5, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=10, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
# FL
parser.add_argument('--fed', default='Momentum', choices=['None', 'WeightedFedAvg', 'WeightedFedAvg2', 'FedAvg', 'FedOpt', 'FedProx', 'Momentum'])
parser.add_argument('--fedopt_epoch', default=5)
parser.add_argument('--fedMu', default=0.005)
parser.add_argument('--client_number', default=4)

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
args.client_number = int(args.client_number)
CLIENT_NUMBER = args.client_number
induct = args.induct
n = args.n
AUX_FOLDER = './data/'

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
Path("log/").mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename='log/{}.log'.format(time.strftime("%Y%m%d-%H%M%S")),
                    filemode='a',
                    format='%(message)s',
                    level=logging.DEBUG)

# logging.info("Running Urban Planning")
logger = logging.getLogger('urbanGUI')
logger.info(args)

### Extract data for training, validation and testing

fl_node_features, fl_edge_features, fl_full_data, fl_train_data, fl_val_data, fl_test_data, fl_new_node_val_data, fl_new_node_test_data = [], [], [], [], [], [], [], []
fl_train_ngh_finder, fl_full_ngh_finder, fl_train_rand_sampler, fl_val_rand_sampler, fl_test_rand_sampler, fl_nn_test_rand_sampler = [], [], [], [], [], []
fl_mean_time_shift_src, fl_std_time_shift_src, fl_mean_time_shift_dst, fl_std_time_shift_dst = [], [], [], []
# client data
for i in range(CLIENT_NUMBER):
  node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data_fl(DATA, induct, n,different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features, logger = logger, client_index=i, client_number=CLIENT_NUMBER)
  # Initialize training neighbor finder to retrieve temporal graph
  train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
  full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

  # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
  # across different runs
  # NB: in the inductive setting, negatives are sampled only amongst other new nodes
  train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
  val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
  # nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
  test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
  nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

  # Set device
  device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
  logger.info("$$$ ==> {}".format(device_string))
  device = torch.device(device_string)
  logger.info("### ==> {}".format(device))

  # Compute time statistics
  mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

  fl_node_features.append(node_features)
  fl_edge_features.append(edge_features)
  fl_full_data.append(full_data)
  fl_train_data.append(train_data)
  fl_val_data.append(val_data)
  fl_test_data.append(test_data)
  fl_new_node_val_data.append(new_node_val_data)
  fl_new_node_test_data.append(new_node_test_data)
  fl_train_ngh_finder.append(train_ngh_finder)
  fl_full_ngh_finder.append(full_ngh_finder)
  fl_train_rand_sampler.append(train_rand_sampler)
  fl_val_rand_sampler.append(val_rand_sampler)
  fl_test_rand_sampler.append(test_rand_sampler)
  fl_nn_test_rand_sampler.append(nn_test_rand_sampler)
  fl_mean_time_shift_src.append(mean_time_shift_src)
  fl_std_time_shift_src.append(std_time_shift_src)
  fl_mean_time_shift_dst.append(mean_time_shift_dst)
  fl_std_time_shift_dst.append(std_time_shift_dst)


client_model = dict()
client_opts = dict()
client_bests = dict()
client_early_stopper = dict()

fl_num_batch, fl_num_instance = [], []
cluster_fname = 'lanl_' + str(args.client_number) + '_' + str(args.client_number) +'.json'
args.mc = load_machine_clusters(AUX_FOLDER + cluster_fname)
if (args.fed == 'WeightedFedAvg2') or (args.fed == 'Momentum'):
    global_G = build_random_graph(args.mc)

for i in range(CLIENT_NUMBER):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)
  # Initialize Model
  print("edge feature shape: ", fl_edge_features[i].shape)
  tgn = TGN(neighbor_finder=fl_train_ngh_finder[i], node_features=fl_node_features[i],
            edge_features=fl_edge_features[i], device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=fl_mean_time_shift_src[i], std_time_shift_src=fl_std_time_shift_src[i],
            mean_time_shift_dst=fl_mean_time_shift_dst[i], std_time_shift_dst=fl_std_time_shift_dst[i],
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep)
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  client_opts.update({i: optimizer})
  tgn = tgn.to(device)
  client_model.update({i: tgn})

  num_instance = len(fl_train_data[i].sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  print('num of training instances: {}'.format(num_instance))
  print('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  client_early_stopper.update({i: early_stopper})
  fl_num_batch.append(num_batch)
  fl_num_instance.append(num_instance)
  if args.fed == 'WeightedFedAvg2':
    similarities = compute_graph_weights(fl_train_data[i], global_G)
    if not hasattr(args, 'similarity'):
        args.similarity = []
    args.similarity.append(similarities[2])
  if args.fed == 'Momentum':
      similarities = compute_graph_weights(fl_train_data[i], global_G)
      if not hasattr(args, 'similarity'):
          args.similarity = []
      args.similarity.append(similarities[2])
# fed dataset and model
# whole data
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(DATA, induct, n,different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features, logger = logger)
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

full_ngh_finder = get_neighbor_finder(full_data, args.uniform)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

fed_model = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep)
fed_model = fed_model.to(device)

# start to train
new_nodes_val_aps = [[]*CLIENT_NUMBER]
val_aps = []
epoch_times = []
total_epoch_times = []
train_losses = []

for epoch in range(NUM_EPOCH):
  start_epoch = time.time()
  for i in range(CLIENT_NUMBER):
    # Train using only training graph
    if args.fed != 'FedOpt':
      client_model[i].set_neighbor_finder(fl_train_ngh_finder[i])
      m_loss = []
      for k in range(0, fl_num_batch[i], args.backprop_every):
        loss = 0
        client_opts[i].zero_grad()

        # Custom loop to allow to perform backpropagation only every a certain number of batches
        for j in range(args.backprop_every):
          batch_idx = k + j

          if batch_idx >= fl_num_batch[i]:
            continue

          start_idx = batch_idx * BATCH_SIZE
          end_idx = min(fl_num_instance[i], start_idx + BATCH_SIZE)
          sources_batch, destinations_batch = fl_train_data[i].sources[start_idx:end_idx], fl_train_data[i].destinations[start_idx:end_idx]
          edge_idxs_batch = fl_train_data[i].edge_idxs[start_idx: end_idx]
          timestamps_batch = fl_train_data[i].timestamps[start_idx:end_idx]

          size = len(sources_batch)
          _, negatives_batch = fl_train_rand_sampler[i].sample(size)

          with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

          client_model[i] = client_model[i].train()
          pos_prob, neg_prob = client_model[i].compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
          loss += criterion(pos_prob.squeeze(1), pos_label) + criterion(neg_prob.squeeze(1), neg_label)
        loss /= args.backprop_every

        if args.fed == 'FedProx':
          proximal_term = 0.0
          for param, global_param in zip(client_model[i].parameters(), fed_model.parameters()):
              proximal_term += ((args.fedMu / 2) * torch.norm((param - global_param.data))**2)

          loss += proximal_term
        loss.backward()
        client_opts[i].step()
        m_loss.append(loss.item())
    else:
        for e in range(args.fedopt_epoch):
          client_model[i].set_neighbor_finder(fl_train_ngh_finder[i])
          m_loss = []
          for k in range(0, fl_num_batch[i], args.backprop_every):
            if (k%10000 == 0):
                logger.info('Batch number: {}/{}'.format(k, fl_num_batch[i]))
            loss = 0
            client_opts[i].zero_grad()

            # Custom loop to allow to perform backpropagation only every a certain number of batches
            for j in range(args.backprop_every):
              batch_idx = k + j

              if batch_idx >= fl_num_batch[i]:
                continue

              start_idx = batch_idx * BATCH_SIZE
              end_idx = min(fl_num_instance[i], start_idx + BATCH_SIZE)
              sources_batch, destinations_batch = fl_train_data[i].sources[start_idx:end_idx], \
                                                  fl_train_data[i].destinations[start_idx:end_idx]
              edge_idxs_batch = fl_train_data[i].edge_idxs[start_idx: end_idx]
              timestamps_batch = fl_train_data[i].timestamps[start_idx:end_idx]

              size = len(sources_batch)
              _, negatives_batch = fl_train_rand_sampler[i].sample(size)

              with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)

              client_model[i] = client_model[i].train()
              pos_prob, neg_prob = client_model[i].compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

              loss += criterion(pos_prob.squeeze(1), pos_label) + criterion(neg_prob.squeeze(1), neg_label)

            loss /= args.backprop_every

            loss.backward()
            client_opts[i].step()
            m_loss.append(loss.item())

    print("client {} loss: {}".format(i, np.mean(m_loss)))
    logger.info("client {} loss: {}".format(i, np.mean(m_loss)))

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    # Validation
    if False:
        client_model[i].set_neighbor_finder(fl_full_ngh_finder[i])
        val_ap, val_auc, val_recall, val_precision, val_fp, val_fn, val_tp, val_tn, _, _ = eval_edge_prediction(model=client_model[i], negative_edge_sampler=fl_val_rand_sampler[i], data=fl_val_data[i], n_neighbors=NUM_NEIGHBORS)
        # Validate on unseen nodes
        nn_val_ap, nn_val_auc, nn_val_recall, nn_val_precision, nn_val_fp, nn_val_fn, nn_val_tp, nn_val_tn, _, _ = eval_edge_prediction(model=client_model[i], negative_edge_sampler=fl_val_rand_sampler[i], data=fl_new_node_val_data[i], n_neighbors=NUM_NEIGHBORS)

        new_nodes_val_aps.append(nn_val_ap)
        val_aps.append(val_ap)
        train_losses.append(np.mean(m_loss))

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('Client: {} epoch: {} took {:.2f}s'.format(i, epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info(
        'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
        logger.info(
        'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
        logger.info(
        'val recall: {}, new node val recall: {}'.format(val_recall, nn_val_recall))
        logger.info(
        'val precision: {}, new node val precision: {}'.format(val_precision, nn_val_precision))

        print('Client: {} epoch: {} loss: {}'.format(i, epoch, np.mean(m_loss)))
        print('val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
        print('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

        logger.info(
        'val fp: {}, new node val fp: {}'.format(val_fp, nn_val_fp))
        logger.info(
        'val fn: {}, new node val fn: {}'.format(val_fn, nn_val_fn))
        logger.info(
        'val tp: {}, new node val tp: {}'.format(val_tp, nn_val_tp))
        logger.info(
        'val tn: {}, new node val tn: {}'.format(val_tn, nn_val_tn))
  fed_model, client_models = fed_combine(client_model, args, fed_model)
  fed_model.embedding_module.neighbor_finder = full_ngh_finder
  val_ap, val_auc, val_recall, val_precision, val_fp, val_fn, val_tp, val_tn, _, _ = eval_edge_prediction(model=fed_model, negative_edge_sampler=test_rand_sampler, data=test_data, n_neighbors=NUM_NEIGHBORS)
  nn_val_ap, nn_val_auc, nn_val_recall, nn_val_precision, nn_val_fp, nn_val_fn, nn_val_tp, nn_val_tn, _, _ = eval_edge_prediction(model=fed_model, negative_edge_sampler=test_rand_sampler, data=new_node_test_data, n_neighbors=NUM_NEIGHBORS)
  print('Fed model on full data epoch: {}'.format(epoch))
  print('val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
  print('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
  logger.info('Fed moel on full data epoch: {}'.format(epoch))
  logger.info('val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
  logger.info('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))



### Test local models
logger.info('Started Testing')
if False:
  client_model[i].embedding_module.neighbor_finder = fl_full_ngh_finder[i]
  test_ap, test_auc, test_recall, test_precision, test_fp, test_fn, test_tp, test_tn, _, _ = eval_edge_prediction(model=client_model[i], negative_edge_sampler=fl_test_rand_sampler[i], data=fl_test_data[i], n_neighbors=NUM_NEIGHBORS)

  # Test on unseen nodes
  nn_test_ap, nn_test_auc, nn_test_recall, nn_test_precision, nn_test_fp, nn_test_fn, nn_test_tp, nn_test_tn = eval_edge_prediction(model=client_model[i], negative_edge_sampler=fl_nn_test_rand_sampler[i], data=fl_new_node_test_data[i], n_neighbors=NUM_NEIGHBORS)

  logger.info(
    'local model {} Old nodes -- auc: {}, ap: {}, recall: {}, precision: {}, fp: {}, fn: {}, tp: {}, tn: {}'.format(i, test_auc, test_ap, test_recall, test_precision, test_fp, test_fn, test_tp, test_tn))
  logger.info(
    'local model {} New nodes -- auc: {}, ap: {}, recall: {}, precision: {}, fp: {}, fn: {}, tp: {}, tn: {}'.format(i, nn_test_auc, nn_test_ap, nn_test_recall, nn_test_precision, nn_test_fp, nn_test_fn, nn_test_tp, nn_test_tn))


  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "new_node_test_ap": nn_test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

### Test fed models
logger.info('Started Testing Fed Model on full dataset.')
fed_model.embedding_module.neighbor_finder = full_ngh_finder
test_ap, test_auc, test_recall, test_precision, test_fp, test_fn, test_tp, test_tn, _, _ = eval_edge_prediction(model=fed_model, negative_edge_sampler=test_rand_sampler, data=test_data, n_neighbors=NUM_NEIGHBORS)

# Test on unseen nodes
nn_test_ap, nn_test_auc, nn_test_recall, nn_test_precision, nn_test_fp, nn_test_fn, nn_test_tp, nn_test_tn, _, _ = eval_edge_prediction(model=fed_model, negative_edge_sampler=nn_test_rand_sampler, data=new_node_test_data, n_neighbors=NUM_NEIGHBORS)

logger.info(
  'Fed Test on full statistics: Old nodes -- auc: {}, ap: {}, recall: {}, precision: {}, tn: {}, fp: {}, fn: {}, tp: {}'.format(test_auc, test_ap, test_recall, test_precision, test_tn, test_fp, test_fn, test_tp))
logger.info(
  'Fed Test on full statistics: New nodes -- auc: {}, ap: {}, recall: {}, precision: {}, tn: {}, fp: {}, fn: {}, tp: {}'.format(nn_test_auc, nn_test_ap, nn_test_recall, nn_test_precision, nn_test_tn, nn_test_fp, nn_test_fn, nn_test_tp, nn_test_tn))

print('Fed Test on full: Old nodes -- ap: {:.4f}, auc: {:.4f}, tn: {}, fp: {}, fn: {}, tp: {}'.format(test_ap, test_auc, test_tn, test_fp, test_fn, test_tp))

print('Fed Test on full: New nodes -- ap: {:.4f}, auc: {:.4f}, tn: {}, fp: {}, fn: {}, tp: {}'.format(nn_test_ap, nn_test_auc, nn_test_tn, nn_test_fp, nn_test_fn, nn_test_tp, nn_test_tn))

exit()
all_seen_true_labels = []
all_seen_pred_scores = []
all_unseen_true_labels = []
all_unseen_pred_scores = []
for i in range(CLIENT_NUMBER):
  fed_model.embedding_module.neighbor_finder = fl_full_ngh_finder[i]
  test_ap, test_auc, test_recall, test_precision, test_fp, test_fn, test_tp, test_tn, seen_true_labels, seen_pred_scores = eval_edge_prediction(model=fed_model, negative_edge_sampler=fl_test_rand_sampler[i], data=fl_test_data[i], n_neighbors=NUM_NEIGHBORS)
  all_seen_true_labels.append(seen_true_labels)
  all_seen_pred_scores.append(seen_pred_scores)

  # Test on unseen nodes
  nn_test_ap, nn_test_auc, nn_test_recall, nn_test_precision, nn_test_fp, nn_test_fn, nn_test_tp, nn_test_tn, unseen_true_labels, unseen_pred_scores = eval_edge_prediction(model=fed_model, negative_edge_sampler=fl_nn_test_rand_sampler[i], data=fl_new_node_test_data[i], n_neighbors=NUM_NEIGHBORS)
  all_unseen_true_labels.append(unseen_true_labels)
  all_unseen_pred_scores.append(unseen_pred_scores)

  logger.info('Fed Test on local client {} statistics: Old nodes -- auc: {}, ap: {}, recall: {}, precision: {}, fp: {}, fn: {}, tp: {}, tn: {}'.format(i, test_auc, test_ap, test_recall, test_precision, test_fp, test_fn, test_tp, test_tn))
  logger.info('Fed Test on local client {} statistics: New nodes -- auc: {}, ap: {}, recall: {}, precision: {}, fp: {}, fn: {}, tp: {}, tn: {}'.format(i, nn_test_auc, nn_test_ap, nn_test_recall, nn_test_precision, nn_test_fp, nn_test_fn, nn_test_tp, nn_test_tn))
  print('Fed Test on local client {}: Old nodes -- ap: {}, auc: {}, tn: {}, fp: {}, fn: {}, tp: {}'.format(i, test_ap, test_auc, test_tn, test_fp, test_fn, test_tp))
  print('Fed Test on local client {}: New nodes -- ap: {}, auc: {}, tn: {}, fp: {}, fn: {}, tp: {}'.format(i, nn_test_ap, nn_test_auc, nn_test_tn, nn_test_fp, nn_test_fn, nn_test_tp, nn_test_tn))

true_labels = np.concatenate(all_seen_true_labels)
pred_scores = np.concatenate(all_seen_pred_scores)
ap = average_precision_score(true_labels, pred_scores)
auc = roc_auc_score(true_labels, pred_scores)
tn, fp, fn, tp = confusion_matrix(true_labels, pred_scores).ravel()
print('seen ap: {}, auc: {}, tn: {}, fp: {}, fn: {}, tp: {}'.format(ap, auc, tn, fp, fn, tp))
logger.info('seen ap: {}, auc: {}, tn: {}, fp: {}, fn: {}, tp: {}'.format(ap, auc, tn, fp, fn, tp))

true_labels = np.concatenate(all_unseen_true_labels)
pred_scores = np.concatenate(all_unseen_pred_scores)
ap = average_precision_score(true_labels, pred_scores)
auc = roc_auc_score(true_labels, pred_scores)
tn, fp, fn, tp = confusion_matrix(true_labels, pred_scores).ravel()
logger.info('unseen ap: {}, auc: {}, tn: {}, fp: {}, fn: {}, tp: {}'.format(ap, auc, tn, fp, fn, tp))
print('unseen ap: {}, auc: {}, tn: {}, fp: {}, fn: {}, tp: {}'.format(ap, auc, tn, fp, fn, tp))

torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
logger.info('Jbeil model saved')
