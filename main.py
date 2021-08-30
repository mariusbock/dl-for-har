import argparse
import os
import time
import sys
import numpy as np
import pickle

from data_processing.preprocess_data import load_dataset, compute_mean_and_std
from misc.osutils import mkdir_if_missing
from model.cross_validation import cross_participant_cv, per_participant_cv, normal_cv

from misc.logging import Logger
from misc.torchutils import seed_torch


"""
DATASET OPTIONS:
- DATASET:
    - opportunity: full Opportunity dataset; preprocessing as described in Ordonez et al.
    - opportunity_ordonez: Opportunity dataset as described and preprocessed in Ordonez et al.
    - wetlab: Wetlab dataset
    - rwhar: RealWorld HAR dataset
    - sbhar: SBHAR dataset
    - hhar: HHAR dataset
- PRED_TYPE:
    - Opportunity: 'gestures' or 'locomotion'
    - Wetlab: 'actions' or 'tasks'
- CUTOFF_TRAIN: subject where the train dataset is cut off, i.e. last subject within train dataset
- CUTOFF_VALID: subject where the validation dataset is cut off, i.e. last subject within validation dataset
- SW_LENGTH: length of sliding window
- SW_UNIT: unit in which length of sliding window is measured
- SW_OVERLAP: overlap ratio between sliding windows (in percent, i.e. 60 = 60%)
- MEANS_AND_STDS: boolean whether to append means and standard deviations of feature columns to dataset
- INCLUDE_NULL: boolean whether to include null class in datasets
"""

DATASET = 'opportunity_ordonez'
PRED_TYPE = 'gestures'
CUTOFF_TRAIN = 20
CUTOFF_VALID = 30
SW_LENGTH = 24
SW_UNIT = 'units'
SW_OVERLAP = 50
MEANS_AND_STDS = False
INCLUDE_NULL = True

"""
NETWORK OPTIONS:
- NETWORK: network architecture to be used (e.g. 'deepconvlstm')
- NB_UNITS_LSTM: number of hidden units in each LSTM layer
- NB_LAYERS_LSTM: number of layers in LSTM
- CONV_BLOCK_TYPE: type of convolution blocks employed ('normal', 'skip' or 'fixup')
- NB_CONV_BLOCKS: number of convolution blocks employed
- NB_FILTERS: number of convolution filters employed in each layer of convolution blocks
- FILTER_WIDTH: width of convolution filters (e.g. 11 = 11x1 filter)
- DILATION: dilation factor employed on convolutions (set 1 for not dilation)
- DROP_PROB: dropout probability in dropout layers
- POOLING: boolean whether to employ a pooling layer after convolution layers
- BATCH_NORM: boolean whether to apply batch normalisation in convolution blocks
- REDUCE_LAYER: boolean whether to employ a reduce layer after convolution layers
- POOL_TYPE: type of pooling employed in pooling layer
- POOL_KERNEL_WIDTH: width of pooling kernel (e.g. 2 = 2x1 pooling kernel)
- REDUCE_LAYER_OUTPUT: size of the output after the reduce layer (i.e. what reduction is to be applied) 
"""

NETWORK = 'deepconvlstm'
NB_UNITS_LSTM = 128
NB_LAYERS_LSTM = 2
CONV_BLOCK_TYPE = 'normal'
NB_CONV_BLOCKS = 2
NB_FILTERS = 64
FILTER_WIDTH = 5
DILATION = 1
DROP_PROB = 0.5
POOLING = False
BATCH_NORM = False
REDUCE_LAYER = False
POOL_TYPE = 'max'
POOL_KERNEL_WIDTH = 2
REDUCE_LAYER_OUTPUT = 8

"""
TRAINING OPTIONS:
- SEED: 
- VALID_TYPE:
- BATCH_SIZE:
- EPOCHS:
- OPTIMIZER: 
- LR: 
- WEIGHT_DECAY: 
- WEIGHTS_INIT:
- LOSS: 
- GPU: 
- SPLITS_SSS: 
- USE_WEIGHTS:
- ADJ_LR: 
- EARLY_STOPPING: 
- ADJ_LR_PATIENCE: 
- ES_PATIENCE: 
"""

SEED = 1
VALID_TYPE = 'cross-participant'
BATCH_SIZE = 100
EPOCHS = 1
OPTIMIZER = 'adam'
LR = 1e-4
WEIGHT_DECAY = 1e-6
WEIGHTS_INIT = 'xavier_normal'
LOSS = 'cross-entropy'
GPU = 'cuda:0'
SPLITS_SSS = 2
SIZE_SSS = 0.6
USE_WEIGHTS = True
ADJ_LR = False
EARLY_STOPPING = False
ADJ_LR_PATIENCE = 2
ES_PATIENCE = 5

"""
LOGGING OPTIONS:
- LOGGING: boolean whether to log console outputs in a text file
- PRINT_COUNTS: boolean whether to print the distribution of predicted labels after each epoch 
- PLOT_GRADIENT: boolean whether to plot the gradient flow throughout epochs
- VERBOSE: boolean whether to print batchwise results during epochs
- PRINT_FREQ: number of batches after which batchwise results are printed
- SAVE_TEST_PREDICTIONS: boolean whether to save test predictions
- SAVE_MODEL: boolean whether to save the model after last epoch as pickle file
"""

LOGGING = False
PRINT_COUNTS = False
PLOT_GRADIENT = False
VERBOSE = False
PRINT_FREQ = 100
SAVE_TEST_PREDICTIONS = True
SAVE_MODEL = True


def main(args):
    # parameters used to calculate runtime
    start = time.time()
    log_date = time.strftime('%Y%m%d')
    log_timestamp = time.strftime('%H%M%S')

    # saves logs to a file (standard output redirected)
    if args.logging:
        sys.stdout = Logger(os.path.join('logs', log_date, log_timestamp, 'log.txt'))

    print('Applied settings: ')
    print(args)

    ################################################## DATA LOADING ####################################################

    print('Loading data...')
    X_train, y_train, X_val, y_val, X_test, y_test, nb_classes, class_names, sampling_rate, has_null = \
        load_dataset(dataset=args.dataset,
                     cutoff_train=args.cutoff_train,
                     cutoff_valid=args.cutoff_valid,
                     pred_type=args.pred_type,
                     include_null=args.include_null
                     )
    args.sampling_rate = sampling_rate
    args.nb_classes = nb_classes
    args.class_names = class_names
    args.has_null = has_null

    if args.dataset == 'opportunity_ordonez':
        args.valid_type = 'normal'

    if args.means_and_stds:
        X_train = np.concatenate((X_train, compute_mean_and_std(X_train[:, 1:])), axis=1)
        X_val = np.concatenate((X_val, compute_mean_and_std(X_val[:, 1:])), axis=1)
        X_test = np.concatenate((X_test, compute_mean_and_std(X_test[:, 1:])), axis=1)

    ############################################# TRAINING #############################################################

    seed_torch(args.seed)
    if X_test.size != 0:
        X = np.concatenate((X_train, X_val, X_test), axis=0)
        y = np.concatenate((y_train, y_val, y_test), axis=0)
    else:
        X = np.concatenate((X_train, X_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)
    data = np.concatenate((X, (np.array(y)[:, None])), axis=1)

    if args.valid_type == 'cross-participant':
        trained_net = cross_participant_cv(data, args, log_date, log_timestamp)
    elif args.valid_type == 'per-participant':
        trained_net = per_participant_cv(data, args, log_date, log_timestamp)
    elif args.valid_type == 'normal':
        trained_net = normal_cv(X_train, y_train, X_val, y_val, X_test, y_test, args, log_date, log_timestamp)

    # TODO: implement k-fold

    if args.save_model:
        mkdir_if_missing(os.path.join('logs', log_date, log_timestamp))
        pickle.dump(trained_net, open(os.path.join('logs', log_date, log_timestamp, 'trained_model.sav'), 'wb'))

    # calculate time data creation took
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nFinal time elapsed: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DATASET, type=str,
                        help='dataset to be used (rwhar, sbhar, wetlab or hhar)')
    parser.add_argument('--pred_type', default=PRED_TYPE, type=str,
                        help='prediction type for wetlab dataset (actions or tasks)')
    parser.add_argument('--save_test_preds', default=SAVE_TEST_PREDICTIONS, type=bool, help='save predictions in separate file')
    parser.add_argument('--logging', default=LOGGING, type=bool, help='log terminal output into text file')
    parser.add_argument('--verbose', default=VERBOSE, type=bool, help='verbose training output (batchwise)')
    parser.add_argument('--print_freq', default=PRINT_FREQ, type=int,
                        help='if verbose, frequency of which is printed (batches)')
    parser.add_argument('--print_counts', default=PRINT_COUNTS, type=bool,
                        help='print class distribution of train and validation set after epochs')
    parser.add_argument('--plot_gradient', default=PLOT_GRADIENT, type=bool, help='plot gradient development as plot')
    parser.add_argument('--include_null', default=INCLUDE_NULL, type=bool,
                        help='include null class (if dataset has one) in training/ prediction')
    parser.add_argument('--cutoff_train', default=CUTOFF_TRAIN, type=int,
                        help='cutoff point (subject-wise) for train dataset')
    parser.add_argument('--cutoff_valid', default=CUTOFF_VALID, type=int,
                        help='cutoff point (subject-wise) for validation dataset')
    parser.add_argument('--splits_sss', default=SPLITS_SSS, type=int, help='no. splits for per-participant eval')
    parser.add_argument('--size_sss', default=SIZE_SSS, type=float,
                        help='size of validation set in per-participant eval')
    parser.add_argument('--network', default=NETWORK, type=str, help='network to be used (e.g. deepconvlstm)')
    parser.add_argument('--valid_type', default=VALID_TYPE, type=str,
                        help='validation type to be used (per-participant, cross-participant, k-fold)')
    parser.add_argument('--seed', default=SEED, type=int, help='seed to be employed')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='no. epochs to use during training')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size to use during training')
    parser.add_argument('--means_and_stds', default=MEANS_AND_STDS, type=bool,
                        help='append means and stds of columns to dataset')
    parser.add_argument('--sw_length', default=SW_LENGTH, type=float, help='length of sliding window')
    parser.add_argument('--sw_unit', default=SW_UNIT, type=str, help='sliding window unit used (units, seconds)')
    parser.add_argument('--sw_overlap', default=SW_OVERLAP, type=float, help='overlap employed between sliding windows')
    parser.add_argument('--weights_init', default=WEIGHTS_INIT, type=str,
                        help='weight initialization method used (normal, orthogonal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal)')
    parser.add_argument('--batch_norm', default=BATCH_NORM, type=bool,
                        help='use batch normalisation after each convolution')
    parser.add_argument('--reduce_layer', default=REDUCE_LAYER, type=bool, help='use reduce layer after convolutions')
    parser.add_argument('--reduce_layer_output', default=REDUCE_LAYER_OUTPUT, type=bool,
                        help='size of reduce layer output')
    parser.add_argument('--pooling', default=POOLING, type=bool, help='apply pooling after convolutions')
    parser.add_argument('--pool_type', default=POOL_TYPE, type=str, help='type of pooling applied (max, average)')
    parser.add_argument('--pool_kernel_width', default=POOL_KERNEL_WIDTH, help='size of pooling kernel')
    parser.add_argument('--use_weights', default=USE_WEIGHTS, type=bool, help='use weighted loss')
    parser.add_argument('--filter_width', default=FILTER_WIDTH, type=int, help='filter size (convolutions)')
    parser.add_argument('--drop_prob', default=DROP_PROB, type=float, help='dropout probability before classifier')
    parser.add_argument('--optimizer', default=OPTIMIZER, type=str,
                        help='optimizer to be used (adam, rmsprop, adadelta)')
    parser.add_argument('--loss', default=LOSS, type=str, help='loss to be used (cross-entropy)')
    parser.add_argument('--lr', default=LR, type=float, help='learning rate to be used')
    parser.add_argument('--gpu', default=GPU, type=str, help='gpu to be used (e.g. cuda:1)')
    parser.add_argument('--adj_lr', default=ADJ_LR, type=bool, help='adjust learning rate')
    parser.add_argument('--adj_lr_patience', default=ADJ_LR_PATIENCE, type=int,
                        help='patience when learning rate is to be adjusted (e.g. after 5 epochs of no improvement)')
    parser.add_argument('--early_stopping', default=EARLY_STOPPING, type=bool, help='employ early stopping')
    parser.add_argument('--es_patience', default=ES_PATIENCE, type=int,
                        help='patience for early stopping (e.g. after 5 epochs of no improvement)')
    parser.add_argument('--weight_decay', default=WEIGHT_DECAY, type=float, help='weight decay to be used')
    parser.add_argument('--nb_units_lstm', default=NB_UNITS_LSTM, type=int,
                        help='number of units within each LSTM layer')
    parser.add_argument('--nb_layers_lstm', default=NB_LAYERS_LSTM, type=int, help='number of layers in LSTM')
    parser.add_argument('--nb_conv_blocks', default=NB_CONV_BLOCKS, type=int, help='number of convolution blocks')
    parser.add_argument('--conv_block_type', default=CONV_BLOCK_TYPE, type=str,
                        help='type of convolution blocks used (normal, skip, fixup)')
    parser.add_argument('--nb_filters', default=NB_FILTERS, type=int, help='number of convolution filters')
    parser.add_argument('--dilation', default=DILATION, type=int, help='dilation applied in convolution filters')
    parser.add_argument('--save_model', default=SAVE_MODEL, type=bool, help='whether to save the trained model as a pickle file')

    args = parser.parse_args()

    main(args)
