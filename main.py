##################################################
# Main script used to commence experiments
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import argparse
import os
import time
import sys
import numpy as np

from sklearn.model_selection import train_test_split

from data_processing.preprocess_data import load_dataset, compute_mean_and_std
from data_processing.sliding_window import apply_sliding_window
from model.validation import cross_participant_cv, per_participant_cv, train_valid_split, k_fold
from model.train import predict

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
- CUTOFF_TYPE: type how dataset is supposed to be split (subject-wise, percentage or record)
- CUTOFF_TRAIN: point where the train dataset is cut off; depends on CUTOFF_TYPE:
    - subject-wise: last subject which is supposed to be contained in train dataset, i.e. 5 = first 5 subjects are in train
    - percentage: percentage value of how large the train dataset is supposed to be compared to full dataset
    - record: last record which is supposed to be contained in train dataset, i.e. 500 = first 500 records are in train
- CUTOFF_VALID: point where the validation dataset is cut off; depends on CUTOFF_TYPE:
    - subject-wise: last subject which is supposed to be contained in validation dataset, i.e. 5 = first 5 subjects are in validation
    - percentage: percentage value of how large the validation dataset is supposed to be compared to full dataset
    - record: last record which is supposed to be contained in validation dataset, i.e. 500 = first 500 records are in validation
- SW_LENGTH: length of sliding window
- SW_UNIT: unit in which length of sliding window is measured
- SW_OVERLAP: overlap ratio between sliding windows (in percent, i.e. 60 = 60%)
- MEANS_AND_STDS: boolean whether to append means and standard deviations of feature columns to dataset
- INCLUDE_NULL: boolean whether to include null class in datasets (does not work with opportunity_ordonez dataset)
"""

DATASET = 'rwhar'
PRED_TYPE = 'gestures'
CUTOFF_TYPE = 'subject'
CUTOFF_TRAIN = 10
CUTOFF_VALID = None
SW_LENGTH = 1
SW_UNIT = 'seconds'
SW_OVERLAP = 60
MEANS_AND_STDS = False
INCLUDE_NULL = True

"""
NETWORK OPTIONS:
- NETWORK: network architecture to be used (e.g. 'deepconvlstm')
- LSTM: boolean whether to employ a lstm after convolution layers
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
NO_LSTM = False
NB_UNITS_LSTM = 128
NB_LAYERS_LSTM = 1
CONV_BLOCK_TYPE = 'normal'
NB_CONV_BLOCKS = 2
NB_FILTERS = 64
FILTER_WIDTH = 11
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
- SEED: random seed which is to be employed
- VALID_TYPE: (cross-)validation type; either 'cross-participant', 'per-participant', 'train-valid-split' or 'k-fold'
- BATCH_SIZE: size of the batches
- EPOCHS: number of epochs during training
- OPTIMIZER: optimizer to use; either 'rmsprop', 'adadelta' or 'adam'
- LR: learning rate to employ for optimizer
- WEIGHT_DECAY: weight decay to employ for optimizer
- WEIGHTS_INIT: weight initialization method to use to initialize network
- LOSS: loss to use ('cross_entropy', 'maxup')
- SMOOTHING: degree of label smoothing employed if cross-entropy used
- GPU: name of GPU to use (e.g. 'cuda:0')
- SPLITS_KFOLD: number of splits for stratified k-fold cross-validation
- SPLITS_PP: number of stratified splits for each subject in per-participant evaluation
- SIZE_PP:
- WEIGHTED: boolean whether to use weighted loss calculation based on support of each class
- ADJ_LR: boolean whether to adjust learning rate if no improvement
- LR_SCHEDULER: type of learning rate scheduler to employ ('step_lr', 'reduce_lr_on_plateau')
- LR_STEP: step size of learning rate scheduler (patience if plateau).
- LR_DECAY: decay factor of learning rate scheduler.
- EARLY_STOPPING: boolean whether to stop the network training early if no improvement 
- ES_PATIENCE: patience (i.e. number of epochs) after which network training is stopped if no improvement
"""

SEED = 1
VALID_TYPE = 'cross-participant'
BATCH_SIZE = 100
EPOCHS = 30
OPTIMIZER = 'adam'
LR = 1e-4
WEIGHT_DECAY = 1e-6
WEIGHTS_INIT = 'xavier_normal'
LOSS = 'cross_entropy'
SMOOTHING = 0.0
GPU = 'cuda:0'
SPLITS_KFOLD = 5
SPLITS_PP = 5
SIZE_PP = 0.6
WEIGHTED = False
ADJ_LR = False
LR_SCHEDULER = 'step_lr'
LR_STEP = 10
LR_DECAY = 0.9
EARLY_STOPPING = False
ES_PATIENCE = 10

"""
LOGGING OPTIONS:
- NAME: name of the experiment; used for logging purposes
- LOGGING: boolean whether to log console outputs in a text file
- PRINT_COUNTS: boolean whether to print the distribution of predicted labels after each epoch 
- VERBOSE: boolean whether to print batchwise results during epochs
- PRINT_FREQ: number of batches after which batchwise results are printed
- SAVE_TEST_PREDICTIONS: boolean whether to save test predictions
- SAVE_MODEL: boolean whether to save the model after last epoch as a checkpoint file
- SAVE_GRADIENT_PLOT: boolean whether to save the gradient flow plot
"""

NAME = None
LOGGING = False
PRINT_COUNTS = False
VERBOSE = False
PRINT_FREQ = 100
SAVE_TEST_PREDICTIONS = False
SAVE_CHECKPOINTS = False
SAVE_ANALYSIS = False
SAVE_GRADIENT_PLOT = False


def main(args):
    # check if valid prediction type chosen for dataset
    if args.dataset == 'opportunity' or args.dataset == 'opportunity_ordonez':
        if args.pred_type != 'gestures' and args.pred_type != 'locomotion':
            print('Did not choose a valid prediction type for Opportunity dataset!')
            exit()
    elif args.dataset == 'wetlab':
        if args.pred_type != 'actions' and args.pred_type != 'tasks':
            print('Did not choose a valid prediction type for Wetlab dataset!')
            exit()

    # parameters used to calculate runtime
    start = time.time()
    log_date = time.strftime('%Y%m%d')
    log_timestamp = time.strftime('%H%M%S')

    # saves logs to a file (standard output redirected)
    if args.logging:
        if args.name:
            sys.stdout = Logger(os.path.join('logs', log_date, log_timestamp, 'log_{}.txt'.format(args.name)))
        else:
            sys.stdout = Logger(os.path.join('logs', log_date, log_timestamp, 'log.txt'))

    print('Applied settings: ')
    print(args)

    ################################################## DATA LOADING ####################################################

    print('Loading data...')
    X, y, nb_classes, class_names, sampling_rate, has_null = \
        load_dataset(dataset=args.dataset,
                     pred_type=args.pred_type,
                     include_null=args.include_null
                     )

    args.sampling_rate = sampling_rate
    args.nb_classes = nb_classes
    args.class_names = class_names
    args.has_null = has_null

    # if selected compute means and standard deviations of each column and append to dataset
    if args.means_and_stds:
        X = np.concatenate((X, compute_mean_and_std(X[:, 1:])), axis=1)

    ############################################# TRAINING #############################################################

    # apply the chosen random seed to all relevant parts
    seed_torch(args.seed)

    # re-create full dataset for splitting purposes
    data = np.concatenate((X, (np.array(y)[:, None])), axis=1)

    # split data according to settings
    if args.dataset == 'opportunity_ordonez':
        args.valid_type = 'train-valid-split'
        train = data[:497014, :]
        valid = data[497014:557963, :]
        test = data[557963:, :]
    # subject-wise splitting
    elif args.cutoff_type == 'subject':
        train = data[(data[:, 0] <= (args.cutoff_train - 1))]
        if args.cutoff_valid:
            valid = data[(data[:, 0] > (args.cutoff_train - 1)) & (data[:, 0] <= (args.cutoff_valid - 1))]
            test = data[(data[:, 0] > (args.cutoff_valid - 1))]
            train_val = np.concatenate((train, valid), axis=0)
        else:
            valid = data[(data[:, 0] > (args.cutoff_train - 1))]
            test = None
            train_val = np.concatenate((train, valid), axis=0)
    # percentage-wise splitting
    elif args.cutoff_type == 'percentage':
        if args.cutoff_valid:
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y,
                                                                        train_size=(args.cutoff_train + args.cutoff_valid),
                                                                        shuffle=False)

            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                              train_size=args.cutoff_train / (args.cutoff_train + args.cutoff_valid),
                                                              shuffle=False)
            test = np.concatenate((X_test, (np.array(y_test)[:, None])), axis=1)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=args.cutoff_train, shuffle=False)
            test = None

        train = np.concatenate((X_train, (np.array(y_train)[:, None])), axis=1)
        valid = np.concatenate((X_val, (np.array(y_val)[:, None])), axis=1)
        train_val = np.concatenate((train, valid), axis=0)
    # record-wise splitting
    elif args.cutoff_type == 'record':
        train = data[:args.cutoff_train, :]
        if args.cutoff_valid:
            valid = data[args.cutoff_train:args.cutoff_valid, :]
            test = data[args.cutoff_valid:, :]
        else:
            valid = data[args.cutoff_train:, :]
            test = None
        train_val = np.concatenate((train, valid), axis=1)

    print("Split datasets with size: | train {0} | valid {1} | test {2} |".format(train.shape, valid.shape, test.shape))

    custom_net = None
    custom_loss = None
    custom_opt = None

    # cross-validation; either cross-participant, per-participant or normal
    if args.valid_type == 'cross-participant':
        trained_net = cross_participant_cv(train_val, custom_net, custom_loss, custom_opt, args, log_date, log_timestamp)
    elif args.valid_type == 'per-participant':
        trained_net = per_participant_cv(train_val, custom_net, custom_loss, custom_opt, args, log_date, log_timestamp)
    elif args.valid_type == 'train-valid-split':
        trained_net = train_valid_split(train, valid, custom_net, custom_loss, custom_opt, args, log_date, log_timestamp)
    elif args.valid_type == 'k-fold':
        trained_net = k_fold(train_val, custom_net, custom_loss, custom_opt, args, log_date, log_timestamp)
    else:
        print('Did not choose a valid validation type dataset!')
        exit()

    ############################################# TESTING ##############################################################

    if test.size != 0:
        X_test, y_test = apply_sliding_window(test[:, :-1], test[:, -1],
                                              args.sw_length, args.sw_unit, args.sampling_rate, args.sw_overlap)
        X_test = X_test[:, :, 1:]
        predict(X_test, y_test, trained_net, vars(args), log_date, log_timestamp)

    # calculate time data creation took
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nFinal time elapsed: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Flags
    parser.add_argument('--save_test_preds', default=SAVE_TEST_PREDICTIONS, action='store_true',
                        help='Flag indicating to save predictions in separate file')
    parser.add_argument('--logging', default=LOGGING, action='store_true',
                        help='Flag indicating to log terminal output into text file')
    parser.add_argument('--verbose', default=VERBOSE, action='store_true',
                        help='Flag indicating to have verbose training output (batchwise)')
    parser.add_argument('--print_counts', default=PRINT_COUNTS, action='store_true',
                        help='Flag indicating to print class distribution of train and validation set after epochs')
    parser.add_argument('--save_gradient_plot', default=SAVE_GRADIENT_PLOT, action='store_true',
                        help='Flag indicating to save gradient flow plot')
    parser.add_argument('--include_null', default=INCLUDE_NULL, action='store_true',
                        help='Flag indicating to include null class (if dataset has one) in training/ prediction')
    parser.add_argument('--means_and_stds', default=MEANS_AND_STDS, action='store_true',
                        help='Flag indicating to append means and stds of columns to dataset')
    parser.add_argument('--batch_norm', default=BATCH_NORM, action='store_true',
                        help='Flag indicating to use batch normalisation after each convolution')
    parser.add_argument('--reduce_layer', default=REDUCE_LAYER, action='store_true',
                        help='Flag indicating to use reduce layer after convolutions')
    parser.add_argument('--weighted', default=WEIGHTED, action='store_true',
                        help='Flag indicating to use weighted loss')
    parser.add_argument('--pooling', default=POOLING, action='store_true',
                        help='Flag indicating to apply pooling after convolutions')
    parser.add_argument('--adj_lr', default=ADJ_LR, action='store_true',
                        help='Flag indicating to adjust learning rate')
    parser.add_argument('--early_stopping', default=EARLY_STOPPING, action='store_true',
                        help='Flag indicating to employ early stopping')
    parser.add_argument('--save_checkpoints', default=SAVE_CHECKPOINTS, action='store_true',
                        help='Flag indicating to save the trained model as a checkpoint file')
    parser.add_argument('--no_lstm', default=NO_LSTM, action='store_true',
                        help='Flag indicating whether to omit LSTM from architecture')
    parser.add_argument('--save_analysis', default=SAVE_ANALYSIS, action='store_true',
                        help='Flag indicating whether to save analysis results.')

    # Strings
    parser.add_argument('--name', default=NAME, type=str,
                        help='Name of the experiment (visible in logging). Default: None')
    parser.add_argument('-d', '--dataset', default=DATASET, type=str,
                        help='Dataset to be used. Options: rwhar, sbhar, wetlab, hhar or opportunity_ordonez. '
                             'Default: rwhar')
    parser.add_argument('-co', '--cutoff_type', default=CUTOFF_TYPE, type=str,
                        help='Type how dataset is split. Options: subject, percentage or record. '
                        'Default: subject')
    parser.add_argument('-p', '--pred_type', default=PRED_TYPE, type=str,
                        help='(If applicable) prediction type for dataset. See dataset documentation for options. '
                        'Default: gestures')
    parser.add_argument('-n', '--network', default=NETWORK, type=str,
                        help='Network to be used. Options: deepconvlstm. '
                             'Default: deepconvlstm')
    parser.add_argument('-vt', '--valid_type', default=VALID_TYPE, type=str,
                        help='Validation type to be used. Options: per-participant, cross-participant, '
                             'train-valid-split, k-fold). '
                             'Default: cross-participant')
    parser.add_argument('-swu', '--sw_unit', default=SW_UNIT, type=str,
                        help='sliding window unit used. Options: units, seconds.'
                             'Default: seconds')
    parser.add_argument('-wi', '--weights_init', default=WEIGHTS_INIT, type=str,
                        help='weight initialization method used. Options: normal, orthogonal, xavier_uniform, '
                             'xavier_normal, kaiming_uniform, kaiming_normal. '
                             'Default: xavier_normal')
    parser.add_argument('-pt', '--pool_type', default=POOL_TYPE, type=str,
                        help='type of pooling applied. Options: max, average. '
                             'Default: max')
    parser.add_argument('-o', '--optimizer', default=OPTIMIZER, type=str,
                        help='Optimizer to be used. Options: adam, rmsprop, adadelta.'
                             'Default: adam')
    parser.add_argument('-l', '--loss', default=LOSS, type=str,
                        help='Loss to be used. Options: cross_entropy, maxup.'
                             'Default: cross_entropy')
    parser.add_argument('-g', '--gpu', default=GPU, type=str,
                        help='GPU to be used. Default: cuda:1')
    parser.add_argument('-lrs', '--lr_scheduler', default=LR_SCHEDULER, type=str,
                        help='Learning rate scheduler to use. Options: step_lr, reduce_lr_on_plateau. '
                             'Default: step_lr')
    parser.add_argument('-cbt', '--conv_block_type', default=CONV_BLOCK_TYPE, type=str,
                        help='type of convolution blocks used. Options: normal, skip, fixup.'
                             'Default: normal')

    # Integers
    parser.add_argument('-pf', '--print_freq', default=PRINT_FREQ, type=int,
                        help='If verbose, frequency of which is printed (batches).'
                             'Default: 100')
    parser.add_argument('-cot', '--cutoff_train', default=CUTOFF_TRAIN, type=int,
                        help='Cutoff point for train dataset. See documentation for further explanations. '
                             'Default: 10')
    parser.add_argument('-cov', '--cutoff_valid', default=CUTOFF_VALID, type=int,
                        help='Cutoff point for validation dataset. If None, no testing is performed. '
                             'See documentation for further explanations. '
                             'Default: None')
    parser.add_argument('-sskf', '--splits_kfold', default=SPLITS_KFOLD, type=int,
                        help='No. splits for k-fold cv. '
                             'Default: 5')
    parser.add_argument('-sppp', '--splits_pp', default=SPLITS_PP, type=int,
                        help='No. splits for per-participant eval.'
                             'Default: 5')
    parser.add_argument('-s', '--seed', default=SEED, type=int,
                        help='Seed to be employed. '
                             'Default: 1')
    parser.add_argument('-e', '--epochs', default=EPOCHS, type=int,
                        help='No. epochs to use during training.'
                             'Default: 30')
    parser.add_argument('-bs', '--batch_size', default=BATCH_SIZE, type=int,
                        help='Batch size to use during training.'
                             'Default: 100')
    parser.add_argument('-rlo', '--reduce_layer_output', default=REDUCE_LAYER_OUTPUT, type=int,
                        help='Size of reduce layer output. '
                             'Default: 8')
    parser.add_argument('-pkw', '--pool_kernel_width', default=POOL_KERNEL_WIDTH, type=int,
                        help='Size of pooling kernel.'
                             'Default: 2')
    parser.add_argument('-fw', '--filter_width', default=FILTER_WIDTH, type=int,
                        help='Filter size (convolutions).'
                             'Default: 11')
    parser.add_argument('-esp', '--es_patience', default=ES_PATIENCE, type=int,
                        help='Patience for early stopping (e.g. after 10 epochs of no improvement). '
                             'Default: 10')
    parser.add_argument('-nbul', '--nb_units_lstm', default=NB_UNITS_LSTM, type=int,
                        help='Number of units within each LSTM layer. '
                             'Default: 128')
    parser.add_argument('-nbll', '--nb_layers_lstm', default=NB_LAYERS_LSTM, type=int,
                        help='Number of layers in LSTM.'
                             'Default: 1')
    parser.add_argument('-nbcb', '--nb_conv_blocks', default=NB_CONV_BLOCKS, type=int,
                        help='Number of convolution blocks. '
                             'Default: 2')
    parser.add_argument('-nbf', '--nb_filters', default=NB_FILTERS, type=int,
                        help='Number of convolution filters.'
                             'Default: 64')
    parser.add_argument('-dl', '--dilation', default=DILATION, type=int,
                        help='Dilation applied in convolution filters.'
                             'Default: 1')
    parser.add_argument('-lrss', '--lr_step', default=LR_STEP, type=int,
                        help='Period of learning rate decay (patience if plateau scheduler).'
                             'Default: 10')

    # Floats
    parser.add_argument('-spp', '--size_pp', default=SIZE_PP, type=float,
                        help='Size of validation set in per-participant eval.'
                             'Default: 0.6')
    parser.add_argument('-swl', '--sw_length', default=SW_LENGTH, type=float,
                        help='Length of sliding window. '
                             'Default: 1')
    parser.add_argument('-swo', '--sw_overlap', default=SW_OVERLAP, type=int,
                        help='Overlap employed between sliding windows.'
                             'Default: 60')
    parser.add_argument('-dp', '--drop_prob', default=DROP_PROB, type=float,
                        help='Dropout probability.'
                             'Default 0.5')
    parser.add_argument('-sm', '--smoothing', default=SMOOTHING, type=float,
                        help='Degree of label smoothing.'
                             'Default: 0.0')
    parser.add_argument('-lr', '--learning_rate', default=LR, type=float,
                        help='Learning rate to be used. '
                             'Default: 1e-04')
    parser.add_argument('-wd', '--weight_decay', default=WEIGHT_DECAY, type=float,
                        help='Weight decay to be used. '
                             'Default: 1e-06')
    parser.add_argument('-lrsd', '--lr_decay', default=LR_DECAY, type=float,
                        help='Multiplicative factor of learning rate decay. '
                             'Default: 0.9')

    args = parser.parse_args()

    main(args)
