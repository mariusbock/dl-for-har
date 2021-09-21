import argparse
import os
import time
import sys
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from data_processing.preprocess_data import load_dataset, compute_mean_and_std
from data_processing.sliding_window import apply_sliding_window
from misc.osutils import mkdir_if_missing
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

DATASET = 'hhar'
PRED_TYPE = 'gestures'
CUTOFF_TYPE = 'subject'
CUTOFF_TRAIN = 3
CUTOFF_VALID = 5
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
NB_LAYERS_LSTM = 1
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
- SEED: random seed which is to be employed
- VALID_TYPE: (cross-)validation type; either 'cross-participant', 'per-participant', 'train-valid-split' or 'k-fold'
- BATCH_SIZE: size of the batches
- EPOCHS: number of epochs during training
- OPTIMIZER: optimizer to use; either 'rmsprop', 'adadelta' or 'adam'
- LR: learning rate to employ for optimizer
- WEIGHT_DECAY: weight decay to employ for optimizer
- WEIGHTS_INIT: weight initialization method to use to initialize network
- LOSS: loss to use; currently only 'cross_entropy' supported
- GPU: name of GPU to use (e.g. 'cuda:0')
- SPLITS_KFOLD: number of splits for stratified k-fold cross-validation
- SPLITS_SSS: number of stratified splits for each subject in per-participant evaluation
- USE_WEIGHTS: boolean whether to use weighted loss calculation based on support of each class
- ADJ_LR: boolean whether to adjust learning rate if no improvement
- EARLY_STOPPING: boolean whether to stop the network training early if no improvement 
- ADJ_LR_PATIENCE: patience (i.e. number of epochs) after which learning is adjusted if no improvement
- ES_PATIENCE: patience (i.e. number of epochs) after which network training is stopped if no improvement
"""

SEED = 1
VALID_TYPE = 'k-fold'
BATCH_SIZE = 100
EPOCHS = 5
OPTIMIZER = 'adam'
LR = 1e-4
WEIGHT_DECAY = 1e-6
WEIGHTS_INIT = 'xavier_normal'
LOSS = 'cross_entropy'
LS_SMOOTHING = 0.1
GPU = 'cuda:0'
SPLITS_KFOLD = 5
SPLITS_SSS = 2
SIZE_SSS = 0.6
USE_WEIGHTS = True
ADJ_LR = False
EARLY_STOPPING = False
ADJ_LR_PATIENCE = 5
ES_PATIENCE = 5

"""
LOGGING OPTIONS:
- LOGGING: boolean whether to log console outputs in a text file
- PRINT_COUNTS: boolean whether to print the distribution of predicted labels after each epoch 
- VERBOSE: boolean whether to print batchwise results during epochs
- PRINT_FREQ: number of batches after which batchwise results are printed
- SAVE_TEST_PREDICTIONS: boolean whether to save test predictions
- SAVE_MODEL: boolean whether to save the model after last epoch as pickle file
- SAVE_GRADIENT_PLOT: boolean whether to save the gradient flow plot
"""

LOGGING = False
PRINT_COUNTS = False
VERBOSE = False
PRINT_FREQ = 100
SAVE_TEST_PREDICTIONS = False
SAVE_MODEL = False
SAVE_GRADIENT_PLOT = False


def main(args):
    # check if valid prediction type chosen for dataset
    if DATASET == 'opportunity' or DATASET == 'opportunity_ordonez':
        if PRED_TYPE != 'gestures' and PRED_TYPE != 'locomotion':
            print('Did not choose a valid prediction type for Opportunity dataset!')
            exit()
    elif DATASET == 'wetlab':
        if PRED_TYPE != 'actions' and PRED_TYPE != 'tasks':
            print('Did not choose a valid prediction type for Wetlab dataset!')
            exit()

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
    elif args.cutoff_type == 'subject':
        train = data[(data[:, 0] <= (args.cutoff_train - 1))]
        valid = data[(data[:, 0] > (args.cutoff_train - 1)) & (data[:, 0] <= (args.cutoff_valid - 1))]
        test = data[(data[:, 0] > (args.cutoff_valid - 1))]
        train_val = np.concatenate((train, valid), axis=0)
    elif args.cutoff_type == 'percentage':
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y,
                                                                    train_size=(args.cutoff_train + args.cutoff_valid),
                                                                    shuffle=False)

        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                          train_size=args.cutoff_train / (args.cutoff_train + args.cutoff_valid),
                                                          shuffle=False)

        train = np.concatenate((X_train, (np.array(y_train)[:, None])), axis=1)
        valid = np.concatenate((X_val, (np.array(y_val)[:, None])), axis=1)
        test = np.concatenate((X_test, (np.array(y_test)[:, None])), axis=1)

        train_val = np.concatenate((X_train_val, (np.array(y_train_val)[:, None])), axis=1)
    elif args.cutoff_type == 'record':
        train = data[:args.cutoff_train, :]
        valid = data[args.cutoff_train:args.cutoff_valid, :]
        test = data[args.cutoff_valid:, :]

        train_val = np.concatenate((train, valid), axis=1)

    print("Split datasets with size: | train {0} | valid {1} | test {2} |".format(train.shape, valid.shape, test.shape))

    # cross-validation; either cross-participant, per-participant or normal
    if args.valid_type == 'cross-participant':
        trained_net = cross_participant_cv(train_val, args, log_date, log_timestamp)
    elif args.valid_type == 'per-participant':
        trained_net = per_participant_cv(train_val, args, log_date, log_timestamp)
    elif args.valid_type == 'train-valid-split':
        trained_net = train_valid_split(train, valid, args, log_date, log_timestamp)
    elif args.valid_type == 'k-fold':
        trained_net = k_fold(train_val, args, log_date, log_timestamp)
    else:
        print('Did not choose a valid validation type dataset!')
        exit()

    # test predictions
    if test.size != 0:
        X_test, y_test = apply_sliding_window(test[:, :-1], test[:, -1],
                                              args.sw_length, args.sw_unit, args.sampling_rate, args.sw_overlap)
        X_test = X_test[:, :, 1:]
        predict(X_test, y_test, trained_net, vars(args), log_date, log_timestamp)

    # if selected, save model as pickle file
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
    parser.add_argument('--save_gradient_plot', default=SAVE_GRADIENT_PLOT, type=bool, help='save gradient flow plot')
    parser.add_argument('--include_null', default=INCLUDE_NULL, type=bool,
                        help='include null class (if dataset has one) in training/ prediction')
    parser.add_argument('--cutoff_type', default=CUTOFF_TYPE, type=str, help='type how dataset is split (subject, percentage or record)')
    parser.add_argument('--cutoff_train', default=CUTOFF_TRAIN, type=int,
                        help='cutoff point for train dataset. See documentation for further explanations.')
    parser.add_argument('--cutoff_valid', default=CUTOFF_VALID, type=int,
                        help='cutoff point for validation dataset. See documentation for further explanations.')
    parser.add_argument('--splits_kfold', default=SPLITS_KFOLD, type=int, help='no. splits for k-fold cv')
    parser.add_argument('--splits_sss', default=SPLITS_SSS, type=int, help='no. splits for per-participant eval')
    parser.add_argument('--size_sss', default=SIZE_SSS, type=float,
                        help='size of validation set in per-participant eval')
    parser.add_argument('--network', default=NETWORK, type=str, help='network to be used (e.g. deepconvlstm)')
    parser.add_argument('--valid_type', default=VALID_TYPE, type=str,
                        help='validation type to be used (per-participant, cross-participant, train-valid-split, k-fold)')
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
    parser.add_argument('--optimizer', default=OPTIMIZER, type=str, help='optimizer to be used (adam, rmsprop, adadelta)')
    parser.add_argument('--loss', default=LOSS, type=str, help='loss to be used (e.g. cross_entropy)')
    parser.add_argument('--ls_smoothing', default=LS_SMOTHING, type=float, help='degree of label smoothing (if employed)')
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
