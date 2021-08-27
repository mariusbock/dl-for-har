import argparse
import os
import time
import sys
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight as cw

from data_processing.sliding_window import apply_sliding_window
from data_processing.preprocess_data import load_dataset, compute_mean_and_std
from model.evaluate import evaluate_participant_scores

from misc.logging import Logger
from misc.torchutils import seed_torch

# data
"""
DATA CHOICES: 
    - wetlab: 'actions' or 'tasks'
    - opportunity: 'gestures' or 'locomotion'
    - opportunity_ordonez: 'gestures' or 'locomotion'
    - rwhar
    - sbhar
    - hhar
"""
DATASET = 'opportunity_ordonez'
PRED_TYPE = 'gestures'

CUTOFF_TRAIN = 4
CUTOFF_TEST = 6
SW_LENGTH = 50
SW_UNIT = 'units'
SW_OVERLAP = 60
MEANS_AND_STDS = False
INCLUDE_NULL = True

# network
NETWORK = 'deepconvlstm'
NB_UNITS_LSTM = 128
NB_LAYERS_LSTM = 1
CONV_BLOCK_TYPE = 'normal'
NB_CONV_BLOCKS = 2
NB_FILTERS = 64
FILTER_WIDTH = 11
DILATION = 1
DROP_PROB = 0.5
POOLING = False
POOL_TYPE = 'max'
POOL_KERNEL_WIDTH = 2
BATCH_NORM = False
REDUCE_LAYER = False
REDUCE_LAYER_OUTPUT = 8

# training
VALID_TYPE = 'cross-participant'
BATCH_SIZE = 100
EPOCHS = 5
OPTIMIZER = 'adam'
LR = 1e-4
WEIGHT_DECAY = 1e-6
WEIGHTS_INIT = 'None'
LOSS = 'cross-entropy'
USE_WEIGHTS = True
ADJ_LR = False
EARLY_STOPPING = False
ADJ_LR_PATIENCE = 2
ES_PATIENCE = 5

# print settings
LOGGING = False
PRINT_COUNTS = False
PLOT_GRADIENT = False
VERBOSE = False
SAVE_PREDICTIONS = False

# misc
GPU = 'cuda:0'
SPLITS_SSS = 2
SIZE_SSS = 0.6
SEED = 1
PRINT_FREQ = 100


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
                     cutoff_sequence_train=args.cutoff_train,
                     cutoff_sequence_test=args.cutoff_test,
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

    if args.network == 'deepconvlstm':
        from model.train import train
        from model.DeepConvLSTM import DeepConvLSTM
        seed_torch(args.seed)
        orig_lr = args.lr
        if args.valid_type == 'cross-participant':
            print(' Calculating cross-participant scores using LOSO CV.')
            X = np.concatenate((X_train, X_val), axis=0)
            y = np.concatenate((y_train, y_val), axis=0)
            data = np.concatenate((X, (np.array(y)[:, None])), axis=1)
            cp_scores = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
            train_val_gap = np.zeros((4, int(np.max(data[:, 0]) + 1)))
            all_eval_output = None
            for i, sbj in enumerate(np.unique(data[:, 0])):
                # for i, sbj in enumerate([0, 1]):
                print('\n VALIDATING FOR SUBJECT {0} OF {1}'.format(int(sbj) + 1, int(np.max(data[:, 0])) + 1))
                train_data = data[data[:, 0] != sbj][:, :]
                val_data = data[data[:, 0] == sbj][:, :]
                args.lr = orig_lr
                # Sensor data is segmented using a sliding window mechanism
                X_train, y_train = apply_sliding_window(args.dataset, train_data[:, :-1], train_data[:, -1],
                                                        sliding_window_size=args.sw_length,
                                                        unit=args.sw_unit,
                                                        sampling_rate=args.sampling_rate,
                                                        sliding_window_overlap=args.sw_overlap,
                                                        )

                X_val, y_val = apply_sliding_window(args.dataset, val_data[:, :-1], val_data[:, -1],
                                                    sliding_window_size=args.sw_length,
                                                    unit=args.sw_unit,
                                                    sampling_rate=args.sampling_rate,
                                                    sliding_window_overlap=args.sw_overlap,
                                                    )
                args.window_size = X_train.shape[1]
                args.nb_channels = X_train.shape[2]

                net = DeepConvLSTM(config=vars(args))

                val_output, train_output, _ = train(X_train, y_train, X_val, y_val, X_test, y_test,
                                                              network=net, config=vars(args))

                if all_eval_output is None:
                    all_eval_output = val_output
                else:
                    all_eval_output = np.concatenate((all_eval_output, val_output), axis=0)

                # fill values for normal evaluation
                cls = np.array(range(args.nb_classes))
                cp_scores[0, :, int(sbj)] = jaccard_score(val_output[:, 1], val_output[:, 0], average=None,
                                                          labels=cls)
                cp_scores[1, :, int(sbj)] = precision_score(val_output[:, 1], val_output[:, 0], average=None,
                                                            labels=cls)
                cp_scores[2, :, int(sbj)] = recall_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)
                cp_scores[3, :, int(sbj)] = f1_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)

                # fill values for train val gap evaluation
                train_val_gap[0, int(sbj)] = jaccard_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                             jaccard_score(val_output[:, 1], val_output[:, 0], average='macro')
                train_val_gap[1, int(sbj)] = precision_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                             precision_score(val_output[:, 1], val_output[:, 0], average='macro')
                train_val_gap[2, int(sbj)] = recall_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                             recall_score(val_output[:, 1], val_output[:, 0], average='macro')
                train_val_gap[3, int(sbj)] = f1_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                             f1_score(val_output[:, 1], val_output[:, 0], average='macro')

                print("SUBJECT {0} VALIDATION RESULTS: ".format(int(sbj) + 1))
                print("Accuracy: {0}".format(
                    jaccard_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)))
                print("Precision: {0}".format(
                    precision_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)))
                print(
                    "Recall: {0}".format(recall_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)))
                print("F1: {0}".format(f1_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)))

            evaluate_participant_scores(participant_scores=cp_scores,
                                        gen_gap_scores=train_val_gap,
                                        input_cm=all_eval_output,
                                        class_names=args.class_names,
                                        nb_subjects=int(np.max(data[:, 0]) + 1),
                                        filepath=os.path.join('logs', log_date, log_timestamp),
                                        filename='cross-participant',
                                        args=args
                                        )

        elif args.valid_type == 'per-participant':
            # TODO: Currently not working properly!
            print('Calculating per-participant scores using stratified random split.')
            X = np.concatenate((X_train, X_val), axis=0)
            y = np.concatenate((y_train, y_val), axis=0)
            data = np.concatenate((X, (np.array(y)[:, None])), axis=1)
            pp_scores = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
            all_eval_output = None
            train_val_gap = np.zeros((4, int(np.max(data[:, 0]) + 1)))

            for i, sbj in enumerate(np.unique(data[:, 0])):
                print('\n VALIDATING FOR SUBJECT {0} OF {1}'.format(int(sbj) + 1, int(np.max(data[:, 0])) + 1))

                sss = StratifiedShuffleSplit(train_size=args.size_sss,
                                             n_splits=args.splits_sss,
                                             random_state=args.seed)

                subject_data = data[data[:, 0] == sbj]
                X, y = subject_data[:, :-1], subject_data[:, -1]

                classes = np.array(range(args.nb_classes))
                subject_accuracy = np.zeros(args.nb_classes)
                subject_precision = np.zeros(args.nb_classes)
                subject_recall = np.zeros(args.nb_classes)
                subject_f1 = np.zeros(args.nb_classes)

                subject_accuracy_gap = 0
                subject_precision_gap = 0
                subject_recall_gap = 0
                subject_f1_gap = 0
                for j, (train_index, test_index) in enumerate(sss.split(X, y)):
                    print('SPLIT {0}/{1}'.format(j + 1, args.splits_sss))

                    X_train, X_val = X[train_index], X[test_index]
                    y_train, y_val = y[train_index], y[test_index]
                    args.lr = orig_lr

                    # Sensor data is segmented using a sliding window mechanism
                    X_train, y_train = apply_sliding_window(args.dataset, X_train, y_train,
                                                            sliding_window_size=args.sw_length,
                                                            unit=args.sw_unit,
                                                            sampling_rate=args.sampling_rate,
                                                            sliding_window_overlap=args.sw_overlap,
                                                            )

                    X_val, y_val = apply_sliding_window(args.dataset, X_val, y_val,
                                                        sliding_window_size=args.sw_length,
                                                        unit=args.sw_unit,
                                                        sampling_rate=args.sampling_rate,
                                                        sliding_window_overlap=args.sw_overlap,
                                                        )

                    args.window_size = X_train.shape[1]
                    args.nb_channels = X_train.shape[2]

                    net = DeepConvLSTM(config=vars(args))

                    val_output, train_output, _ = train(X_train, y_train, X_val, y_val, X_test, y_test,
                                                        network=net, config=vars(args))

                    if all_eval_output is None:
                        all_eval_output = val_output
                    else:
                        all_eval_output = np.concatenate((all_eval_output, val_output), axis=0)

                    subject_accuracy += jaccard_score(val_output[:, 1], val_output[:, 0], average=None,
                                                      labels=classes)
                    subject_precision += precision_score(val_output[:, 1], val_output[:, 0], average=None,
                                                         labels=classes)
                    subject_recall += recall_score(val_output[:, 1], val_output[:, 0], average=None, labels=classes)
                    subject_f1 += f1_score(val_output[:, 1], val_output[:, 0], average=None, labels=classes)

                    # add up train val gap evaluation
                    subject_accuracy_gap = jaccard_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                                 jaccard_score(val_output[:, 1], val_output[:, 0], average='macro')
                    subject_precision_gap = precision_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                                 precision_score(val_output[:, 1], val_output[:, 0], average='macro')
                    subject_recall_gap = recall_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                                 recall_score(val_output[:, 1], val_output[:, 0], average='macro')
                    subject_f1_gap = f1_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                                 f1_score(val_output[:, 1], val_output[:, 0], average='macro')

                pp_scores[0, :, int(sbj)] = subject_accuracy / args.splits_sss
                pp_scores[1, :, int(sbj)] = subject_precision / args.splits_sss
                pp_scores[2, :, int(sbj)] = subject_recall / args.splits_sss
                pp_scores[3, :, int(sbj)] = subject_f1 / args.splits_sss

                train_val_gap[0, int(sbj)] = subject_accuracy_gap / args.splits_sss
                train_val_gap[1, int(sbj)] = subject_precision_gap / args.splits_sss
                train_val_gap[2, int(sbj)] = subject_recall_gap / args.splits_sss
                train_val_gap[3, int(sbj)] = subject_f1_gap / args.splits_sss

                print("SUBJECT {0} VALIDATION RESULTS: ".format(int(sbj)))
                print("Accuracy: {0}".format(pp_scores[0, :, int(sbj)]))
                print("Precision: {0}".format(pp_scores[1, :, int(sbj)]))
                print("Recall: {0}".format(pp_scores[2, :, int(sbj)]))
                print("F1: {0}".format(pp_scores[3, :, int(sbj)]))

            if args.save_preds:
                np.save('predictions/' + args.pred_type + '.npy', val_output)

            evaluate_participant_scores(participant_scores=pp_scores,
                                        gen_gap_scores=train_val_gap,
                                        input_cm=all_eval_output,
                                        class_names=args.class_names,
                                        nb_subjects=int(np.max(data[:, 0]) + 1),
                                        filepath=os.path.join('logs', log_date, log_timestamp),
                                        filename='per-participant',
                                        args=args
                                        )

        elif args.valid_type == 'normal':
            # Sensor data is segmented using a sliding window mechanism
            X_train, y_train = apply_sliding_window(args.dataset, X_train, y_train,
                                                    sliding_window_size=args.sw_length,
                                                    unit=args.sw_unit,
                                                    sampling_rate=args.sampling_rate,
                                                    sliding_window_overlap=args.sw_overlap,
                                                    )

            X_val, y_val = apply_sliding_window(args.dataset, X_val, y_val,
                                                sliding_window_size=args.sw_length,
                                                unit=args.sw_unit,
                                                sampling_rate=args.sampling_rate,
                                                sliding_window_overlap=args.sw_overlap,
                                                )

            args.window_size = X_train.shape[1]
            args.nb_channels = X_train.shape[2]

            net = DeepConvLSTM(config=vars(args))

            if X_test.size == 0:
                val_output, train_output, _ = train(X_train, y_train, X_val, y_val, X_test, y_test,
                                                    network=net, config=vars(args))
            else:
                X_test, y_test = apply_sliding_window(args.dataset, X_test, y_test,
                                                      sliding_window_size=args.sw_length,
                                                      unit=args.sw_unit,
                                                      sampling_rate=args.sampling_rate,
                                                      sliding_window_overlap=args.sw_overlap,
                                                      )

                val_output, train_output, test_output = train(X_train, y_train, X_val, y_val, X_test, y_test,
                                                              network=net, config=vars(args))

            cls = np.array(range(args.nb_classes))
            print('VALIDATION RESULTS: ')
            print("Avg. Accuracy: {0}".format(jaccard_score(val_output[:, 1], val_output[:, 0], average='macro')))
            print("Avg. Precision: {0}".format(precision_score(val_output[:, 1], val_output[:, 0], average='macro')))
            print("Avg. Recall: {0}".format(recall_score(val_output[:, 1], val_output[:, 0], average='macro')))
            print("Avg. F1: {0}".format(f1_score(val_output[:, 1], val_output[:, 0], average='macro')))

            print("VALIDATION RESULTS (PER CLASS): ")
            print("Accuracy: {0}".format(jaccard_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)))
            print("Precision: {0}".format(precision_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)))
            print("Recall: {0}".format(recall_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)))
            print("F1: {0}".format(f1_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)))

            if X_test.size != 0:
                print('TEST RESULTS: ')
                print("Avg. Accuracy: {0}".format(jaccard_score(test_output[:, 1], test_output[:, 0], average='macro')))
                print("Avg. Precision: {0}".format(precision_score(test_output[:, 1], test_output[:, 0], average='macro')))
                print("Avg. Recall: {0}".format(recall_score(test_output[:, 1], test_output[:, 0], average='macro')))
                print("Avg. F1: {0}".format(f1_score(test_output[:, 1], test_output[:, 0], average='macro')))

                print("TEST RESULTS (PER CLASS): ")
                print("Accuracy: {0}".format(jaccard_score(test_output[:, 1], test_output[:, 0], average=None, labels=cls)))
                print("Precision: {0}".format(precision_score(test_output[:, 1], test_output[:, 0], average=None, labels=cls)))
                print("Recall: {0}".format(recall_score(test_output[:, 1], test_output[:, 0], average=None, labels=cls)))
                print("F1: {0}".format(f1_score(test_output[:, 1], test_output[:, 0], average=None, labels=cls)))

            print("GENERALIZATION GAP ANALYSIS: ")
            print("Train-Val-Accuracy Difference: {0}".format(
                jaccard_score(train_output[:, 1], train_output[:, 0], average='macro') -
                jaccard_score(val_output[:, 1], val_output[:, 0], average='macro')))
            print("Train-Val-Precision Difference: {0}".format(
                precision_score(train_output[:, 1], train_output[:, 0], average='macro') -
                precision_score(val_output[:, 1], val_output[:, 0], average='macro')))
            print("Train-Val-Recall Difference: {0}".format(
                recall_score(train_output[:, 1], train_output[:, 0], average='macro') -
                recall_score(val_output[:, 1], val_output[:, 0], average='macro')))
            print("Train-Val-F1 Difference: {0}".format(
                f1_score(train_output[:, 1], train_output[:, 0], average='macro') -
                f1_score(val_output[:, 1], val_output[:, 0], average='macro')))

            # TODO: implement k-fold
    else:
        print('Error: Did not provide a valid network name!')

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
    parser.add_argument('--save_preds', default=SAVE_PREDICTIONS, type=bool, help='save predictions in separate file')
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
    parser.add_argument('--cutoff_test', default=CUTOFF_TEST, type=int,
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
    parser.add_argument('--dilation', default=DILATION, type=int, help='dilation applied in covolution filters')

    args = parser.parse_args()

    main(args)
