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
DATASET = 'rwhar'
PRED_TYPE = 'actions'
CUTOFF_TRAIN = 4
CUTOFF_TEST = 30
SW_LENGTH = 1
SW_UNIT = 'seconds'
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
VALID_TYPE = 'k-fold'
BATCH_SIZE = 100
EPOCHS = 30
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
PRINT_COUNTS = True
PLOT_GRADIENT = False
VERBOSE = True
SAVE_PREDICTIONS = False

# misc
GPU = 'cuda:0'
SPLITS_SSS = 10
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

    ################################################## DATA LOADING ########################################################

    print('Loading data...')
    X_train, y_train, X_val, y_val, X_test, y_test, nb_classes, class_names, sampling_rate = \
        load_dataset(dataset=args.dataset,
                     cutoff_sequence_train=args.cutoff_train,
                     cutoff_sequence_test=args.cutoff_test,
                     pred_type=args.pred_type,
                     include_null=args.include_null
                     )
    args.sampling_rate = sampling_rate
    args.nb_classes = nb_classes
    args.class_names = class_names
    if args.means_and_stds:
        X_train = np.concatenate((X_train, compute_mean_and_std(X_train[:, 1:])), axis=1)
        X_val = np.concatenate((X_val, compute_mean_and_std(X_val[:, 1:])), axis=1)
        X_test = np.concatenate((X_test, compute_mean_and_std(X_test[:, 1:])), axis=1)

    ############################################### TRAINING ###############################################################

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
            all_eval_output = None
            for i, sbj in enumerate(np.unique(data[:, 0])):
                # for i, sbj in enumerate([0, 1]):
                print('\n VALIDATING FOR SUBJECT {0} OF {1}'.format(int(sbj) + 1, int(np.max(data[:, 0])) + 1))
                train_data = data[data[:, 0] != sbj][:, :]
                val_data = data[data[:, 0] == sbj][:, :]
                args.lr = orig_lr
                # Sensor data is segmented using a sliding window mechanism
                X_train, y_train = apply_sliding_window(train_data[:, :-1], train_data[:, -1],
                                                        sliding_window_size=args.sw_length,
                                                        unit=args.sw_unit,
                                                        sampling_rate=args.sampling_rate,
                                                        sliding_window_overlap=args.sw_overlap,
                                                        )

                X_val, y_val = apply_sliding_window(val_data[:, :-1], val_data[:, -1],
                                                    sliding_window_size=args.sw_length,
                                                    unit=args.sw_unit,
                                                    sampling_rate=args.sampling_rate,
                                                    sliding_window_overlap=args.sw_overlap,
                                                    )
                args.window_size = X_train.shape[1]
                args.nb_channels = X_train.shape[2]

                net = DeepConvLSTM(config=vars(args))

                train_losses, val_losses, eval_output = train(X_train, y_train, X_val, y_val, network=net,
                                                              config=vars(args))

                if all_eval_output is None:
                    all_eval_output = eval_output
                else:
                    all_eval_output = np.concatenate((all_eval_output, eval_output), axis=0)

                cls = np.array(range(args.nb_classes))
                cp_scores[0, :, int(sbj)] = jaccard_score(eval_output[:, 1], eval_output[:, 0], average=None,
                                                          labels=cls)
                cp_scores[1, :, int(sbj)] = precision_score(eval_output[:, 1], eval_output[:, 0], average=None,
                                                            labels=cls)
                cp_scores[2, :, int(sbj)] = recall_score(eval_output[:, 1], eval_output[:, 0], average=None, labels=cls)
                cp_scores[3, :, int(sbj)] = f1_score(eval_output[:, 1], eval_output[:, 0], average=None, labels=cls)

                print("SUBJECT {0} VALIDATION RESULTS: ".format(int(sbj) + 1))
                print("Accuracy: {0}".format(
                    jaccard_score(eval_output[:, 1], eval_output[:, 0], average=None, labels=cls)))
                print("Precision: {0}".format(
                    precision_score(eval_output[:, 1], eval_output[:, 0], average=None, labels=cls)))
                print(
                    "Recall: {0}".format(recall_score(eval_output[:, 1], eval_output[:, 0], average=None, labels=cls)))
                print("F1: {0}".format(f1_score(eval_output[:, 1], eval_output[:, 0], average=None, labels=cls)))

            evaluate_participant_scores(participant_scores=cp_scores,
                                        input_cm=all_eval_output,
                                        class_names=args.class_names,
                                        nb_subjects=int(np.max(data[:, 0]) + 1),
                                        filepath=os.path.join('logs', log_date, log_timestamp),
                                        filename='cross-participant'
                                        )

        elif args.valid_type == 'per-participant':
            print('Calculating per-participant scores using stratified random split.')
            X = np.concatenate((X_train, X_val), axis=0)
            y = np.concatenate((y_train, y_val), axis=0)
            data = np.concatenate((X, (np.array(y)[:, None])), axis=1)
            pp_scores = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
            all_eval_output = None
            for i, sbj in enumerate(np.unique(data[:, 0])):
                print('\n VALIDATING FOR SUBJECT {0} OF {1}'.format(int(sbj) + 1, int(np.max(data[:, 0])) + 1))

                sss = StratifiedShuffleSplit(train_size=args.size_sss,
                                             n_splits=args.splits_sss,
                                             random_state=args.seed)

                subject_data = data[data[:, 0] == sbj]
                non_subject_data = data[data[:, 0] != sbj]
                X, y = subject_data[:, :-1], subject_data[:, -1]

                if args.use_weights:
                    class_weights = cw.compute_class_weight('balanced',
                                                            classes=np.unique(non_subject_data[:, -1] + 1),
                                                            y=non_subject_data[:, -1] + 1
                                                            )
                else:
                    class_weights = None

                classes = np.array(range(args.nb_classes))
                subject_accuracy = np.zeros(args.nb_classes)
                subject_precision = np.zeros(args.nb_classes)
                subject_recall = np.zeros(args.nb_classes)
                subject_f1 = np.zeros(args['nb_classes'])
                for j, (train_index, test_index) in enumerate(sss.split(X, y)):
                    print('SPLIT {0}/{1}'.format(j + 1, args.splits_sss))

                    X_train, X_val = X[train_index], X[test_index]
                    y_train, y_val = y[train_index], y[test_index]
                    args.lr = orig_lr

                    # Sensor data is segmented using a sliding window mechanism
                    X_train, y_train = apply_sliding_window(X_train, y_train,
                                                            sliding_window_size=args.sw_length,
                                                            unit=args.sw_unit,
                                                            sampling_rate=args.sampling_rate,
                                                            sliding_window_overlap=args.sw_overlap,
                                                            )

                    X_val, y_val = apply_sliding_window(X_val, y_val,
                                                        sliding_window_size=args.sw_length,
                                                        unit=args.sw_unit,
                                                        sampling_rate=args.sampling_rate,
                                                        sliding_window_overlap=args.sw_overlap,
                                                        )

                    args.window_size = X_train.shape[1]
                    args.nb_channels = X_train.shape[2]

                    net = DeepConvLSTM(config=vars(args))

                    train_losses, val_losses, eval_output = train(X_train, y_train, X_val, y_val,
                                                                  network=net, config=vars(args), cw=class_weights)

                    if all_eval_output is None:
                        all_eval_output = eval_output
                    else:
                        all_eval_output = np.concatenate((all_eval_output, eval_output), axis=0)

                    subject_accuracy += jaccard_score(eval_output[:, 1], eval_output[:, 0], average=None,
                                                      labels=classes)
                    subject_precision += precision_score(eval_output[:, 1], eval_output[:, 0], average=None,
                                                         labels=classes)
                    subject_recall += recall_score(eval_output[:, 1], eval_output[:, 0], average=None, labels=classes)
                    subject_f1 += f1_score(eval_output[:, 1], eval_output[:, 0], average=None, labels=classes)

                pp_scores[0, :, int(sbj)] = subject_accuracy / args.splits_sss
                pp_scores[1, :, int(sbj)] = subject_precision / args.splits_sss
                pp_scores[2, :, int(sbj)] = subject_recall / args.splits_sss
                pp_scores[3, :, int(sbj)] = subject_f1 / args.splits_sss

                print("SUBJECT {0} VALIDATION RESULTS: ".format(int(sbj)))
                print("Accuracy: {0}".format(pp_scores[0, :, int(sbj)]))
                print("Precision: {0}".format(pp_scores[1, :, int(sbj)]))
                print("Recall: {0}".format(pp_scores[2, :, int(sbj)]))
                print("F1: {0}".format(pp_scores[3, :, int(sbj)]))

            if args.save_preds:
                np.save('predictions/' + args.pred_type + '.npy', eval_output)

            evaluate_participant_scores(participant_scores=pp_scores,
                                        input_cm=all_eval_output,
                                        class_names=args.class_names,
                                        nb_subjects=int(np.max(data[:, 0]) + 1),
                                        filepath=os.path.join('logs', log_date, log_timestamp),
                                        filename='per-participant'
                                        )
        elif args.valid_type == 'k-fold':
            # Sensor data is segmented using a sliding window mechanism
            X_train, y_train = apply_sliding_window(X_train, y_train,
                                                    sliding_window_size=args.sw_length,
                                                    unit=args.sw_unit,
                                                    sampling_rate=args.sampling_rate,
                                                    sliding_window_overlap=args.sw_overlap,
                                                    )

            X_val, y_val = apply_sliding_window(X_val, y_val,
                                                sliding_window_size=args.sw_length,
                                                unit=args.sw_unit,
                                                sampling_rate=args.sampling_rate,
                                                sliding_window_overlap=args.sw_overlap,
                                                )

            args.window_size = X_train.shape[1]
            args.nb_channels = X_train.shape[2]

            net = DeepConvLSTM(config=vars(args))
            # TODO: implement k-fold

            train_losses, val_losses, eval_output = train(X_train, y_train, X_val, y_val, network=net,
                                                          config=vars(args))

            cls = np.array(range(args.nb_classes))

            print("VALIDATION RESULTS: ")
            print("Accuracy: {0}".format(jaccard_score(eval_output[:, 1], eval_output[:, 0], average=None, labels=cls)))
            print("Precision: {0}".format(
                precision_score(eval_output[:, 1], eval_output[:, 0], average=None, labels=cls)))
            print("Recall: {0}".format(recall_score(eval_output[:, 1], eval_output[:, 0], average=None, labels=cls)))
            print("F1: {0}".format(f1_score(eval_output[:, 1], eval_output[:, 0], average=None, labels=cls)))

            # TODO: include test predictions
    else:
        print('Error: Did not provide a valid network name!')

    # calculate time data creation took
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nFinal time elapsed: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--pred_type', default=PRED_TYPE, type=str)
    parser.add_argument('--save_preds', default=SAVE_PREDICTIONS, type=bool)
    parser.add_argument('--logging', default=LOGGING, type=bool)
    parser.add_argument('--verbose', default=VERBOSE, type=bool)
    parser.add_argument('--print_freq', default=PRINT_FREQ, type=int)
    parser.add_argument('--print_counts', default=PRINT_COUNTS, type=bool)
    parser.add_argument('--plot_gradient', default=PLOT_GRADIENT, type=bool)
    parser.add_argument('--include_null', default=INCLUDE_NULL, type=bool)
    parser.add_argument('--cutoff_train', default=CUTOFF_TRAIN, type=int)
    parser.add_argument('--cutoff_test', default=CUTOFF_TEST, type=int)
    parser.add_argument('--splits_sss', default=SPLITS_SSS, type=int)
    parser.add_argument('--size_sss', default=SIZE_SSS, type=float)
    parser.add_argument('--network', default=NETWORK, type=str)
    parser.add_argument('--valid_type', default=VALID_TYPE, type=str)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--epochs', default=EPOCHS, type=int)
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
    parser.add_argument('--means_and_stds', default=MEANS_AND_STDS, type=bool)
    parser.add_argument('--sw_length', default=SW_LENGTH, type=float)
    parser.add_argument('--sw_unit', default=SW_UNIT, type=str)
    parser.add_argument('--sw_overlap', default=SW_OVERLAP, type=float)
    parser.add_argument('--weights_init', default=WEIGHTS_INIT, type=str)
    parser.add_argument('--batch_norm', default=BATCH_NORM, type=bool)
    parser.add_argument('--reduce_layer', default=REDUCE_LAYER, type=bool)
    parser.add_argument('--reduce_layer_output', default=REDUCE_LAYER_OUTPUT, type=bool)
    parser.add_argument('--pooling', default=POOLING, type=bool)
    parser.add_argument('--pool_type', default=POOL_TYPE, type=str)
    parser.add_argument('--pool_kernel_width', default=POOL_KERNEL_WIDTH)
    parser.add_argument('--use_weights', default=USE_WEIGHTS, type=bool)
    parser.add_argument('--filter_width', default=FILTER_WIDTH, type=int)
    parser.add_argument('--drop_prob', default=DROP_PROB, type=float)
    parser.add_argument('--optimizer', default=OPTIMIZER, type=str)
    parser.add_argument('--loss', default=LOSS, type=str)
    parser.add_argument('--lr', default=LR, type=float)
    parser.add_argument('--gpu', default=GPU, type=str)
    parser.add_argument('--adj_lr', default=ADJ_LR, type=bool)
    parser.add_argument('--adj_lr_patience', default=ADJ_LR_PATIENCE, type=int)
    parser.add_argument('--early_stopping', default=EARLY_STOPPING, type=bool)
    parser.add_argument('--es_patience', default=ES_PATIENCE, type=int)
    parser.add_argument('--weight_decay', default=WEIGHT_DECAY, type=float)
    parser.add_argument('--nb_units_lstm', default=NB_UNITS_LSTM, type=int)
    parser.add_argument('--nb_layers_lstm', default=NB_LAYERS_LSTM, type=int)
    parser.add_argument('--nb_conv_blocks', default=NB_CONV_BLOCKS, type=int)
    parser.add_argument('--conv_block_type', default=CONV_BLOCK_TYPE, type=str)
    parser.add_argument('--nb_filters', default=NB_FILTERS, type=int)
    parser.add_argument('--dilation', default=DILATION, type=int)

    args = parser.parse_args()

    main(args)
