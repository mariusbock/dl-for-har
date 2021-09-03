import os

import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from data_processing.sliding_window import apply_sliding_window
from misc.osutils import mkdir_if_missing
from model.DeepConvLSTM import DeepConvLSTM
from model.evaluate import evaluate_participant_scores
from model.train import train


def cross_participant_cv(data, args, log_date, log_timestamp):
    """
    Method to apply cross-participant cross-validation (also known as leave-one-subject-out cross-validation).

    :param data: data used for applying cross-validation
    :param args: args object containing all relevant hyperparameters and settings
    :param log_date: date information needed for saving
    :param log_timestamp: timestamp information needed for saving

    :return trained network
    """

    print(' Calculating cross-participant scores using LOSO CV.')
    cp_scores = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
    train_val_gap = np.zeros((4, int(np.max(data[:, 0]) + 1)))
    all_eval_output = None
    orig_lr = args.lr

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

        X_test, y_test = pd.DataFrame(), pd.DataFrame()

        args.window_size = X_train.shape[1]
        args.nb_channels = X_train.shape[2]

        if args.network == 'deepconvlstm':
            net = DeepConvLSTM(config=vars(args))
        else:
            print("Did not provide a valid network name!")

        net, val_output, train_output, _ = train(X_train, y_train, X_val, y_val, X_test, y_test,
                                                 network=net, config=vars(args), log_date=log_date,
                                                 log_timestamp=log_timestamp)

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

    return net


def per_participant_cv(data, args, log_date, log_timestamp):
    """
    Method to apply per-participant cross-validation.

    :param data: data used for applying cross-validation
    :param args: args object containing all relevant hyperparameters and settings
    :param log_date: date information needed for saving
    :param log_timestamp: timestamp information needed for saving

    :return trained network
    """

    print('Calculating per-participant scores using stratified random split.')
    pp_scores = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
    all_eval_output = None
    train_val_gap = np.zeros((4, int(np.max(data[:, 0]) + 1)))
    orig_lr = args.lr

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

            X_test, y_test = pd.DataFrame(), pd.DataFrame()

            args.window_size = X_train.shape[1]
            args.nb_channels = X_train.shape[2]

            if args.network == 'deepconvlstm':
                net = DeepConvLSTM(config=vars(args))
            else:
                print("Did not provide a valid network name!")

            net, val_output, train_output, _ = train(X_train, y_train, X_val, y_val, X_test, y_test,
                                                     network=net, config=vars(args), log_date=log_date,
                                                     log_timestamp=log_timestamp)

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

    evaluate_participant_scores(participant_scores=pp_scores,
                                gen_gap_scores=train_val_gap,
                                input_cm=all_eval_output,
                                class_names=args.class_names,
                                nb_subjects=int(np.max(data[:, 0]) + 1),
                                filepath=os.path.join('logs', log_date, log_timestamp),
                                filename='per-participant',
                                args=args
                                )

    return net


def train_valid_test_split(X_train, y_train, X_val, y_val, X_test, y_test, args, log_date, log_timestamp):
    """
    Method to apply normal cross-validation, i.e. one set split into train, validation and testing data.

    :param X_train: train features used for applying cross-validation
    :param y_train: train labels used for applying cross-validation
    :param X_val: validation features used for applying cross-validation
    :param y_val: validation labels used for applying cross-validation
    :param X_test: test features used for applying cross-validation
    :param y_test: test labels used for applying cross-validation
    :param args: args object containing all relevant hyperparameters and settings
    :param log_date: date information needed for saving
    :param log_timestamp: timestamp information needed for saving

    :return trained network
    """

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
    if X_test.size != 0:
        X_test, y_test = apply_sliding_window(args.dataset, X_test, y_test,
                                              sliding_window_size=args.sw_length,
                                              unit=args.sw_unit,
                                              sampling_rate=args.sampling_rate,
                                              sliding_window_overlap=args.sw_overlap,
                                              )

    args.window_size = X_train.shape[1]
    args.nb_channels = X_train.shape[2]

    if args.network == 'deepconvlstm':
        net = DeepConvLSTM(config=vars(args))
    else:
        print("Did not provide a valid network name!")

    net, val_output, train_output, test_output = train(X_train, y_train, X_val, y_val, X_test, y_test,
                                                       network=net, config=vars(args), log_date=log_date,
                                                       log_timestamp=log_timestamp)

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

        if args.save_test_preds:
            mkdir_if_missing(os.path.join('logs', log_date, log_timestamp))
            np.save(os.path.join('logs', log_date, log_timestamp, 'test_preds.npy'), test_output)

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

    return net
