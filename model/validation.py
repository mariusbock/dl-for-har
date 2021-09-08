import os

import numpy as np
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from data_processing.sliding_window import apply_sliding_window
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

    print('\nCALCULATING CROSS-PARTICIPANT SCORES USING LOSO CV.\n')
    cp_scores = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
    train_val_gap = np.zeros((4, int(np.max(data[:, 0]) + 1)))
    all_eval_output = None
    orig_lr = args.lr

    for i, sbj in enumerate(np.unique(data[:, 0])):
        # for i, sbj in enumerate([0, 1]):
        print('\n VALIDATING FOR SUBJECT {0} OF {1}'.format(int(sbj) + 1, int(np.max(data[:, 0])) + 1))
        train_data = data[data[:, 0] != sbj]
        val_data = data[data[:, 0] == sbj]
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

        X_train, X_val = X_train[:, :, 1:], X_val[:, :, 1:]

        args.window_size = X_train.shape[1]
        args.nb_channels = X_train.shape[2]

        if args.network == 'deepconvlstm':
            net = DeepConvLSTM(config=vars(args))
        else:
            print("Did not provide a valid network name!")

        net, val_output, train_output = train(X_train, y_train, X_val, y_val,
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

    print('\nCALCULATING PER-PARTICIPANT SCORES USING STRATIFIED SHUFFLE SPLIT.\n')
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

        # sensor data is segmented using a sliding window mechanism
        X, y = apply_sliding_window(subject_data[:, :-1], subject_data[:, -1],
                                    sliding_window_size=args.sw_length,
                                    unit=args.sw_unit,
                                    sampling_rate=args.sampling_rate,
                                    sliding_window_overlap=args.sw_overlap,
                                    )

        X = X[:, :, 1:]

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

            args.window_size = X_train.shape[1]
            args.nb_channels = X_train.shape[2]

            if args.network == 'deepconvlstm':
                net = DeepConvLSTM(config=vars(args))
            else:
                print("Did not provide a valid network name!")

            net, val_output, train_output = train(X_train, y_train, X_val, y_val, network=net, config=vars(args),
                                                  log_date=log_date, log_timestamp=log_timestamp)

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
            subject_accuracy_gap += jaccard_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                    jaccard_score(val_output[:, 1], val_output[:, 0], average='macro')
            subject_precision_gap += precision_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                     precision_score(val_output[:, 1], val_output[:, 0], average='macro')
            subject_recall_gap += recall_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                  recall_score(val_output[:, 1], val_output[:, 0], average='macro')
            subject_f1_gap += f1_score(train_output[:, 1], train_output[:, 0], average='macro') - \
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


def train_valid_split(train_data, valid_data, args, log_date, log_timestamp):
    """
    Method to apply normal cross-validation, i.e. one set split into train, validation and testing data.

    :param train_data: train features & labels used for applying cross-validation
    :param valid_data: validation features & labels used for applying cross-validation
    :param args: args object containing all relevant hyperparameters and settings
    :param log_date: date information needed for saving
    :param log_timestamp: timestamp information needed for saving

    :return trained network
    """
    print('\nCALCULATING TRAIN-VALID-SPLIT SCORES.\n')

    # Sensor data is segmented using a sliding window mechanism
    X_train, y_train = apply_sliding_window(train_data[:, :-1], train_data[:, -1],
                                            sliding_window_size=args.sw_length,
                                            unit=args.sw_unit,
                                            sampling_rate=args.sampling_rate,
                                            sliding_window_overlap=args.sw_overlap,
                                            )

    X_val, y_val = apply_sliding_window(valid_data[:, :-1], valid_data[:, -1],
                                        sliding_window_size=args.sw_length,
                                        unit=args.sw_unit,
                                        sampling_rate=args.sampling_rate,
                                        sliding_window_overlap=args.sw_overlap,
                                        )

    X_train, X_val = X_train[:, :, 1:], X_val[:, :, 1:]

    args.window_size = X_train.shape[1]
    args.nb_channels = X_train.shape[2]

    if args.network == 'deepconvlstm':
        net = DeepConvLSTM(config=vars(args))
    else:
        print("Did not provide a valid network name!")

    net, val_output, train_output = train(X_train, y_train, X_val, y_val,
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


def k_fold(data, args, log_date, log_timestamp):
    """
    Method to apply per-participant cross-validation.

    :param data: data used for applying cross-validation
    :param args: args object containing all relevant hyperparameters and settings
    :param log_date: date information needed for saving
    :param log_timestamp: timestamp information needed for saving

    :return trained network
    """
    print('\nCALCULATING K-FOLD SCORES USING STRATIFIED K-FOLD.\n')
    # sensor data is segmented using a sliding window mechanism
    X, y = apply_sliding_window(data[:, :-1], data[:, -1],
                                sliding_window_size=args.sw_length,
                                unit=args.sw_unit,
                                sampling_rate=args.sampling_rate,
                                sliding_window_overlap=args.sw_overlap,
                                )

    orig_lr = args.lr

    skf = StratifiedKFold(n_splits=args.splits_kfold, shuffle=True, random_state=args.seed)

    cls = np.array(range(args.nb_classes))
    kfold_accuracy = np.zeros(args.nb_classes)
    kfold_precision = np.zeros(args.nb_classes)
    kfold_recall = np.zeros(args.nb_classes)
    kfold_f1 = np.zeros(args.nb_classes)

    kfold_accuracy_gap = 0
    kfold_precision_gap = 0
    kfold_recall_gap = 0
    kfold_f1_gap = 0
    for j, (train_index, test_index) in enumerate(skf.split(X, y)):
        print('FOLD {0}/{1}'.format(j + 1, args.splits_kfold))

        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        args.lr = orig_lr

        args.window_size = X_train.shape[1]
        args.nb_channels = X_train.shape[2]

        if args.network == 'deepconvlstm':
            net = DeepConvLSTM(config=vars(args))
        else:
            print("Did not provide a valid network name!")

        net, val_output, train_output = train(X_train, y_train, X_val, y_val,
                                              network=net, config=vars(args), log_date=log_date,
                                              log_timestamp=log_timestamp)

        fold_acc = jaccard_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)
        fold_prec = precision_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)
        fold_rec = recall_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)
        fold_f1 = f1_score(val_output[:, 1], val_output[:, 0], average=None, labels=cls)

        fold_acc_gap = jaccard_score(train_output[:, 1], train_output[:, 0], average='macro') - jaccard_score(val_output[:, 1], val_output[:, 0], average='macro')
        fold_prec_gap = precision_score(train_output[:, 1], train_output[:, 0], average='macro') - precision_score(val_output[:, 1], val_output[:, 0], average='macro')
        fold_rec_gap = recall_score(train_output[:, 1], train_output[:, 0], average='macro') - recall_score(val_output[:, 1], val_output[:, 0], average='macro')
        fold_f1_gap = f1_score(train_output[:, 1], train_output[:, 0], average='macro') - f1_score(val_output[:, 1], val_output[:, 0], average='macro')

        print("\nFOLD {0} VALIDATION RESULTS: ".format(j + 1))
        print("Accuracy: {0}".format(np.mean(fold_acc)))
        print("Precision: {0}".format(np.mean(fold_prec)))
        print("Recall: {0}".format(np.mean(fold_rec)))
        print("F1: {0}".format(np.mean(fold_f1)))

        # add up fold evaluation results
        kfold_accuracy += fold_acc
        kfold_precision += fold_prec
        kfold_recall += fold_rec
        kfold_f1 += fold_f1

        # add up train val gap evaluation
        kfold_accuracy_gap += fold_acc_gap
        kfold_precision_gap += fold_prec_gap
        kfold_recall_gap += fold_rec_gap
        kfold_f1_gap += fold_f1_gap

    print("\nK-FOLD VALIDATION RESULTS: ")
    print("Accuracy: {0}".format(np.mean(kfold_accuracy / args.splits_kfold)))
    print("Precision: {0}".format(np.mean(kfold_precision / args.splits_kfold)))
    print("Recall: {0}".format(np.mean(kfold_recall / args.splits_kfold)))
    print("F1: {0}".format(np.mean(kfold_f1 / args.splits_kfold)))

    print("\nVALIDATION RESULTS (PER CLASS): ")
    print("Accuracy: {0}".format(kfold_accuracy / args.splits_kfold))
    print("Precision: {0}".format(kfold_precision / args.splits_kfold))
    print("Recall: {0}".format(kfold_recall / args.splits_kfold))
    print("F1: {0}".format(kfold_f1 / args.splits_kfold))

    print("\nGENERALIZATION GAP ANALYSIS:")
    print("Accuracy: {0}".format(kfold_accuracy_gap / args.splits_kfold))
    print("Precision: {0}".format(kfold_precision_gap / args.splits_kfold))
    print("Recall: {0}".format(kfold_recall_gap / args.splits_kfold))
    print("F1: {0}".format(kfold_f1_gap / args.splits_kfold))
