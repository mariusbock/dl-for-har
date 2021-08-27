import matplotlib.pyplot as plt
import numpy as np
import itertools

import os
from sklearn.metrics import confusion_matrix

from misc.osutils import mkdir_if_missing


def plot_confusion_matrix(input, target_names, title='Confusion matrix', cmap=None, normalize=True, output_path=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    input:        confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    cm = confusion_matrix(input[:, 0], input[:, 1])
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if output_path is not None:
        plt.savefig(output_path)


def evaluate_participant_scores(participant_scores, gen_gap_scores, input_cm, class_names, nb_subjects, filepath, filename, args):
    print('\nPREDICTION RESULTS')
    print('-------------------')
    print('Average results')
    avg_acc = np.mean(participant_scores[0, :, :])
    std_acc = np.std(participant_scores[0, :, :])
    avg_prc = np.mean(participant_scores[1, :, :])
    std_prc = np.std(participant_scores[1, :, :])
    avg_rcll = np.mean(participant_scores[2, :, :])
    std_rcll = np.std(participant_scores[2, :, :])
    avg_f1 = np.mean(participant_scores[3, :, :])
    std_f1 = np.std(participant_scores[3, :, :])
    print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
          'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
          'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
          'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    if args.include_null:
        print('Average results (no null)')
        avg_acc = np.mean(participant_scores[0, 1:, :])
        std_acc = np.std(participant_scores[0, 1:, :])
        avg_prc = np.mean(participant_scores[1, 1:, :])
        std_prc = np.std(participant_scores[1, 1:, :])
        avg_rcll = np.mean(participant_scores[2, 1:, :])
        std_rcll = np.std(participant_scores[2, 1:, :])
        avg_f1 = np.mean(participant_scores[3, 1:, :])
        std_f1 = np.std(participant_scores[3, 1:, :])
        print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
              'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
              'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
              'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    print('Average class results')
    for i, class_name in enumerate(class_names):
        avg_acc = np.mean(participant_scores[0, i, :])
        std_acc = np.std(participant_scores[0, i, :])
        avg_prc = np.mean(participant_scores[1, i, :])
        std_prc = np.std(participant_scores[1, i, :])
        avg_rcll = np.mean(participant_scores[2, i, :])
        std_rcll = np.std(participant_scores[2, i, :])
        avg_f1 = np.mean(participant_scores[3, i, :])
        std_f1 = np.std(participant_scores[3, i, :])
        print('Class {}: Avg. Accuracy {:.4f} (±{:.4f}), '.format(class_name, avg_acc, std_acc),
              'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
              'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
              'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    print('Subject-wise results')
    for subject in range(nb_subjects):
        print('Subject ', subject + 1, ' results: ')
        for i, class_name in enumerate(class_names):
            acc = participant_scores[0, i, subject]
            prc = participant_scores[1, i, subject]
            rcll = participant_scores[2, i, subject]
            f1 = participant_scores[3, i, subject]
            print('Class {}: Accuracy {:.4f}, '.format(class_name, acc),
                  'Precision {:.4f}, '.format(prc),
                  'Recall {:.4f}, '.format(rcll),
                  'F1-Score {:.4f}'.format(f1))

    print('\nGENERALIZATION GAP ANALYSIS')
    print('-------------------')
    print('Average results')
    avg_acc = np.mean(gen_gap_scores[0, :])
    std_acc = np.std(gen_gap_scores[0, :])
    avg_prc = np.mean(gen_gap_scores[1, :])
    std_prc = np.std(gen_gap_scores[1, :])
    avg_rcll = np.mean(gen_gap_scores[2, :])
    std_rcll = np.std(gen_gap_scores[2, :])
    avg_f1 = np.mean(gen_gap_scores[3, :])
    std_f1 = np.std(gen_gap_scores[3, :])
    print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
          'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
          'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
          'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    print('Subject-wise results')
    for subject in range(nb_subjects):
        print('Subject ', subject + 1, ' results: ')
        acc = gen_gap_scores[0, subject]
        prc = gen_gap_scores[1, subject]
        rcll = gen_gap_scores[2, subject]
        f1 = gen_gap_scores[3, subject]
        print('Accuracy {:.4f}, '.format(acc),
              'Precision {:.4f}, '.format(prc),
              'Recall {:.4f}, '.format(rcll),
              'F1-Score {:.4f}'.format(f1))

    # create boxplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.suptitle('Average Participant Results', size=16)
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].boxplot(participant_scores[0, :, :].T, labels=class_names, showmeans=True)
    axs[0, 1].set_title('Precision')
    axs[0, 1].boxplot(participant_scores[1, :, :].T, labels=class_names, showmeans=True)
    axs[1, 0].set_title('Recall')
    axs[1, 0].boxplot(participant_scores[2, :, :].T, labels=class_names, showmeans=True)
    axs[1, 1].set_title('F1-Score')
    axs[1, 1].boxplot(participant_scores[3, :, :].T, labels=class_names, showmeans=True)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)
    fig.subplots_adjust(hspace=0.5)
    mkdir_if_missing(filepath)
    plt.savefig(os.path.join(filepath, filename + '_bx.png'))

    # create confusion matrix
    plot_confusion_matrix(input_cm, class_names, normalize=False, output_path=os.path.join(filepath, filename + '_cm.png'))
