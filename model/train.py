import math
import os

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.utils import class_weight
from torch import nn
from torch.utils.data import DataLoader

from misc.osutils import mkdir_if_missing
from misc.torchutils import count_parameters
from model.DeepConvLSTM import ConvBlock, ConvBlockSkip, ConvBlockFixup


def init_weights(network):
    """
    Weight initialization of network (initialises all LSTM, Conv2D and Linear layers according to weight_init parameter
    of network.

    :param network: network of which weights are to be initialised
    :return: network with initialised weights
    """
    for m in network.modules():
        # normal convblock and skip convblock initialisation
        if isinstance(m, (ConvBlock, ConvBlockSkip)):
            if network.weights_init == 'normal':
                torch.nn.init.normal_(m.conv1.weight)
                torch.nn.init.normal_(m.conv2.weight)
            elif network.weights_init == 'orthogonal':
                torch.nn.init.orthogonal_(m.conv1.weight)
                torch.nn.init.orthogonal_(m.conv2.weight)
            elif network.weights_init == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.conv1.weight)
                torch.nn.init.xavier_uniform_(m.conv2.weight)
            elif network.weights_init == 'xavier_normal':
                torch.nn.init.xavier_normal_(m.conv1.weight)
                torch.nn.init.xavier_normal_(m.conv2.weight)
            elif network.weights_init == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.conv1.weight)
                torch.nn.init.kaiming_uniform_(m.conv2.weight)
            elif network.weights_init == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(m.conv1.weight)
                torch.nn.init.kaiming_normal_(m.conv2.weight)
            m.conv1.bias.data.fill_(0.0)
            m.conv2.bias.data.fill_(0.0)
        # fixup block initialisation (see fixup paper for details)
        elif isinstance(m, ConvBlockFixup):
            nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(
                2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * network.nb_conv_blocks ** (-0.5))
            nn.init.constant_(m.conv2.weight, 0)
        # linear layers
        elif isinstance(m, nn.Linear):
            if network.use_fixup:
                nn.init.constant_(m.weight, 0)
            elif network.weights_init == 'normal':
                torch.nn.init.normal_(m.weight)
            elif network.weights_init == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight)
            elif network.weights_init == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight)
            elif network.weights_init == 'xavier_normal':
                torch.nn.init.xavier_normal_(m.weight)
            elif network.weights_init == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.weight)
            elif network.weights_init == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # LSTM initialisation
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    if network.weights_init == 'normal':
                        torch.nn.init.normal_(param.data)
                    elif network.weights_init == 'orthogonal':
                        torch.nn.init.orthogonal_(param.data)
                    elif network.weights_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif network.weights_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param.data)
                    elif network.weights_init == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif network.weights_init == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    if network.weights_init == 'normal':
                        torch.nn.init.normal_(param.data)
                    elif network.weights_init == 'orthogonal':
                        torch.nn.init.orthogonal_(param.data)
                    elif network.weights_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif network.weights_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param.data)
                    elif network.weights_init == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif network.weights_init == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.0)
    return network


def plot_grad_flow(network):
    """
    Function which plots the average gradient of a network.

    :param network: network used to obtain gradient
    :return: plot containing the plotted average gradient
    """
    named_parameters = network.named_parameters()
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def init_loss(config):
    """
    Initialises an loss object for a given network.

    :return: loss object
    """
    if config.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config.loss == 'label_smoothing':
        class LabelSmoothingLoss(nn.Module):
            def __init__(self, smoothing=0.0):
                super(LabelSmoothingLoss, self).__init__()
                self.smoothing = smoothing

            def forward(self, prediction, target):
                assert 0 <= self.smoothing < 1
                neglog_softmaxPrediction = -prediction.log_softmax(dim=1)

                with torch.no_grad():
                    smoothedLabels = self.smoothing / (prediction.size(1) - 1) * torch.ones_like(prediction)
                    smoothedLabels.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)
                return torch.mean(torch.sum(smoothedLabels * neglog_softmaxPrediction, dim=1))
        criterion = LabelSmoothingLoss(smoothing=config.ls_smoothing)
    return criterion


def init_optimizer(network, config):
    """
    Initialises an optimizer object for a given network.

    :param network: network for which optimizer and loss are to be initialised
    :return: optimizer object
    """
    # define optimizer and loss
    if config.optimizer == 'adadelta':
        opt = torch.optim.Adadelta(network.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        opt = torch.optim.Adam(network.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(network.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return opt


def train(train_features, train_labels, val_features, val_labels, network, optimizer, loss, config, log_date, log_timestamp):
    """
    Method to train a PyTorch network.

    :param train_features: training features
    :param train_labels: training labels
    :param val_features: validation features
    :param val_labels: validation labels
    :param network: DeepConvLSTM network object
    :param optimizer: optimizer object
    :param loss: loss object
    :param config: config file which contains all training and hyperparameter settings
    :param log_date: date used for logging
    :param log_timestamp: timestamp used for logging

    :return three numpy arrays containing validation, training and test predictions (predictions, gt labels)
    """

    # prints the number of learnable parameters in the network
    count_parameters(network)

    # init network using weight initialization of choice
    network = init_weights(network)
    # send network to GPU
    network.to(config['gpu'])
    network.train()

    # if weighted loss chosen, calculate weights based on training dataset; else each class is weighted equally
    if config['use_weights']:
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels + 1), y=train_labels + 1)
        if config['loss'] == 'cross_entropy':
            loss.weights = class_weights
        print('Applied weighted class weights: ')
        print(class_weights)
    else:
        class_weights = class_weight.compute_class_weight(None, classes=np.unique(train_labels + 1), y=train_labels + 1)
        if config['loss'] == 'cross_entropy':
            loss.weights = class_weights


    # initialize optimizer and loss
    opt, criterion = optimizer, loss

    # counters and objects used for early stopping and learning rate adjustment
    best_loss = np.inf
    best_network = None
    best_val_losses = None
    best_train_losses = None
    best_val_preds = None
    best_train_preds = None
    early_stop = False
    lr_pt_counter = 0
    es_pt_counter = 0

    # training loop; iterates through epochs
    for e in range(config['epochs']):
        """
        TRAINING
        """
        # helper objects
        train_preds = []
        train_gt = []
        train_losses = []
        start_time = time.time()
        batch_num = 1

        # initialize train dataset and loader
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))
        trainloader = DataLoader(dataset,
                                 batch_size=config['batch_size'],
                                 num_workers=2,
                                 shuffle=False,
                                 )

        # iterate over train dataset
        for i, (x, y) in enumerate(trainloader):
            # send x and y to GPU
            inputs, targets = x.to(config['gpu']), y.to(config['gpu'])
            # zero accumulated gradients
            opt.zero_grad()
            # send inputs through network to get predictions, calculate loss and backpropagate
            train_output = network(inputs)
            loss = criterion(train_output, targets.long())
            loss.backward()
            opt.step()
            # append train loss to list
            train_losses.append(loss.item())

            # create predictions and append them to final list
            y_preds = np.argmax(train_output.cpu().detach().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            train_preds = np.concatenate((np.array(train_preds, int), np.array(y_preds, int)))
            train_gt = np.concatenate((np.array(train_gt, int), np.array(y_true, int)))

            # if verbose print out batch wise results (batch number, loss and time)
            if config['verbose']:
                if batch_num % config['print_freq'] == 0 and batch_num > 0:
                    cur_loss = np.mean(train_losses)
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d} batches | ms/batch {:5.2f} | '
                          'train loss {:5.2f}'.format(e, batch_num, elapsed * 1000 / config['batch_size'], cur_loss))
                    start_time = time.time()
                batch_num += 1

            # plot gradient flow if wanted
            if config['save_gradient_plot']:
                plot_grad_flow(network)

        """
        VALIDATION
        """

        # helper objects
        val_preds = []
        val_gt = []
        val_losses = []

        # initialize train dataset and loader
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features).float(), torch.from_numpy(val_labels))
        valloader = DataLoader(dataset,
                               batch_size=config['batch_size'],
                               num_workers=2,
                               shuffle=False,
                               )

        # set network to eval mode
        network.eval()
        with torch.no_grad():
            # iterate over validation dataset
            for i, (x, y) in enumerate(valloader):
                # send x and y to GPU
                inputs, targets = x.to(config['gpu']), y.to(config['gpu'])

                # send inputs through network to get predictions, loss and calculate softmax probabilities
                val_output = network(inputs)
                val_loss = criterion(val_output, targets.long())
                val_output = torch.nn.functional.softmax(val_output, dim=1)

                # append validation loss to list
                val_losses.append(val_loss.item())

                # create predictions and append them to final list
                y_preds = np.argmax(val_output.cpu().numpy(), axis=-1)
                y_true = targets.cpu().numpy().flatten()
                val_preds = np.concatenate((np.array(val_preds, int), np.array(y_preds, int)))
                val_gt = np.concatenate((np.array(val_gt, int), np.array(y_true, int)))

            # print epoch evaluation results for train and validation dataset
            print("EPOCH: {}/{}".format(e + 1, config['epochs']),
                  "Train Loss: {:.4f}".format(np.mean(train_losses)),
                  "Train Acc: {:.4f}".format(jaccard_score(train_gt, train_preds, average='macro')),
                  "Train Prec: {:.4f}".format(precision_score(train_gt, train_preds, average='macro')),
                  "Train Rcll: {:.4f}".format(recall_score(train_gt, train_preds, average='macro')),
                  "Train F1: {:.4f}".format(f1_score(train_gt, train_preds, average='macro')),
                  "Val Loss: {:.4f}".format(np.mean(val_losses)),
                  "Val Acc: {:.4f}".format(jaccard_score(val_gt, val_preds, average='macro')),
                  "Val Prec: {:.4f}".format(precision_score(val_gt, val_preds, average='macro')),
                  "Val Rcll: {:.4f}".format(recall_score(val_gt, val_preds, average='macro')),
                  "Val F1: {:.4f}".format(f1_score(val_gt, val_preds, average='macro')))

            # if chosen, print the value counts of the predicted labels for train and validation dataset
            if config['print_counts']:
                y_train = np.bincount(train_preds)
                ii_train = np.nonzero(y_train)[0]
                y_val = np.bincount(val_preds)
                ii_val = np.nonzero(y_val)[0]
                print('Predicted Train Labels: ')
                print(np.vstack((ii_train, y_train[ii_train])).T)
                print('Predicted Val Labels: ')
                print(np.vstack((ii_val, y_val[ii_val])).T)

        # if adjust learning rate is enabled
        if config['adj_lr'] or config['early_stopping']:
            if best_loss < np.mean(val_losses):
                lr_pt_counter += 1
                es_pt_counter += 1

                # adjust learning rate check
                if lr_pt_counter >= config['adj_lr_patience'] and config['adj_lr']:
                    config['lr'] *= 0.1
                    for param_group in opt.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.1
                    print('Changing learning rate to {} since no loss improvement over {} epochs.'
                          .format(config['lr'], str(lr_pt_counter)))

                # early stopping check
                if es_pt_counter >= config['es_patience'] and config['early_stopping']:
                    print('Stopping training early since no loss improvement over {} epochs.'
                          .format(str(es_pt_counter)))
                    early_stop = True
                    # print results of best epoch
                    print('Final (best) results: ')
                    print("Train Loss: {:.4f}".format(np.mean(best_train_losses)),
                          "Train Acc: {:.4f}".format(jaccard_score(train_gt, best_train_preds, average='macro')),
                          "Train Prec: {:.4f}".format(precision_score(train_gt, best_train_preds, average='macro')),
                          "Train Rcll: {:.4f}".format(recall_score(train_gt, best_train_preds, average='macro')),
                          "Train F1: {:.4f}".format(f1_score(train_gt, best_train_preds, average='macro')),
                          "Val Loss: {:.4f}".format(np.mean(best_val_losses)),
                          "Val Acc: {:.4f}".format(jaccard_score(val_gt, best_val_preds, average='macro')),
                          "Val Prec: {:.4f}".format(precision_score(val_gt, best_val_preds, average='macro')),
                          "Val Rcll: {:.4f}".format(recall_score(val_gt, best_val_preds, average='macro')),
                          "Val F1: {:.4f}".format(f1_score(val_gt, best_val_preds, average='macro')))

            else:
                lr_pt_counter = 0
                es_pt_counter = 0
                best_network = network
                best_loss = np.mean(val_losses)
                best_train_losses = train_losses
                best_train_preds = train_preds
                best_val_losses = val_losses
                best_val_preds = val_preds
        else:
            best_network = network
            best_train_losses = train_losses
            best_train_preds = train_preds
            best_val_losses = val_losses
            best_val_preds = val_preds

        # set network to train mode again
        network.train()

        if early_stop:
            break

    # if plot_gradient gradient plot is shown at end of training
    if config['save_gradient_plot']:
        mkdir_if_missing(os.path.join('logs', log_date, log_timestamp))
        plt.savefig(os.path.join('logs', log_date, log_timestamp, 'grad_flow.png'))

    # return validation, train and test predictions as numpy array with ground truth
    return best_network, np.vstack((best_val_preds, val_gt)).T, np.vstack((best_train_preds, train_gt)).T


def predict(test_features, test_labels, network, config, log_date, log_timestamp):
    """
    Method that applies a trained network to obtain predictions on a test dataset. If selected, saves predictions.

    :param test_features: test features
    :param test_labels: test labels
    :param network: trained network object
    :param config: config file which contains all training and hyperparameter settings
    :param log_date: date used for saving predictions
    :param log_timestamp: timestamp used for saving predictions
    """
    # set network to eval mode
    network.eval()
    # helper objects
    test_preds = []
    test_gt = []

    # initialize test dataset and loader
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_features).float(), torch.from_numpy(test_labels))
    testloader = DataLoader(dataset,
                            batch_size=config['batch_size'],
                            num_workers=2,
                            shuffle=False,
                            )

    with torch.no_grad():
        # iterate over test dataset
        for i, (x, y) in enumerate(testloader):
            # send x and y to GPU
            inputs, targets = x.to(config['gpu']), y.to(config['gpu'])

            # send inputs through network to get predictions and calculate softmax probabilities
            test_output = network(inputs)
            test_output = torch.nn.functional.softmax(test_output, dim=1)

            # create predictions and append them to final list
            y_preds = np.argmax(test_output.cpu().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            test_preds = np.concatenate((np.array(test_preds, int), np.array(y_preds, int)))
            test_gt = np.concatenate((np.array(test_gt, int), np.array(y_true, int)))

    cls = np.array(range(config['nb_classes']))
    print('\nTEST RESULTS: ')
    print("Avg. Accuracy: {0}".format(jaccard_score(test_gt, test_preds, average='macro')))
    print("Avg. Precision: {0}".format(precision_score(test_gt, test_preds, average='macro')))
    print("Avg. Recall: {0}".format(recall_score(test_gt, test_preds, average='macro')))
    print("Avg. F1: {0}".format(f1_score(test_gt, test_preds, average='macro')))

    print("\nTEST RESULTS (PER CLASS): ")
    print("Accuracy: {0}".format(jaccard_score(test_gt, test_preds, average=None, labels=cls)))
    print("Precision: {0}".format(precision_score(test_gt, test_preds, average=None, labels=cls)))
    print("Recall: {0}".format(recall_score(test_gt, test_preds, average=None, labels=cls)))
    print("F1: {0}".format(f1_score(test_gt, test_preds, average=None, labels=cls)))

    if config['save_test_preds']:
        mkdir_if_missing(os.path.join('logs', log_date, log_timestamp))
        np.save(os.path.join('logs', log_date, log_timestamp, 'test_preds.npy'), test_output)