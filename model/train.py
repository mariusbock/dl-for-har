##################################################
# All functions related to training a model
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# Author: Michael Moeller
# Email: michael.moeller(at)uni-siegen.de
##################################################

import os
import random

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

from misc.osutils import mkdir_if_missing
from misc.torchutils import count_parameters, seed_worker
from model.DeepConvLSTM import ConvBlock, ConvBlockSkip, ConvBlockFixup


def init_weights(network):
    """
    Weight initialization of network (initialises all LSTM, Conv2D and Linear layers according to weight_init parameter
    of network.

    :param network: pytorch model
        Network of which weights are to be initialised
    :return: pytorch model
        Network with initialised weights
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


class Maxup(torch.nn.Module):
    """
    A meta-augmentation, returning the worst result from a range of augmentations.
    As in the orignal paper, https://arxiv.org/abs/2002.09024,
    Implementation inspired by https://github.com/JonasGeiping/data-poisoning/
    see forest / data / mixing_data_augmentations.py
    """

    def __init__(self, given_data_augmentation, ntrials=4):
        """Initialize with a given data augmentation module."""
        super().__init__()
        self.augment = given_data_augmentation
        self.ntrials = ntrials
        self.max_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, y):
        additional_x, additional_labels = [], []
        for trial in range(self.ntrials):
            x_out, y_out = self.augment(x, y)
            additional_x.append(x_out)
            additional_labels.append(y_out)

        additional_x = torch.cat(additional_x, dim=0)
        additional_labels = torch.cat(additional_labels, dim=0)

        return additional_x, additional_labels

    def maxup_loss(self, outputs, extra_labels):
        """Compute loss. Here the loss is computed as worst-case estimate over the trials."""
        batch_size = outputs.shape[0] // self.ntrials
        correct_preds = (torch.argmax(outputs.data, dim=1) == extra_labels).sum().item() / self.ntrials
        stacked_loss = self.max_criterion(outputs, extra_labels).view(batch_size, self.ntrials, -1)
        loss = stacked_loss.max(dim=1)[0].mean()

        return loss, correct_preds


def my_noise_addition_augmenter(x, y):
    """
    Noise augmenter for maxup loss

    :param x: numpy array
        Features
    :param y: numpy array
        Labels
    :return: numpy array, numpy array
        Features with added noise and labels
    """
    sigma = 0.5
    return x + sigma*torch.randn_like(x), y


def plot_grad_flow(network):
    """
    Function which plots the average gradient of a network.

    :param network: pytorch model
        Network used to obtain gradient
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

    :param config: dict
        General setting dictionary
    :return: loss object
    """
    if config.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(label_smoothing=config.smoothing)
    elif config.loss == 'maxup':
        return None
    else:
        print("Did not provide a valid loss name!")
        return None
    return criterion


def init_optimizer(network, config):
    """
    Initialises an optimizer object for a given network.

    :param network: pytorch model
        Network for which optimizer and loss are to be initialised
    :param config: dict
        General setting dictionary
    :return: optimizer object
    """
    # define optimizer and loss
    if config.optimizer == 'adadelta':
        opt = torch.optim.Adadelta(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        opt = torch.optim.Adam(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        print("Did not provide a valid optimizer name!")
        return None
    return opt


def init_scheduler(optimizer, config):
    """

    :param optimizer: optimizer object
        Optimizer object used during training
    :param config: dict
        General setting dictionary
    :return:
    """
    if config.lr_scheduler == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_step, config.lr_decay)
    elif config.lr_scheduler == 'reduce_lr_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.lr_step, factor=config.lr_decay)
    else:
        print("Did not provide a valid learning scheduler name!")
        return None
    return scheduler


def train(train_features, train_labels, val_features, val_labels, network, optimizer, loss, config, log_date, log_timestamp, lr_scheduler=None):
    """
    Method to train a PyTorch network.

    :param train_features: numpy array
        Training features
    :param train_labels: numpy array
        Training labels
    :param val_features: numpy array
        Validation features
    :param val_labels: numpy array
        Validation labels
    :param network: pytorch model
        DeepConvLSTM network object
    :param optimizer: optimizer object
        Optimizer object
    :param loss: loss object
        Loss object
    :param config: dict
        Config file which contains all training and hyperparameter settings
    :param log_date: string
        Date used for logging
    :param log_timestamp: string
        Timestamp used for logging
    :param lr_scheduler: scheduler object, default: None
        Learning rate scheduler object
    :return pytorch model, numpy array, numpy array
        Trained network and training and validation predictions with ground truth
    """

    # prints the number of learnable parameters in the network
    count_parameters(network)

    # init network using weight initialization of choice
    network = init_weights(network)
    # send network to GPU
    network.to(config['gpu'])
    network.train()

    # if weighted loss chosen, calculate weights based on training dataset; else each class is weighted equally
    if config['weighted']:
        class_weights = torch.from_numpy(
            compute_class_weight('balanced', classes=np.unique(train_labels + 1), y=train_labels + 1)).float()
        if config['loss'] == 'cross_entropy':
            loss.weight = class_weights.cuda()
        print('Applied weighted class weights: ')
        print(class_weights)
    else:
        class_weights = torch.from_numpy(
            compute_class_weight(None, classes=np.unique(train_labels + 1), y=train_labels + 1)).float()
        if config['loss'] == 'cross_entropy':
            loss.weight = class_weights.cuda()

    # initialize optimizer and loss
    opt, criterion = optimizer, loss

    if config['loss'] == 'maxup':
        maxup = Maxup(my_noise_addition_augmenter, ntrials=4)

    # initialize training and validation dataset, define DataLoaders
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))

    g = torch.Generator()
    g.manual_seed(config['seed'])

    trainloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=config['shuffling'],
                             worker_init_fn=seed_worker, generator=g, pin_memory=True)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
    valloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                           worker_init_fn=seed_worker, generator=g, pin_memory=True)

    # counters and objects used for early stopping and learning rate adjustment
    best_metric = 0.0
    best_network = None
    best_val_losses = None
    best_train_losses = None
    best_val_preds = None
    best_train_preds = None
    early_stop = False
    es_pt_counter = 0
    labels = list(range(0, config['nb_classes']))

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

        # iterate over train dataset
        for i, (x, y) in enumerate(trainloader):
            # send x and y to GPU
            inputs, targets = x.to(config['gpu']), y.to(config['gpu'])
            # zero accumulated gradients
            opt.zero_grad()

            if config['loss'] == 'maxup':
                # Increase the inputs via data augmentation
                inputs, targets = maxup(inputs, targets)

            # send inputs through network to get predictions, calculate loss and backpropagate
            train_output = network(inputs)

            if config['loss'] == 'maxup':
                # calculates loss
                train_loss = maxup.maxup_loss(train_output, targets.long())[0]
            else:
                train_loss = criterion(train_output, targets.long())

            train_loss.backward()
            opt.step()
            # append train loss to list
            train_losses.append(train_loss.item())

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

        # set network to eval mode
        network.eval()
        with torch.no_grad():
            # iterate over validation dataset
            for i, (x, y) in enumerate(valloader):
                # send x and y to GPU
                inputs, targets = x.to(config['gpu']), y.to(config['gpu'])

                if config['loss'] == 'maxup':
                    # Increase the inputs via data augmentation
                    inputs, targets = maxup(inputs, targets)

                # send inputs through network to get predictions, loss and calculate softmax probabilities
                val_output = network(inputs)
                if config['loss'] == 'maxup':
                    # calculates loss
                    val_loss = maxup.maxup_loss(val_output, targets.long())[0]
                else:
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
                  "\nTrain Loss: {:.4f}".format(np.mean(train_losses)),
                  "Train Acc (M): {:.4f}".format(jaccard_score(train_gt, train_preds, average='macro', labels=labels)),
                  "Train Prc (M): {:.4f}".format(precision_score(train_gt, train_preds, average='macro', labels=labels)),
                  "Train Rcl (M): {:.4f}".format(recall_score(train_gt, train_preds, average='macro', labels=labels)),
                  "Train F1 (M): {:.4f}".format(f1_score(train_gt, train_preds, average='macro', labels=labels)),
                  "Train Acc (W): {:.4f}".format(jaccard_score(train_gt, train_preds, average='weighted', labels=labels)),
                  "Train Prc (W): {:.4f}".format(precision_score(train_gt, train_preds, average='weighted', labels=labels)),
                  "Train Rcl (W): {:.4f}".format(recall_score(train_gt, train_preds, average='weighted', labels=labels)),
                  "Train F1 (W): {:.4f}".format(f1_score(train_gt, train_preds, average='weighted', labels=labels)),
                  "\nValid Loss: {:.4f}".format(np.mean(val_losses)),
                  "Valid Acc (M): {:.4f}".format(jaccard_score(val_gt, val_preds, average='macro', labels=labels)),
                  "Valid Prc (M): {:.4f}".format(precision_score(val_gt, val_preds, average='macro', labels=labels)),
                  "Valid Rcl (M): {:.4f}".format(recall_score(val_gt, val_preds, average='macro', labels=labels)),
                  "Valid F1 (M): {:.4f}".format(f1_score(val_gt, val_preds, average='macro', labels=labels)),
                  "Valid Acc (W): {:.4f}".format(jaccard_score(val_gt, val_preds, average='weighted', labels=labels)),
                  "Valid Prc (W): {:.4f}".format(precision_score(val_gt, val_preds, average='weighted', labels=labels)),
                  "Valid Rcl (W): {:.4f}".format(recall_score(val_gt, val_preds, average='weighted', labels=labels)),
                  "Valid F1 (W): {:.4f}".format(f1_score(val_gt, val_preds, average='weighted', labels=labels))
                  )

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

        # adjust learning rate if enabled
        if config['adj_lr']:
            if config['lr_scheduler'] == 'reduce_lr_on_plateau':
                lr_scheduler.step(np.mean(val_losses))
            else:
                lr_scheduler.step()

        # employ early stopping if employed
        metric = f1_score(val_gt, val_preds, average='macro')
        if best_metric >= metric:
            if config['early_stopping']:
                es_pt_counter += 1
                # early stopping check
                if es_pt_counter >= config['es_patience']:
                    print('Stopping training early since no loss improvement over {} epochs.'
                          .format(str(es_pt_counter)))
                    early_stop = True
                    # print results of best epoch
                    print('Final (best) results: ')
                    print("Train Loss: {:.4f}".format(np.mean(best_train_losses)),
                          "Train Acc: {:.4f}".format(jaccard_score(train_gt, best_train_preds, average='macro', labels=labels)),
                          "Train Prec: {:.4f}".format(precision_score(train_gt, best_train_preds, average='macro', labels=labels)),
                          "Train Rcll: {:.4f}".format(recall_score(train_gt, best_train_preds, average='macro', labels=labels)),
                          "Train F1: {:.4f}".format(f1_score(train_gt, best_train_preds, average='macro', labels=labels)),
                          "Val Loss: {:.4f}".format(np.mean(best_val_losses)),
                          "Val Acc: {:.4f}".format(jaccard_score(val_gt, best_val_preds, average='macro', labels=labels)),
                          "Val Prec: {:.4f}".format(precision_score(val_gt, best_val_preds, average='macro', labels=labels)),
                          "Val Rcll: {:.4f}".format(recall_score(val_gt, best_val_preds, average='macro', labels=labels)),
                          "Val F1: {:.4f}".format(f1_score(val_gt, best_val_preds, average='macro', labels=labels)))
        else:
            print(f"Performance improved... ({best_metric}->{metric})")
            if config['early_stopping']:
                es_pt_counter = 0
                best_train_losses = train_losses
                best_val_losses = val_losses
            best_metric = metric
            best_network = network
            checkpoint = {
                "model_state_dict": network.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "random_rnd_state": random.getstate(),
                "numpy_rnd_state": np.random.get_state(),
                "torch_rnd_state": torch.get_rng_state(),
            }
            best_train_preds = train_preds
            best_val_preds = val_preds

        # set network to train mode again
        network.train()

        if early_stop:
            break

    # if plot_gradient gradient plot is shown at end of training
    if config['save_gradient_plot']:
        if config['name']:
            plt.savefig(os.path.join(log_dir, 'grad_flow_{}.png'.format(config['name'])))
        else:
            plt.savefig(os.path.join(log_dir, 'grad_flow.png'))

    # return validation, train and test predictions as numpy array with ground truth
    if config['valid_epoch'] == 'best':
        return best_network, checkpoint, np.vstack((best_val_preds, val_gt)).T, \
               np.vstack((best_train_preds, train_gt)).T
    else:
        checkpoint = {
            "model_state_dict": network.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }
        return network, checkpoint, np.vstack((val_preds, val_gt)).T, np.vstack((train_preds, train_gt)).T


def predict(test_features, test_labels, network, config, log_date, log_timestamp):
    """
    Method that applies a trained network to obtain predictions on a test dataset. If selected, saves predictions.

    :param test_features: numpy array
        Test features
    :param test_labels: numpy array
        Test labels
    :param network: pytorch model
        Trained network object
    :param config: dict
        Config file which contains all training and hyperparameter settings
    :param log_date: string
        Date used for saving predictions
    :param log_timestamp: string
        Timestamp used for saving predictions
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
                            shuffle=False,
                            worker_init_fn=np.random.seed(int(config['seed']))
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

    labels = list(range(0, config['nb_classes']))
    print('\nTEST RESULTS: ')
    print("Avg. Accuracy: {0}".format(jaccard_score(test_gt, test_preds, average='macro', labels=labels)))
    print("Avg. Precision: {0}".format(precision_score(test_gt, test_preds, average='macro', labels=labels)))
    print("Avg. Recall: {0}".format(recall_score(test_gt, test_preds, average='macro', labels=labels)))
    print("Avg. F1: {0}".format(f1_score(test_gt, test_preds, average='macro', labels=labels)))

    print("\nTEST RESULTS (PER CLASS): ")
    print("Accuracy: {0}".format(jaccard_score(test_gt, test_preds, average=None, labels=labels)))
    print("Precision: {0}".format(precision_score(test_gt, test_preds, average=None, labels=labels)))
    print("Recall: {0}".format(recall_score(test_gt, test_preds, average=None, labels=labels)))
    print("F1: {0}".format(f1_score(test_gt, test_preds, average=None, labels=labels)))

    if config['save_test_preds']:
        if config['name']:
            np.save(os.path.join(log_dir, 'test_preds_{}.npy'.format(config['name'])), test_output.cpu().numpy())
        else:
            np.save(os.path.join(log_dir, 'test_preds.npy'), test_output.cpu().numpy())

    return test_preds, test_gt
