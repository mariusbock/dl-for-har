import math
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.utils import class_weight
from torch import nn
from torch.utils.data import DataLoader

from model.DeepConvLSTM import ConvBlock, ConvBlockSkip, ConvBlockFixup


def init_weights(network):
    """
    Weight initialization of network (initialises all LSTM, Conv2D and Linear layers according to weight_init parameter
    of network.

    :param network: network of which weights are to be initialised
    :return: network with initialised weights
    """
    for m in network.modules():
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
        elif isinstance(m, ConvBlockFixup):
            nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(
                2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * network.nb_conv_blocks ** (-0.5))
            nn.init.constant_(m.conv2.weight, 0)
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


def plot_grad_flow(named_parameters):
    """
    Funtion which plots the average gradient of a network.

    :param named_parameters: parameters of the network (used to obtain gradient)
    :return: plot containing the plotted average gradient
    """
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def init_optimizer_and_loss(network, optimizer, loss, lr, weight_decay, class_weights, gpu_name):
    """
    Initialises an optimizer and loss object for a given network.

    :param network: network for which optimizer and loss are to be initialised
    :param optimizer: type of optimizer to initialise (choose between 'adadelta' 'adam' or 'rmsprop')
    :param loss: type of loss to initialise (currently only 'cross-entropy' supported)
    :param lr: learning rate employed in optimizer
    :param weight_decay: weight decay employed in optimizer
    :param class_weights: class weights array to use during CEE loss calculation
    :param gpu_name: name of the gpu which optimizer and loss are to be transferred to
    :return: optimizer and loss object
    """
    # define optimizer and loss
    if optimizer == 'adadelta':
        opt = torch.optim.Adadelta(network.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'adam':
        opt = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(network.parameters(), lr=lr, weight_decay=weight_decay)
    if loss == 'cross-entropy':
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(gpu_name))
    return opt, criterion


def adjust_lr(opt, lr_pt_counter, es_pt_counter, best_loss, curr_loss, args):
    """
    Function to adjust learning rate inbetween epochs.

    Args:
        opt -- update parameters of optimizer
        epoch -- epoch number
        args -- train arguments
    """
    if best_loss < curr_loss:
        lr_pt_counter += 1
        es_pt_counter += 1
        if lr_pt_counter > args['adj_lr_patience'] and args['adj_lr']:
            args['lr'] *= 0.1
            print('Changing learning rate to {} since no loss improvement over {} epochs.'.format(args['lr'], str(lr_pt_counter)))
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
        if es_pt_counter > args['es_patience'] and args['early_stopping']:
            print('Stopping training early since no loss improvement over {} epochs.'.format(str(es_pt_counter)))
            return best_loss, lr_pt_counter, es_pt_counter, False, True
        return best_loss, lr_pt_counter, es_pt_counter, False, False
    else:
        lr_pt_counter = 0
        es_pt_counter = 0
        best_loss = curr_loss
        return best_loss, lr_pt_counter, es_pt_counter, True, False


def train(train_features, train_labels, val_features, val_labels, network, config, cw=None):
    """
    Method to train a PyTorch network.

    :param train_features: training features
    :param train_labels: training labels
    :param val_features: validation features
    :param val_labels: validation labels
    :param network: DeepConvLSTM network object
    :param config: config file which contains all training and hyperparameter settings; these include:
        - epochs: number of epochs used during training
        - batch_size: employed batch size
        - optimizer: employed optimizer (choose between 'adadelta' 'adam' or 'rmsprop')
        - loss: employed loss (currently only 'cross-entropy' supported)
        - lr: employed learning rate
        - weight_decay: employed weight decay
        - class_weights: class weights used to calculate CE loss
        - gpu: name of the GPU to use for training/ prediction
        - verbose: boolean whether to print losses within epochs
        - print_freq: frequency (no. batches) in which losses are provided within epochs
        - print_counts: boolean whether to print predicted classes for train and test dataset
        - plot_gradient: boolean whether to print gradient
    :return: numpy array containing (predictions, gt labels)
    """
    print("Number of Parameters: ")
    print(sum(p.numel() for p in network.parameters()))
    network = init_weights(network)
    network.to(config['gpu'])
    network.train()
    if config['use_weights']:
        if cw is None:
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels + 1), y=train_labels + 1)
            print('Applied weighted class weights: ')
            print(class_weights)
        else:
            class_weights = cw
            print('Applied weighted class weights: ')
            print(class_weights)
    else:
        class_weights = class_weight.compute_class_weight(None, classes=np.unique(train_labels + 1), y=train_labels + 1)

    opt, criterion = init_optimizer_and_loss(network, config['optimizer'], config['loss'], config['lr'],
                                             config['weight_decay'], class_weights, config['gpu'])
    best_loss = 999999
    best_val_losses = None
    best_train_losses = None
    best_preds = None
    lr_pt_counter = 0
    es_pt_counter = 0
    # iterate through number of epochs
    for e in range(config['epochs']):
        train_losses = []
        start_time = time.time()
        batch_num = 1
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))

        trainloader = DataLoader(dataset,
                                 batch_size=config['batch_size'],
                                 num_workers=2,
                                 shuffle=False,
                                 )
        for i, (x, y) in enumerate(trainloader):
            inputs, targets = x.to(config['gpu']), y.to(config['gpu'])
            # zero accumulated gradients
            opt.zero_grad()
            output = network(inputs)
            loss = criterion(output, targets.long())
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            if config['verbose']:
                if batch_num % config['print_freq'] == 0 and batch_num > 0:
                    cur_loss = np.mean(train_losses)
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d} batches | ms/batch {:5.2f} | '
                          'train loss {:5.2f}'.format(e, batch_num, elapsed * 1000 / config['batch_size'], cur_loss))
                    start_time = time.time()
                batch_num += 1
            if config['plot_gradient']:
                plot_grad_flow(network.named_parameters())

        """
        VALIDATION
        """
        # initialise hidden state for validation
        val_preds = []
        val_gt = []
        val_losses = []
        train_preds = []
        train_gt = []

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features).float(),
                                                 torch.from_numpy(val_labels))
        valloader = DataLoader(dataset,
                               batch_size=config['batch_size'],
                               num_workers=2,
                               shuffle=False,
                               )

        # set network to eval mode
        network.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(valloader):
                inputs, targets = x.to(config['gpu']), y.to(config['gpu'])
                val_output = network(inputs)
                val_loss = criterion(val_output, targets.long())
                if math.isnan(val_loss):
                    print(val_output)
                    print(targets.long())
                val_losses.append(val_loss.item())
                val_output = torch.nn.functional.softmax(val_output, dim=1)
                y_preds = np.argmax(val_output.cpu().numpy(), axis=-1)
                y_true = targets.cpu().numpy().flatten()
                val_preds = np.concatenate((np.array(val_preds, int), np.array(y_preds, int)))
                val_gt = np.concatenate((np.array(val_gt, int), np.array(y_true, int)))
            for i, (x, y) in enumerate(trainloader):
                inputs, targets = x.to(config['gpu']), y.to(config['gpu'])
                train_output = network(inputs)
                train_output = torch.nn.functional.softmax(train_output, dim=1)
                y_preds = np.argmax(train_output.cpu().numpy(), axis=-1)
                y_true = targets.cpu().numpy().flatten()
                train_preds = np.concatenate((np.array(train_preds, int), np.array(y_preds, int)))
                train_gt = np.concatenate((np.array(train_gt, int), np.array(y_true, int)))
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

            if config['print_counts']:
                y_train = np.bincount(train_preds)
                ii_train = np.nonzero(y_train)[0]
                y_val = np.bincount(val_preds)
                ii_val = np.nonzero(y_val)[0]
                print('Predicted Train Labels: ')
                print(np.vstack((ii_train, y_train[ii_train])).T)
                print('Predicted Val Labels: ')
                print(np.vstack((ii_val, y_val[ii_val])).T)

        network.train()
        if config['adj_lr'] or config['early_stopping']:
            best_loss, lr_pt_counter, es_pt_counter, improvement, stop = adjust_lr(opt, lr_pt_counter, es_pt_counter, best_loss, np.mean(val_losses), config)
            if improvement:
                best_train_losses = train_losses
                best_val_losses = val_losses
                best_preds = val_preds
            if stop:
                return best_train_losses, best_val_losses, np.vstack((best_preds, val_gt)).T
    # if plot_gradient gradient plot is shown at end of training
    if config['plot_gradient']:
        plt.show()

    return train_losses, val_losses, np.vstack((val_preds, val_gt)).T
