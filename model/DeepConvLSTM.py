##################################################
# Pytorch implementation (with extensions) of the DeepConvLSTM as proposed by by Ordonez and Roggen (https://www.mdpi.com/1424-8220/16/1/115)
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

from torch import nn
import torch
import warnings

warnings.filterwarnings('ignore')


class ConvBlockFixup(nn.Module):
    """
    Fixup convolution block
    """
    def __init__(self, filter_width, input_filters, nb_filters, dilation):
        super(ConvBlockFixup, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1),
                               bias=False, padding='same')
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1),
                               bias=False, padding='same')
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        out += identity
        out = self.relu(out)

        return out


class ConvBlockSkip(nn.Module):
    """
    Convolution block with skip connection
    """
    def __init__(self, window_size, filter_width, input_filters, nb_filters, dilation, batch_norm):
        super(ConvBlockSkip, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1), padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1), padding='same')
        self.seq_len = window_size - (filter_width + 1) * 2
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_filters)
            self.norm2 = nn.BatchNorm2d(self.nb_filters)

    def forward(self, x):
        identity = x
        if self.batch_norm:
            out = self.conv1(x)
            out = self.relu(out)
            out = self.norm1(out)
            out = self.conv2(out)
            out += identity
            out = self.relu(out)
            out = self.norm2(out)
        else:
            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)
            out += identity
            out = self.relu(out)
        return out


class ConvBlock(nn.Module):
    """
    Normal convolution block
    """
    def __init__(self, filter_width, input_filters, nb_filters, dilation, batch_norm):
        super(ConvBlock, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1))
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_filters)
            self.norm2 = nn.BatchNorm2d(self.nb_filters)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm1(out)
        out = self.conv2(out)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm2(out)
        return out


class DeepConvLSTM(nn.Module):
    def __init__(self, config):
        """
        DeepConvLSTM model based on architecture suggested by Ordonez and Roggen (https://www.mdpi.com/1424-8220/16/1/115)

        :param config: config dictionary containing all settings needed to initialize DeepConvLSTM; these include:
            - no_lstm:              whether not to use an lstm
            - pooling:              whether to use pooling layer
            - reduce_layer:         whether to use reduce layer
            - reduce_layer_output:  size of output of reduce layer
            - pool_type:            type of pooling
            - pool_kernel_width:    width of pooling kernel
            - window_size:          number of samples contained in each sliding window
            - final_seq_len:        length of the sequence after the applying each convolution layer
            - nb_channels:          number of sensor channels used (i.e. number of features)
            - nb_classes:           number of classes which are to be predicted (e.g. 2 if binary classification problem)
            - nb_units_lstm:        number of units within each hidden layer of the LSTM
            - nb_layers_lstm:       number of hidden layers of the LSTM
            - nb_conv_blocks:       number of convolution blocks
            - conv_block_type:      type of convolution blocks used
            - nb_filters:           number of filters employed in convolution blocks
            - filter_width:         size of the filters (1 x filter_width) applied within the convolution layer
            - dilation:             dilation factor for convolutions
            - batch_norm:           whether to use batch normalization
            - drop_prob:            dropout probability employed after each dense layer (e.g. 0.5 for 50% dropout probability)
            - weights_init:         type of weight initialization used
            - seed:                 random seed employed to ensure reproducibility of results
        """
        super(DeepConvLSTM, self).__init__()
        # parameters
        self.no_lstm = config['no_lstm']
        self.pooling = config['pooling']
        self.reduce_layer = config['reduce_layer']
        self.reduce_layer_output = config['reduce_layer_output']
        self.pool_type = config['pool_type']
        self.pool_kernel_width = config['pool_kernel_width']
        self.window_size = config['window_size']
        self.drop_prob = config['drop_prob']
        self.nb_channels = config['nb_channels']
        self.nb_classes = config['nb_classes']
        self.weights_init = config['weights_init']
        self.seed = config['seed']
        # convolution settings
        self.nb_conv_blocks = config['nb_conv_blocks']
        self.conv_block_type = config['conv_block_type']
        if self.conv_block_type == 'fixup':
            self.use_fixup = True
        else:
            self.use_fixup = False
        self.nb_filters = config['nb_filters']
        self.filter_width = config['filter_width']
        self.dilation = config['dilation']
        self.batch_norm = config['batch_norm']
        # lstm settings
        self.nb_units_lstm = config['nb_units_lstm']
        self.nb_layers_lstm = config['nb_layers_lstm']

        # define conv layers
        self.conv_blocks = []
        for i in range(self.nb_conv_blocks):
            if i == 0:
                input_filters = 1
            else:
                input_filters = self.nb_filters
            if self.conv_block_type == 'fixup':
                self.conv_blocks.append(ConvBlockFixup(self.filter_width, input_filters, self.nb_filters, self.dilation))
            elif self.conv_block_type == 'skip':
                self.conv_blocks.append(
                    ConvBlockSkip(self.window_size, self.filter_width, input_filters, self.nb_filters, self.dilation,
                                  self.batch_norm))
            elif self.conv_block_type == 'normal':
                self.conv_blocks.append(
                    ConvBlock(self.filter_width, input_filters, self.nb_filters, self.dilation, self.batch_norm))
        self.conv_blocks = nn.ModuleList(self.conv_blocks)
        # define max pool layer
        if self.pooling:
            if self.pool_type == 'max':
                self.pool = nn.MaxPool2d((self.pool_kernel_width, 1))
            elif self.pool_type == 'avg':
                self.pool = nn.AvgPool2d((self.pool_kernel_width, 1))
        if self.reduce_layer:
            self.reduce = nn.Conv2d(self.nb_filters, self.reduce_layer_output, (self.filter_width, 1))
        self.final_seq_len = self.window_size - (self.filter_width - 1) * (self.nb_conv_blocks * 2)
        # define lstm layers
        if not self.no_lstm:
            self.lstm_layers = []
            for i in range(self.nb_layers_lstm):
                if i == 0:
                    if self.reduce_layer:
                        self.lstm_layers.append(nn.LSTM(self.nb_channels * self.reduce_layer_output, self.nb_units_lstm))
                    else:
                        self.lstm_layers.append(nn.LSTM(self.nb_channels * self.nb_filters, self.nb_units_lstm))
                else:
                    self.lstm_layers.append(nn.LSTM(self.nb_units_lstm, self.nb_units_lstm))
            self.lstm_layers = nn.ModuleList(self.lstm_layers)
        # define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)
        # define classifier
        if self.no_lstm:
            if self.reduce_layer:
                self.fc = nn.Linear(self.reduce_layer_output * self.nb_channels, self.nb_classes)
            else:
                self.fc = nn.Linear(self.nb_filters * self.nb_channels, self.nb_classes)
        else:
            self.fc = nn.Linear(self.nb_units_lstm, self.nb_classes)

    def forward(self, x):
        # reshape data for convolutions
        x = x.view(-1, 1, self.window_size, self.nb_channels)
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
        if self.pooling:
            x = self.pool(x)
            self.final_seq_len = x.shape[2]
        if self.reduce_layer:
            x = self.reduce(x)
            self.final_seq_len = x.shape[2]
        # permute dimensions and reshape for LSTM
        x = x.permute(0, 2, 1, 3)
        if self.reduce_layer:
            x = x.reshape(-1, self.final_seq_len, self.nb_channels * self.reduce_layer_output)
        else:
            x = x.reshape(-1, self.final_seq_len, self.nb_filters * self.nb_channels)
        if self.no_lstm:
            if self.reduce_layer:
                x = x.view(-1, self.nb_channels * self.reduce_layer_output)
            else:
                x = x.view(-1, self.nb_filters * self.nb_channels)
        else:
            for lstm_layer in self.lstm_layers:
                x, _ = lstm_layer(x)
            # reshape data for classifier
            x = x.view(-1, self.nb_units_lstm)
        x = self.dropout(x)
        x = self.fc(x)
        # reshape data and return predicted label of last sample within final sequence (determines label of window)
        out = x.view(-1, self.final_seq_len, self.nb_classes)

        return out[:, -1, :]

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
