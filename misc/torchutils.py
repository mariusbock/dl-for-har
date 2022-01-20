##################################################
# Helper functions (pytorch-related)
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import os
import random
import numpy as np
import torch
from prettytable import PrettyTable


def seed_torch(seed):
    """
    Function which seeds torch and all related random aspects of a python distribution

    :param seed: int
        Random seed to employ
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.use_deterministic_algorithms(True)


def count_parameters(model):
    """
    Function which prints the amount of learnable parameters per network element + total

    :param model: pytorch model
        Model to be analysed
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_learn_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        table.add_row([name, param])
        total_learn_params += param
    print(table)
    print(f"Total Params: {total_learn_params}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
