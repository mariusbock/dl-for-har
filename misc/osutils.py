from __future__ import absolute_import
import os
import errno

import numpy as np


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print('Assigning workflow to GPU: ' + str(np.argmax(memory_available)))
    return np.argmax(memory_available)
