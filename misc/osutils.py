##################################################
# Helper functions (os-related)
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

from __future__ import absolute_import
import os
import errno


def mkdir_if_missing(dir_path):
    """
    Function that creates directory path if missing

    :param dir_path: string
        Path to be created
    """
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
