#!/usr/bin/env python

"""
Generates PhotonID H5 file.
"""

import argparse
import glob
import h5py
import os
from atl02v.shared.photonid import PhotonID, save_ids
from atl02v.shared.paths import path_to_data, path_to_outputs
from atl02v.shared.tools import make_file_dir


def generate(path_in):
    """
    """
    atl01_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL01_*.h5'))[0]
    atl02_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL02_*.h5'))[0]  

    out_filename = save_ids(atl01_file, atl02_file, path_out=make_file_dir(os.path.join(path_to_outputs, 'data'), atl02_file))

    return out_filename


def parse_args():
    """ Parses command line arguments.

    No option to designate ATL01 and ATL02 separately because that would make
    the foundational assumption behind the PhotonID class void.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--p', dest='path_in',
        action='store', type=str, required=True, default='',
        help="Path relative to the data/ directory, to the input ATL01 and ATL02 files.")  

    args = parser.parse_args()
     
    return args

if __name__=='__main__':
    args = parse_args()

    generate(path_in=args.path_in)