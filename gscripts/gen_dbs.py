#!/usr/bin/env python

""" Generates downlink band-related fields such as channel mask.
"""

import argparse
import glob
import h5py
import os
from atl02v.shared.downlink_bands import DownlinkBands
from atl02v.shared.paths import path_to_data, path_to_outputs
from atl02v.shared.tools import make_file_dir, pickle_in


def generate(path_in, tof_pickle):
    """
    """
    atl01_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL01_*.h5'))[0]

    if tof_pickle != None:
        tof_pickle = os.path.join(path_to_outputs, 'data', tof_pickle)   

    atl02_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL02_*.h5'))[0]

    dlbs = DownlinkBands(atl01_file, tof_pickle=tof_pickle, verbose=True, very_verbose=False)

    out_filename = pickle_in(dlbs, out_location=make_file_dir(os.path.join(path_to_outputs, 'data'), atl02_file))

    return out_filename

def parse_args():
    """ Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--p', dest='path_in',
        action='store', type=str, required=True, default='',
        help="Path relative to the data/ directory, to the input ATL01 file.")  
    parser.add_argument('--tof', dest='tof_pickle',
        action='store', type=str, required=False, default=None,
        help="Path + filename, relative to the outputs/data/ directory, to the TOF pickle.")    

    args = parser.parse_args()
     
    return args

if __name__=='__main__':
    args = parse_args()

    generate(path_in=args.path_in, tof_pickle=args.tof_pickle)