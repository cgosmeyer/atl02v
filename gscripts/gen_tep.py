#!/usr/bin/env python

""" Generates TEP pickle file.
"""

import argparse
import glob
import h5py
import os
from atl02v.tep.tep import TEP
from atl02v.shared.paths import path_to_data, path_to_outputs
from atl02v.shared.tools import make_file_dir, pickle_in, pickle_out


def generate(path_in, tod_pickle, tof_pickle, anc27_path=None, 
    atl01_file=None):
    """
    """
    # The ATL01 and ANC files have the option to be individually specified.
    if atl01_file == None:
        atl01_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL01_*.h5'))[0]
    if anc27_path == None:
        anc27_path = os.path.join(path_to_data, path_in)
    atl02_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL02_*.h5'))[0]

    # The TOD and TOF pickles must be specified.
    tod_pickle = os.path.join(path_to_outputs, 'data', tod_pickle)
    tof_pickle = os.path.join(path_to_outputs, 'data', tof_pickle)

    tep = TEP(atl01_file=atl01_file, anc27_path=anc27_path, tod_pickle=tod_pickle, tof_pickle=tof_pickle, 
        verbose=True, very_verbose=False, multiprocess=False)

    out_filename = pickle_in(tep, out_location=make_file_dir(os.path.join(path_to_outputs, 'data'), atl02_file))

    return out_filename

def parse_args():
    """ Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--p', dest='path_in',
        action='store', type=str, required=True, default='',
        help="Path relative to the data/ directory, to the input ATL01 and ANC27 files.")

    parser.add_argument('--tod', dest='tod_pickle',
        action='store', type=str, required=True,
        help="Path + filename, relative to the outputs/data/ directory, to the TOD pickle.")
    parser.add_argument('--tof', dest='tof_pickle',
        action='store', type=str, required=True,
        help="Path + filename, relative to the outputs/data/ directory, to the TOF pickle.")  
    parser.add_argument('--anc27', dest='anc27_path',
        action='store', type=str, required=False, default=None,
        help="Path to directory of the ANC27.")
    parser.add_argument('--atl01', dest='atl01_file',
        action='store', type=str, required=False, default=None,
        help="Path + filename to directory of the ATL01.")

    args = parser.parse_args()
     
    return args

if __name__ == '__main__':
    args = parse_args()

    print("path_in={}".format(args.path_in))
    print("tod_pickle={}".format(args.tod_pickle))
    print("tof_pickle={}".format(args.tof_pickle))
    print("anc27_path={}".format(args.anc27_path))
    print("atl01_file={}".format(args.atl01_file))

    ## Make the required option path to all the files.
    ## Non-required could be to specify individually a file (like ANC13) which would override the required path name
    generate(path_in=args.path_in, tod_pickle=args.tod_pickle,
        tof_pickle=args.tof_pickle, anc27_path=args.anc27_path,
        atl01_file=args.atl01_file)
