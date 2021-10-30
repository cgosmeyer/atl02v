#!/usr/bin/env python

""" Verifying datasets in /atlas/tx_pulse_width

Author:

    C.M. Gosmeyer

Notes:

    Add command line args?

"""

import argparse
import h5py
import numpy as np
import os
from pydl import uniq
from atl02v.shared.paths import path_to_outputs
from atl02v.shared.tools import make_file_dir, make_timestamp_dir, pickle_out
from atl02v.verification.atlas_tx_pulse_width_verification import VerifyTxPulseWidth
from atl02v.verification.verification_tools import VFileReader


def verify_tx_pulse_width(vfiles, path_out):
    """ Function for verifying datasets in tx_pulse_width
    """
    tof = pickle_out(vfiles.tof)

    def remove_nan(arr):
        nan_idx = np.where(np.isnan(arr))
        new_arr = np.delete(arr, nan_idx)
        return new_arr

    def match_pulse_skew(arr):
        """ Because pulse skew ends up being the correct length when NANs removed, 
        make the others match...

        Should I have NANed out all the same indices in the first place when 
        I calculated them in the TOF class?
        """
        nan_idx = np.where(np.isnan(tof.pulse_skew))
        new_arr = np.delete(arr, nan_idx)
        return new_arr

    d = {'tx_pulse_skew_est' : [1e-11, (lambda : remove_nan(tof.pulse_skew)), 'calculation'],
         'tx_pulse_width_lower' : [1e-10, (lambda : match_pulse_skew(tof.pulse_width_lower)), 'calculation'],
         'tx_pulse_width_upper' : [1e-10, (lambda : match_pulse_skew(tof.pulse_width_upper)), 'calculation']
        }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyTxPulseWidth(vfiles, tolerance, atl02_dataset, path_out).do_verify(custom_func)


def parse_args():
    """ Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--p', dest='path_out',
        action='store', type=str, required=False, default='',
        help="Overrides the default path to which to write output files.")
    parser.add_argument('--v', dest='vfiles',
        action='store', type=str, required=True,
        help="Name of the verification CSV file.") 
    
    args = parser.parse_args()
     
    return args

if __name__=="__main__":
    args = parse_args()

    print("path_out={}".format(args.path_out))
    print("vfiles={}".format(args.vfiles))

    vfiles = VFileReader(vfile_name=args.vfiles).read()

    if args.path_out == '':
        # Default setting
        path_out = make_timestamp_dir(make_file_dir(os.path.join(path_to_outputs, 'reports'), vfiles.atl02))
    else:
        path_out = make_timestamp_dir(make_file_dir(args.path_out, vfiles.atl02))

    verify_tx_pulse_width(vfiles=vfiles, path_out=path_out)
