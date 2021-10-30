#!/usr/bin/env python

""" Wrapper to all generation functions for PhotonID, TOD, TOF, TEP, Radiometry, 
and DuplicateMask.

Will generate them all in one go, with the ATL01, ATL02, ANC, and CAL
files available at the path specified by the user.  If the user wishes to
customize which files are used (if they are split between directories),
the user will need run the generation script for each component seperately.

Creates a verification "vfile".

To Do:

    Automate adding entry to outfile_log.csv?
"""

import argparse
import glob
import h5py
import os

from datetime import datetime

from atl02v.shared.paths import path_to_data, path_to_outputs
from atl02v.shared.tools import make_file_dir, pickle_in, pickle_out
from atl02v.verification.verification_tools import VerificationContainer as VC

from gen_photonid import generate as gen_photonid
from gen_duplicatemask import generate as gen_dupmask 
from gen_tod import generate as gen_tod
from gen_tof import generate as gen_tof
from gen_radiometry import generate as gen_rad


def generate_all(path_in, vfile_descr=None):
    """
    Currently the user can only customize the base path where the ATL01, ATL02,
    ANC, and CAL files are. They must all be in the same base directory in 
    data/!
    """
    # Begin timer.
    start_time = datetime.now()
    print("--GEN_ALL start time: {}--".format(start_time))

    atl01_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL01_*.h5'))[0]
    atl02_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL02_*.h5'))[0]  

    # Generate PhotonID H5
    print(" ")
    print("--Start PHOTONID--")
    photonid_filename = gen_photonid(path_in)
    print("--End PHOTONID with {}".format(photonid_filename))

    # Generate TOD pickle
    print(" ")
    print("--Start TOD--")
    tod_filename = gen_tod(path_in)
    print("End TOD with {}".format(tod_filename))

    # Generate TOF pickle
    print(" ")
    print("--Start TOF--")
    tof_filename = gen_tof(path_in, tod_pickle=tod_filename)
    print("End TOF with {}".format(tof_filename))

    # Generate TEP pickle
    print(" ")
    print("--Start TEP--")
    tep_filename = gen_tep(path_in, tod_pickle=tod_filename, tof_pickle=tof_filename)
    print("End TEP with {}".format(tep_filename))

    # Generate DuplicateMask H5
    print(" ")
    print("--Start DUPLICATEMASK--")
    dupmask_filename = gen_dupmask(path_in, tof_pickle=tof_filename)
    print("End DUPLICATEMASK with {}".format(dupmask_filename))

    # Generate Radiometry pickle
    print(" ")
    print("--Start RADIOMETRY--")
    rad_filename = gen_rad(path_in)
    print("End RADIOMETRY with {}".format(rad_filename))

    # Create a vfile.
    if vfile_descr==None:
        vfile_descr='GEN_ALL.py files created beginning on {}'.format(start_time)
    vc = VC(descr=vfile_descr,
            atl01=atl01_file,
            atl02=atl02_file,
            photonids ='/'.join(photonid_filename.split('/')[-2:]),
            tod='/'.join(tod_filename.split('/')[-2:]),
            tof='/'.join(tof_filename.split('/')[-2:]),
            tep='/'.join(tep_filename.split('/')[-2:]),
            dupmask='/'.join(dupmask_filename.split('/')[-2:]),
            radiometry='/'.join(rad_filename.split('/')[-2:]))
    vfile = vc.write()
    print("--Created vfile: {}".format(vfile))

    run_time = datetime.now() - start_time
    print(" ")
    print("--GEN_ALL run time: {}--".format(run_time))


def parse_args():
    """ Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--p', dest='path_in',
        action='store', type=str, required=True, default='',
        help="Path relative to the data/ directory, to the input ATL01 and ANC27 files.")
    parser.add_argument('--vd', dest='vfile_descr',
        action='store', type=str, required=False, default=None,
        help="Optional description of the generated dataset to be included in the output vfile.")    

    args = parser.parse_args()
     
    return args

if __name__ == '__main__':
    args = parse_args()

    print("path_in={}".format(args.path_in))
    print("vfile_descr={}".format(args.vfile_descr))

    generate_all(path_in=args.path_in, vfile_descr=args.vfile_descr)


