#!/usr/bin/env python

""" Generates TOD pickle file.
"""

import argparse
import glob
import os
from atl02v.tod.tod import TOD
from atl02v.shared.paths import path_to_data, path_to_outputs
from atl02v.shared.tools import make_file_dir, pickle_in
from gen_tof import get_size


def generate(path_in, atl01_file=None, anc13_path=None, anc27_path=None):
    """
    """
    if atl01_file == None:
        atl01_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL01_*.h5'))[0]
    if anc13_path == None:
        anc13_path = os.path.join(path_to_data, path_in)
    if anc27_path == None:
        anc27_path = os.path.join(path_to_data, path_in)
    atl02_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL02_*.h5'))[0]

    tod = TOD(atl01_file=atl01_file, anc13_path=anc13_path, anc27_path=anc27_path, 
              verbose=False, mf_limit=None)

    #s = get_size(tod)
    #print("TOD size: {} bytes".format(s))

    out_filename = pickle_in(tod, out_location=make_file_dir(os.path.join(path_to_outputs, 'data'), atl02_file))

    return out_filename

def parse_args():
    """ Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--p', dest='path_in',
        action='store', type=str, required=True, default='',
        help="Path relative to the data/ directory, to the input ATL01, ANC13, and ANC27 files.")
    parser.add_argument('--atl01', dest='atl01_file',
        action='store', type=str, required=False, default=None,
        help="Path + filename to directory of the ATL01.")
    parser.add_argument('--anc13', dest='anc13_path',
        action='store', type=str, required=False, default=None,
        help="Path to outputs directory of the ANC13.")    
    parser.add_argument('--anc27', dest='anc27_path',
        action='store', type=str, required=False, default=None,
        help="Path to directory of the ANC27.")

    args = parser.parse_args()
     
    return args

if __name__ == '__main__':
    args = parse_args()

    print("path_in={}".format(args.path_in))
    print("atl01_file={}".format(args.atl01_file))
    print("anc13_path={}".format(args.anc13_path))
    print("anc27_path={}".format(args.anc27_path))

    ## Make the required option path to all the files.
    ## Non-required could be to specify individually a file (like ANC13) which would override the required path name
    generate(path_in=args.path_in, atl01_file=args.atl01_file, 
        anc13_path=args.anc13_path, anc27_path=args.anc27_path)


