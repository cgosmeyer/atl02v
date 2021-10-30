#!/usr/bin/env python

""" Generates Radiometry pickle file.
"""

import argparse
import glob
import h5py
import os

from atl02v.radiometry.radiometry import Radiometry
from atl02v.shared.paths import path_to_data, path_to_outputs
from atl02v.shared.tools import make_file_dir, pickle_in, pickle_out


def generate(path_in, atl01_file=None, atl02_file=None, 
        anc13_path=None, anc27_path=None, cal30_path=None,
        cal45_path=None, cal46_path=None, cal47_path=None,
        cal54_path=None, cal61_path=None):
    # The ATL01, ANC, and CAL files have the option to be individually specified.
    if atl01_file == None:
        atl01_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL01_*.h5'))[0]
    if atl02_file == None:
        atl02_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL02_*.h5'))[0]
    if anc13_path == None:
        anc13_path = os.path.join(path_to_data, path_in)
    if anc27_path == None:
        anc27_path = os.path.join(path_to_data, path_in)

    if cal30_path == None:
        cal30_path = os.path.join(path_to_data, path_in, 'CAL30')
    if cal45_path == None:
        cal45_path = os.path.join(path_to_data, path_in, 'CAL45')
    if cal46_path == None:
        cal46_path = os.path.join(path_to_data, path_in, 'CAL46')
    if cal47_path == None:
        cal47_path = os.path.join(path_to_data, path_in, 'CAL47')
    if cal54_path == None:
        cal54_path = os.path.join(path_to_data, path_in, 'CAL54')
    if cal61_path == None:
        cal61_path = os.path.join(path_to_data, path_in, 'CAL61')

    radiometry = Radiometry(
        atl01_file=atl01_file, 
        atl02_file=atl02_file,
        anc13_path=anc13_path, 
        anc27_path=anc27_path, 
        cal30_path=cal30_path, 
        cal45_path=cal45_path, 
        cal46_path=cal46_path, 
        cal47_path=cal47_path,
        cal54_path=cal54_path, 
        cal61_path=cal61_path,
        verbose=True)

    out_filename = pickle_in(radiometry, out_location=make_file_dir(os.path.join(path_to_outputs, 'data'), atl02_file))

    return out_filename

def parse_args():
    """ Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--p', dest='path_in',
        action='store', type=str, required=True, default='',
        help="Path relative to the data/ directory, to the input ATL01 and ANC27 files.")

    parser.add_argument('--atl01', dest='atl01_file',
        action='store', type=str, required=False, default=None,
        help="Path + filename to directory of the ATL01.")
    parser.add_argument('--atl02', dest='atl02_file',
        action='store', type=str, required=False, default=None,
        help="Path + filename to directory of the ATL02.")
    parser.add_argument('--anc13', dest='anc13_path',
        action='store', type=str, required=False, default=None,
        help="Path to directory of the ANC13.")
    parser.add_argument('--anc27', dest='anc27_path',
        action='store', type=str, required=False, default=None,
        help="Path to directory of the ANC27.")
    parser.add_argument('--cal30', dest='cal30_path',
        action='store', type=str, required=False, default=None,
        help="Path to directory of the CAL30 folder.")
    parser.add_argument('--cal45', dest='cal45_path',
        action='store', type=str, required=False, default=None,
        help="Path to directory of the CAL45 folder.")
    parser.add_argument('--cal46', dest='cal46_path',
        action='store', type=str, required=False, default=None,
        help="Path to directory of the CAL46 folder.")
    parser.add_argument('--cal47', dest='cal47_path',
        action='store', type=str, required=False, default=None,
        help="Path to directory of the CAL47 folder.")
    parser.add_argument('--cal54', dest='cal54_path',
        action='store', type=str, required=False, default=None,
        help="Path to directory of the CAL54 folder.")
    parser.add_argument('--cal61', dest='cal61_path',
        action='store', type=str, required=False, default=None,
        help="Path to directory of the CAL61 folder.")

    args = parser.parse_args()
     
    return args

if __name__ == '__main__':
    args = parse_args()

    print("path_in={}".format(args.path_in))
    print("atl01_file={}".format(args.atl01_file))
    print("atl02_file={}".format(args.atl02_file))
    print("anc13_path={}".format(args.anc13_path))
    print("anc27_path={}".format(args.anc27_path))

    print("cal30_path={}".format(args.cal30_path))
    print("cal45_path={}".format(args.cal45_path))
    print("cal46_path={}".format(args.cal46_path))
    print("cal47_path={}".format(args.cal47_path))
    print("cal54_path={}".format(args.cal54_path))
    print("cal61_path={}".format(args.cal61_path))

    ## Make the required option path to all the files.
    ## Non-required could be to specify individually a file (like ANC13) which would override the required path name
    generate(path_in=args.path_in, 
        atl01_file=args.atl01_file, atl02_file=args.atl02_file, 
        anc13_path=args.anc13_path, anc27_path=args.anc27_path,
        cal30_path=args.cal30_path, cal45_path=args.cal45_path,
        cal46_path=args.cal46_path, cal47_path=args.cal47_path,
        cal54_path=args.cal54_path, cal61_path=args.cal61_path)

