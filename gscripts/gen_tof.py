#!/usr/bin/env python

""" Generates TOF pickle file.
"""

import argparse
import glob
import h5py
import os
import sys
from atl02v.shared.paths import path_to_data, path_to_outputs
from atl02v.shared.tools import make_file_dir, pickle_in, pickle_out
from atl02v.tof.tof import TOF


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def generate(path_in, tod_pickle, atl01_file=None, atl02_file=None, 
        anc13_path=None, anc27_path=None, cal17_path=None, 
        cal44_path=None, cal49_path=None):

    # The TOD pickle must be specified.
    tod_pickle = os.path.join(path_to_outputs, 'data', tod_pickle)
    print(tod_pickle)

    # The ATL01, ANC, and CAL files have the option to be individually specified.
    if atl01_file == None:
        print(os.path.join(path_to_data, path_in, 'ATL01_*.h5'))
        print(glob.glob(os.path.join(path_to_data, path_in, 'ATL01_*.h5')))
        atl01_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL01_*.h5'))[0]
    if atl02_file == None:
        atl02_file = glob.glob(os.path.join(path_to_data, path_in, 'ATL02_*.h5'))[0]
    if anc13_path == None:
        anc13_path = os.path.join(path_to_data, path_in)
    if anc27_path == None:
        anc27_path = os.path.join(path_to_data, path_in)
    if cal17_path == None:
        cal17_path = os.path.join(path_to_data, path_in, 'CAL17')
    if cal44_path == None:
        cal44_path = os.path.join(path_to_data, path_in, 'CAL44')
    if cal49_path == None:
        cal49_path = os.path.join(path_to_data, path_in, 'CAL49')

    tof = TOF(atl01_file=atl01_file, atl02_file=atl02_file, anc13_path=anc13_path, 
        anc27_path=anc27_path, cal17_path=cal17_path, cal44_path=cal44_path, 
        cal49_path=cal49_path, tod_pickle=tod_pickle, multiprocess=True,
        verbose=False, very_verbose=False, qtest_mode=False, mf_limit=None)

    #s = get_size(tof)
    #print("TOF size: {} bytes".format(s))

    print('atl01_file: ', atl01_file)
    print('directory: ', os.path.join(path_to_outputs, 'data'))
    print("tof: ", tof)

    out_filename = pickle_in(tof, out_location=make_file_dir(os.path.join(path_to_outputs, 'data'), atl02_file))

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
    parser.add_argument('--cal17', dest='cal17_path',
        action='store', type=str, required=False, default=None,
        help="Path to directory of the CAL17 folder.")
    parser.add_argument('--cal44', dest='cal44_path',
        action='store', type=str, required=False, default=None,
        help="Path to directory of the CAL44 folder.")
    parser.add_argument('--cal49', dest='cal49_path',
        action='store', type=str, required=False, default=None,
        help="Path to directory of the CAL49 folder.")

    args = parser.parse_args()
     
    return args

if __name__ == '__main__':
    args = parse_args()

    print("path_in={}".format(args.path_in))
    print("tod_pickle={}".format(args.tod_pickle))
    print("atl01_file={}".format(args.atl01_file))
    print("atl02_file={}".format(args.atl02_file))
    print("anc13_path={}".format(args.anc13_path))
    print("anc27_path={}".format(args.anc27_path))
    print("cal17_path={}".format(args.cal17_path))
    print("cal44_path={}".format(args.cal44_path))
    print("cal49_path={}".format(args.cal49_path))    

    ## Make the required option path to all the files.
    ## Non-required could be to specify individually a file (like ANC13) which would override the required path name
    generate(path_in=args.path_in, tod_pickle=args.tod_pickle, 
        atl01_file=args.atl01_file, atl02_file=args.atl02_file, 
        anc13_path=args.anc13_path, anc27_path=args.anc27_path,
        cal17_path=args.cal17_path, cal44_path=args.cal44_path,
        cal49_path=args.cal49_path)


