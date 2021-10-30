#!/usr/bin/env python

""" Verify 'delta_time' fields in all ATL02 groups, except atlas/pcex.

Author:

    C.M. Gosmeyer

"""

import argparse
import h5py
import numpy as np
import os
import pandas as pd
from atl02v.shared.paths import path_to_outputs
from atl02v.shared.tools import make_file_dir, make_timestamp_dir, pickle_out
from atl02v.verification.verification_tools import VFileReader

from atl02v.shared.delta_time import seek_delta_times, DeltaTime


def verify_delta_time(vfiles, path_out):
    """ Verify 'delta_time' fields in all ATL02 groups, except atlas/pcex.

    We can't verify them with diffs, so instead check that data rates
    and limits fall within granule bounds.
    """

    atl02 = h5py.File(vfiles.atl02, 'r', driver=None)

    delta_time_ls = seek_delta_times(atl02)

    delta_time_labels = []
    status = []
    start_limits_results = []
    end_limits_results = []
    rate_results = []
    match = [] # True if all results True, False if any results False.
    starts = []
    ends = []
    rates = []
    delta_time_arrays = []

    for delta_time_label in delta_time_ls:
        
        if 'delta_time_start' not in delta_time_label and \
            'delta_time_end' not in delta_time_label and \
            'start_delta_time' not in delta_time_label and \
            'end_delta_time' not in delta_time_label and \
            'photons' not in delta_time_label and \
            'tep' not in delta_time_label:

            dt = DeltaTime(delta_time_label, atl02)

            # Define the tolerance for the rate check [Hz]
            rate_tolerance=0.5

            # Define the rate check bounds [Hz]
            if 'stellar_window' in delta_time_label:
                    rate_check = 300.0
            elif 'mce_position' in delta_time_label:
                    rate_check = 200.0
            elif 'laser_energy_lrs' in delta_time_label or 'laser_centroid' in delta_time_label or \
                'orbit_info' in delta_time_label or 'altimetry' in delta_time_label or \
                'algorithm_science' in delta_time_label or 'background' in delta_time_label or \
                'inertial_measurement_unit' in delta_time_label or 'laser_image' in delta_time_label or \
                'stellar_image' in delta_time_label:
                rate_check = 50.0
            elif 'atmosphere_s' in delta_time_label:
                rate_check = 25.0
            elif 'stellar_centroid' in delta_time_label or 'star_tracker' in delta_time_label:
                rate_check = 10.0
            elif 'laser_window' in delta_time_label:
                rate_check = 20.0
            else:
                # example: gpsr
                rate_check = 1.0

            # Perform the limit check, for whether the given delta_times fall within granule.
            start_limit_result, end_limit_result = dt.check_limits()
            rate_result = dt.check_rate(rate_check=rate_check, rate_tolerance=rate_tolerance)

            # Perform remainder of checks and append to lists.
            delta_time_labels.append(delta_time_label)
            status.append('COMPLETED')
            start_limits_results.append(start_limit_result)
            end_limits_results.append(end_limit_result)
            rate_results.append(rate_result)
            match.append(bool(start_limit_result*end_limit_result*rate_result))
            starts.append(dt.start)
            ends.append(dt.end)
            rates.append(dt.rate)
            delta_time_arrays.append(dt.delta_times)

    # Create a report summarizing all delta_times
    data = {'label': delta_time_labels, 
            'status':status,
            'start_limits_check': start_limits_results,
            'end_limits_check': end_limits_results,
            'rate_check': rate_results,
            'match?':match,
            'start_[s]': starts,
            'end_[s]': ends,
            'rate_[Hz]': rates,
            'delta_times': delta_time_arrays
            }
    df = pd.DataFrame(data)

    csv_name = os.path.join(path_out, 'delta_time.csv')
    df.to_csv(csv_name)
    print("Created {}".format(csv_name))


    # Create report for the data limits.
    data = {'start': [dt.min_start],
            'end': [dt.max_end],
            'diff': [dt.max_end-dt.min_start],
            'rate_tolerance': [rate_tolerance]}
    df = pd.DataFrame(data)

    csv_name = os.path.join(path_out, 'limits.csv')
    df.to_csv(csv_name)
    print("Created {}".format(csv_name))

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

    verify_delta_time(vfiles=vfiles, path_out=path_out)
