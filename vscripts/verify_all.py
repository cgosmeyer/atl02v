#!/usr/bin/env python

""" Wrapper to verify all following ATL02 groups in one go.

"""

import argparse
import glob
import os
import shutil

from datetime import datetime

import verify_ancillary_data
import verify_atlas_housekeeping
import verify_atlas_pcex
import verify_atlas_tx_pulse_width
import verify_gpsr
import verify_lrs
import verify_orbit_info
import verify_sc
import verify_delta_time

from atl02v.shared.paths import path_to_outputs
from atl02v.shared.tools import make_file_dir, make_timestamp_dir, pickle_out
from atl02v.verification.verification_tools import VFileReader, SummarizeGroup


def verify_all(vfiles, path_out):
    """ Runs verification functions over all groups, organizes outputs
    by group-based folders, and outputs summary reports and plots.
    """
    start_time = datetime.now()
    print("--VERIFY_ALL start time: {}--".format(start_time))
    print(" ")

    # ancillary_data
    print("--Begin ancillary_data--")
    verify_ancillary_data.verify_ancillary_data(vfiles, path_out)
    verify_ancillary_data.verify_calibrations(vfiles, path_out)
    verify_ancillary_data.verify_housekeeping(vfiles, path_out)
    verify_ancillary_data.verify_isf(vfiles, path_out)
    verify_ancillary_data.verify_tep(vfiles, path_out)
    verify_ancillary_data.verify_tod_tof(vfiles, path_out)
    package_group(path_out, group_name='ancillary_data')

    # Create major folder for atlas group.
    if not os.path.isdir(os.path.join(path_out, 'atlas')):
        os.mkdir(os.path.join(path_out, 'atlas'))

    # atlas
    print("--Begin atlas/housekeeping--")

    verify_atlas_housekeeping.verify_laser_energy_internal(vfiles, path_out)
    verify_atlas_housekeeping.verify_laser_energy_lrs(vfiles, path_out)
    verify_atlas_housekeeping.verify_laser_energy_spd(vfiles, path_out)
    verify_atlas_housekeeping.verify_meb_pdu_thermal(vfiles, path_out, atl02_group='meb')
    verify_atlas_housekeeping.verify_meb_pdu_thermal(vfiles, path_out, atl02_group='pdu')
    verify_atlas_housekeeping.verify_meb_pdu_thermal(vfiles, path_out, atl02_group='thermal')
    verify_atlas_housekeeping.verify_pointing(vfiles, path_out)
    verify_atlas_housekeeping.verify_position_velocity(vfiles, path_out)
    verify_atlas_housekeeping.verify_radiometry(vfiles, path_out)
    verify_atlas_housekeeping.verify_status(vfiles, path_out)
    verify_atlas_housekeeping.verify_time_at_the_tone(vfiles, path_out)
    package_group(path_out, group_name='atlas/housekeeping')

    print("--Begin atlas/pcex--")
    verify_atlas_pcex.verify_photons(vfiles, path_out)
    verify_atlas_pcex.verify_strongweak(vfiles, path_out)
    verify_atlas_pcex.verify_tep(vfiles, path_out)
    verify_atlas_pcex.verify_altimetry(vfiles, path_out)
    verify_atlas_pcex.verify_algorithm_science(vfiles, path_out)
    verify_atlas_pcex.verify_algorithm_science_strongweak(vfiles, path_out)
    verify_atlas_pcex.verify_atmosphere_s(vfiles, path_out)
    verify_atlas_pcex.verify_background(vfiles, path_out)
    package_group(path_out, group_name='atlas/pcex')

    print("--Begin atlas/tx_pulse_width--")
    verify_atlas_tx_pulse_width.verify_tx_pulse_width(vfiles, path_out)
    package_group(path_out, group_name='atlas/tx_pulse_width')

    # gpsr
    print("--Begin gpsr--")
    verify_gpsr.verify_carrier_amplitude(vfiles, path_out)
    verify_gpsr.verify_carrier_phase(vfiles, path_out)
    verify_gpsr.verify_channel_status(vfiles, path_out)
    verify_gpsr.verify_code_phase(vfiles, path_out)
    verify_gpsr.verify_hk(vfiles, path_out)
    verify_gpsr.verify_navigation(vfiles, path_out)
    verify_gpsr.verify_noise_histogram(vfiles, path_out)
    verify_gpsr.verify_time_correlation(vfiles, path_out)
    package_group(path_out, group_name='gpsr')

    # lrs
    print("--Begin lrs--")
    verify_lrs.verify_hk_1120(vfiles, path_out)
    verify_lrs.verify_laser_centroid(vfiles, path_out)
    verify_lrs.verify_stellar_centroid(vfiles, path_out)
    package_group(path_out, group_name='lrs')

    # orbit_info
    print("--Begin orbit_info--")
    verify_orbit_info.verify_orbit_info(vfiles, path_out)
    package_group(path_out, group_name='orbit_info')

    #sc
    print("--Begin sc--")
    verify_sc.verify_sc(vfiles, path_out)
    verify_sc.verify_attitude_control_system(vfiles, path_out)
    verify_sc.verify_ephemeris(vfiles, path_out)
    verify_sc.verify_hk(vfiles, path_out)
    verify_sc.verify_inertial_measurement_unit(vfiles, path_out)
    verify_sc.verify_inertial_measurement_unit_gyro_x(vfiles, path_out)
    verify_sc.verify_solar_array(vfiles, path_out)
    verify_sc.verify_star_tracker(vfiles, path_out)
    verify_sc.verify_star_tracker_optical_head_x(vfiles, path_out)
    package_group(path_out, group_name='sc')

    # delta_times
    print("--Begin delta_time--")
    verify_delta_time.verify_delta_time(vfiles, path_out)
    package_group(path_out, group_name='delta_time')

    run_time = datetime.now() - start_time
    print(" ")
    print("--VERIFY_ALL run time: {}--".format(run_time))

def package_group(path_out, group_name):
    """ Packages a group (its PNG and CSV files) in one folder 
    under 'path_out' and summarizes the results.

    Parameters
    ----------
    group_name : string
        Name of the group.
    """
    items = glob.glob(os.path.join(path_out, '*csv')) + \
        glob.glob(os.path.join(path_out, '*png'))
    folder = os.path.join(path_out, group_name)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    for item in items:
        shutil.move(item, folder)

    # Summarize the results.
    sg = SummarizeGroup(folder, group_name)
    sg.write_summary()
    sg.plot_summary()


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

    verify_all(vfiles=vfiles, path_out=path_out)
