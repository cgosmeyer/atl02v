#!/usr/bin/env python

""" Verifying datasets in {pce}

Author:

    C.M. Gosmeyer

"""

import argparse
import h5py
import numpy as np
import os
import pandas as pd
from pydl import uniq
from atl02v.shared.paths import path_to_outputs
from atl02v.shared.tools import make_file_dir, make_timestamp_dir, pickle_out
from atl02v.tep.tep import map_strong2tep
from atl02v.verification.atlas_pcex_verification import VerifyPhotons, \
    VerifyStrongWeak, VerifyTep, VerifyAltimetry, VerifyAlgorithmScience, \
    VerifyAlgorithmScienceStrongWeak, VerifyAtmosphereS, VerifyBackground, \
    calculate_useflag
from atl02v.verification.verification_tools import VFileReader


def verify_algorithm_science(vfiles, path_out):
    """ Function for verifying datasets in {pce}/algorithm_science
    """
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)
    atl02 = h5py.File(vfiles.atl02, 'r', driver=None)

    def build_useflag(pce):
        return calculate_useflag(atl01, atl02['pce{}/algorithm_science/delta_time'.format(pce)].value, pce)

    def calculate_amet_time(pce):
        """
        Notes
        -----
        Convert by deviding 4294967296.0 because JLee says so (29 May 2019).
        """
        secs = atl01['pce{}/a_pmf_algorithm_science/raw_mframe_amet_time_sec'.format(pce)].value
        subsecs = atl01['pce{}/a_pmf_algorithm_science/raw_mframe_amet_time_subsec'.format(pce)].value 
        return secs + (subsecs/4294967296.0)

    def calculate_gps_time(pce):
        """
        Notes
        -----
        Convert by deviding 4294967296.0 because JLee says so (29 May 2019).
        """
        secs = atl01['pce{}/a_pmf_algorithm_science/raw_mframe_gps_time_sec'.format(pce)].value
        subsecs = atl01['pce{}/a_pmf_algorithm_science/raw_mframe_gps_time_subsec'.format(pce)].value 
        return secs + (subsecs/4294967296.0)

    d = {'amet_time':       [0, calculate_amet_time, 'calculation'], 
         'gps_time':        [0, calculate_gps_time, 'calculation'], 
         'pce_mframe_cnt':  [0, (lambda pce : atl01['pce{}/a_pmf_algorithm_science/raw_pce_mframe_cnt'.format(pce)].value), 'passthrough'],
         'useflag':         [0, (lambda pce : build_useflag(pce)), 'calculation']
         }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyAlgorithmScience(vfiles, tolerance, atl02_dataset, path_out).do_verify(custom_func)
    atl01.close()
    atl02.close()

def verify_algorithm_science_strongweak(vfiles, path_out):
    """ Function for verifying datasets in {pce}/algorithm_science/{strong/weak}
    """
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)

    d = {'daynight_flag':   [0, (lambda pce, spot : atl01['pce{}/a_pmf_algorithm_science/{}/raw_alt_daynight_flag'.format(pce, spot)].value), 'passthrough'],
         'decisionflags':   [0, (lambda pce, spot : atl01['pce{}/a_pmf_algorithm_science/{}/raw_alt_decisionflags'.format(pce, spot)].value), 'passthrough'],
         'ds_4bytes':       [0, (lambda pce, spot : np.arange(1,5)), 'definition'],
         'flywheel':        [0, (lambda pce, spot : atl01['pce{}/a_pmf_algorithm_science/{}/raw_alt_flywheel'.format(pce, spot)].value), 'passthrough'],
         'signalflags':     [0, (lambda pce, spot : atl01['pce{}/a_pmf_algorithm_science/{}/raw_alt_signalflags'.format(pce, spot)].value), 'passthrough']
        }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyAlgorithmScienceStrongWeak(vfiles, tolerance, atl02_dataset, path_out).do_verify(custom_func)
    atl01.close()

def verify_altimetry(vfiles, path_out):
    """ Function for verifying datasets in {pce}/altimetry
    """
    dlbs = pickle_out(vfiles.dlbs)
    tod = pickle_out(vfiles.tod)
    tof = pickle_out(vfiles.tof)
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)
    atl02 = h5py.File(vfiles.atl02, 'r', driver=None)
    photonids = h5py.File(vfiles.photonids, 'r', driver=None)

    def build_useflag(pce):
        return calculate_useflag(atl01, atl02['pce{}/altimetry/delta_time'.format(pce)].value, pce, verbose=True)

    def calc_delta_time(pce):
        """ The DeltaTime_ll at the first index of each major frame.
        The time of the first TX pulse in MF, relative to ATLAS SDP GPS epoch 
        """
        # Find index of first unique mframe_cnt
        mframe_cnt_uniq = uniq(atl01['pce{}/a_alt_science_ph/raw_pce_mframe_cnt'.format(pce)].value)

        # Use unique delta_time to index into DeltaTime_ll
        delta_time = tod.map_pce(pce).DeltaTime_ll[mframe_cnt_uniq]

        return delta_time

    d = {#'cal_fall_sm':      [0, (lambda pce : ), 'calculation'], ## can't verify; values hard-coded
         #'cal_rise_sm':      [0, (lambda pce : ), 'calculation'],  ## can't verify; values hard-coded
         'ch_mask_s':        [0,  (lambda pce : dlbs.map_pce(pce).ch_mask_s), 'calculation'],
         'ch_mask_w':        [0,  (lambda pce : dlbs.map_pce(pce).ch_mask_w), 'calculation'],
         'ds_strong_channel_index': [0, (lambda pce : np.arange(1,17)), 'definition'],
         'ds_weak_channel_index': [0, (lambda pce : np.arange(1,5)), 'definition'],
         'n_bands':          [0, (lambda pce : atl01['pce{}/a_alt_science/raw_alt_n_bands'.format(pce)].value), 'passthrough'],
         'pce_mframe_cnt':   [0, (lambda pce : atl01['pce{}/a_alt_science/raw_pce_mframe_cnt'.format(pce)].value), 'passthrough'],
         'useflag':          [0, (lambda pce : build_useflag(pce)), 'calculation']
         }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyAltimetry(vfiles, tolerance, atl02_dataset, path_out).do_verify(custom_func)
    atl01.close()
    atl02.close()

def verify_strongweak(vfiles, path_out):
    """ Function for verifying datasets in {pce}/altimetry/{strong_weak}
    """
    dlbs = pickle_out(vfiles.dlbs)
    tod = pickle_out(vfiles.tod)
    tof = pickle_out(vfiles.tof)
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)
    photonids_df = h5py.File(vfiles.photonids, 'r', driver=None)

    # These functions need to be inside function so that they can
    # see the openend verification files.
    def calc_n_mf_ph(pce, spot):
        """ Number photons per MF.
        """
        verifier_values = tof.map_pce(pce).majorframe
        photonids = photonids_df['pce{}/{}'.format(pce, spot)].value
        verifier_matched_values = verifier_values[photonids]
        return np.bincount(verifier_matched_values)

    def calc_ph_ndx_beg(pce, spot):
        """ Index of first photon in each MF.
        """
        verifier_values = tof.map_pce(pce).majorframe
        photonids = photonids_df['pce{}/{}'.format(pce, spot)].value
        verifier_matched_values = verifier_values[photonids]

        # Get index of the *last* of each unique item.
        verifier_uniq = uniq(verifier_matched_values)
        # Add two so that get the index of the first photon on MF instead of last.
        # I believe it must be two because it appears to be starting index from 1,
        # whereas my array is indexed from 0.
        verifier_uniq += 2
        # Insert "1" at 0th index
        verifier_uniq = np.insert(verifier_uniq, 0, 1)
        # Finally remove the last item (since it isn't showing the first place of anything)
        verifier_uniq = verifier_uniq[:(len(verifier_uniq)-1)]
        return verifier_uniq

    d = {'alt_rw_start':    [1e-7, (lambda pce, spot : tof.map_pce(pce).RWS_T), 'calculation'], #float32!
         'alt_rw_width':    [1e-7, (lambda pce, spot : tof.map_pce(pce).RWW_T), 'calculation'],
         'band1_offset':    [1e-7, (lambda pce, spot : dlbs.map_spot(pce, spot).band1_offset), 'calculation'], #float32!
         'band1_width':     [1e-7, (lambda pce, spot : dlbs.map_spot(pce, spot).band1_width), 'calculation'],
         'band2_offset':    [1e-7, (lambda pce, spot : dlbs.map_spot(pce, spot).band2_offset), 'calculation'],
         'band2_width':     [1e-7, (lambda pce, spot : dlbs.map_spot(pce, spot).band2_width), 'calculation'],
         'n_mf_ph':         [0, calc_n_mf_ph, 'calculation'], 
         'ph_ndx_beg':      [0, calc_ph_ndx_beg, 'calculation']
        } 

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyStrongWeak(vfiles, tolerance, atl02_dataset, path_out).do_verify(custom_func)
    atl01.close()

def verify_photons(vfiles, path_out):
    """ Function for verifying datasets in {pce}/altimetry/{strong_weak}/photons

    Notes
    -----
    * Within VerifyPhotons, there is a special case to verify that 
      "rx_band_id" is all one value. So you will see in the output
      plot and summary that, if the array was valid, it reduced to
      a singleton array.
    """
    tod = pickle_out(vfiles.tod)
    tof = pickle_out(vfiles.tof)
    tep = pickle_out(vfiles.tep)
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)

    def configure_tof_flag(pce, spot):
        if pce == 3:
            return tof.map_pce(pce).tof_flag
        else:
            if spot == 'strong':
                return tep.map_pce(pce).tof_flag
            elif spot == 'weak':
                return tof.map_pce(pce).tof_flag

    d = {'delta_time':      [1e-6, (lambda pce : tod.map_pce(pce).DeltaTime_ll), 'calculation'],
         'rx_band_id':      [0,     (lambda pce : atl01['pce{}/a_alt_science_ph/raw_rx_band_id'.format(pce)].value), 'passthrough'],
         'pce_mframe_cnt':  [0,     (lambda pce : atl01['pce{}/a_alt_science_ph/raw_pce_mframe_cnt'.format(pce)].value), 'passthrough'],
         'ph_id_channel':   [0,     (lambda pce : atl01['pce{}/a_alt_science_ph/ph_id_channel'.format(pce)].value), 'passthrough'],
         'ph_id_count':     [0,     (lambda pce : atl01['pce{}/a_alt_science_ph/ph_id_count'.format(pce)].value), 'passthrough'],
         'ph_id_pulse':     [0,     (lambda pce : atl01['pce{}/a_alt_science_ph/ph_id_pulse'.format(pce)].value), 'passthrough'],
         'ph_tof':          [1e-15, (lambda pce : tof.map_pce(pce).TOF.all), 'calculation'],
         'tof_flag':        [0, (lambda pce, spot : configure_tof_flag(pce, spot)), 'calculation'],
         'tx_ll_tof':       [1e-12, (lambda pce : tof.map_pce(pce).TX_T_ll), 'calculation'],
         'tx_other_tof':    [1e-15, (lambda pce : tof.map_pce(pce).TX_T_other), 'calculation']
         }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyPhotons(vfiles, tolerance, atl02_dataset, path_out).do_verify(custom_func)
    atl01.close()
        
def verify_tep(vfiles, path_out):
    """ Function for verifying datasets in {pce}/tep
    """
    tod = pickle_out(vfiles.tod)
    tof = pickle_out(vfiles.tof)
    tep = pickle_out(vfiles.tep)
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)
    atl02 = h5py.File(vfiles.atl02, 'r', driver=None)
    dupmask = h5py.File(vfiles.dupmask, 'r', driver=None)
    photonids = h5py.File(vfiles.photonids, 'r', driver=None)

    # Note that need to use the NON-filtered version of TEP arrays for `map_strong2tep` to work.
    d = {'delta_time':      [1e-6,  (lambda pce : tep.map_pce(pce).delta_time), 'calculation'],
         'pce_mframe_cnt':  [0,  (lambda pce : atl01['pce{}/a_alt_science_ph/raw_pce_mframe_cnt'.format(pce)].value), 'passthrough'],
         'ph_id_channel':   [0,  (lambda pce : atl01['pce{}/a_alt_science_ph/ph_id_channel'.format(pce)].value), 'passthrough'],    
         'ph_id_count':     [0,  (lambda pce : atl01['pce{}/a_alt_science_ph/ph_id_count'.format(pce)].value), 'passthrough'],
         'ph_id_pulse':     [0,  (lambda pce : atl01['pce{}/a_alt_science_ph/ph_id_pulse'.format(pce)].value), 'passthrough'],
         'rx_band_id':      [0,  (lambda pce : atl01['pce{}/a_alt_science_ph/raw_rx_band_id'.format(pce)].value), 'passthrough'],
         'rx_channel_id':   [0,  (lambda pce : atl01['pce{}/a_alt_science_ph/raw_rx_channel_id'.format(pce)].value), 'passthrough'],   
         'tep_pulse_num':   [0,  (lambda pce : map_strong2tep(pce, tep, photonids, tep.map_pce(pce).N)), ''],
         'tof_tep':         [1e-15,  (lambda pce : map_strong2tep(pce, tep, photonids, tep.map_pce(pce).TOF_TEP)), 'calculation'],
         'tx_ll_tof_tep':   [1e-12,  (lambda pce : map_strong2tep(pce, tep, photonids, tep.map_pce(pce).TX_T_ll_jplusN)), 'calculation'],
         'tx_other_tof_tep':[1e-15,  (lambda pce : map_strong2tep(pce, tep, photonids, tep.map_pce(pce).TX_T_other_jplusN)), 'calculation']
         }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyTep(vfiles, tolerance, atl02_dataset, path_out).do_verify(custom_func)
    atl01.close()
    atl02.close()

def verify_atmosphere_s(vfiles, path_out):
    """ Function for verifying datasets in {pce}/atmosphere_s
    """
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)
    atl02 = h5py.File(vfiles.atl02, 'r', driver=None)

    def build_useflag(pce):
        return calculate_useflag(atl01, atl02['pce{}/atmosphere_s/delta_time'.format(pce)].value, pce)

    d = {'atm_bins':        [0,  (lambda pce : atl01['pce{}/a_atm_hist_s/raw_atm_bins'.format(pce)].value), 'passthrough'],
         'atm_rw_start':    [1e-6,  (lambda pce : atl01['pce{}/a_atm_hist_s/raw_atm_rw_start'.format(pce)].value*1e-8), 'conversion'], # clock cylces to sec?
         'atm_rw_width':    [1e-6,  (lambda pce : atl01['pce{}/a_atm_hist_s/raw_atm_rw_width'.format(pce)].value*1e-8), 'conversion'], # clock cylces to sec?
         'atm_shift_amount':    [0,  (lambda pce : atl01['pce{}/a_atm_hist_s/raw_atm_shift_amount'.format(pce)].value), 'passthrough'],
         'ds_hist_bin_index':   [0, (lambda pce : np.arange(1,468)), 'definition'],
         'pce_mframe_cnt':  [0,  (lambda pce : atl01['pce{}/a_atm_hist_s/pce_mframe_cnt'.format(pce)].value), 'passthrough'],
         'useflag':         [0, (lambda pce : build_useflag(pce)), 'calculation']
        }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyAtmosphereS(vfiles, tolerance, atl02_dataset, path_out).do_verify(custom_func)
    atl01.close()
    atl02.close()

def verify_background(vfiles, path_out):
    """ Function for verifying datasets in {pce}/background
    """
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)
    atl02 = h5py.File(vfiles.atl02, 'r', driver=None)

    def build_useflag(pce):
        delta_time = atl02['pce{}/background/delta_time'.format(pce)].value
        return calculate_useflag(atl01, np.vstack([delta_time, delta_time, delta_time, delta_time]).T.flatten(), pce)

    def build_pce_mframe_cnt(pce):
        """
        """
        arr = atl01['pce{}/a_pmf_algorithm_science/raw_pce_mframe_cnt'.format(pce)].value
        out = np.vstack([arr, arr, arr, arr])
        return out.T.flatten()

    d = {'bg_cnt_50shot_s': [0, (lambda pce : atl01['pce{}/a_pmf_algorithm_science/strong/raw_alt_50shot_bg_cnt'.format(pce)].value.flatten()), 'calculation'],
         'bg_cnt_50shot_w': [0, (lambda pce : atl01['pce{}/a_pmf_algorithm_science/weak/raw_alt_50shot_bg_cnt'.format(pce)].value.flatten()), 'calculation'],
         'pce_mframe_cnt':  [0, (lambda pce : build_pce_mframe_cnt(pce)), 'calculation'], # need repeat four times
         'useflag':         [0, (lambda pce : build_useflag(pce)), 'calculation']
        }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyBackground(vfiles, tolerance, atl02_dataset, path_out).do_verify(custom_func)
    atl01.close()

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
    parser.add_argument('--f', dest='fields', 
        default=['altimetry', 'strongweak', 'photons', 'tep', 'algorithm_science', \
            'atmosphere_s', 'background'],
        action='store', type=str, nargs='+', required=False, 
        help="Select groups in pcex to verify ['altimetry', 'strongweak', 'photons', \
            'tep', 'algorithm_science', 'atmosphere_s', 'background']. By default all.")
    
    args = parser.parse_args()
     
    return args

if __name__=="__main__":
    args = parse_args()

    print("path_out={}".format(args.path_out))
    print("vfiles={}".format(args.vfiles))
    print("fields={}".format(args.fields))

    vfiles = VFileReader(vfile_name=args.vfiles).read()

    if args.path_out == '':
        # Default setting
        path_out = make_timestamp_dir(make_file_dir(os.path.join(path_to_outputs, 'reports'), vfiles.atl02))
    else:
        path_out = make_timestamp_dir(make_file_dir(args.path_out, vfiles.atl02))
   
    if 'altimetry' in args.fields:
        verify_altimetry(vfiles=vfiles, path_out=path_out) 
    if 'strongweak' in args.fields:
        verify_strongweak(vfiles=vfiles, path_out=path_out)  
    if 'photons' in args.fields:
        verify_photons(vfiles=vfiles, path_out=path_out)
    if 'tep' in args.fields:
        verify_tep(vfiles=vfiles, path_out=path_out)
    if 'algorithm_science' in args.fields:
        verify_algorithm_science(vfiles=vfiles, path_out=path_out) 
        verify_algorithm_science_strongweak(vfiles=vfiles, path_out=path_out) 
    if 'atmosphere_s' in args.fields:
        verify_atmosphere_s(vfiles=vfiles, path_out=path_out) 
    if 'background' in args.fields:
        verify_background(vfiles=vfiles, path_out=path_out)

