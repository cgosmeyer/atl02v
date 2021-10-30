#!/usr/bin/env python

""" Verifying the duplicate identication algorithm.

Author:

    C.M. Gosmeyer

"""

import argparse
import h5py
import numpy as np
import os
import pandas as pd
import pylab as plt
from pydl import uniq
from atl02v.shared.constants import pces
from atl02v.shared.paths import path_to_outputs
from atl02v.shared.tools import make_file_dir, make_timestamp_dir, pickle_out
from atl02v.verification.verification_tools import VFileReader


# Compare master ids from duplicate mask to IDs of ATL02's
#   a. TOF
#   b. TEP TOF

# Output:
#   1. Overplots of both IDs (and the TOF values too?)
#   2. Diff plot
#   3. % of array that is identified as duplicate. Raise warning if more than 5-6%.


class VerifyDuplicates(object):
    """
    """
    def __init__(self, vfiles, path_out=''):
        self.path_out = path_out

        self.tof = pickle_out(vfiles.tof)
        self.tep = pickle_out(vfiles.tep)
        self.dupmask = h5py.File(vfiles.dupmask, 'r', driver=None)
        self.photonids = h5py.File(vfiles.photonids, 'r', driver=None)
        self.atl01 = h5py.File(vfiles.atl01, 'r', driver=None)
        self.atl02 = h5py.File(vfiles.atl02, 'r', driver=None)

    def make_comparision_matrix(self, pce):
        """
        To pinpoint what is going on with the duplicates, need be able to print together, for
        ATL02 and myself, channel, rx_fc, rx_cc, FC_TDC, and TOF, along with the location 
        of identified duplicates.
        """
        # Retrieve calculated paraeters, in RX-space
        # Index so only strong included.
        # Duplicates are not removed.
        strong_idx = np.where(self.tof.map_pce(pce).channel < 17)[0]
        calculated_channel = self.tof.map_pce(pce).channel[strong_idx]
        calculated_rx_fc = self.tof.map_pce(pce).rx_fc[strong_idx]
        calculated_rx_cc = self.tof.map_pce(pce).rx_cc[strong_idx]
        calculated_tof = self.tof.map_pce(pce).TOF.all[strong_idx]

        calculated_dupmask_s = self.dupmask['pce{}/all'.format(pce)].value[strong_idx]
        calculated_dups = strong_idx[np.where(calculated_dupmask_s==0)[0]]
        # Total number of calculated duplicates for pcex strong.
        n_calculated_dups = len(calculated_dups)

        # Retrieve ATL01 parameters
        atl01_channel = self.atl01['pce{}/a_alt_science_ph/raw_rx_channel_id'.format(pce)].value[strong_idx]
        atl01_rx_fc = self.atl01['pce{}/a_alt_science_ph/raw_rx_leading_fine'.format(pce)].value[strong_idx]
        atl01_rx_cc = self.atl01['pce{}/a_alt_science_ph/raw_rx_leading_coarse'.format(pce)].value[strong_idx]
        atl02_tof = self.atl02['pce{}/altimetry/strong/photons/ph_tof'.format(pce)].value

        # Assume all 'missing' photons from PhotonID are duplicates.
        # So use this to find the IDs of assumed duplicates.
        master_ids = np.arange(len(self.tof.map_pce(pce).channel))
        # Master (ATL01) indices of all strong photons, duplicates included.
        master_ids_s = master_ids[strong_idx]
        # Master (ATL01) indices of all strong photons, ATL02 duplicates removed.
        phid_s = self.photonids['pce{}/strong'.format(pce)].value
        # Now need figure out the indices of the photons in master_id_s 
        # excluded from phid_s.  These will be the assunmed 'duplicates'
        # Which master_ids_s values are missing from phid_s.
        atl02_dups = np.setdiff1d(master_ids_s, phid_s)
        # Total numbber of duplicates in ATL02 pcex strong.
        n_atl02_dups = len(atl02_dups)

        # The ATL02 duplicates indexed in strong space
        atl02_dups_s_idx = np.searchsorted(master_ids_s, atl02_dups)
        # Mask for ATL02 strong duplicates
        atl02_dupmask_s = np.ones(len(strong_idx))
        atl02_dupmask_s[atl02_dups_s_idx] = 0

        # Map ATL02 TOF to ATL01 master strong space.
        atl02_tof_full = np.zeros(len(strong_idx))
        atl02_tof_s_idx = np.searchsorted(master_ids_s, phid_s)
        atl02_tof_full[atl02_tof_s_idx] = atl02_tof      

        # Now finally matrix them together in ATL01 RX strong space.
        df = pd.DataFrame.from_dict({'strong_idx':strong_idx, 
                           'calc_channel':calculated_channel,
                           'atl01_channel':atl01_channel,
                           'calc_rx_fc':calculated_rx_fc,
                           'atl01_rx_fc':atl01_rx_fc,
                           'calc_rx_cc':calculated_rx_cc,
                           'atl01_rx_cc':atl01_rx_cc,
                           'calc_dupmask_s':calculated_dupmask_s,
                           'atl02_dupmask_s':atl02_dupmask_s,
                           'calc_tof':calculated_tof,
                           'atl02_tof':atl02_tof_full
                          })

    def check_tof(self, pce):
        """
        """
        # This should contain all IDs, duplicates included. 
        tof_ids = np.arange(len(self.tof.map_pce(pce).TOF.all))
        
        # IDs of my calculated duplicates in RX-space.
        print("")
        v_dupmask = np.array(self.dupmask['pce{}/all'.format(pce)].value, dtype=bool)
        print("v_dupmask: ", len(v_dupmask), v_dupmask)
        v_num_duplicates = len(np.where(v_dupmask == False)[0])
        print("numb duplicates in verify calculation: ", v_num_duplicates)
        v_perc_duplicates = (v_num_duplicates / len(tof_ids)) * 100.0
        print("perc duplicates in verify calculation: ", v_perc_duplicates)

        # IDs of the ATL02 TOFs (with duplicates removed)
        photonids_s = self.photonids['pce{}/strong'.format(pce)].value
        photonids_w = self.photonids['pce{}/weak'.format(pce)].value

        # Find the 'missing' events by comparing to the tof_ids "master list".
        # The events missing from the master list will be the duplicates.
        photonids = np.sort(np.append(photonids_s, photonids_w))

        # The 'duplicate mask' for ATL02.
        a_dupmask = np.zeros(len(tof_ids), np.bool)
        a_dupmask[photonids] = True
        print("a_dupmask: ", len(a_dupmask), a_dupmask)
        a_num_duplicates = len(np.where(a_dupmask == False)[0])
        print("numb duplicates in ATL02: ", a_num_duplicates)
        a_perc_duplicates = (a_num_duplicates / len(tof_ids)) * 100.0
        print("perc duplicates in ATL02: ", a_perc_duplicates)

        # Take the OR of the arrays and convert from bool to int.
        diff = v_dupmask ^ a_dupmask
        diff = np.array(diff, dtype=int)
        print('diff: ', len(diff), diff)

        # Find number of mismatches (where array is 1)
        mismatches = len(np.where(diff==1)[0])
        print("mismatches: ", mismatches)

        # Plot each of the arrays and the diff.
        lim=2000
        fig, ax1 = plt.subplots(figsize=(12,10))
        v_dupmask_int = np.array(v_dupmask, dtype=int)
        a_dupmask_int = np.array(a_dupmask, dtype=int)
        ax1.scatter(tof_ids[:lim], v_dupmask_int[:lim], color='blue', label='calculated', marker='d')
        ax1.scatter(tof_ids[:lim], a_dupmask_int[:lim], color='orange', label='atl02', marker='+')
        ax1.scatter(tof_ids[:lim], diff[:lim], color='grey', label='diff', marker='x')

        ax1.set_title('PCE {} Duplicates from TOF'.format(pce), size=20)
        ax1.set_xlabel('Index', size=14)
        ax1.set_ylabel('Duplicate Flag', size=14)
        ax1.tick_params(labelsize=12)
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.scatter(tof_ids[:lim], self.tof.map_pce(pce).TOF.all[:lim], color='red', label='tof', marker='.')
        ax2.set_ylabel('TOF', size=14)

        png_name = os.path.join(self.path_out, 'pce{}_tof_duplicates.png'.format(pce))
        #plt.savefig(png_name)
        plt.show()
        print('Created {}'.format(png_name))


    def check_tep(self, pce):
        """
        """
        tep_ids = tep.map_pce(pce).filtered.master_ids

        #verify_tep_ids = map_tep2rx(pce, tep.map_pce(pce).filtered.master_ids, tep)


def verify_duplicates(vfiles, path_out):
    """
    """
    vd = VerifyDuplicates(vfiles, path_out)

    # Verify TOF duplicates
    for pce in pces:
        vd.check_tof(pce)

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
   
    verify_duplicates(vfiles=vfiles, path_out=path_out)
