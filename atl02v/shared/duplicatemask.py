""" Class to mask out duplicates in TOF arrays. 

Author:

    C.M. Gosmeyer

"""

import h5py
import itertools
import numpy as np
import os
import pandas as pd
from datetime import datetime
from atl02v.shared.constants import pces
from atl02v.shared.tools import find_nearest_bracketing, pickle_out

class DuplicateMask(object):
    __name__ = 'DuplicateMask'
    def __init__(self, pce, ph_id_count, tof, verbose=False):
        """ Creates mask of duplicates in a TOF array.

        Parameters
        ----------
        pce : int
            Either 1, 2, or 3.
        ph_id_count : array
            The array from ATL01 for give PCE:
            pce{}/a_alt_science_ph/ph_id_count
        tof : TOF
            A TOF object.
        verbose : {True, False} 
            Turn on for verbose. Default is off.
        """
        self.verbose = verbose
        self.pce = pce
        self.tof = tof
        self.ph_id_count = ph_id_count

        # Define the tolerance for the number of cells in the delay chain.
        self.tolerance = np.float64(0.8)

        # Read arrays from TOF object.
        self.rx_cc = self.tof.map_pce(self.pce).rx_cc
        self.rx_fc = self.tof.map_pce(self.pce).rx_fc
        self.FC_TDC = self.tof.map_pce(self.pce).FC_TDC
        self.channel = np.array(self.tof.map_pce(self.pce).channel, dtype='int64')
        self.toggle = np.array(self.tof.map_pce(self.pce).toggle, dtype='int64')

    def make_mask(self):
        """ Table 17

        With all non-duplicates set to 1, can put mask over the master list of 
        photon IDs, and then compare the IDs with ASAS's mapped via PhotonID.

        Returns
        -------
        duplicate_mask : array
            Length of all TOFs per PCE. All 1's, assigning 0's to duplicates.
        """
        master_ids = np.arange(len(self.FC_TDC))  #(self.ph_id_count))
        duplicate_mask = np.ones(len(self.FC_TDC))  #(self.ph_id_count))

        # Identify all returns that come from same shot via 'ph_id_count'.
        # soooo, find indexes of all non-1's, but also include the index of the 1 before them.
        # Then group them together.
        for rx in master_ids:
            # Set a static value for later.
            rx_static = rx
            if self.verbose:
                print("rx: ", rx)
                print("self.ph_id_count[rx]: ", self.ph_id_count[rx])

            # First check for channel == 0, which is invalid and should set the 
            # event to 0  in duplicate mask.
            if self.channel[rx] == 0:
                print("Removing event {}".format(rx))
                print("channel: {}, rx_fc: {}".format(self.channel[rx], self.rx_fc[rx]))
                duplicate_mask[rx] = 0

            else:
                # Find each 'group', which will be a list of the master indices of the
                # suspected group of duplicates.
                group = []

                if self.ph_id_count[rx] >= 2:
                    # The 2 will be the special case because then need make sure capturing
                    # the '1' count before it.
                    group.append(rx-1)                                
                    group.append(rx)
                    ch = self.channel[rx_static]
                    rx+=1
                    while self.ph_id_count[rx] > 2:
                        group.append(rx)
                        rx+=1
                    # Need also capture all the potentially misplaced ph_id_counts. That is,
                    # need to check for other matching channels that were misplaced from 
                    # the consecutive list.
                    # Find index of nearest channel=1 before and after.
                    # NOTE: Can comment out this bracket-finding section if want a faster
                    # run. It doesn't necessarily find many more duplicates. Moreover,
                    # there is a bug in it. See the function's comments for details.
                    rx0, rx1 = find_nearest_bracketing(self.channel, search_idx=rx_static, bracket_value=1)
                    if self.verbose:
                        print("channel: {}".format(ch))
                        print("rx0: {}, rx1: {}".format(rx0, rx1))
                    for rxn in np.arange(rx0, rx1, 1):
                        if self.channel[rxn] == ch and rxn not in group:
                            group.append(rxn)

                    if self.verbose:
                        print("group: ", len(group), group)

                    # Compare the n event with all other events in case of ph_id_count >= 3
                    for n0, n1 in itertools.combinations(range(len(group)), 2):
                        duplicate_mask = self.test_for_duplicate(n0, n1, group, duplicate_mask)

        return duplicate_mask


    def test_for_duplicate(self, n0, n1, group, duplicate_mask):
        """
        n1 = nth value
        n0 = nth-1 value
        """
        # Compare the first course count to the other course counts in the 'group'
        cc_diff = self.rx_cc[group[n1]] - self.rx_cc[group[n0]]

        if self.verbose:
            print("cc_diff: ", cc_diff)
            print("abs(rx_fc[group[n]] - rx_fc[group[n-1]]): ", abs(self.rx_fc[group[n1]] - self.rx_fc[group[n0]]))
            print("FC_TDC[group[n]]*tolerance: ", self.FC_TDC[group[n1]]*self.tolerance)
            print("self.toggle[group[n1]]: {} and self.toggle[group[n0]]: {}".format(abs(self.toggle[group[n1]]), abs(self.toggle[group[n0]])))
            print("self.channel[group[n1]]: {} and self.channel[group[n0]]: {}".format(abs(self.channel[group[n1]]), abs(self.channel[group[n0]])))

        # If the fine count is zero the event has an error and should be discarded.
        if self.channel[group[n1]] > 20:
            print("Removing event {}".format(group[n1]))
            print("channel: {}, rx_fc: {}".format(self.channel[group[n1]], self.rx_fc[group[n1]]))
            duplicate_mask[group[n1]] = 0

        # If event is ok, check whether meets criteria for a duplicate.
        elif (abs(cc_diff) == 1) and (abs(self.rx_fc[group[n1]] - self.rx_fc[group[n0]]) > self.FC_TDC[group[n1]]*self.tolerance) \
            and (abs(self.toggle[group[n1]] - self.toggle[group[n0]]) == 0) \
            and (abs(self.channel[group[n1]] - self.channel[group[n0]]) == 0):
            # Pair contains a duplicate.
            if cc_diff == -1:
                # Keep the nth value and remove the other value because
                # it is a duplicate.
                if self.verbose:
                    print("duplicate at n0 removed.")
                duplicate_mask[group[n0]] = 0
            else:
                # Keep the other value and the nth value is the duplicate and 
                # should be removed.
                # (so should be keeping the event with smaller CC value)
                if self.verbose:
                    print("duplicate at n1 removed.")
                duplicate_mask[group[n1]] = 0

        return duplicate_mask


########################################
# For saving the masks in an HDF5 file.
########################################

def save_masks(atl01_file, tof_pickle, out_location='', verbose=False):
    """ Saves the DuplicateMask output to an H5 file.

    To get strong/weak, just use PhotonIDs

    pi_strong = pi_df('pce1/strong').value
    dm = dm_df('pce1/all').value
    dm_strong = dm[pi_strong]

    Parameters
    ----------
    atl01_file : str
        The path and filename of the ATL01 file.
    tof_pickle : str
        The name of the TOF pickle file.
    out_location : str
        The location to write the output H5 file.
    verbose : {True, False} 
        Turn on for verbose. Default is off.
    """
    start_time = datetime.now()
    print("DuplicateMask start time: ", start_time)

    atl01 = h5py.File(atl01_file, 'r', driver=None)
    tof = pickle_out(tof_pickle)

    ph_id_count_dict = {
        1:np.array(atl01['pce1/a_alt_science_ph/ph_id_count'].value),
        2:np.array(atl01['pce2/a_alt_science_ph/ph_id_count'].value),
        3:np.array(atl01['pce3/a_alt_science_ph/ph_id_count'].value)}
    
    atl01.close()

    # Open file to store duplicate masks.
    timenow = start_time.strftime('%Y-%jT%H-%M-%S')
    filename = os.path.join(out_location, "duplicate_masks_{}.hdf5".format(timenow))
    f = h5py.File(filename, "w")

    # Loop to find duplicates and write to file
    for pce in pces:
        print("PCE:", pce)

        dm = DuplicateMask(pce, ph_id_count_dict[pce], tof, verbose=verbose)

        duplicate_mask = dm.make_mask()
        print("duplicate_mask: ", len(duplicate_mask), duplicate_mask)
        print("number of duplicates: ", len(np.where(duplicate_mask == 0)[0]))
        print("percent events removed: {}%".format(100*len(np.where(duplicate_mask == 0)[0])/float(len(duplicate_mask))))
        grp = f.create_group("pce{}".format(pce))
        dset_all = grp.create_dataset("all", data=duplicate_mask)

    print("Finished writing {}".format(filename))

    run_time = datetime.now() - start_time
    print("Run-time: ", run_time)

    f.close()

    return filename

