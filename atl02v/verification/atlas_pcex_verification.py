""" Module for atlas/pcex verification classes.

Author:

    C.M. Gosmeyer

"""

import h5py
import numpy as np
import pylab as plt
from atl02qa.verification.verification_tools import Verify
from atl02qa.shared.constants import pces
from atl02qa.shared.tools import find_nearest

def find_runs(x):
    """Find runs of consecutive items in an array.
    
    Source
    ------
    https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths

def calculate_useflag(atl01, delta_time, pce, verbose=False):
    """ Calculates useflag in groups atlas/altimetry, atlas/algorithm_science, 
    atlas/atmosphere_s, atlas/background.

    Parameters
    ----------
    atl01 : 
        Opened ATL01 H5 file.
    delta_time : array
        The ATL01 delta_time array of group to be given useflag in ATL02.
    pce : int
        1, 2, or 3.
    """
    pmf_mode = np.array(atl01['pce{}/a_pce_pmf_hk/mode'.format(pce)].value, dtype=np.float64)
    amcs_mode = np.array(atl01['a_sbs_hk_1030/amcs_mode'].value, dtype=np.float64)
    pmf_delta_time = np.array(atl01['pce{}/a_pce_pmf_hk/delta_time'.format(pce)].value, dtype=np.float64)
    amcs_delta_time = np.array(atl01['a_sbs_hk_1030/delta_time'].value, dtype=np.float64)

    # Initalize useflag of ones at length (date rate) of target group (input delta_time)
    useflag = np.ones(len(delta_time))

    # Pad at end of a transition period.
    pad = 1 # seconds
    
    # Evaluate PMF and AMCS modes for mode transitions from 1 (science).
    
    # First start with PMF mode
    pmf_chunk_values, pmf_chunk_starts, pmf_chunk_lengths = find_runs(pmf_mode)
    
    # Find whether there are PMF values other than 1.
    # If so, loop over the values to find start and ending delta_times.
    pmf_values = list(set(pmf_chunk_values))
    
    if pmf_values[0] != 1:
        for v, s, l in zip(pmf_chunk_values, pmf_chunk_starts, pmf_chunk_lengths):
            if v != 1:
                # Check for chunk start equally start of PMF array.
                # Want to ensure that start on PMF delta_time in that case.
                if s == 0:
                    pmf_time_chunk_beg = delta_time[0]
                else:
                    pmf_time_chunk_beg = pmf_delta_time[s]
                # Check whether chunk length equal to size of the PMF array.
                # If so, decrease by 1 so don't get error.
                if l == len(pmf_mode):
                    l =- 1
                pmf_time_chunk_end = pmf_delta_time[s+l] + pad
                beg_idx = np.where(delta_time >= pmf_time_chunk_beg)[0][0]
                end_idx = np.where(delta_time < pmf_time_chunk_end)[0][-1] + 1
                if verbose:
                    print("pmf_time_chunk_beg: ", pmf_time_chunk_beg)
                    print("pmf_time_chunk_end: ", pmf_time_chunk_end)  
                    print("beg_idx: ", beg_idx)
                    print("end_idx: ", end_idx)
                useflag[beg_idx:end_idx] = v 

    # Second find AMCS values other than 1, and when found, add 10 to useflag.
    amcs_not_one = np.where(amcs_mode != 1)[0]
    if len(amcs_not_one) != 0:
        amcs_chunk_values, amcs_chunk_starts, amcs_chunk_lengths = find_runs(amcs_mode)
        for v, s, l in zip(amcs_chunk_values, amcs_chunk_starts, amcs_chunk_lengths):
            if v != 1: 
                # Check for chunk start equally start of AMCS array.
                # Want to ensure that start on AMCS delta_time in that case.
                if s == 0:
                    amcs_time_chunk_beg = delta_time[0]
                else:
                    amcs_time_chunk_beg = amcs_delta_time[s]
                # Check whether chunk length equal to size of the AMCS array.
                # If so, decrease by 1 so don't get error.
                if l == len(amcs_mode):
                    l =- 1
                amcs_time_chunk_end = amcs_delta_time[s+l] + pad
                beg_idx = np.where(delta_time >= amcs_time_chunk_beg)[0][0]
                end_idx = np.where(delta_time < amcs_time_chunk_end)[0][-1] + 1 
                if verbose:
                    print("amcs_time_chunk_beg: ", amcs_time_chunk_beg)
                    print("amcs_time_chunk_end: ", amcs_time_chunk_end)  
                    print("beg_idx: ", beg_idx)
                    print("end_idx: ", end_idx)                    
                useflag[beg_idx:end_idx] += 10
    
    if verbose:
        print("useflag: ", useflag)

    return useflag 

class VerifyPhotons(Verify):
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)

    def __verify_single(self, pce, spot, custom_func):
        """
        """
        print("pce {}, spot {}".format(pce,spot))

        # Create the base out name.
        self.base_filename = 'atlas.pce{}.altimetry.{}.photons.{}'.format(pce, spot, self.atl02_dataset)

        # Read the values from ATL02.
        atl02_values = self.atl02['pce{}/altimetry/{}/photons/{}'.format(pce, spot, self.atl02_dataset)].value

        # Read the dataset to be compared against ATL02.
        if self.atl02_dataset == 'tof_flag':
            # Need special case for strong/weak
            verifier_values = custom_func(pce, spot)
        else:
            verifier_values = custom_func(pce)

        # Read the Photon IDs.
        photonids = np.array(self.photonids['pce{}/{}'.format(pce, spot)].value)

        # Index my array so match ATL02's using PhotonID hdf5 file.
        verifier_matched_values = verifier_values[photonids]

        # Special case for rx_band_id, to make sure all values identical.
        # So reduces the array to unique values, which if valid, should be
        # only a single value.
        if self.atl02_dataset == 'rx_band_id':
            atl02_values = np.array(list(set(atl02_values)))
            verifier_matched_values = np.array(list(set(verifier_matched_values)))

        # Diff the arrays, plot diff, and record statistics.
        self.compare_arrays(atl02_values, verifier_matched_values)

    def do_verify(self, custom_func):
        for pce in pces:
            for spot in ['strong', 'weak']:
                try:
                    self.__verify_single(pce, spot, custom_func)
                except Exception as error:
                        self.record_exception(error)

class VerifyStrongWeak(Verify):
    """ The strong/weak levels.
    """
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)

    def __verify_single(self, pce, spot, custom_func):
        """
        """
        print("pce {}, spot {}".format(pce,spot))

        # Create the base out name.
        self.base_filename = 'pce{}.altimetry.{}.{}'.format(pce, spot, self.atl02_dataset)

        # Read the values from ATL02.
        atl02_values = self.atl02['pce{}/altimetry/{}/{}'.format(pce, spot, self.atl02_dataset)].value

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func(pce, spot)

        # Diff the arrays, plot diff, and record statistics.
        self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        for pce in pces:
            for spot in ['strong', 'weak']:
                try:
                    self.__verify_single(pce, spot, custom_func)
                except Exception as error:
                    self.record_exception(error)

class VerifyAltimetry(Verify):
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)

    def __verify_single(self, pce, custom_func):
        """ 
        """
        # Create the base out name.
        self.base_filename = 'pce{}.altimetry.{}'.format(pce, self.atl02_dataset)

        # Read the values from ATL02.
        atl02_values = self.atl02['pce{}/altimetry/{}'.format(pce, self.atl02_dataset)].value

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func(pce)

        # Special case for n_bands. It should fall in range 1-4.
        if self.atl02_dataset == 'n_bands':
            if len(np.where((atl02_values < 1) | (atl02_values > 4))[0]) != 0:
                self.record_exception("n_bands out of range [1,4]")

        # Diff the arrays, plot diff, and record statistics.
        self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        for pce in pces:
            try:
                self.__verify_single(pce, custom_func)
            except Exception as error:
                self.record_exception(error)

class VerifyTep(Verify):
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)
        self.atl02_dataset = atl02_dataset

    def __verify_single(self, pce, custom_func):
        """
        """
        # Create the base out name.
        self.base_filename = 'pce{}.tep.{}'.format(pce, self.atl02_dataset)

        # Read the values from ATL02.
        atl02_values = self.atl02['pce{}/tep/{}'.format(pce, self.atl02_dataset)].value
        print("atl02_values: ", len(atl02_values), atl02_values)

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func(pce)
        print("verifier_values: ", len(verifier_values), verifier_values)

        # Read the Photon IDs.
        photonids = self.photonids['pce{}/tep'.format(pce)].value
        print("photonids: ", len(photonids), photonids)

        # Index ATL01 array so match ATL02's using PhotonID hdf5 file.
        # My values do not need be indexed because in theory we should have same lengths already.
        if self.atl02_dataset not in ['tep_pulse_num', 'tof_tep', 'tx_ll_tof_tep', 'tx_other_tof_tep']:
            verifier_matched_values = verifier_values[photonids]

            # Diff the arrays, plot diff, and record statistics.
            self.compare_arrays(atl02_values, verifier_matched_values)

        else:
            ## may need do something different because in reality our indices
            ## probably won't match exactly...
            ## How to characterize that case?
            self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        for pce in pces[:2]:
            try:
                self.__verify_single(pce, custom_func)
            except Exception as error:
                self.record_exception(error)

class VerifyAlgorithmScience(Verify):
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)

    def __verify_single(self, pce, custom_func):
        """ 
        """
        # Create the base out name.
        self.base_filename = 'pce{}.algorithm_science.{}'.format(pce, self.atl02_dataset)

        # Read the values from ATL02.
        atl02_values = self.atl02['pce{}/algorithm_science/{}'.format(pce, self.atl02_dataset)].value

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func(pce)

        # Diff the arrays, plot diff, and record statistics.
        self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        for pce in pces:
            try:
                self.__verify_single(pce, custom_func)
            except Exception as error:
                self.record_exception(error)

class VerifyAlgorithmScienceStrongWeak(Verify):
    """ The strong/weak levels.
    """
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)

    def __verify_single(self, pce, spot, custom_func):
        """
        """
        print("pce {}, spot {}".format(pce, spot))

        # Create the base out name.
        self.base_filename = 'pce{}.algorithm_science.{}.{}'.format(pce, spot, self.atl02_dataset)

        # Read the values from ATL02.
        atl02_values = self.atl02['pce{}/algorithm_science/{}/{}'.format(pce, spot, self.atl02_dataset)].value

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func(pce, spot)

        # Diff the arrays, plot diff, and record statistics.
        self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        for pce in pces:
            for spot in ['strong', 'weak']:
                try:
                    self.__verify_single(pce, spot, custom_func)
                except Exception as error:
                    self.record_exception(error)

class VerifyAtmosphereS(Verify):
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)

    def __verify_single(self, pce, custom_func):
        """ n
        """
        # Create the base out name.
        self.base_filename = 'pce{}.atmosphere_s.{}'.format(pce, self.atl02_dataset)

        # Read the values from ATL02.
        atl02_values = self.atl02['pce{}/atmosphere_s/{}'.format(pce, self.atl02_dataset)].value

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func(pce)

        # Diff the arrays, plot diff, and record statistics.
        self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        for pce in pces:
            try:
                self.__verify_single(pce, custom_func)
            except Exception as error:
                self.record_exception(error)

class VerifyBackground(Verify):
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)

    def __verify_single(self, pce, custom_func):
        """ 
        """
        # Create the base out name.
        self.base_filename = 'pce{}.background.{}'.format(pce, self.atl02_dataset)

        # Read the values from ATL02.
        atl02_values = self.atl02['pce{}/background/{}'.format(pce, self.atl02_dataset)].value

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func(pce)

        # Diff the arrays, plot diff, and record statistics.
        self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        for pce in pces:
            try:
                self.__verify_single(pce, custom_func)
            except Exception as error:
                self.record_exception(error)


