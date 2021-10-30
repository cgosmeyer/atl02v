""" Transmitter Echo Path.

To detect a TE return, it must be in range:

RWS + DLBO < 10000N + tep_tol and 10000N - tep_tol < RWS + DLBO + DLBW

Author:
    
    C.M. Gosmeyer

Date:

    Apr 2018

References:

    ATL02 ATBD chapter 4.

Required Inputs:

    TOF calculation

"""

import h5py
import math
import numpy as np 
from datetime import datetime
from multiprocessing import Pool
from pydl import uniq
from atl02v.tof import tof
from atl02v.shared.constants import d_USO, SF_USO, pces, channels
from atl02v.shared.tools import flatten_mf_arrays, map_channel2spot, map_mf2rx, pickle_out

 
class PCEVariables(object):
    def __init__(self, TX_CENT_T_j=None, TX_CENT_T_jplusN=None, 
        TX_T_ll_jplusN=None, TX_T_other_jplusN=None, N=None, TX_T_ll=None, 
        TX_T_other=None, Elapsed_T=None, TOF_TEP=None, delta_time=None, 
        master_ids=None, tof_flag=None, filtered_TOF_TEP=None, 
        filtered_N=None, filtered_delta_time=None, filtered_TX_T_ll=None, 
        filtered_TX_T_other=None, filtered_TX_T_ll_jplusN=None,
        filtered_TX_T_other_jplusN=None, filtered_master_ids=None, 
        filtered_tof_flag=None):
        self.TX_CENT_T_j = TX_CENT_T_j
        self.TX_CENT_T_jplusN = TX_CENT_T_jplusN
        self.TX_T_ll_jplusN = TX_T_ll_jplusN
        self.TX_T_other_jplusN = TX_T_other_jplusN
        self.N = N
        self.TX_T_ll = TX_T_ll
        self.TX_T_other = TX_T_other
        self.Elapsed_T = Elapsed_T
        self.TOF_TEP = TOF_TEP
        self.delta_time = delta_time
        self.master_ids = master_ids
        self.tof_flag = tof_flag
        self.filtered = FilterContainer(filtered_TOF_TEP, filtered_N, 
            filtered_delta_time,
            filtered_TX_T_ll, filtered_TX_T_other, 
            filtered_TX_T_ll_jplusN, filtered_TX_T_other_jplusN, 
            filtered_master_ids, filtered_tof_flag)

class FilterContainer(object):
    def __init__(self, TOF_TEP=None, N=None, delta_time=None, TX_T_ll=None, 
        TX_T_other=None, TX_T_ll_jplusN=None, TX_T_other_jplusN=None, 
        master_ids=None, tof_flag=None):
        self.TOF_TEP = TOF_TEP
        self.N = N
        self.delta_time=delta_time
        self.TX_T_ll = TX_T_ll
        self.TX_T_other = TX_T_other
        self.TX_T_ll_jplusN = TX_T_ll_jplusN
        self.TX_T_other_jplusN = TX_T_other_jplusN
        self.master_ids = master_ids
        self.tof_flag = tof_flag

class ReadATL01(object):
    """ Read all needed ATL01 datasets.
    """
    def __init__(self, atl01, pce):
        self.raw_pce_mframe_cnt_ph = np.array(atl01['pce{}/a_alt_science_ph/raw_pce_mframe_cnt'.format(pce)].value, dtype=np.float64)


class TEP(object):
    __name__ = 'TEP'
    def __init__(self, atl01_file, anc27_path=None,  tod_pickle=None, 
        tof_pickle=None, verbose=False, very_verbose=False, multiprocess=True):
        """ 
        Parameters
        ----------
        atl01_file : str
            The name of the ATL01 H5 file.
        anc27_path : str
            [Optional] The path to the ANC27 file, used for retrieving CAL-10 (f_cal).
            Use only if calculating TOF from scratch.
        tod_pickle : str
            [Optional/Recommended] Name of the pickled TOD instance, for calculating
            from scratch TOD.
        tof_pickle : str
            [Optional/Recommended] Name of the pickled TOF instance. If this
            parameter is filled, TOD will not be calculated here.
        verbose : {True, False}
            On by default. Print read-in and calculated final arrays.
        very_verbose : {True, False}
            Off by default. Print in-between calculations in for loops.
        multiprocess: {True, False}
            Set on to multiprocess betweeen PCEs the TEP calculation.
            If verbose is False, will cut time from ~10 min to ~6 min.
        """ 
        # Begin timer.
        start_time = datetime.now()
        print("TEP start time: ", start_time)

        # Begin defining attributes.
        self.multiprocess = multiprocess
        self.verbose = verbose
        self.very_verbose = very_verbose

        self.atl01_file = atl01_file
        self.atl01 = h5py.File(self.atl01_file, 'r', driver=None)

        self.anc27_path = anc27_path
        self.tod_pickle = tod_pickle

        # Define nominal number of corse clocks per T0 interval.
        self.cc_per_T0 = np.float64(10000)

        # Tolerance value which takes up the maximum deviation of
        # laser fires from 10,000 and the maximum expected time between
        # laser fire and TE return. 70ns is nominal value.
        self.tep_tol = np.float64(70e-9)

        # From TOF output read RWS, DLBO, TX_T, T_center, TOFs
        if tof_pickle == None:
            self.tof = tof.TOF(self.atl01_file, tod_pickle=self.tod_pickle, anc27_path=self.anc27_path)
        else:
            self.tof = pickle_out(tof_pickle)

        self.tod = pickle_out(tod_pickle)

        # Read attributes from tof object and assign to self.
        self.side = self.tof.side
        self.d_USO = self.tof.d_USO
        self.f_cal = self.tof.f_cal
        self.SF_USO = self.tof.SF_USO
        self.aligner = self.tof.aligner

        # Read the ATL01 dataframe.
        self.atl01_dict = {}
        self.atl01_dict[1] = ReadATL01(self.atl01, pce=1)
        self.atl01_dict[2] = ReadATL01(self.atl01, pce=2)
        # Close ATL01 and set to None, since can't pickle an H5 file.
        # In addition, can't multiprocess with atl01 not None.
        self.atl01.close()
        self.atl01 = None

        # Find TEP-corrected TOF for each PCE.
        # TEP is only in PCEs 1 and 2.
        if self.multiprocess:
            # Parallelized TEP calculation.
            p = Pool(8)
            results = p.map(self.calculate_TEPs, [1,2])
            print(results)
            self.pce1 = results[0]
            self.pce2 = results[1]
        else:
            # Single-thread TEP calculation.
            self.pce1 = self.calculate_TEPs(pce=1)
            self.pce2 = self.calculate_TEPs(pce=2)

        print("pce1.filtered.TOF_TEP: ", len(self.pce1.filtered.TOF_TEP), self.pce1.filtered.TOF_TEP)
        print("pce2.filtered.TOF_TEP: ", len(self.pce2.filtered.TOF_TEP), self.pce2.filtered.TOF_TEP)

        print("--TEP is complete.--")
        self.run_time = datetime.now() - start_time
        print("Run-time: ", self.run_time)

    def map_pce(self, pce):
        """ If you need a way to map PCE number to PCE attribute.

        Parameters
        ----------
        pce : int
            1 or 2.
        """
        if pce == 1:
            return self.pce1
        if pce == 2:
            return self.pce2

    def calculate_TEPs(self, pce):
        """ Calculate TEPs and all variables leading to it.
        
        Note 1: 1/10 of the TOFs should be associated with a TEP.

        Note 2: TEP is only present in the STRONG beams of PCE1 and PCE2.

        Parameters
        ----------
        pce : int
            1 or 2.
        """
        # Retrieve the mf-indexed arrays.
        # Note these are 1xn_mf, already selected by band id.
        DLBO = self.tof.map_pce(pce).DLBO
        DLBW = self.tof.map_pce(pce).DLBW
        RWS = self.tof.map_pce(pce).RWS
        RWS_T = self.tof.map_pce(pce).RWS_T
        # Retrieve the Rx-indexed arrays.
        TOF = self.tof.map_pce(pce).TOF.all
        tof_flag = self.tof.map_pce(pce).tof_flag
        DeltaTime_ll = self.tod.map_pce(pce).DeltaTime_ll

        if self.verbose:
            print("pce: ", pce)
            print("DLBO: ", len(DLBO), DLBO)
            print("DLBW: ", len(DLBW), DLBW)
            print("RWS: ", len(RWS), RWS)
            print("RWS_T: ", len(RWS_T), RWS_T)
            print("TOF: ", len(TOF), TOF)
            print("tof_flag: ", len(tof_flag), tof_flag)

        # Initialize master ID; this will get indexed down same as TOF,
        # so in the end, I should be able to map directly back to the original
        # ATL01 photons.
        master_ids = np.arange(len(TOF))

        # Retrieve the Tx-indexed arrays.
        TX_T_ll = np.nan_to_num(self.tof.map_pce(pce).aligned.TX_T_ll)
        TX_T_other = np.nan_to_num(self.tof.map_pce(pce).aligned.TX_T_other)
        T_center = self.tof.T_center

        # Create an array of ints the length of Tx's
        transmits = np.arange(len(TX_T_ll))

        # Create an array of ints the length of Rx's, whose values map directly back to transmits.
        receives = self.aligner.maptx2rx(pce=pce, aligned_data=transmits)

        # Shift so that no zero values are included.
        nonzeros = np.where(TX_T_ll != 0)[0]
        transmits = transmits[nonzeros]

        # Initialize first index of receives.
        start_receive = 0

        if self.verbose:
            print("len(TX_T_ll): ", len(TX_T_ll))
            print("transmits: ", len(transmits), transmits)
            print("receives: ", len(receives), receives)

        # Find indexes of strong spot returns.
        strong_ind = np.where(self.tof.map_pce(pce).channel < 17)[0]
        # Index receives, TOF, and master_ids to include only strongs.
        strong_receives = receives[strong_ind]
        strong_TOF = TOF[strong_ind]
        strong_master_ids = master_ids[strong_ind]

        if self.verbose:
            print("strong_TOF: ", len(strong_TOF), strong_TOF)
            print("strong_receives: ", len(strong_receives), strong_receives)
            print("strong_master_ids: ", len(strong_master_ids), strong_master_ids)

        # Need to map RWS and DLBO first from MF to returns.
        RWS_rx = map_mf2rx(RWS, self.atl01_dict[pce].raw_pce_mframe_cnt_ph)
        RWS_T_rx = map_mf2rx(RWS_T, self.atl01_dict[pce].raw_pce_mframe_cnt_ph)
        DLBO_rx = map_mf2rx(DLBO, self.atl01_dict[pce].raw_pce_mframe_cnt_ph)
        DLBW_rx = map_mf2rx(DLBW, self.atl01_dict[pce].raw_pce_mframe_cnt_ph)

        # Then map to transmit space.
        RWS_tx = np.nan_to_num(self.aligner.align2timetags(pce, RWS_rx))
        RWS_T_tx = np.nan_to_num(self.aligner.align2timetags(pce, RWS_T_rx))
        DLBO_tx = np.nan_to_num(self.aligner.align2timetags(pce, DLBO_rx))
        DLBW_tx = np.nan_to_num(self.aligner.align2timetags(pce, DLBW_rx))

        if self.verbose:
            print("RWS_tx: ", len(RWS_tx), RWS_tx)
            print("DLBO_tx: ", len(DLBO_tx), DLBO_tx)
            print("DLBW_tx: ", len(DLBW_tx), DLBW_tx)

        # Initialize output lists and arrays.
        # 'j+N' is the real location of the TEP; j is where is recorded
        TX_CENT_T_j_list = []
        TX_CENT_T_jplusN_list = []
        TX_T_ll_jplusN_list = np.zeros(len(strong_receives))
        TX_T_other_jplusN_list = np.zeros(len(strong_receives))
        Elapsed_T_list = []
        # TOF TEP and N need be length of strong receives.
        N_list = np.zeros(len(strong_receives))
        TOF_TEP = np.zeros(len(strong_receives))
        # And initialize a delta_time list.
        delta_time = np.copy(DeltaTime_ll)

        # Calculate equations 4-6 to 4-9 per Tx. Then for each Rx in that
        # Tx, calculate equation 4-10.

        # j starts at transmit index of whatever is the first non-zero value
        # from the aligned TX_T_ll. Therefore, all the tx-space arrays that
        # were filled using the 'aligner' object don't need to be filtered
        # to remove pre- and post-pended zeros (which were present so that
        # arrays are same size across PCEs).
        for j in transmits:

            ## This should only be done for the first event in MF?
            RWS_j = RWS_tx[j]
            RWS_T_j = RWS_T_tx[j]
            DLBO_j = DLBO_tx[j]
            DLBW_j = DLBW_tx[j]

            # Determine N, the integer value that indicates there may
            # be a TE return in the down link band.
            # N is calculated once per major frame.
            N0, N1 = self.N(RWS_j, DLBO_j, DLBW_j) 

            if (N0 == N1) and ((N0+j) < len(transmits)):

                N = N0

                # Equation 4-7
                TX_CENT_T_j = (TX_T_ll[j] + T_center[j])    

                # Equation 4-8
                TX_CENT_T_jplusN = (TX_T_ll[j+N] + T_center[j+N])

                # Return and the one it was improperly associated with.
                Elapsed_T = self.Elapsed_T(N)

                # Count how many Rx's are in the jth Tx.
                receives_in_transmit = (strong_receives == j).sum()

                if self.very_verbose:
                    print("N: {}, j: {}".format(N,j))
                    print("TX_T_ll[j]: ", TX_T_ll[j])
                    print("T_center[j]: ", T_center[j])
                    print("TX_CENT_T_j: ", TX_CENT_T_j)
                    print("TX_T_ll[j+N]: ",  TX_T_ll[j+N])
                    print("T_center[j+N]: ", T_center[j+N])
                    print("TX_CENT_T_jplusN: ", TX_CENT_T_jplusN)
                    print("Elapsed_T: ", Elapsed_T)
                    print("start_receive: {}, receives_in_transmit: {}"\
                        .format(start_receive, receives_in_transmit))

                # Equation 4-10.
                # Index over all Rx's in the jth Tx to calculate TEP TOF.
                TOF_TEP[start_receive:start_receive+receives_in_transmit] = \
                    strong_TOF[start_receive:start_receive+receives_in_transmit] - \
                    (TX_CENT_T_jplusN - TX_CENT_T_j) - Elapsed_T

                # Likewise index N_list, TX_T_ll_jplusN, and TX_T_other_jplusN, 
                # for record-keeping's sake.
                N_list[start_receive:start_receive+receives_in_transmit] = N
                TX_T_ll_jplusN_list[start_receive:start_receive+receives_in_transmit] =  TX_T_ll[j+N]
                TX_T_other_jplusN_list[start_receive:start_receive+receives_in_transmit] = TX_T_other[j+N]

                # Add the MF-specific RWS to the delta_time.
                delta_time[start_receive:start_receive+receives_in_transmit] += RWS_T_j

                # Increment the start receive so it will fall at the next transmit.
                start_receive += receives_in_transmit

                # Append intermediate-value lists.
                TX_CENT_T_j_list.append(TX_CENT_T_j)
                TX_CENT_T_jplusN_list.append(TX_CENT_T_jplusN)

                Elapsed_T_list.append(Elapsed_T)

            else:
                # End the loop.
                N_list[start_receive:] = N
                TX_T_ll_jplusN_list[start_receive:] = TX_T_ll[j+N]
                TX_T_other_jplusN_list[start_receive:] = TX_T_other[j+N]
                delta_time[start_receive:] += RWS_T_j

                # Filter so only include values between 0 and 100 ns.
                filtered_ind = self.filter_indices(TOF_TEP)
                filtered_TOF_TEP = self.__filter(TOF_TEP, filtered_ind)
                if self.verbose:
                    print("filtered_ind: ", len(filtered_ind), filtered_ind)
                    print("filtered_TOF_TEP: ", len(filtered_TOF_TEP), filtered_TOF_TEP)

                # Filter remainder of receive arrays so same shape.
                filtered_N = self.__filter(N_list, filtered_ind)
                filtered_master_ids = self.__filter(strong_master_ids, filtered_ind)
                filtered_delta_time = self.__filter(delta_time, filtered_ind)
                filtered_TX_T_ll_jplusN = self.__filter(TX_T_ll_jplusN_list, filtered_ind)
                filtered_TX_T_other_jplusN = self.__filter(TX_T_other_jplusN_list, filtered_ind)
                if self.verbose:
                    print("filtered_N: ", len(filtered_N), filtered_N)
                    print("filtered_master_ids: ", len(filtered_master_ids), filtered_master_ids)
                    print("filtered_delta_times: ", len(filtered_delta_time), filtered_delta_time)
                    print("filtered_TX_T_ll_jplusN: ", len(filtered_TX_T_ll_jplusN), filtered_TX_T_ll_jplusN)
                    print("filtered_TX_T_other_jplusN: ", len(filtered_TX_T_other_jplusN), filtered_TX_T_other_jplusN)

                # To filter the aligned-transmit arrays, need first fill them out to
                # receive space. 
                TX_T_ll_rx = self.aligner.maptx2rx(pce, TX_T_ll) 
                TX_T_other_rx = self.aligner.maptx2rx(pce, TX_T_other)
                if self.verbose:
                    print("TX_T_ll_rx: ", len(TX_T_ll_rx), TX_T_ll_rx)
                    print("TX_T_other_rx: ", len(TX_T_other_rx), TX_T_other_rx)
                # Then index down to strong-only receives.
                TX_T_ll_rx_strong = TX_T_ll_rx[strong_ind]
                TX_T_other_rx_strong = TX_T_other_rx[strong_ind]
                if self.verbose:
                    print("TX_T_ll_rx_strong: ", len(TX_T_ll_rx_strong), TX_T_ll_rx_strong)
                    print("TX_T_other_rx_strong: ", len(TX_T_other_rx_strong), TX_T_other_rx_strong)
                # Finally filter them.
                filtered_TX_T_ll = self.__filter(TX_T_ll_rx_strong, filtered_ind)
                filtered_TX_T_other = self.__filter(TX_T_other_rx_strong, filtered_ind)
                if self.verbose:
                    print("filtered_TX_T_ll", len(filtered_TX_T_ll), filtered_TX_T_ll)
                    print("filtered_TX_T_other", len(filtered_TX_T_other), filtered_TX_T_other)

                # Update the tof_flag.
                print("master_ids: ", len(master_ids), master_ids)
                print("strong_ind: ", len(strong_ind), strong_ind)
                print("filtered_ind: ", len(filtered_ind), filtered_ind)
                tep_ids = strong_ind[filtered_ind]
                print("tep_ids = strong_ind[filtered_ind]: ", len(tep_ids), tep_ids)
                filtered_tof_flag = np.copy(tof_flag)
                filtered_tof_flag[tep_ids] += 10
                print("filtered_tof_flag: ", len(filtered_tof_flag), filtered_tof_flag)
                # strong_ids, strong_arr, rx_arr
                tof_flag = map_tep2rx(tep_ids=tep_ids, tep_arr=filtered_tof_flag[tep_ids], rx_arr=tof_flag)
                print("tof_flag: ", len(tof_flag), tof_flag)

                # Store values for return.
                pce_variables = PCEVariables(
                    TX_CENT_T_j =  np.array(TX_CENT_T_j_list),
                    TX_CENT_T_jplusN = np.array(TX_CENT_T_jplusN_list),
                    TX_T_ll_jplusN = np.array(TX_T_ll_jplusN_list),
                    TX_T_other_jplusN = np.array(TX_T_other_jplusN_list),
                    N = N_list,
                    TX_T_ll=TX_T_ll_rx_strong,
                    TX_T_other=TX_T_other_rx_strong,
                    Elapsed_T = np.array(Elapsed_T_list),
                    TOF_TEP = np.array(TOF_TEP, dtype=np.float64),
                    delta_time = np.array(delta_time, dtype=np.float64),
                    master_ids = strong_master_ids,
                    tof_flag = tof_flag,
                    filtered_TOF_TEP = np.array(filtered_TOF_TEP, dtype=np.float64),
                    filtered_N = np.array(filtered_N),
                    filtered_delta_time = np.array(filtered_delta_time, dtype=np.float64),
                    filtered_TX_T_ll = np.array(filtered_TX_T_ll, dtype=np.float64),
                    filtered_TX_T_other = np.array(filtered_TX_T_other, dtype=np.float64),
                    filtered_TX_T_ll_jplusN = np.array(filtered_TX_T_ll_jplusN, dtype=np.float64),
                    filtered_TX_T_other_jplusN = np.array(filtered_TX_T_other_jplusN, dtype=np.float64),
                    filtered_master_ids = filtered_master_ids,
                    filtered_tof_flag = filtered_tof_flag)

                return pce_variables

    def filter_indices(self, TOF_TEP):
        """ Filter to only include TOFs of TEPs between 0 and 100e-9.
        These indices are based on strong-space, not rx-space.

        Parameters
        ----------
        TOF_TEP : array
            The TOFs of TEPs.

        Returns
        -------
        filter_indices : array
            Filtered TOF_TEPs.
        """
        filter_indices = np.where(np.logical_and(np.greater(TOF_TEP,0),np.less_equal(TOF_TEP,100e-9)))[0]
        return filter_indices

    def __filter(self, arr, ind):
        """ Filters array by indices.

        Parameters
        ----------
        arr : array
            Input array to be filtered.
        ind : array
            Indices to include in output array.
        """
        return arr[ind]

    def N(self, RWS, DLBO, DLBW):
        """ Equation 4-6.
        Helps determine whether or not a TE event can be observed within
        a given downlink band. If N0 is equal to N1 then assume there
        is a TE event present.

        Parameters
        ----------
        RWS : int
            Range window start of TE event.
        DLBO : int
            Downlink band offset of TE event.
        DLBW : int
            Downlink band width of TE event.
        """
        N0 = math.ceil((RWS + DLBO - self.tep_tol) / self.cc_per_T0)
        N1 = math.floor((RWS + DLBO + DLBW + self.tep_tol) / self.cc_per_T0)
        return N0, N1

    def Elapsed_T(self, N):
        """ Equation 4-9.
        The time elapsed between the shot which made the TE return and
        the one with which it was improperly associated.

        Parameters
        ----------
        N : int
            Value that indicates there may be a TE return in the down
            link band.
        """
        return N*self.cc_per_T0*self.d_USO*self.SF_USO


def map_tep2rx(tep_ids, tep_arr, rx_arr):
    """ Maps photons from TEP strong rx-space to full rx-space.

    tep_ids : array
        The TEP strong-space indices.
    tep_arr : array
        Array indexed by TEP strong rx-space.
    rx_arr : array
        Array indexed by full rx-space.
    """
    rx_arr[tep_ids] = tep_arr

    return rx_arr


def map_strong2tep(pce, tep, photonids, strong_arr):
    """ Maps photons from strong rx-space to tep-space.
    
    Parameters
    ----------
    pce : int
        1 or 2.
    tep : TEP
        TEP object.
    photonids : PhotonIDs
        PhotonIDs object.
    strong_arr : array
        An array in strong rx-space. Non-filtered.
    """
    master_ids = np.arange(len(tep.tof.map_pce(pce).TOF.all))

    # Find the ATL02 indices corresponding to each TEP.
    # These are just the master IDs of the strong photons
    tep_strong_master_ids = tep.map_pce(pce).master_ids

    # Create an array of length of all returns. Set all values
    # to -1. Then populate the strong positions with the
    # non-filtered TEP TOFs.
    rxspace = np.ones(len(master_ids)) * -1
    rxspace[tep_strong_master_ids] = strong_arr
    
    # Finally, filter down the array with the TEP master ids
    # from the photonid object.
    tep_arr = rxspace[photonids['pce{}/tep'.format(pce)].value]

    return tep_arr
