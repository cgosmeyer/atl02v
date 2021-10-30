""" Time of Day calculation for the leading lower threshold crossing of a laser
fire.

Author:
    
    C.M. Gosmeyer

Date:

    Apr 2018

References:

    ATL02 ATBD chapter 2.

Required Inputs:

    CAL-10

"""

import h5py
import numpy as np 
import os
from datetime import datetime
from atl02v.shared.calproducts import ReadANC13, ReadANC27
from atl02v.shared.constants import d_USO, SF_USO, pces
from atl02v.shared.tools import find_nearest, flatten_mf_arrays, flatten_mf_arrays, pickle_out


class PCEVariables(object):
    def __init__(self, mf_amet_upper=None, mf_amet_lower=None, tx_cc=None, 
        TX_CC=None, amet_FirstT0MF=None, T0_effective=None, GPSTime_FirstT0MF=None,
        delta_GPSTime=None, GPSTime_T0=None, GPSTime_ll=None, DeltaTime_ll=None):
        """ Container for all PCE-specific variables/arrays.
        """
        self.mf_amet_upper=mf_amet_upper
        self.mf_amet_lower=mf_amet_lower
        self.tx_cc=tx_cc
        self.TX_CC=TX_CC
        self.amet_FirstT0MF=amet_FirstT0MF
        self.T0_effective=T0_effective
        self.GPSTime_FirstT0MF=GPSTime_FirstT0MF
        self.delta_GPSTime=delta_GPSTime
        self.GPSTime_T0=GPSTime_T0
        self.GPSTime_ll=GPSTime_ll
        self.DeltaTime_ll=DeltaTime_ll

class ReadATL01(object):
    """ Read all needed datasets.
    """
    def __init__(self, atl01, pce):

        self.mf_amet_upper = np.array(atl01['pce{}/a_alt_science/raw_pce_amet_mframe_hi'.format(pce)].value, dtype=np.float64)
        self.mf_amet_lower = np.array(atl01['pce{}/a_alt_science/raw_pce_amet_mframe_lo'.format(pce)].value, dtype=np.float64)
        self.raw_pce_mframe_cnt_ph = np.array(atl01['pce{}/a_alt_science_ph/raw_pce_mframe_cnt'.format(pce)].value, dtype=np.int64)
        self.ph_id_pulse = np.array(atl01['pce{}/a_alt_science_ph/ph_id_pulse'.format(pce)].value, dtype=np.int64)    
        self.raw_tx_leading_coarse = np.array(atl01['pce{}/a_alt_science_ph/raw_tx_leading_coarse'.format(pce)].value, dtype=np.float64)

class TOD(object):
    __name__ = 'TOD'
    def __init__(self, atl01_file, anc13_path=None, anc27_path=None, verbose=False, mf_limit=500):
        """ TOD class.

        Parameters
        ----------
        atl01_file : str
            The name of the ATL01 H5 file.
        anc13_path : str
            Path to ANC13 files, used to retrieve laser, side, and
            laser energy level.
        anc27_path : str
            Path to the ANC27 file, used for retrieving CAL-10 (f_cal).
        verbose : {True, False}
            On by default. Print read-in and calculated final arrays.
        mf_limit : int
            Limit to number of major frames processed. By default,
            no limit and whole ATL01 is used.
        """
        # Begin timer.
        start_time = datetime.now()
        print("TOD start time: ", start_time)

        # Begin defining attributes.
        self.verbose = verbose

        self.atl01_file = atl01_file
        self.atl01 = h5py.File(self.atl01_file, 'r', driver=None)
        self.anc13_path = anc13_path
        self.anc27_path = anc27_path
        self.mf_limit = mf_limit

        if self.mf_limit != None:
            print("Limiting number of MFs processed to {}".format(self.mf_limit))

        # Get date range of ATL01.
        self.start_utc = self.atl01['data_start_utc'].value[0].decode()

        # Retrieve constants.
        self.anc13 = ReadANC13(self.anc13_path)
        self.side = self.anc13.side_spd
        self.anc27 = ReadANC27(self.anc27_path, side=self.side, atl02_date=self.start_utc)
        self.f_cal = self.anc27.select_value('uso_freq')
        #self.f_cal = f_cal(self.anc27_file, self.side)
        self.d_USO = d_USO
        self.GPSTime_Epoch =  np.array(self.atl01['atlas_sdp_gps_epoch'].value, dtype=np.float64) # seconds
        self.SF_USO = np.float64(1.0) #np.float64(0.999999999999902)  # np.float64(SF_USO(self.f_cal))  # temporary test value

        # Retrieve amet arrays.
        self.simhk_amet_64high = np.array(self.atl01['a_sim_hk_1026/raw_amet_64_bit_hi'].value, dtype=np.float64)
        self.simhk_amet_64low = np.array(self.atl01['a_sim_hk_1026/raw_amet_64_bit_lo'].value, dtype=np.float64)
        self.simhk_amet_sc1pps = np.array(self.atl01['a_sim_hk_1026/raw_amet_at_sc_{}_1pps'.format(self.side.lower())].value, dtype=np.float64)

        # Retrieve GPS arrays.
        self.gps_sec_sc1pps = np.array(self.atl01['a_sim_hk_1026/raw_gps_of_used_sc_1pps_secs'].value, dtype=np.float64)
        self.gps_subsec_sc1pps = np.array(self.atl01['a_sim_hk_1026/raw_gps_of_used_sc_1pps_sub_secs'].value, dtype=np.float64)

        # Equation 2-2.
        for i in range(len(self.simhk_amet_64low)):
            if self.simhk_amet_64low[i] < self.simhk_amet_sc1pps[i]:
                # Reduce by 1 count.
                self.simhk_amet_64high[i] -= 1

        self.amet_sc1pps = self.amet_sc1pps()

        if self.verbose:
            print("read H5 file: ", self.atl01_file)
            print("side: ", self.side)
            print("simhk_amet_64high: ", len(self.simhk_amet_64high), self.simhk_amet_64high)
            print("simhk_amet_64low: ", len(self.simhk_amet_64low), self.simhk_amet_64low)
            print("simhk_amet_sc1pps: ", len(self.simhk_amet_sc1pps), self.simhk_amet_sc1pps)
            print("gps_sec_sc1pps: ", len(self.gps_sec_sc1pps), self.gps_sec_sc1pps)
            print("gps_subsec_sc1pps: ", len(self.gps_subsec_sc1pps), self.gps_subsec_sc1pps)
            print("amet_sc1pps: ", len(self.amet_sc1pps), self.amet_sc1pps)
            print("f_cal: ", self.f_cal)

        # Read the atl01 dataframe.
        self.atl01_dict = {}
        self.atl01_dict[1] = ReadATL01(self.atl01, pce=1)
        self.atl01_dict[2] = ReadATL01(self.atl01, pce=2)
        self.atl01_dict[3] = ReadATL01(self.atl01, pce=3)
        
        # Close ATL01 and set to None, since can't pickle an H5 file.
        self.atl01.close() 
        self.atl01 = None

        # Calculate TODs.
        self.pce1 = self.calculate_TODs(pce=1)
        self.pce2 = self.calculate_TODs(pce=2)
        self.pce3 = self.calculate_TODs(pce=3)

        print("pce1.DeltaTime_ll: ", len(self.pce1.DeltaTime_ll), self.pce1.DeltaTime_ll)
        print("pce2.DeltaTime_ll: ", len(self.pce2.DeltaTime_ll), self.pce2.DeltaTime_ll)
        print("pce3.DeltaTime_ll: ", len(self.pce3.DeltaTime_ll), self.pce3.DeltaTime_ll)

        # Clear ATL01 dictionary because large.
        self.atl01_dict = None

        print("--TOD is complete.--")
        self.run_time = datetime.now() - start_time
        print("Run-time: ", self.run_time)

    def map_pce(self, pce):
        """ If you need a way to map PCE number to PCE attribute.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        if pce == 1:
            return self.pce1
        if pce == 2:
            return self.pce2
        if pce == 3:
            return self.pce3

    def calculate_TODs(self, pce):
        """ Calculates TOD for each PCE.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        mf_amet_upper = self.atl01_dict[pce].mf_amet_upper
        mf_amet_lower = self.atl01_dict[pce].mf_amet_lower
        # This tags each photon in the Major Frame
        raw_pce_mframe_cnt_ph = self.atl01_dict[pce].raw_pce_mframe_cnt_ph
        # This maps which laser fire "made" the returns. Many photons can have same pulse id.
        # It is not always true that pulse id is 1-200, though usually is.
        ph_id_pulse = self.atl01_dict[pce].ph_id_pulse      

        # This gives the major frame numnber of each shot
        mframe_counts = raw_pce_mframe_cnt_ph - raw_pce_mframe_cnt_ph[0]
        # This gives the list of major frames. Length = number of major frames.
        mframes = list(set(mframe_counts))

        # Leading lower element coarse counts.
        tx_cc = self.tx_cc(pce) #equations.tx_cc(self.atl01, pce)
        TX_CC = self.TX_CC(tx_cc)

        if self.verbose:
            print(" ")
            print("pce: ", pce)
            print("mframe_counts: ", len(mframe_counts), mframe_counts)
            print("tx_cc: ", len(tx_cc), tx_cc)
            print("TX_CC: ", len(TX_CC), TX_CC)

        # Initialize lists to store values per MF.
        amet_FirstT0MF_permf = []
        GPSTime_FirstT0MF_permf = []
        delta_GPSTime_permf = []
        GPSTime_T0_permf = []
        GPSTime_ll_permf = []
        DeltaTime_ll_permf = []
        mf_amet_upper_permf = []
        mf_amet_lower_permf = []
        tx_cc_permf = []
        TX_CC_permf = []

        if self.mf_limit != None:
            end_mf = self.mf_limit
        else:
            end_mf = len(mframes)

        for mframe in mframes[:end_mf]:
            print("PCE {}, mframe {}".format(pce, mframe))

            mframe = int(mframe)
            # mask out all events not in major frame.
            mframe_mask = np.where(mframe_counts == mframe)[0]
            TX_CC_mframe = TX_CC[mframe_mask]
            ph_id_pulse_mframe = ph_id_pulse[mframe_mask]
            # List unique pulse IDs available in the major frame
            pulses = list(set(ph_id_pulse_mframe))

            amet_FirstT0MF = self.amet_FirstT0MF(mf_amet_upper[mframe], mf_amet_lower[mframe])
            GPSTime_FirstT0MF = self.GPSTime_FirstT0MF(pce, amet_FirstT0MF)

            # Do a first calculation of T0_effective and GPSTime_T0
            # Need the GPS time for each T0 in order to establish the GPS
            # time of each transmit (Tx) that uses T0 as reference
            T0_effective = self.T0_effective(pce, TX_CC_mframe, ph_id_pulse_mframe)
            GPSTime_T0 = self.GPSTime_T0(GPSTime_FirstT0MF, T0_effective) 

            # Determine GPS time of the LL now that know each shot's T0effective GPS time.
            GPSTime_ll = self.GPSTime_ll(pce, TX_CC_mframe, GPSTime_T0) 
            # ToD value for PCE, each MF?
            DeltaTime_ll = self.DeltaTime_ll(GPSTime_ll)

            # Append to lists.
            GPSTime_T0_permf.append(GPSTime_T0)
            GPSTime_ll_permf.append(GPSTime_ll)
            DeltaTime_ll_permf.append(DeltaTime_ll)
            amet_FirstT0MF_permf.append(amet_FirstT0MF)
            GPSTime_FirstT0MF_permf.append(GPSTime_FirstT0MF)

            mf_amet_upper_permf.append(mf_amet_upper[mframe])
            mf_amet_lower_permf.append(mf_amet_lower[mframe])
            tx_cc_permf.append(tx_cc[mframe_mask])
            TX_CC_permf.append(TX_CC[mframe_mask])

            if self.verbose:
                print("    mframe: ", mframe)
                print("    mframe_mask: ", len(mframe_mask), mframe_mask)
                print("    TX_CC_mframe: ", len(TX_CC_mframe), TX_CC_mframe)
                print("    ph_id_pulse_mframe: ", len(ph_id_pulse_mframe), ph_id_pulse_mframe)
                print("    amet_FirstT0MF: ", amet_FirstT0MF)
                print("    GPSTime_FirstT0MF: ", GPSTime_FirstT0MF)
                print("    T0_effective: ", T0_effective)
                print("    GPSTime_T0: ", GPSTime_T0)
                print("    GPSTime_ll: ", len(GPSTime_ll), GPSTime_ll)
                print("    DeltaTime_ll: ", len(DeltaTime_ll), DeltaTime_ll)
 
        # Flatten the arrays in major frame key. Only want one array.
        GPSTime_T0_permf = flatten_mf_arrays(GPSTime_T0_permf)
        GPSTime_ll_permf = flatten_mf_arrays(GPSTime_ll_permf)
        DeltaTime_ll_permf = flatten_mf_arrays(DeltaTime_ll_permf)
        amet_FirstT0MF_permf = np.array(amet_FirstT0MF_permf).flatten()
        GPSTime_FirstT0MF_permf = np.array(GPSTime_FirstT0MF_permf).flatten()
        mf_amet_upper_permf = np.array(mf_amet_upper_permf).flatten()
        mf_amet_lower_permf = np.array(mf_amet_lower_permf).flatten()
        tx_cc_permf = np.array(tx_cc_permf).flatten()
        TX_CC_permf = np.array(TX_CC_permf).flatten()     

        # Return all values for this PCE in a named tuple.
        pce_variables = PCEVariables(
            mf_amet_upper=mf_amet_upper_permf, #mf_amet_upper, 
            mf_amet_lower=mf_amet_lower_permf, #mf_amet_lower, 
            tx_cc=tx_cc_permf, #tx_cc, 
            TX_CC=TX_CC_permf, #TX_CC,
            amet_FirstT0MF=amet_FirstT0MF_permf, 
            T0_effective=T0_effective, 
            GPSTime_FirstT0MF=GPSTime_FirstT0MF_permf, 
            GPSTime_T0=GPSTime_T0_permf, 
            GPSTime_ll=GPSTime_ll_permf, 
            DeltaTime_ll=DeltaTime_ll_permf) 

        return pce_variables
        
    ###################################
    # 0. Construct AMET counter values
    ###################################

    def amet_sc1pps(self):
        """ Equation 2-1.
        64 bit expression of the latched AMET at spacecraft 1pps.
        """
        return np.float64((256**4 * self.simhk_amet_64high) + self.simhk_amet_sc1pps)

    def amet_FirstT0MF(self, mf_amet_upper, mf_amet_lower):
        """ Equation 2-3.
        Value of a given PCE’s first T0 in a major frame.

        Parameters
        ----------
        mf_amet_upper : array
            Highest 32 bits of 64 bit AMET counter, generated by Flight
            Software at time of the first T0 in the major frame
        mf_amet_lower : array
            Lowest 32 bits of 64 bit AMET counter, generated by Flight
            Software at time of the first T0 in the major frame    
        """
        return np.float64((256**4 * mf_amet_upper) + mf_amet_lower)

    ################################################
    # Determine GPS time of first T0 in Major Frame
    ################################################

    def GPSTime_FirstT0MF(self, pce, amet_FirstT0MF):
        """ Equation 2-4.
        GPS time of the first T0 in each major frame.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        amet_FirstT0MF : array
            Value of a given PCE’s first T0 in each major frame.
        """
        # select gps_sc_sc1pps and gps_subsec_sc1pps 
        # based on amet_sc1pps which is nearest neighbor to amet_FirstT0MF
        nearest_amet_sc1pps, nearest_idx = find_nearest(self.amet_sc1pps, 
            amet_FirstT0MF)

        return np.float64((self.gps_sec_sc1pps[nearest_idx] + \
            self.gps_subsec_sc1pps[nearest_idx]) + \
            (amet_FirstT0MF - nearest_amet_sc1pps)*self.d_USO*self.SF_USO)

    #########################################################
    # Determine effective shot (T0) count within Major Frame
    #########################################################

    def tx_cc(self, pce):
        """ Equation 3-7.
        Start event value in units of USO course-clock cycles.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        return self.atl01_dict[pce].raw_tx_leading_coarse

    def TX_CC(self, tx_cc, tx_cal_ccoffset=-1, tx_smbit=0): 
        """ Equation 2-9.

        The calibrated TX course-count.

        If ll: tx_cc + tx_cal_cccoffset
        If other: tx_cc + tx_cal_ccoffset + tx_smbit

        Parameters
        ----------
        tx_cc : 
            Raw tx course count.
        tx_cal_ccoffset : int
            Start time offset.
        tx_smbit : int
            Start time marker.
        """
        return np.float64(tx_cc + tx_cal_ccoffset + tx_smbit)

    def T0_effective(self, pce, TX_CC_mframe, ph_id_pulse_mframe):
        """ Equation 2-5.

        T0_effective is the number of shots in major frame.
        Important to get this right because each coarse count it 
        expressed relative to a T0.

        To calculate the effective T0 of every laser fire within a major 
        frame, begin by constructing an array of elements, j = [j0…jn],
        where all values of j are set to 0 and the length of j is equal 
        to the number of Tx tags in the major frame.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        TX_CC_mframe : array
            The calibrated TX course-counts for given major frame.
        ph_id_pulse_mframe : array
            The pulse IDs for given major frame.
        """
        # j is length of the major frame
        j = np.zeros(len(TX_CC_mframe)) 

        # Loop over each event in one major frame
        for i in range(0, len(j)-1):
            T_diff = TX_CC_mframe[i+1] - TX_CC_mframe[i] 
            
            # Add only if the next pulse ID is different from the previous
            # Note this algorithm is no longer in the ATBD.
            if ph_id_pulse_mframe[i] != ph_id_pulse_mframe[i+1]:
                if T_diff >= -10 and T_diff <= 10:
                    j[i+1] = 1
                elif T_diff >= (10000-10) and T_diff <= (10000+10):
                    j[i+1] = 0
                elif T_diff >= (-10000-10) and T_diff <= (-10000+10):
                    j[i+1] = 2 

        # Return sum per position, a cumulative sum the size of the input array.
        return np.float64(np.cumsum(j))

    ########################################
    # Calculation of GPS Time at T0 of shot
    ########################################

    def GPSTime_T0(self, GPSTime_FirstT0MF, T0_effective):
        """ Equation 2-6.
        Calculate GPS time once each shot's effective T0 is found.

        GPSTime_FirstT0MF : array
            GPS time of the first T0 in each major frame.
        T0_effective : float
            Number of shots in major frame.
        """
        return np.float64(GPSTime_FirstT0MF + (T0_effective*np.float64(10000)*self.d_USO*self.SF_USO))

    ###################################################
    # Calculation of GPS Time of Leading Lower of shot
    ###################################################

    def GPSTime_ll(self, pce, TX_CC_mframe, GPSTime_T0_list):
        """ Equation 2-10.
        The time of an individual laser fires.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        TX_CC_mframe : array
            The calibrated TX course-counts for given major frame.
        GPSTime_T0_list : array
            GPS time of each T0 in given major frame.
        """
        return np.float64(np.array(GPSTime_T0_list) + (TX_CC_mframe*self.d_USO*self.SF_USO))

    def DeltaTime_ll(self, GPSTime_ll):
        """ Equation 2-11. *available in ATL02*

        The GPS times relative to the ATLAS standard data product (SDP) GPS Epoch.

        Parameters
        ----------
        GPSTime_ll : array
            The time of an individual laser fires.
        """
        return np.float64(GPSTime_ll - self.GPSTime_Epoch)

