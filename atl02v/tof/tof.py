""" Time of Flight, per PCE, per channel.

    Start events (Tx) and receive events (Rx) are found using a
    fine + course time component.
    Tx is expressed relative to T0
    Rx is expressed relative to the LL of the start

Author:
    
    C.M. Gosmeyer

Date: 

    Apr 2018

References:

    ATL02 ATBD chapter 2.

    https://stackoverflow.com/questions/3253966/python-string-to-attribute?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

Current Shortcuts:

    1. Pretend no duplicates.
    2. No smoothing of FC_TDC. Note that rising and falling are read 
       directly from ATL02 pce{}/altimetry/cal_{}_sm
    3. SF_USO is hard-corded to 1.0.
    4. Any "0" in a_alt_science_ph/raw_* is setting all values in that row to 0.
       Which may be the correct thing to do, but it has not been evaluated. 

Possible Speed-Ups:

    1. Parallelize calculate_TOFs by PCE. 
    2. Write all calibrations into an H5 files, open it once and keep it open.
       JLee uses "anc21_20180403_01.h5"

Required Inputs:

    ANC13
    ANC27
    CAL-17
    CAL-49
    ALT01 file
    TOD calculation

"""

import h5py
import math
import numpy as np 
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
from atl02v.tod import tod
from atl02v.shared.constants import d_USO, SF_USO, pces, channels, dt_imet, dt_t0, lrs_clock
from atl02v.shared.calproducts import ReadANC13, ReadANC27, ReadKs, ReadCal17, ReadCal44, ReadCal49
from atl02v.shared.alignment import Aligner
from atl02v.shared.tools import flatten_mf_arrays, map_channel2spot, map_mf2rx, nan2zero, pickle_out

class PCEVariables(object):
    def __init__(self, tx_cc=None, TX_CC_ll=None, TX_CC_other=None, 
        tx_cal_ccoffset=None, tx_smbit=None, fc_TDC_falling=None, 
        FC_TDC_falling=None, fc_TDC_rising=None, FC_TDC_rising=None,
        tx_fc_ll=None, TX_FC_ll=None, tx_fc_other=None, TX_FC_other=None, 
        tx_t_ll=None, tx_t_other=None, spd_cal=None, TX_T_other=None, 
        TX_T_ll=None, TX_T_fine=None, rx_cc=None,  rx_cal_ccoffset=None,
        RX_CC=None, rx_fc=None,
        RX_FC=None, rx_t=None, RX_T=None, ch_cal=None, DLBO=None, 
        DLBW=None, RWW=None, RWS=None, RWW_T=None, RWS_T=None, 
        DLBO_T=None, DLBW_T=None, FC_TDC=None, channel=None, toggle=None, 
        majorframe=None, T_center_rx_mapped=None, aligned=None, TOF=None,
        tof_flag=None):
        """ Container for all PCE-specific variables/arrays.
        """
        self.tx_cc = tx_cc
        self.tx_cal_ccoffset=tx_cal_ccoffset
        self.TX_CC_ll=TX_CC_ll
        self.tx_smbit=tx_smbit
        self.TX_CC_other=TX_CC_other
        self.fc_TDC_falling=fc_TDC_falling
        self.FC_TDC_falling=FC_TDC_falling
        self.fc_TDC_rising=fc_TDC_rising
        self.FC_TDC_rising=FC_TDC_rising
        self.tx_fc_ll=tx_fc_ll
        self.TX_FC_ll=TX_FC_ll
        self.tx_fc_other=tx_fc_other
        self.TX_FC_other=TX_FC_other
        self.tx_t_ll=tx_t_ll
        self.tx_t_other=tx_t_other
        self.spd_cal=spd_cal
        self.TX_T_other=TX_T_other
        self.TX_T_ll=TX_T_ll
        self.TX_T_fine=TX_T_fine
        self.rx_cc=rx_cc
        self.rx_cal_ccoffset=rx_cal_ccoffset
        self.RX_CC=RX_CC
        self.rx_fc=rx_fc
        self.RX_FC=RX_FC
        self.rx_t=rx_t
        self.RX_T=RX_T
        self.ch_cal=ch_cal
        self.DLBO=DLBO
        self.DLBW=DLBW
        self.RWW=RWW
        self.RWS=RWS
        self.DLBO_T=DLBO_T
        self.DLBW_T=DLBW_T
        self.RWW_T=RWW_T
        self.RWS_T=RWS_T
        self.FC_TDC=FC_TDC
        self.channel=channel
        self.toggle=toggle
        self.majorframe=majorframe
        self.T_center_rx_mapped=T_center_rx_mapped
        # A sub-container to contain the variables that get aligned across PCEs along with delta_time
        self.aligned = AlignedContainer()
        ## A sub-container containing strong, weak, and "all" TOFs.
        self.TOF = TOFContainer()
        self.tof_flag = tof_flag

class AlignedContainer(object):
    def __init__(self, TX_T_other=None, TX_T_ll=None):
        """ Container for aligned arrays.
        """
        self.TX_T_other = TX_T_other
        self.TX_T_ll = TX_T_ll

class TOFContainer(object):
    def __init__(self, all=None, strong=None, weak=None,
        strong_ind=None, weak_ind=None):
        """ Container for TOF-specific arrays.
        """
        self.all = all
        self.strong = strong
        self.weak = weak
        self.strong_ind = strong_ind
        self.weak_ind = weak_ind

class ReadATL01(object):
    """ Read all needed ATL01 datasets.
    """
    def __init__(self, atl01, pce):
        self.raw_rx_channel_id = np.array(atl01['pce{}/a_alt_science_ph/raw_rx_channel_id'.format(pce)].value)
        self.raw_rx_toggle_flg = np.array(atl01['pce{}/a_alt_science_ph/raw_rx_toggle_flg'.format(pce)].value)
        self.raw_pce_mframe_cnt_ph = np.array(atl01['pce{}/a_alt_science_ph/raw_pce_mframe_cnt'.format(pce)].value, dtype=np.float64)
        self.ph_id_pulse = np.array(atl01['pce{}/a_alt_science_ph/ph_id_pulse'.format(pce)].value, dtype=np.int32)  

        self.raw_pce_mframe_cnt_ph = np.array(atl01['pce{}/a_alt_science_ph/raw_pce_mframe_cnt'.format(pce)].value)
        self.raw_rx_band_id = np.array(atl01['pce{}/a_alt_science_ph/raw_rx_band_id'.format(pce)].value)
        self.raw_rx_channel_id = np.array(atl01['pce{}/a_alt_science_ph/raw_rx_channel_id'.format(pce)].value)
        self.raw_rx_toggle_flg = np.array(atl01['pce{}/a_alt_science_ph/raw_rx_toggle_flg'.format(pce)].value)

        self.raw_tx_leading_coarse = np.array(atl01['pce{}/a_alt_science_ph/raw_tx_leading_coarse'.format(pce)].value, dtype=np.int32)
        self.raw_tx_start_marker = np.array(atl01['pce{}/a_alt_science_ph/raw_tx_start_marker'.format(pce)].value, dtype=np.int32)
        self.raw_alt_cal_rise = np.array(atl01['pce{}/a_alt_science/raw_alt_cal_rise'.format(pce)].value, dtype=np.float64)
        self.raw_alt_cal_fall = np.array(atl01['pce{}/a_alt_science/raw_alt_cal_fall'.format(pce)].value, dtype=np.float64)
        self.raw_tx_leading_fine = np.array(atl01['pce{}/a_alt_science_ph/raw_tx_leading_fine'.format(pce)].value, dtype=np.int32)
        self.raw_tx_trailing_fine = np.array(atl01['pce{}/a_alt_science_ph/raw_tx_trailing_fine'.format(pce)].value, dtype=np.int32)

        self.raw_rx_leading_coarse = np.array(atl01['pce{}/a_alt_science_ph/raw_rx_leading_coarse'.format(pce)].value, dtype=np.int32)
        self.raw_rx_leading_fine = np.array(atl01['pce{}/a_alt_science_ph/raw_rx_leading_fine'.format(pce)].value, dtype=np.int32)

        self.raw_alt_band_offset = pd.DataFrame(atl01['pce{}/a_alt_science/raw_alt_band_offset'.format(pce)].value)
        self.raw_alt_band_width = np.array(atl01['pce{}/a_alt_science/raw_alt_band_width'.format(pce)].value)
        self.raw_alt_ch_mask = np.array(atl01['pce{}/a_alt_science/raw_alt_ch_mask'.format(pce)].value)
        self.raw_alt_rw_start_s = np.array(atl01['pce{}/a_alt_science/raw_alt_rw_start_s'.format(pce)].value)
        self.raw_alt_rw_start_w = np.array(atl01['pce{}/a_alt_science/raw_alt_rw_start_w'.format(pce)].value)
        self.raw_alt_rw_width_s = np.array(atl01['pce{}/a_alt_science/raw_alt_rw_width_s'.format(pce)].value)
        self.raw_alt_rw_width_w = np.array(atl01['pce{}/a_alt_science/raw_alt_rw_width_w'.format(pce)].value)

class ReadATL02(object):
    """ Read all needed ATL02 datasets.
    """
    def __init__(self, atl02, pce):
        self.cal_rise_sm = np.array(atl02['pce{}/altimetry/cal_rise_sm'.format(pce)].value, dtype=np.float64)
        self.cal_fall_sm = np.array(atl02['pce{}/altimetry/cal_fall_sm'.format(pce)].value, dtype=np.float64)

class TOF(object):
    __name__ = 'TOF'
    def __init__(self, atl01_file, atl02_file=None, tod_pickle=None, 
        anc13_path=None, anc27_path=None, cal17_path=None, cal44_path=None, 
        cal49_path=None, verbose=False, very_verbose=False, multiprocess=True,
        qtest_mode=False, mf_limit=None):
        """ TOF class.

        Parameters
        ----------
        atl01_file : str
            The name of the ATL01 H5 file.
        atl02_file : str
            The name of the ATL02 H5 file.
        tod_pickle : str
            [Optional] Name of the pickled TOD instance. If this
            parameter is filled, TOD will not be calculated here.
        anc13_path : str
            Path to ANC13 files, used to retrieve laser, side, and
            laser energy level.
        anc27_path : str
            Path to ANC27 files, used for retrieving CAL-10 (f_cal).
        cal17_path : str
            Path to the CAL17 products, where its index file lives.
        cal44_path : str
            Path to the CAL44 products, where its index file lives.
        cal49_path : str
            Path to the CAL49 products, where its index file lives.
        verbose : {True, False}
            Off by default. Print read-in and calculated final arrays.
        very_verbose : {True, False}
            Off by default. Print in-between calculations in for loops.
        multiprocess: {True, False}
            Set on to multiprocess betweeen PCEs the TOF calculation.
            If verbose is False, will cut time from {...} to {...}.
        qtest_mode: {True, False}
            Quick Test Mode. Set on if want to run a quick 1-MF, 2-
            photon TOF calculation.
        mf_limit : int
            Limit to number of major frames processed. By default,
            no limit and whole ATL01 is used.
        """ 
        # Begin timer.
        start_time = datetime.now()
        print("TOF start time: ", start_time)

        # Begin defining attributes.
        self.multiprocess = multiprocess
        self.verbose = verbose
        self.very_verbose = very_verbose
        self.qtest_mode = qtest_mode
        self.mf_limit = mf_limit

        if self.qtest_mode:
            print("In qtest_mode! Will only generate TOF for 1 MF, 2 photons.")
        if self.mf_limit != None:
            print("Limiting number of MFs processed to {}".format(self.mf_limit))

        # Read the ATL01 and ATL02
        self.atl01_file = atl01_file
        self.atl01 = h5py.File(self.atl01_file, 'r', driver=None)

        self.atl02_file = atl02_file
        if self.atl02_file != None:
            self.atl02 = h5py.File(self.atl02_file, 'r', driver=None)
        else:
            self.atl02 = None
        
        self.anc13_path = anc13_path
        self.anc27_path = anc27_path
        self.cal17_path = cal17_path
        self.cal44_path = cal44_path
        self.cal49_path = cal49_path

        # Get date range of ATL02.
        self.start_utc = self.atl02['data_start_utc'].value[0].decode()
        self.end_utc = self.atl02['data_end_utc'].value[0].decode()

        # Initialize calibration objects and retrieve constants.
        self.anc13 = ReadANC13(self.anc13_path)  #valid_date_range=[self.start_utc, self.end_utc]) ## For now, use last value
        self.laser = self.anc13.laser
        self.mode = self.anc13.mode
        self.side = self.anc13.side_spd
        self.cal17 = ReadCal17(self.cal17_path, verbose=self.verbose)
        self.cal44 = ReadCal44(self.cal44_path, side=self.side, 
            atl02=self.atl02, verbose=self.verbose)
        self.cal49 = ReadCal49(self.cal49_path, side=self.side, 
            atl02=self.atl02, verbose=self.verbose)
        self.anc27 = ReadANC27(self.anc27_path, side=self.side, 
            atl02_date=self.start_utc, verbose=self.verbose)
        self.ks = ReadKs(self.anc27.anc27_file)
        self.f_cal = self.anc27.select_value('uso_freq')
        self.d_USO = d_USO
        self.SF_USO = np.float64(1.0) # np.float64(SF_USO(self.f_cal))  # temporary test value
        self.dt_imet = dt_imet
        self.dt_t0 = dt_t0
        self.lrs_clock = lrs_clock

        if self.verbose:
            print("atl01 file: ", self.atl01_file)
            print("side: ", self.side)
            print("d_USO: ", self.d_USO)
            print("f_cal: ", self.f_cal)
            print("SF_USO: ", self.SF_USO)
            print("cal44.temperature: ", self.cal44.temperature)
            print("cal44.nearest_temp: ", self.cal44.nearest_temp)
            print("cal49.temperature: ", self.cal49.temperature)
            print("cal49.nearest_temp: ", self.cal49.nearest_temp)

        # Read the atl01 and atl02 dataframes.
        self.atl01_dict = {}
        self.atl01_dict[1] = ReadATL01(self.atl01, pce=1)
        self.atl01_dict[2] = ReadATL01(self.atl01, pce=2)
        self.atl01_dict[3] = ReadATL01(self.atl01, pce=3)

        self.atl02_dict = {}
        self.atl02_dict[1] = ReadATL02(self.atl02, pce=1)
        self.atl02_dict[2] = ReadATL02(self.atl02, pce=2)
        self.atl02_dict[3] = ReadATL02(self.atl02, pce=3)

        # Close open H5 files.
        self.atl01.close()
        self.atl02.close()
        # Set to None since can't pickle an h5 file
        self.atl01 = None 
        self.atl02 = None

        # Return the number of major frames.
        self.nmf = self.return_nmf()

        # Calculate all variables that depend on PCE only.
        if self.multiprocess:
            # Parallelized majorframe-level calculations.
            p = Pool(8)
            results = p.map(self.calculate_majorframe, pces)
            print(results)
            self.pce1 = results[0]
            self.pce2 = results[1]
            self.pce3 = results[2]
        else:
            # Single-threaded majorframe-level calculations.
            self.pce1 = self.calculate_majorframe(pce=1)
            self.pce2 = self.calculate_majorframe(pce=2)
            self.pce3 = self.calculate_majorframe(pce=3)

        # Read a pre-calculated TOD; otherwise calculate it.
        if tod_pickle == None:
            self.TOD = tod.TOD(self.atl01_file, anc27_path=self.anc27_path)
        else:
            self.TOD = pickle_out(tod_pickle)  

        if not self.qtest_mode:
            # Run all usual alignment and aligned-dependent calculations 
            # if not in quick test mode.

            # Align data using DeltaTime_ll time tags
            # Limit to the MF limit, if any.
            self.aligner = Aligner(self.TOD.pce1.DeltaTime_ll[:len(self.pce1.TX_T_ll)], 
                self.TOD.pce2.DeltaTime_ll[:len(self.pce2.TX_T_ll)], 
                self.TOD.pce3.DeltaTime_ll[:len(self.pce3.TX_T_ll)], verbose=self.verbose)

            # Store aligned data, matching alignment found by time tags across given arrays.
            self.pce1.aligned.TX_T_other, self.pce2.aligned.TX_T_other, self.pce3.aligned.TX_T_other = \
                self.aligner.align2timetags_all(self.pce1.TX_T_other, self.pce2.TX_T_other, self.pce3.TX_T_other)
            self.pce1.aligned.TX_T_ll, self.pce2.aligned.TX_T_ll, self.pce3.aligned.TX_T_ll = \
                self.aligner.align2timetags_all(self.pce1.TX_T_ll, self.pce2.TX_T_ll, self.pce3.TX_T_ll)

            print("self.pce1.aligned.TX_T_other: ", len(self.pce1.aligned.TX_T_other), self.pce1.aligned.TX_T_other)
            print("self.pce2.aligned.TX_T_other: ", len(self.pce2.aligned.TX_T_other), self.pce2.aligned.TX_T_other)
            print("self.pce3.aligned.TX_T_other: ", len(self.pce3.aligned.TX_T_other), self.pce3.aligned.TX_T_other)

            print("self.pce1.aligned.TX_T_ll: ", len(self.pce1.aligned.TX_T_ll), self.pce1.aligned.TX_T_ll)
            print("self.pce2.aligned.TX_T_ll: ", len(self.pce2.aligned.TX_T_ll), self.pce2.aligned.TX_T_ll)
            print("self.pce3.aligned.TX_T_ll: ", len(self.pce3.aligned.TX_T_ll), self.pce3.aligned.TX_T_ll)

            # For each event find centroid time relative to LL (seconds)
            self.T_center = self.T_center(self.pce1.aligned.TX_T_other, self.pce2.aligned.TX_T_other, 
                self.pce3.aligned.TX_T_other)

            print("self.T_center: ", len(self.T_center), self.T_center)

            # Fill T_center out from Tx-indices to Rx-indices for this PCE.
            self.pce1.T_center_rx_mapped, self.pce2.T_center_rx_mapped, self.pce3.T_center_rx_mapped = \
                self.aligner.maptx2rx_all(self.T_center, self.T_center, self.T_center) 

            print("self.pce1.T_center_rx_mapped: ", len(self.pce1.T_center_rx_mapped), self.pce1.T_center_rx_mapped)
            print("self.pce2.T_center_rx_mapped: ", len(self.pce2.T_center_rx_mapped), self.pce2.T_center_rx_mapped)
            print("self.pce3.T_center_rx_mapped: ", len(self.pce3.T_center_rx_mapped), self.pce3.T_center_rx_mapped)

            # Calculate the pulse width lower and upper, and pulse skew.
            # Not needed for TOF, but will be needed for comparisions with ATL02.
            self.pulse_width_lower = self.pulse_width_lower(self.pce3.aligned.TX_T_other)
            self.pulse_width_upper = self.pulse_width_upper(self.pce1.aligned.TX_T_other, 
                self.pce2.aligned.TX_T_other)
            self.pulse_skew = self.pulse_skew(self.pce1.aligned.TX_T_other, self.pce2.aligned.TX_T_other, 
                self.pce3.aligned.TX_T_other)

            # Determine the TOF flags.
            self.tof_flag = self.determine_tof_flag()
            self.pce1.tof_flag = self.aligner.maptx2rx(1,self.tof_flag)
            self.pce2.tof_flag = self.aligner.maptx2rx(2,self.tof_flag)
            self.pce3.tof_flag = self.aligner.maptx2rx(3,self.tof_flag)

            if self.verbose:
                print("T_center: ", len(self.T_center), self.T_center)
                print("pulse_width_lower: ", pulse_width_lower)
                print("pulse_width_upper: ", pulse_width_upper)
                print("pulse_skew: ", pulse_skew)

        if self.qtest_mode:
            # Make dummy T_center values if in quick test mode, centering 
            # depends on aligned arrays of all data, which of course
            # are not available in quick test mode.
            self.T_center = np.ones(len(self.pce1.TX_T_ll))*9.71941905813815E-10
            self.pce1.T_center_rx_mapped = np.ones(len(self.pce1.TX_T_ll))*9.71941905813815E-10
            self.pce2.T_center_rx_mapped = np.ones(len(self.pce2.TX_T_ll))*9.71941905813815E-10
            self.pce3.T_center_rx_mapped = np.ones(len(self.pce3.TX_T_ll))*9.71941905813815E-10

        ## Need also store TOF before center correction?
        ## Maybe not. To recover that value, just add T_center_rx_mapped

        # Finally!
        # Calculate the TOFs for all events, just strong, and just weak.        
        if False: #self.multiprocess:
            ## NOTE multiprocess on calculate_TOFs does not work with
            ## the writing to PCEVariables; these would need to be taken
            ## out of the function. So keep this option FALSE for now.
            # Parallelized TOF calculations.
            p = Pool(8)
            results = p.map(self.calculate_TOFs, pces)
            print(results)
            self.pce1.TOF.all, self.pce1.TOF.strong, self.pce1.TOF.weak, \
                self.pce1.TOF.strong_ind, self.pce1.TOF.weak_ind = results[0]
            self.pce2.TOF.all, self.pce2.TOF.strong, self.pce2.TOF.weak, \
                self.pce2.TOF.strong_ind, self.pce2.TOF.weak_ind = results[1]
            self.pce3.TOF.all, self.pce3.TOF.strong, self.pce3.TOF.weak, \
                self.pce3.TOF.strong_ind, self.pce3.TOF.weak_ind = results[2]                
        else:
            # Single-threaded TOF calculations.
            self.pce1.TOF.all, self.pce1.TOF.strong, self.pce1.TOF.weak, \
                self.pce1.TOF.strong_ind, self.pce1.TOF.weak_ind = self.calculate_TOFs(pce=1)
            self.pce2.TOF.all, self.pce2.TOF.strong, self.pce2.TOF.weak, \
                self.pce2.TOF.strong_ind, self.pce2.TOF.weak_ind = self.calculate_TOFs(pce=2)
            self.pce3.TOF.all, self.pce3.TOF.strong, self.pce3.TOF.weak, \
                self.pce3.TOF.strong_ind, self.pce3.TOF.weak_ind = self.calculate_TOFs(pce=3)

        print("pce1.TOF.all: ", len(self.pce1.TOF.all), self.pce1.TOF.all)
        print("pce2.TOF.all: ", len(self.pce2.TOF.all), self.pce2.TOF.all)
        print("pce3.TOF.all: ", len(self.pce3.TOF.all), self.pce3.TOF.all)

        if self.verbose:
            print("pce1.TOF.strong: ", len(self.pce1.TOF.strong), self.pce1.TOF.strong)
            print("pce2.TOF.strong: ", len(self.pce2.TOF.strong), self.pce2.TOF.strong)
            print("pce3.TOF.strong: ", len(self.pce3.TOF.strong), self.pce3.TOF.strong)

            print("pce1.TOF.weak: ", len(self.pce1.TOF.weak), self.pce1.TOF.weak)
            print("pce2.TOF.weak: ", len(self.pce2.TOF.weak), self.pce2.TOF.weak)
            print("pce3.TOF.weak: ", len(self.pce3.TOF.weak), self.pce3.TOF.weak)

        # No need to pickle TOD.
        self.TOD = None
        self.atl01_dict = None
        self.atl02_dict = None

        print("--TOF is complete.--")
        self.run_time = datetime.now() - start_time
        print("Run-time: ", self.run_time)

    def map_pce(self, pce):
        """ If you need a way to map PCE number to PCE attribute.
        """
        if pce == 1:
            return self.pce1
        if pce == 2:
            return self.pce2
        if pce == 3:
            return self.pce3

    def parallelize_pces(self):
        """
        """
        results = map(self.calculate_TOFs, pces)

    def return_nmf(self):
        """ Returns number of major frames.
        """
        return len(self.atl01_dict[1].raw_alt_band_offset)

    def calculate_majorframe(self, pce):
        """ Calculate all PCE-dependent, major-frame level variables.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        # Read ATL01 datasets. 
        raw_rx_channel_id = self.atl01_dict[pce].raw_rx_channel_id
        raw_rx_toggle_flg = self.atl01_dict[pce].raw_rx_toggle_flg
        raw_pce_mframe_cnt_ph = self.atl01_dict[pce].raw_pce_mframe_cnt_ph
        ph_id_pulse = self.atl01_dict[pce].ph_id_pulse

        # This gives the major frame numnber of each shot
        mframe_counts = raw_pce_mframe_cnt_ph - raw_pce_mframe_cnt_ph[0]
        # This gives the list of major frames. Length = number of major frames.
        mframes = list(set(mframe_counts))

        spd_cal = self.spd_cal(pce)

        # Calculate start event coarse time in clock cycles.
        tx_cc = self.tx_cc(pce)
        # Calculate start time offset
        tx_cal_ccoffset = self.tx_cal_ccoffset(pce)
        # Calculate ll start event course time in seconds.
        TX_CC_ll = self.TX_CC(tx_cc, tx_cal_ccoffset)
        # Find other element course time via start marker bit
        tx_smbit = self.tx_smbit(pce)
        # Calculate other start event course time in seconds
        TX_CC_other = self.TX_CC(tx_cc, tx_cal_ccoffset, tx_smbit)
        
        print(" ")
        print("CALCULATE_MAJORFRAME")
        if self.verbose:
            print("mframe_counts: ", len(mframe_counts), mframe_counts)
            print("spd_cal: ", spd_cal)
            print("tx_cc: ", len(tx_cc), tx_cc)
            print("tx_cal_ccoffset: ", tx_cal_ccoffset)
            print("TX_CC_ll: ", len(TX_CC_ll), TX_CC_ll)
            print("tx_smbit: ", len(tx_smbit), tx_smbit)
            print("TX_CC_other: ", len(TX_CC_other), TX_CC_other)
            print("raw_rx_channel_id: ", len(raw_rx_channel_id), raw_rx_channel_id)

        # Determine number of fine counts in the delay chain
        fc_TDC_falling = self.fc_TDC(pce, 'fall')
        FC_TDC_falling = self.FC_TDC(pce, 'fall', fc_TDC_falling)
        fc_TDC_rising = self.fc_TDC(pce, 'rise')
        FC_TDC_rising = self.FC_TDC(pce, 'rise', fc_TDC_rising)

        # Calculate the event fine counts
        tx_fc_ll = self.tx_fc(pce, 'll')
        tx_fc_other = self.tx_fc(pce, 'other')

        # Calculate receive event in clock cycles relative to the DPB offset
        rx_cc = self.rx_cc(pce)
        rx_cal_ccoffset = self.rx_cal_ccoffset(pce)
        RX_CC = self.RX_CC(rx_cc, rx_cal_ccoffset)
        # Calculate fine time of receive event in clock cycles.
        rx_fc = self.rx_fc(pce)
        # Note that RX_FC is found in following loop.

        if self.verbose:
            print("fc_TDC_falling[:10]: ", len(fc_TDC_falling), fc_TDC_falling[:10])
            print("FC_TDC_falling[:10]: ", len(FC_TDC_falling), FC_TDC_falling[:10])
            print("fc_TDC_rising[:10]: ", len(fc_TDC_rising), fc_TDC_rising[:10])
            print("FC_TDC_rising[:10]: ", len(FC_TDC_rising), FC_TDC_rising[:10])
            print("tx_fc_ll: ", len(tx_fc_ll), tx_fc_ll)
            print("tx_fc_other: ", len(tx_fc_other), tx_fc_other)

            print("rx_cc: ", len(rx_cc), rx_cc)
            print("rx_cal_ccoffset: ", rx_cal_ccoffset)
            print("RX_CC: ", len(RX_CC), RX_CC)
            print("rx_fc: ", len(rx_fc), rx_fc)

        # Initialize lists.
        TX_FC_ll_permf = np.array([], dtype=np.float32)
        TX_FC_other_permf = np.array([], dtype=np.float32)
        tx_t_ll_permf = np.array([])
        tx_t_other_permf = np.array([])
        TX_T_other_permf = np.array([])
        TX_T_ll_permf = np.array([])
        TX_T_fine_permf = np.array([])
        RX_FC_permf = np.array([], dtype=np.float32)

        if self.qtest_mode:
            end_mf = 1
        elif self.mf_limit != None:
            end_mf = self.mf_limit
        else:
            end_mf = len(mframes)

        for mframe in mframes[:end_mf]:
            print("PCE {}, mframe {}".format(pce, mframe))

            mframe = int(mframe)
            # mask out all events not in major frame.
            mframe_mask = np.where(mframe_counts == mframe)[0]

            # Convert fine counts.
            TX_FC_ll = self.TX_FC(pce, tx_fc_ll[mframe_mask],
                FC_TDC_rising[mframe], FC_TDC_falling[mframe], 
                raw_rx_channel_id[mframe_mask], raw_rx_toggle_flg[mframe_mask],
                component='ll')
            TX_FC_other = self.TX_FC(pce, tx_fc_other[mframe_mask], 
                FC_TDC_rising[mframe], FC_TDC_falling[mframe], 
                raw_rx_channel_id[mframe_mask], raw_rx_toggle_flg[mframe_mask],
                component='other')

            # Calculate precise start event times in ruler clock cycles.
            tx_t_ll = self.tx_t(TX_CC=TX_CC_ll[mframe_mask], TX_FC=TX_FC_ll, 
                FC_TDC_rising=FC_TDC_rising[mframe])
            tx_t_other = self.tx_t(TX_CC=TX_CC_other[mframe_mask], TX_FC=TX_FC_other, 
                FC_TDC_rising=FC_TDC_rising[mframe])

            # Compute time in seconds between ll and other components
            TX_T_other = self.TX_T(tx_t=(tx_t_other - tx_t_ll), spd_cal=spd_cal)

            # Calculate the time in seconds of LL of given PCE relative to its T0
            TX_T_ll =  self.TX_T(tx_t=tx_t_ll)

            # Calculate the fine time of the ll event which originated the return
            TX_T_fine = self.TX_T_fine(TX_FC_ll, FC_TDC_rising[mframe])
  
            # Find calibrated rx_fc
            RX_FC = self.RX_FC(pce, rx_fc[mframe_mask],
                FC_TDC_rising[mframe], FC_TDC_falling[mframe], 
                raw_rx_channel_id[mframe_mask], raw_rx_toggle_flg[mframe_mask])

            if self.verbose:
                print("    mframe: ", mframe)
                print("    mframe_mask: ", len(mframe_mask), mframe_mask)
                print("    TX_FC_ll: ", len(TX_FC_ll), TX_FC_ll)
                print("    TX_FC_other: ", len(TX_FC_other), TX_FC_other)
                print("    tx_t_ll: ", len(tx_t_ll), tx_t_ll)
                print("    tx_t_other: ", len(tx_t_other), tx_t_other)
                print("    TX_T_other: ", len(TX_T_other), TX_T_other)
                print("    TX_T_ll: ", len(TX_T_ll), TX_T_ll)
                print("    TX_T_fine: ", len(TX_T_fine), TX_T_fine)
                print("    RX_FC: ", len(RX_FC), RX_FC)
 
            TX_FC_ll_permf = np.append(TX_FC_ll_permf, TX_FC_ll)
            TX_FC_other_permf = np.append(TX_FC_other_permf, TX_FC_other)
            tx_t_ll_permf = np.append(tx_t_ll_permf, tx_t_ll)
            tx_t_other_permf = np.append(tx_t_other_permf, tx_t_other)
            TX_T_other_permf = np.append(TX_T_other_permf, TX_T_other)
            TX_T_ll_permf = np.append(TX_T_ll_permf, TX_T_ll)
            TX_T_fine_permf = np.append(TX_T_fine_permf, TX_T_fine)
            RX_FC_permf = np.append(RX_FC_permf, RX_FC)

        len_final = len(RX_FC_permf)

        pce_variables = PCEVariables(
            tx_cc=tx_cc[:len_final], 
            TX_CC_ll=TX_CC_ll[:len_final],
            TX_CC_other=TX_CC_other[:len_final],
            tx_cal_ccoffset=tx_cal_ccoffset,
            tx_smbit=tx_smbit[:len_final], 
            fc_TDC_falling=fc_TDC_falling[:len_final],
            FC_TDC_falling=FC_TDC_falling[:len_final], 
            fc_TDC_rising=fc_TDC_rising[:len_final],
            FC_TDC_rising=FC_TDC_rising[:len_final],
            tx_fc_ll=tx_fc_ll[:len_final],
            TX_FC_ll=TX_FC_ll_permf, 
            tx_fc_other=tx_fc_other, 
            TX_FC_other=TX_FC_other_permf, 
            tx_t_ll=tx_t_ll_permf, 
            tx_t_other=tx_t_other_permf, 
            spd_cal=spd_cal,
            TX_T_other=TX_T_other_permf, 
            TX_T_ll=TX_T_ll_permf, 
            TX_T_fine=TX_T_fine_permf, 
            rx_cc=rx_cc[:len_final],
            rx_cal_ccoffset= rx_cal_ccoffset,
            RX_CC=RX_CC[:len_final],
            rx_fc=rx_fc[:len_final],
            RX_FC=RX_FC_permf)

        return pce_variables

    def calculate_TOFs(self, pce):
        """ Calculate all event-level variables that lead to TOF.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        TOFs_list = [] # does this need to be nx4 to account for each band?
        rx_t_list = []
        ch_cal_list = []
        RX_T_list = []
        RWS_list = []
        RWW_list = []
        DLBO_list = []
        DLBW_list = []
        FC_TDC_list = []
        channel_list = []
        toggle_list = []
        majorframe_list = []
        spots_list = []

        # Obtain all the Major Frame level arrays, length number of MFs.
        DLBO_all_bands = self.DLBO(pce)
        DLBW_all_bands = self.DLBW(pce)
        FC_TDC_rising = self.map_pce(pce).FC_TDC_rising
        FC_TDC_falling = self.map_pce(pce).FC_TDC_falling
        RWS_strong = self.RWS(pce, 's')
        RWS_weak = self.RWS(pce, 'w')
        RWW_strong = self.RWW(pce, 's')
        RWW_weak = self.RWW(pce, 'w')

        # Now obtain all event level arrays, length number of events, and mappable to MF #.
        RX_CC = self.map_pce(pce).RX_CC
        RX_FC = self.map_pce(pce).RX_FC
        raw_pce_mframe_cnt_ph = self.atl01_dict[pce].raw_pce_mframe_cnt_ph
        raw_rx_band_id = self.atl01_dict[pce].raw_rx_band_id
        raw_rx_channel_id = self.atl01_dict[pce].raw_rx_channel_id
        raw_rx_toggle_flg = self.atl01_dict[pce].raw_rx_toggle_flg

        print(" ")
        print("CALCULATE_TOFs")
        if self.verbose:
            print("RWS_strong[:10]: ", len(RWS_strong), RWS_strong[:10])
            print("RWS_weak[:10]: ", len(RWS_weak), RWS_weak[:10])
            print("RX_CC: ", len(RX_CC), RX_CC)
            print("RX_FC: ", len(RX_FC), RX_FC)

        # This gives the major frame number of each event.
        mframe_counts = raw_pce_mframe_cnt_ph - raw_pce_mframe_cnt_ph[0]
        # This gives the list of major frames. Length = number of major frames.
        mframes = list(set(mframe_counts))

        if self.qtest_mode:
            end_event = 2
        elif self.mf_limit != None:
            end_mf = mframes[self.mf_limit-1]
            end_event = len(np.where(mframe_counts <= end_mf)[0])
            print("With mf_limit {}, will end at event {}".format(self.mf_limit, end_event))
        else:
            end_event = len(mframe_counts)

        # Loop over receive events.
        for event in range(len(mframe_counts))[:end_event]:
            print("PCE {}, event {}".format(pce, event))

            # Find the major frame number for event.
            mf = mframe_counts[event]

            # Obtain downlink band id.
            band = self.atl01_dict[pce].raw_rx_band_id[event]
            # From band id and major frame number map to DLBO and DLBW (which are 4xn_mf)
            DLBO = np.float32(DLBO_all_bands[mf][band])
            DLBW = np.float32(DLBW_all_bands[mf][band])

            # From major frame number map to channel.
            # When channel is 0, there are Tx timetags without any associated Rx timetags.
            # Everything relating to channel, including TOF, will get set to 0.
            channel = raw_rx_channel_id[event]
            # From channel map to spot (strong vs weak)
            spot = map_channel2spot(channel)
            # From event map to toggle (1=rising, 0=falling)
            toggle = raw_rx_toggle_flg[event]
            # Retrieve the channel skew correction for side/PCE/channel.
            ch_cal = self.ch_cal(pce=pce, channel=channel, toggle=toggle)

            if spot == 's':
            # Select whether RWS strong or weak.
                RWS = RWS_strong[mf]
                RWW = RWW_strong[mf]
            elif spot == 'w':
                RWS = RWS_weak[mf]
                RWW = RWW_weak[mf]
            else:
                RWS = 0
                RWW = 0

            # Select whether FC_TDC rising or falling.
            if toggle == 1:
                FC_TDC = FC_TDC_rising[mf]
            elif toggle == 0:
                FC_TDC = FC_TDC_falling[mf] 
            else:
                FC_TDC = 0             

            # Find rx_t for event (clock cycles).
            rx_t = self.rx_t(RWS, DLBO, RX_CC[event], RX_FC[event], FC_TDC)

            # Convert rx_t from clock cycles to seconds.
            RX_T = self.RX_T(rx_t, ch_cal)
            
            # Find TOF of event.
            tof_per_event = self.tof_per_event(RX_T, self.map_pce(pce).TX_T_fine[event], 
                self.map_pce(pce).T_center_rx_mapped[event])

            if self.very_verbose:
                print(" ")
                print("    event: ", event)
                print("    mf: ", mf)
                print("    band: ", band)
                print("    DLBO: ", DLBO)
                print("    DLBW: ", DLBW)
                print("    channel: ", channel)
                print("    spot: ", spot)
                print("    toggle: ", toggle)
                print("    ch_cal: ", ch_cal)
                print("    RWS: ", RWS)
                print("    FC_TDC: ", FC_TDC)
                print("    RX_FC: ", RX_FC[event])
                print("    RX_CC[event]: ", RX_CC[event])
                print("    rx_t: ", rx_t)
                print("    RX_T: ", RX_T)
                print("    tof_per_event: ", tof_per_event)

            TOFs_list.append(tof_per_event) # (also per band??)

            rx_t_list.append(rx_t)
            RX_T_list.append(RX_T)
            ch_cal_list.append(ch_cal)
            FC_TDC_list.append(FC_TDC)
            channel_list.append(channel)
            toggle_list.append(toggle)
            majorframe_list.append(mf)
            spots_list.append(spot)

            # Only append these if new mf
            if mf != mframe_counts[event-1]:
                RWS_list.append(RWS)
                RWW_list.append(RWW)
                DLBO_list.append(DLBO)
                DLBW_list.append(DLBW)

        # Additional values to be added to PCEVariables
        self.map_pce(pce).rx_t = np.array(rx_t_list)
        self.map_pce(pce).RX_T = np.array(RX_T_list)
        self.map_pce(pce).ch_cal = np.array(ch_cal_list)
        self.map_pce(pce).RWS = np.array(RWS_list)
        self.map_pce(pce).RWW = np.array(RWW_list)
        self.map_pce(pce).DLBO = np.array(DLBO_list)
        self.map_pce(pce).DLBW = np.array(DLBW_list)
        self.map_pce(pce).FC_TDC = np.array(FC_TDC_list)
        self.map_pce(pce).channel = np.array(channel_list)
        self.map_pce(pce).toggle = np.array(toggle_list)
        self.map_pce(pce).majorframe = np.array(majorframe_list)

        # These can be calculated outside loop because they are
        # only multiplied by constants.
        self.map_pce(pce).RWS_T = self.RWS_T(np.array(RWS_list))
        self.map_pce(pce).RWW_T = self.RWW_T(np.array(RWW_list))
        self.map_pce(pce).DLBO_T = self.DLBO_T(np.array(RWS_list), np.array(DLBO_list))
        self.map_pce(pce).DLBW_T = self.DLBW_T(np.array(DLBW_list))

        spots_list = np.array(spots_list)
        TOFs_list = np.array(TOFs_list)
        # Divide TOF between strong and weak.
        strong_indices = np.where(spots_list == 's')[0]
        weak_indices = np.where(spots_list == 'w')[0]
        TOFs_strong = TOFs_list[strong_indices]
        TOFs_weak = TOFs_list[weak_indices]

        return TOFs_list, np.array(TOFs_strong), np.array(TOFs_weak), \
            strong_indices, weak_indices

    ###########################################################
    # Determine times of range window + downlink band features
    ###########################################################

    def RWS_T(self, RWS):
        """ Equation 3-35. *ATL02 output*
        Conversion from sec to nsec.
        """
        return RWS * self.d_USO * self.SF_USO

    def RWW_T(self, RWW):
        """ Equation 3-36. *ATL02 output*
        Conversion from sec to nsec.
        """
        return RWW * self.d_USO * self.SF_USO

    def DLBO_T(self, RWS, DLBO):
        """ Ewuation 3-37. *ATL02 output*

        Is this correct?
        """
        return RWS + DLBO * self.d_USO * self.SF_USO  # remove RWS.

    def DLBW_T(self, DLBW):
        """ Equation 3-38. *ATL02 output*
        """
        return DLBW * self.d_USO * self.SF_USO

    ###################################################################
    # Calculation of leading lower and other element start even coarse 
    # times.
    ###################################################################

    def tx_cc(self, pce):
        """ Equation 3-7.
        Start event value in units of USO course-clock cycles.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        return self.atl01_dict[pce].raw_tx_leading_coarse

    def tx_cal_ccoffset(self, pce):
        """ Equation 3-8.
        Start time offset is an integer-valued correction to the coarse-clock 
        counter to account for the difference between the reported number and 
        the actual number. This offset can be unique for each PCE, but is 
        nominally set to -1. This is a calibration value defined by the DFC 
        spec.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        return -1

    def tx_smbit(self, pce):
        """  Equation 3-5.
        The start time marker.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        return self.atl01_dict[pce].raw_tx_start_marker

    def TX_CC(self, tx_cc, tx_cal_ccoffset, tx_smbit=0): 
        """ Equations 3-4 and 3-6.
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

    ##########################################
    # Calculation of transmit event fine time
    ##########################################

    def fc_TDC(self, pce, toggle):
        """ Equation 3-7.
        Total number of delay-line cells (fine-clock counts)
        per USO clock cycle for 256 consecutive samples.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        toggle : str
            'rise' or 'fall'
        """
        if toggle == 'rise':
            return self.atl01_dict[pce].raw_alt_cal_rise          
        elif toggle == 'fall':
            return self.atl01_dict[pce].raw_alt_cal_fall

    def FC_TDC(self, pce, toggle, fc_TDC):
        """ Equation 3-8.
        The calibrated and smoothed fc_TDC.

        smoothed = boxcar of 120 seconds, or 6000 samples

        But too much trouble to try to match smoothing to ATL02's,
        so just use the ATL02 value. Note that it is in ns, need 
        to convert to s.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        toggle : str
            'rise' or 'fall'.
        fc_TDC : array
            *At this time, not used.*
        """
        ##signal = fc_TDC / 256.0
        #if toggle == 'rise':
        #    smoothed_signal = self.atl02_dict[pce].cal_rise_sm
        #elif toggle == 'fall':
        #    smoothed_signal = self.atl02_dict[pce].cal_fall_sm            
        # Convert from ns to s.
        #return smoothed_signal*10**8
        #return (self.d_USO * self.SF_USO) / smoothed_signal

        ## For now, hard-code the values per-PCE, per-Toggle
        ## Not as important for fall to be exact, since not used in calculations, only for selecting CAL-17
        FC_TDC_dict = {1:{'rise':np.float64(51.4721946716308594), 'fall':np.float64(51.986270904541016)},
                       2:{'rise':np.float64(56.0463562011718750), 'fall':np.float64(54.9649810791015625)},
                       3:{'rise':np.float64(54.9882469177246094), 'fall':np.float64(53.21328905)}}
        return FC_TDC_dict[pce][toggle]*np.ones(len(fc_TDC))

    def tx_fc(self, pce, component):
        """ Generalization of Equations 3-9 and 3-12.
        The raw tx fine-count.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        component : string
            'll' or 'other'.
        """
        if component == 'll':
            return self.atl01_dict[pce].raw_tx_leading_fine
        elif component == 'other':
            return self.atl01_dict[pce].raw_tx_trailing_fine

    def TX_FC(self, pce, tx_fc, FC_TDC_rising, FC_TDC_falling, 
        raw_rx_channel_id, raw_rx_toggle_flg, component):
        """ Generalization of Equations 3-10 and 3-13.
        Look up in CAL-17 using criteria from Table 12.
        
        The calibrated TX fine-count.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        tx_fc : list
            Raw tx fine counts.
        FC_TDC_rising : float
            The number of rising delay line cells in fine-clock counts.
        FC_TDC_falling : float
            The number of falling delay line cells in fine-clock counts.
        raw_rx_channel_id : list
            Channel ids.
        raw_rx_toggle_flg : list
            Toggle flags.
        component : string 
            'll' or 'other'.
        """
        return self.cal17.select_values(pce, FC_TDC_rising, FC_TDC_falling, 
            raw_rx_channel_id, raw_rx_toggle_flg, tx_fc=tx_fc, component=component)

    ###########################################
    # Calculation of precise start event times
    ###########################################

    def tx_t(self, TX_CC, TX_FC, FC_TDC_rising):
        """ Equations 3-15 abd 3-16.
        The precise start time of the LL/other event in ruler clock
        cycles.

        Parameters
        ----------
        TX_CC : list
            The calibrated TX course counts.
        TX_FC : list
            The calibrated TX fine counts.
        FC_TDC_rising : float
            The calibrated rising delay-line cell count.
        """
        return np.float64(TX_CC - (TX_FC / FC_TDC_rising)) 

    def spd_cal(self, pce):
        """ Equation 3-17.
        Look up using Table 13 and CAL-44.

        Calibrated offset to correct for the SPD path delay for the other
        transmitpulse threshold crossing on PCE

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        return self.cal44.select_value(pce)

    ####################################
    # Calculation of Precise Start Time
    ####################################

    def TX_T(self, tx_t, spd_cal=0):
        """ Equations 3-18 and 3-19.
        The calibrated precise start time.

        spd_cal is 0 if ll

        If ll, tx_t=tx_t_ll
        If other, tx_t=tx_t_ll-tx_t_other

        Parameters
        ----------
        tx_t : list
            The raw precise start times.
        spd_cal : float
            The CAL-44 value for the PCE.
        """
        return np.float64(tx_t*self.d_USO*self.SF_USO + spd_cal)

    def T_center(self, TX_T_LU, TX_T_TU, TX_T_TL):
        """ Equation 3-20. *ATL02 output*
        The precise centroid time relative to LL.

        Parameters
        ----------
        TX_T_LU : list
            The calibrated precise start times for LU (PCE1).
        TX_T_TU : list
            The calibrated precise start times for TU (PCE2).
        TX_T_TL : list
            The calibrated precise start times for TL (PCE3).
        """
        T_centers = []
        for i in range(len(TX_T_LU)):
            # select k depending on scenario.
            # The NANs in each array determine which scenerio of k is used.
            k = self.ks.select_k(TX_T_LU[i], TX_T_TU[i], TX_T_TL[i]) 
            T_centers.append(k[0] + k[1]*nan2zero(TX_T_LU[i]) + k[2]*nan2zero(TX_T_TU[i]) + k[3]*nan2zero(TX_T_TL[i]))
        return np.array(T_centers, dtype=np.float64)

    #############################################
    # Calculation of Pulse Width and Start Skews
    #############################################
  
    def pulse_width_lower(self, TX_T_TL):
        """ Equation 3-21.

        Parameters
        ----------
        TX_T_TL : list
            The calibrated precise start times for TL (PCE3).
        """
        return TX_T_TL

    def pulse_width_upper(self, TX_T_LU, TX_T_TU):
        """ Equation 3-22.

        Parameters
        ----------
        TX_T_LU : list
            The calibrated precise start times for LU (PCE1).
        TX_T_TU : list
            The calibrated precise start times for TU (PCE2).        
        """
        return TX_T_TU - TX_T_LU

    def pulse_skew(self, TX_T_LU, TX_T_TU, TX_T_TL):
        """ Equation 3-23.
        The pulse skew relative to LL.

        Parameters
        ----------
        TX_T_LU : list
            The calibrated precise start times for LU (PCE1).
        TX_T_TU : list
            The calibrated precise start times for TU (PCE2).
        TX_T_TL : list
            The calibrated precise start times for TL (PCE3).
        """
        return (TX_T_TU + TX_T_LU)/2.0 - TX_T_TL/2.0

    ###########################################
    # Calculation of Receive Event Course Time
    ###########################################

    def rx_cc(self, pce):
        """ Equation 3-24.
        Raw rx values in units of USO coarse-clock cycles.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        return self.atl01_dict[pce].raw_rx_leading_coarse

    def rx_cal_ccoffset(self, pce):
        """ Equation 3-25.
        The start time offset.

        Can be unique for each PCE but nominally set to -1.

        Parameters
        ----------
        pce : int
            1, 2, or 3.

        Notes
        -----
        ** how will new values be obtained? **
        """
        return -1

    def RX_CC(self, rx_cc, rx_cal_ccoffset):
        """ Equation 3-26.
        The calibrated rx course counts. 

        Parameters
        ----------
        rx_cc : list
            Raw rx course count values.
        rx_cal_ccoffset : int
            Start time offset.
        """
        return np.int64(rx_cc + rx_cal_ccoffset)

    #########################################
    # Calculation of receive event fine time
    #########################################

    def rx_fc(self, pce):
        """ Equation 3-27.
        Telemetered rx fine counts for start element.

        Parameters
        ----------
        pce : int
            1, 2, or 3. 
        """
        return self.atl01_dict[pce].raw_rx_leading_fine

    def RX_FC(self, pce, rx_fc, FC_TDC_rising, FC_TDC_falling, raw_rx_channel_id, 
        raw_rx_toggle_flg):
        """ Equation 3-28.
        Read from CAL-17.
        The calibrated rx fine counts

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        rx_fc : list
            Raw rx fine counts.
        FC_TDC_rising : float
            The number of rising delay line cells in fine-clock counts.
        FC_TDC_falling : float
            The number of falling delay line cells in fine-clock counts.
        raw_rx_channel_id : list
            Channel ids.
        raw_rx_toggle_flg : list
            Toggle flags.
        """ 
        return self.cal17.select_values(pce, FC_TDC_rising, FC_TDC_falling, 
            raw_rx_channel_id, raw_rx_toggle_flg, rx_fc=rx_fc)

    #####################################
    # Calculation of precise event times
    #####################################

    def RWS(self, pce, spot):
        """ Range window starts for PCE.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        spot : string
            's' or 'w'.
        """
        if spot == 's':
            return np.array(self.atl01_dict[pce].raw_alt_rw_start_s)
        elif spot == 'w':
            return np.array(self.atl01_dict[pce].raw_alt_rw_start_w)            

    def RWW(self, pce, spot):
        """ Range window widths for PCE.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        spot : string
            's' or 'w'.
        """
        if spot == 's':
            return np.array(self.atl01_dict[pce].raw_alt_rw_width_s)
        elif spot == 'w':
            return np.array(self.atl01_dict[pce].raw_alt_rw_width_w)

    def DLBO(self, pce):
        """ Downlink band offsets for PCE.
        Divided into four channels (nx4 array)

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        return np.array(self.atl01_dict[pce].raw_alt_band_offset)

    def DLBW(self, pce):
        """ Downlink band widths for PCE.
        Divided into four channels (nx4 array)

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        return np.array(self.atl01_dict[pce].raw_alt_band_width)

    def rx_t(self, RWS, DLBO, RX_CC, RX_FC, FC_TDC):
        """ Equation 3-30.
        The rx event time relative to LL expressed in clock cycles.

        Parameters
        ----------
        RWS : int
            The range window start of the event.
        DLBO : int
            The downlink band offset of the event.
        RX_CC : float
            The calibrated RX course count of the event.
        RX_FC : float
            The calibrated RX fine count of the event.
        FC_TDC : float
            The calibrated delay-line cell counts.
        Notes
        -----
        ** Need modify for: Index DLBO by 4 bands (for now consider band=0 only -- is this linked to channel?)
        """
        if RWS == 0:
            return 0
        else:
            return np.float64(RWS + DLBO + RX_CC - (RX_FC/FC_TDC))

    def ch_cal(self, pce, channel, toggle):
        """ Equation 3-31.
        Channel skew correction recovered from CAL-49, using directions
        in Table 16.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        channel : int
            The channel number.
        toggle : int
            The toggle flag.
        """ 
        if toggle == 1:
            event_edge = 'R'
        elif toggle == 0:
            event_edge = 'F'
        return self.cal49.select_value(event_edge=event_edge, pce=pce, channel=channel)

    def RX_T(self, rx_t, ch_cal):
        """ Equation 3-32.
        The calibrated rx event time relative to LL expressed in clock
        cycles.

        Parameters
        ----------
        rx_t : list
            The rx event time relative to LL expressed in clock cycles.
        ch_cal : float
            CAL-49 value, the channel skew correction.
        """
        if rx_t == 0 or ch_cal == 0:
            return np.float64(0)
        else:
            return np.float64(rx_t * self.d_USO * self.SF_USO + ch_cal)

    def TX_T_fine(self, TX_FC_ll, FC_TDC_rising):
        """ Equation 3-33.
        The fine time of the LL event which originated the return
        under examination.

        Parameters
        ----------
        TX_FC_ll : list
            The calibrated leading lower TX fine-count.
        FC_TDC_rising : float
            The number of rising delay line cells in fine-clock counts.
        """
        return np.float64((TX_FC_ll / FC_TDC_rising) * self.d_USO * self.SF_USO)

    def tof_per_event(self, RX_T, TX_T_fine, T_center_rx_mapped):
        """ Equation 3-35.

        Parameters
        ----------
        RX_T : float
            The calibrated rx event time relative to LL.
        TX_T_fine : float
            The fine time of the LL event that originated the return.
        T_center_rx_mapped : float
            The precise centroid time relative to LL (mapped to RX space).
        """
        if RX_T == 0:
            return np.float64(0)
        else:
            return RX_T + TX_T_fine - T_center_rx_mapped

    #####################################
    # Misc Stuff
    #####################################

    def determine_tof_flag(self):
        """ Calculate the TOF flag.

        Time Of Flight center correction flag. Values indicate what
        components were used to adjust the TOF to the centroid of the
        Tx pulse, based on the alignment of Tx components across all 3
        PCEs. 
            1=LL_LU_TU_TL
            2=LL_TU_TL
            3=LL_LU_TL
            4=LL_LU_TU
            5=LL_TL
            6=LL_TU
            7=LL_LU
            8=LL
        Values greater than 10 indicate the same
        sequence of conditions indicated for a potential TEP photon.
        """
        # Initialize all flags to 1.
        tof_flag = np.ones(len(self.pce1.aligned.TX_T_ll))

        # Determine flag of the event.
        # In TEP class, add 10 to any TEP photon.

        # 2=LL_TU_TL
        tof_flag = [2 if (np.isnan(self.pce1.aligned.TX_T_ll[i]) \
            and not np.isnan(self.pce2.aligned.TX_T_ll[i]) \
            and not np.isnan(self.pce3.aligned.TX_T_ll[i])) \
            else tof_flag[i] for i in range(len(tof_flag))]
        # 3=LL_LU_TL
        tof_flag = [3 if (not np.isnan(self.pce1.aligned.TX_T_ll[i]) \
            and np.isnan(self.pce2.aligned.TX_T_ll[i]) \
            and not np.isnan(self.pce3.aligned.TX_T_ll[i])) \
            else tof_flag[i] for i in range(len(tof_flag))]
        # 4=LL_LU_TU
        tof_flag = [4 if (not np.isnan(self.pce1.aligned.TX_T_ll[i]) \
            and not np.isnan(self.pce2.aligned.TX_T_ll[i]) \
            and np.isnan(self.pce3.aligned.TX_T_ll[i])) \
            else tof_flag[i] for i in range(len(tof_flag))]
        # 5=LL_TL
        tof_flag = [5 if (np.isnan(self.pce1.aligned.TX_T_ll[i]) \
            and np.isnan(self.pce2.aligned.TX_T_ll[i]) \
            and not np.isnan(self.pce3.aligned.TX_T_ll[i])) \
            else tof_flag[i] for i in range(len(tof_flag))] 
        # 6=LL_TU
        tof_flag = [6 if (np.isnan(self.pce1.aligned.TX_T_ll[i]) \
            and not np.isnan(self.pce2.aligned.TX_T_ll[i]) \
            and np.isnan(self.pce3.aligned.TX_T_ll[i])) \
            else tof_flag[i] for i in range(len(tof_flag))]
        # 7=LL_LU
        tof_flag = [7 if (not np.isnan(self.pce1.aligned.TX_T_ll[i]) \
            and np.isnan(self.pce2.aligned.TX_T_ll[i]) \
            and np.isnan(self.pce3.aligned.TX_T_ll[i])) \
            else tof_flag[i] for i in range(len(tof_flag))]
        # 8=LL
        tof_flag = [8 if (np.isnan(self.pce1.aligned.TX_T_ll[i]) \
            and np.isnan(self.pce2.aligned.TX_T_ll[i]) \
            and np.isnan(self.pce3.aligned.TX_T_ll[i])) \
            else tof_flag[i] for i in range(len(tof_flag))]

        return np.array(tof_flag)

