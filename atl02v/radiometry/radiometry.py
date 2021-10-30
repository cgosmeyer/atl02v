""" Radiometry. Should include all calculations having to do with transmitted 
laser energy and small-signal receiver sensitivity.

The radiometry data products are

    1. Total transmitted energy per pulse in all six beams.
    2. Transmitted energy in *each* of the six beams.
    3. Receiver sensitivity to small signals for each of the six active
       detectors.

From the sensors (SPD A, SPD B, Laser 1 Internal, Laser 2 Internal, LRS), ATL02
reports out three estimates of total transmitted energy from
    1. Active SPD energy monitor,
    2. Active laser internal energy monitor, and
    3. LRS.

Author:
    
    C.M. Gosmeyer

Date:

    Apr 2018

References:

    ATL02 ATBD chapter 5.

Required Inputs:

    ANC13
    CAL-30
    CAL-45
    CAL-46
    CAL-47
    CAL-54
    CAL-61

"""

import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from atl02v.shared.calproducts import ReadANC13, ReadCal30, ReadCal45, ReadCal46, \
    ReadCal47, ReadCal54, ReadCal61, ReadANC27
from atl02v.shared.constants import spots
from atl02v.shared.tools import pickle_out
from atl02v.conversion.convert import Converter
from atl02v.conversion.eval_expression import eval_expr
from atl02v.verification.lrs_verification import convert_lrs_temp
        

class SpotContainer(object):
    def __init__(self, spot1=None, spot2=None, spot3=None, spot4=None, 
        spot5=None, spot6=None):
        self.spot1 = spot1
        self.spot2 = spot2
        self.spot3 = spot3
        self.spot4 = spot4
        self.spot5 = spot5
        self.spot6 = spot6

class SensorContainer(object):
    def __init__(self, E_total=None, S=None, T=None, E_fract=None):
        """
        """
        self.E_total = E_total
        self.S = S
        self.T = T
        self.E_fract = SpotContainer()

class SensitivityContainer(object):
    def __init__(self, m=None, V_nom=None, b=None, c=None, V=None,
        S_over_S_nom=None, S_max=None, S_BG=None, h=None, 
        D_lam_max=None, D_lam=None, S_over_S0=None, cal47_value=None, 
        theta=None, phi=None, S_RET=None):
        """
        """
        self.m = m
        self.V_nom = V_nom
        self.b = b
        self.c = c
        self.V = V
        self.S_over_S_nom = S_over_S_nom
        self.S_max = S_max
        self.S_BG = S_BG
        self.h = h
        self.D_lam_max = D_lam_max
        self.D_lam = D_lam
        self.S_over_S0 = S_over_S0
        self.cal47_value = cal47_value
        self.theta = theta
        self.phi = phi
        self.S_RET = S_RET

class DataSet(object):
    def __init__(self, dataset, atl01, df=False, channel=None,
        hkt=None, verbose=False):
        """
        """
        self.verbose = verbose
        self.dataset = dataset
        self.atl01 = atl01
        self.df = df
        self.channel = channel
        self.hkt = hkt
        self.label = self.dataset.split('/')[-1]

        # Need have overwritable mnemonic and description

        self.raw, self.converted = self.convert()
        self.atl01 = None

    def convert(self):

        if 'raw_energy_data_shg' in self.label:
            # Don't convert
            return np.array(self.atl01[self.dataset].value, dtype=np.float64), []

        if 'raw_ldc_t' in self.label:
            raw_ldc_t = np.array(self.atl01['/lrs/hk_1120/raw_ldc_t'].value)
            return raw_ldc_t, np.array(convert_lrs_temp('raw_ldc_t', raw_ldc_t))

        if self.df:
            parameters_df = pd.DataFrame(self.atl01[self.dataset].value)
            converted_parameters_ls = []
            parameters_ls = []
            if self.label == 'raw_cent_mag' or self.label == 'raw_quality_f':
                cols = [4,5,6,7,8,9]
            else:
                cols = range(len(parameters_df.columns))
            for col in cols:
                parameters = np.array(parameters_df[col], dtype=np.float64)
                converted_parameters = parameters + np.float64(32768)
                parameters_ls.append(parameters)
                converted_parameters_ls.append(np.float64(converted_parameters))
                if self.verbose:
                    print("parameters: ", parameters)
                    print("converted_parameters: ", converted_parameters)
                    print(" ")

            return np.array(parameters_ls), np.array(converted_parameters_ls)

        else:
            parameters = np.array(self.atl01[self.dataset].value, dtype=np.float64)
            converted_dict, _ = Converter([self.label], [parameters], 
                channel=self.channel, hkt=self.hkt, verbose=self.verbose).convert_all()
            converted_parameters = converted_dict[self.label]
            if self.verbose:
                print("label: ", self.label)
                print("parameters: ", parameters)
                print("converted_parameters: ", converted_parameters)
                print(" ")

            return parameters, np.array(converted_parameters, dtype=np.float64)
        
class ReadATL01(object):
    """ Read all needed datasets.
    """
    def __init__(self, atl01, laser, verbose):
        """
        """
        # 5.2.1.2
        # laser energy is "converted" by +=32768

        self.raw_pri_lsr_energy = DataSet('a_hkt_e_1063/raw_pri_lsr_energy', atl01, df=True, verbose=verbose) # ch 0-18, SPD energy monitor A
        self.raw_red_lsr_energy = DataSet('a_hkt_e_1063/raw_red_lsr_energy', atl01, df=True, verbose=verbose) # ch 1-19, SPD energy monitor B

        self.raw_hkt_beamx_t = DataSet('a_hkt_c_1061/raw_hkt_beamx_t', atl01, verbose=verbose) # ch 82, SPD temperature laser 1
        self.raw_spda_therm_t = DataSet('a_hkt_c_1061/raw_spda_therm_t', atl01, verbose=verbose) # ch 74, MEB-34, SPD A temperature laser 2
        self.raw_spdb_therm_t = DataSet('a_hkt_c_1061/raw_spdb_therm_t', atl01, verbose=verbose) # ch 81, MEB-35, SPD B temperature laser 2
        self.raw_energy_data_shg = DataSet('a_sla_hk_1032/raw_energy_data_shg', atl01, verbose=verbose) # laser internal energy monitor. is not converted!

        self.raw_hkt_cchp_las1_t = DataSet('a_hkt_c_1061/raw_hkt_cchp_las1_t', atl01, verbose=verbose) # ch 59, TCS-14, laser 1 interface temperature
        ## TEMPORARY: substitute the laser 1 value for laser 2
        ## raw_hkt_cchp_las2_t will not exist unless there is SC data. See email from JLee on 29 June 2018.
        self.raw_hkt_cchp_las2_t = DataSet('a_hkt_c_1061/raw_hkt_cchp_las1_t', atl01, verbose=verbose) #DataSet('sc1/hk/raw_at_t') # # HT_ATL_TIB1_LSR2_IF01_Y07_T
        self.raw_ldc_t  = DataSet('hk_1120/raw_ldc_t', atl01, verbose=verbose) # LRS temperature post-rel003
        # LRS magnitudes do not get converted.
        # To use them will need to pull out the desired column for the beam (cols 0-5).
        self.raw_cent_mag = DataSet('laser_centroid/raw_cent_mag', atl01, df=True, verbose=verbose)
        # Used to verify the quality of raw_cent_mag
        self.raw_quality_f =  DataSet('laser_centroid/raw_quality_f', atl01, df=True, verbose=verbose)

        # 5.3.1.5
        self.raw_pri_hvpc_mod_1 = DataSet('a_hkt_a_1059/raw_pri_hvpc_mod_1', atl01, verbose=verbose)
        self.raw_pri_hvpc_mod_2 = DataSet('a_hkt_a_1059/raw_pri_hvpc_mod_2', atl01, verbose=verbose)
        self.raw_pri_hvpc_mod_3 = DataSet('a_hkt_a_1059/raw_pri_hvpc_mod_3', atl01, verbose=verbose)
        self.raw_pri_hvpc_mod_4 = DataSet('a_hkt_a_1059/raw_pri_hvpc_mod_4', atl01, verbose=verbose)
        self.raw_pri_hvpc_mod_5 = DataSet('a_hkt_a_1059/raw_pri_hvpc_mod_5', atl01, verbose=verbose)
        self.raw_pri_hvpc_mod_6 = DataSet('a_hkt_a_1059/raw_pri_hvpc_mod_6', atl01, verbose=verbose)

        self.raw_red_hvpc_mod_1 = DataSet('a_hkt_b_1060/raw_red_hvpc_mod_1', atl01, verbose=verbose)
        self.raw_red_hvpc_mod_2 = DataSet('a_hkt_b_1060/raw_red_hvpc_mod_2', atl01, verbose=verbose)
        self.raw_red_hvpc_mod_3 = DataSet('a_hkt_b_1060/raw_red_hvpc_mod_3', atl01, verbose=verbose)
        self.raw_red_hvpc_mod_4 = DataSet('a_hkt_b_1060/raw_red_hvpc_mod_4', atl01, verbose=verbose)
        self.raw_red_hvpc_mod_5 = DataSet('a_hkt_b_1060/raw_red_hvpc_mod_5', atl01, verbose=verbose)
        self.raw_red_hvpc_mod_6 = DataSet('a_hkt_b_1060/raw_red_hvpc_mod_6', atl01, verbose=verbose)

        # 5.3.3.4
        self.raw_peak_xmtnc = DataSet('a_hkt_e_1063/raw_peak_xmtnc', atl01, hkt='PEAK_XMTNC_MVOLTS', verbose=verbose) # WTEM peak signal (D_peak)
        self.raw_edge_xmtnc = DataSet('a_hkt_e_1063/raw_edge_xmtnc', atl01, hkt='EDGE_XMTNC_MVOLTS', verbose=verbose) # WTEM edge signal (D_edge)

        self.raw_pri_thrhi_rdbk = DataSet('a_hkt_e_1063/raw_pri_thrhi_rdbk', atl01, hkt='SPDPRI_HI_RDBK_MVOLTS', verbose=verbose) #a_spdpri_hi_rdbk_mvolts
        self.raw_pri_thrlo_rdbk = DataSet('a_hkt_e_1063/raw_pri_thrlo_rdbk', atl01, hkt='SPDPRI_LO_RDBK_MVOLTS',verbose=verbose) #a_spdpri_lo_rdbk_mvolts
        self.raw_red_thrhi_rdbk = DataSet('a_hkt_e_1063/raw_red_thrhi_rdbk', atl01, hkt='SPDRED_HI_RDBK_MVOLTS',verbose=verbose) #a_spdred_hi_rdbk_mvolts
        self.raw_red_thrlo_rdbk = DataSet('a_hkt_e_1063/raw_red_thrlo_rdbk', atl01, hkt='SPDRED_LO_RDBK_MVOLTS',verbose=verbose) #a_spdred_lo_rdbk_mvolts

class ReadATL02(object):
    """
    """
    def __init__(self, atl02):
        self.hkt_dem1_t4_et_t = np.array(atl02['housekeeping/thermal/hkt_dem1_t4_et_t'].value)
        self.hkt_dom_rad1_t = np.array(atl02['housekeeping/thermal/hkt_dom_rad1_t'].value)
        self.hkt_dom_rad2_t = np.array(atl02['housekeeping/thermal/hkt_dom_rad2_t'].value)
        self.laser_internal_temp = np.array(atl02['housekeeping/laser_energy_internal/laser_temp'].value)
        self.lrs_temp = np.array(atl02['housekeeping/laser_energy_lrs/lrs_temp'].value)
        self.spd_temp = np.array(atl02['housekeeping/laser_energy_spd/laser_temp'].value)

class Radiometry(object):
    __name__ = 'Radiometry'
    def __init__(self, atl01_file, atl02_file, anc13_path=None, anc27_path=None,
        cal30_path=None, cal45_path=None, cal46_path=None, cal47_path=None, 
        cal54_path=None, cal61_path=None, verbose=False):
        """ Radiometry class.

        Parameters
        ----------
        atl01_file : str
            The name of the ATL01 H5 file.
        atl02_file : str
            The name of the ATL02 H5 file.
        anc13_path : str
            Path to the ANC13 file. Used for reading the instrument operating state.
        anc27_path : str
            Path to the ANC27 files.
        cal30_path : str
            Path to the CAL30 products.
        cal45_path : str
            Path to the CAL45 products.
        cal46_path : str
            Path to the CAL46 products.
        cal47_path : str
            Path to the CAL47 products. Use the image version.
        cal54_path : str
            Path to the CAL54 products.
        cal61_path : str
            Path to the CAL61 products.
        verbose : {True, False}
            On by default. Print read-in and calculated final arrays.
        """
        # Begin timer.
        start_time = datetime.now()
        print("Radiometry start time: ", start_time)

        self.verbose = verbose

        self.anc13_path = anc13_path
        self.anc27_path = anc27_path
        self.cal30_path = cal30_path
        self.cal45_path = cal45_path
        self.cal46_path = cal46_path
        self.cal47_path = cal47_path
        self.cal54_path = cal54_path
        self.cal61_path = cal61_path

        # Open ATL01 and ATL02 files.
        self.atl01_file = atl01_file
        self.atl01 = h5py.File(self.atl01_file, 'r', driver=None)
        self.atl02_file = atl02_file
        self.atl02 = h5py.File(self.atl02_file, 'r', driver=None)

        # Set constants.
        self.anc13 = ReadANC13(self.anc13_path)
        self.laser = self.anc13.laser
        self.mode = self.anc13.mode
        self.side = self.anc13.side_spd 
        self.scaling_factor_bg = np.float64(1.0)  # Change when ISF says to
        self.scaling_factor_ret = np.float64(0.90)

        # Get date range of ATL02.
        self.start_utc = self.atl02['ancillary_data/data_start_utc'].value[0].decode()
        self.end_utc = self.atl02['ancillary_data/data_end_utc'].value[0].decode()

        print("end_utc: ", type(self.end_utc), self.end_utc)

        # Read desired datasets from ATL01 and ATL02 dataframes.
        self.atl01_store = ReadATL01(self.atl01, laser=self.laser, verbose=self.verbose)
        self.atl02_store = ReadATL02(self.atl02)

        # Close ATL01 and ATL02s files since can't pickle an h5 file.
        self.atl01.close()
        self.atl02.close()
        self.atl01 = None
        self.atl02 = None

        # Return the side-specific threshold readback values.
        self.thrhi_rdbk, self.thrlo_rdbk = self.threshold_readbacks()

        # Read the calibration products.
        # ANC27 contains aging_corr, bias values, and so on.
        self.anc27 = ReadANC27(anc27_path, side=self.side, atl02_date=self.start_utc)
        self.cal30 = ReadCal30(self.cal30_path, side=self.side, atl02=self.atl02_store)
        self.cal45 = ReadCal45(self.cal45_path, side=self.side)
        self.cal46 = ReadCal46(self.cal46_path, side=self.side)
        self.cal47 = ReadCal47(self.cal47_path, laser=self.laser, atl01=self.atl01_store,
            atl02=self.atl02_store, verbose=self.verbose)
        # cal54 also includes T0 attribute
        self.cal54 = ReadCal54(self.cal54_path)
        ## temperature is speficific to the sensor, so can this really
        ## be used here?
        ## TEMPORARY: because configuration A1/2 isn't in file yet, use B2 in its place.
        self.cal61 = ReadCal61(self.cal61_path, atl02=self.atl02_store, side='B', 
            laser=2, mode=self.mode, verbose=self.verbose) 

        if self.verbose:
            print("laser: ", self.laser)
            print("mode: ", self.mode)
            print("spd side: ", self.side)
            print("aging_corr: ", self.anc27.select_value('cal46_aging'))
            print("scaling_factor_bg: ", self.scaling_factor_bg)
            print("scaling_factor_ret: ", self.scaling_factor_ret)
            print("bias_offset_x: ", self.anc27.select_value('bias_offset_x')) 
            print("bias_offset_y: ", self.anc27.select_value('bias_offset_y')) 
            print("bias_rate: ", self.anc27.select_value('bias_rate')) 
            print("cal61.temperature: ", self.cal61.temperature)
            print("cal61.nearest_temp: ", self.cal61.nearest_temp)
            print("cal61.cal_file: ", self.cal61.cal_file)
            print("thrhi_rdbk: ", self.thrhi_rdbk)
            print("thrlo_rdbk: ", self.thrlo_rdbk)

        # Initialize per-sensor attributes.
        self.spd = SensorContainer()
        self.lrs = SensorContainer()
        self.laser_internal = SensorContainer()

        # Find the total energy per sensor.
        self.spd.T, self.spd.S, self.spd.E_total = self.spd_energy()
        self.lrs.T, self.lrs.S, self.lrs.E_total = self.lrs_energy()
        self.laser_internal.T, self.laser_internal.S, self.laser_internal.E_total = self.laser_internal_energy()

        if self.verbose:
            print("spd.E_total: ", self.spd.E_total)
            print("lrs.E_total: ", self.lrs.E_total)
            print("laser_internal.E_total: ", self.laser_internal.E_total)
        
        # Find the fractional energy per spot per sensor.
        ## should this change to spd.spot1.E_fract, etc?
        self.spd.E_fract.spot1, self.spd.E_fract.spot2, self.spd.E_fract.spot3, \
            self.spd.E_fract.spot4, self.spd.E_fract.spot5, self.spd.E_fract.spot6 \
            = self.E_fractional(E_total=self.spd.E_total)
        self.lrs.E_fract.spot1, self.lrs.E_fract.spot2, self.lrs.E_fract.spot3, \
            self.lrs.E_fract.spot4, self.lrs.E_fract.spot5, self.lrs.E_fract.spot6 \
            = self.E_fractional(E_total=self.lrs.E_total)
        self.laser_internal.E_fract.spot1, self.laser_internal.E_fract.spot2, \
            self.laser_internal.E_fract.spot3, self.laser_internal.E_fract.spot4, \
            self.laser_internal.E_fract.spot5, self.laser_internal.E_fract.spot6 \
            = self.E_fractional(E_total=self.laser_internal.E_total)

        if self.verbose:
            print("spd.E_fract: ", self.spd.E_fract)
            print("lrs.E_fract: ", self.lrs.E_fract)
            print("laser_internal.E_fract: ", self.laser_internal.E_fract)

        # Calculate the spot-specific return and background sensitivities.
        self.spot1 = self.calculate_sensitivities(spot=1)
        self.spot2 = self.calculate_sensitivities(spot=2)
        self.spot3 = self.calculate_sensitivities(spot=3)
        self.spot4 = self.calculate_sensitivities(spot=4)
        self.spot5 = self.calculate_sensitivities(spot=5)
        self.spot6 = self.calculate_sensitivities(spot=6)

        if self.verbose:
            print("spot1.S_max: ", self.spot1.S_max)
            print("spot1.S_BG: ", self.spot1.S_BG)
            print("spot1.S_RET: ", self.spot1.S_RET)

            print("spot2.S_max: ", self.spot2.S_max)
            print("spot2.S_BG: ", self.spot2.S_BG)
            print("spot2.S_RET: ", self.spot2.S_RET)

            print("spot3.S_max: ", self.spot3.S_max)
            print("spot3.S_BG: ", self.spot3.S_BG)
            print("spot3.S_RET: ", self.spot3.S_RET)

            print("spot4.S_max: ", self.spot4.S_max)
            print("spot4.S_BG: ", self.spot4.S_BG)
            print("spot4.S_RET: ", self.spot4.S_RET)

            print("spot5.S_max: ", self.spot5.S_max)
            print("spot5.S_BG: ", self.spot5.S_BG)
            print("spot5.S_RET: ", self.spot5.S_RET)

            print("spot6.S_max: ", self.spot6.S_max)
            print("spot6.S_BG: ", self.spot6.S_BG)
            print("spot6.S_RET: ", self.spot6.S_RET)

        print("--Radiometry is complete.--")
        self.run_time = datetime.now() - start_time
        print("Run-time: ", self.run_time)

    def map_sensor(self, sensor):
        """ If you need a way to map PCE number to PCE attribute.

        Parameters
        ----------
        sensor : string
            'spd', 'lrs', or 'laser_internal'
        """
        if sensor == 'spd':
            return self.spd
        if sensor == 'lrs':
            return self.lrs
        if sensor == 'laser_internal':
            return self.laser_internal

    def return_atlas_settings(self):
        """ Returns the ATLAS settings as given in the ANC13.
        """
        with open(self.anc13_file) as f: 
            for line in f:
                if line[0] != '#':
                    # Will only record the latest value, which is what you want.
                    data = line
        laser = 1 #int(data.split(' ')[0])
        laser_energy_level = int(data.split(' ')[4])
        spd = 'A' #str(data.split(' ')[1]).upper()

        return laser, laser_energy_level, spd

    def calculate_sensitivities(self, spot):
        """ Loops over spots to calculate return and background sensitivities.
        
        Parameters
        ----------
        spot : int
            Value 1-6.
        """
        pmt_hv_biases = {'A':{1:self.atl01_store.raw_pri_hvpc_mod_1.converted,
                              2:self.atl01_store.raw_pri_hvpc_mod_2.converted,
                              3:self.atl01_store.raw_pri_hvpc_mod_3.converted,
                              4:self.atl01_store.raw_pri_hvpc_mod_4.converted,
                              5:self.atl01_store.raw_pri_hvpc_mod_5.converted,
                              6:self.atl01_store.raw_pri_hvpc_mod_6.converted},
                         'B':{1:self.atl01_store.raw_red_hvpc_mod_1.converted,
                              2:self.atl01_store.raw_red_hvpc_mod_2.converted,
                              3:self.atl01_store.raw_red_hvpc_mod_3.converted,
                              4:self.atl01_store.raw_red_hvpc_mod_4.converted,
                              5:self.atl01_store.raw_red_hvpc_mod_5.converted,
                              6:self.atl01_store.raw_red_hvpc_mod_6.converted}}


        # Start by retrieving from Cal-30 slope.
        m = self.cal30.select_value(spot)

        # Select CAL_46 values.
        V_nom, b, c = self.cal46.select_values(spot)

        # Return the spot-side-specific voltage.
        V = pmt_hv_biases[self.side][spot]

        # Obtain the receiver sensitivity ratio
        S_over_S_nom = self.receiver_sensitivity_ratio(V=V, V_nom=V_nom, b=b, c=c)

        if self.verbose:
            print("m: ", m)
            print("b: ", b)
            print("c: ", c)
            print("V_nom: ", V_nom)
            print("V: ", V)
            print("S_over_S_nom: ", S_over_S_nom)

        # Calculate intermediate value S_max
        S_max = self.S_max(m=m, S_over_S_nom=S_over_S_nom)

        # Calculate the background and return sensitivities.
        S_BG = self.S_BG(S_max=S_max)

        h, D_lam_max = self.cal61.select_value(spot)
        ## TBA: whether in standard or alternate mode.
        D_lam = self.atl01_store.raw_peak_xmtnc.converted / self.atl01_store.raw_edge_xmtnc.converted
        S_over_S0 = self.receiver_sensitivity_ratio_tuning(h, D_lam, D_lam_max)

        # Read the closest upper and lower theta and phi from ANC27.
        thetas = self.anc27.select_value('bias_offset_x', straddle=True)
        phis = self.anc27.select_value('bias_offset_y', straddle=True)

        # Read the file time, and closest upper and lower dates from ANC27.
        anc27_dates_theta = self.anc27.select_value('bias_offset_x', return_dates=True)
        anc27_dates_phi = self.anc27.select_value('bias_offset_y', return_dates=True)

        # Obtain an interpolated theta and phi.
        theta = self.calculate_theta_phi_misalignment(anc27_dates_theta['t'], anc27_dates_theta['t_n'], 
            anc27_dates_theta['t_nplus1'], angle_n=thetas[0], angle_nplus1=thetas[1])
        phi = self.calculate_theta_phi_misalignment(anc27_dates_phi['t'], anc27_dates_phi['t_n'], 
            anc27_dates_phi['t_nplus1'], angle_n=phis[0], angle_nplus1=phis[1])

        # Finally can look up the CAL-47 value with the interpolated theta and phi.
        cal47_value = self.cal47.select_value(spot, theta=theta, phi=phi)

        S_RET = self.S_RET(S_max=S_max, S_over_S0=S_over_S0, cal47_value=cal47_value)

        if self.verbose:
            print("h: ", h)
            print("D_lam_max: ", D_lam_max)
            print("peak: ", self.atl01_store.raw_peak_xmtnc.converted)
            print("peak[0]: ", self.atl01_store.raw_peak_xmtnc.converted[0])
            print("edge: ", self.atl01_store.raw_edge_xmtnc.converted)
            print("edge[0]: ", self.atl01_store.raw_edge_xmtnc.converted[0])
            print("D_lam: ", D_lam)
            print("S_over_S0: ", S_over_S0)
            print("S_over_S0[0]: ", S_over_S0[0])
            print("theta: ", theta)
            print("phi: ", phi)
            print("cal47_value: ", cal47_value)

        sensitivity_container = SensitivityContainer(
            m=m,
            V_nom=V_nom,
            b=b,
            c=c,
            V=V,
            S_over_S_nom=S_over_S_nom,
            S_max=S_max,
            S_BG=S_BG,
            h=h,
            D_lam_max=D_lam_max,
            D_lam=D_lam,
            S_over_S0=S_over_S0,
            cal47_value=cal47_value,
            theta=theta,
            phi=phi,
            S_RET=S_RET)

        return sensitivity_container


    #################
    # Total Energies
    #################

    def spd_energy(self):
        """ Calculated the SPD energy.
        Returns a 6x18 array.
        """
        #  For selection from CAL-54, sensor either SPDA or SPDB
        sensor = 'SPD_{}'.format(self.side)

        # S and T read from telemetry
        if self.laser == 1:
            # A and B are using the same sensor, really an optical bench temperature.
            T = self.atl01_store.raw_hkt_beamx_t.converted
        elif self.laser == 2:
            if self.side == 'A':
                T = self.atl01_store.raw_spda_therm_t.converted
            elif self.side == 'B':
                T = self.atl01_store.raw_spdb_therm_t.converted

        if self.side == 'A':
            S = self.atl01_store.raw_pri_lsr_energy.converted
        elif self.side == 'B':
            S = self.atl01_store.raw_red_lsr_energy.converted

        # Because ASAS does an interpolation of temperature to match
        # tempo of the energy read-outs, use the ATL02 temperature
        # to do calculation, but retain my calculated T to check 
        # that in proper bounds.
        T_atl02 = self.atl02_store.spd_temp

        E_total = self.E_total(sensor, S, T_atl02)

        return T, S, E_total

    def laser_internal_energy(self):
        """ Calculated the laser internal energy.
        Returns a 1x18 array.
        """
        # For selection from CAL-54, sensor either Laser_1 or Laser_2
        sensor = 'Laser_{}'.format(self.laser)

        # S and T read from telemetry
        if self.laser == 1:
            T = self.atl01_store.raw_hkt_cchp_las1_t.converted
        elif self.laser == 2:
            T = self.atl01_store.raw_hkt_cchp_las2_t.converted

        # This records whichever laser is active.
        S = self.atl01_store.raw_energy_data_shg.raw

        # Because ASAS does an interpolation of temperature to match
        # tempo of the energy read-outs, use the ATL02 temperature
        # to do calculation, but retain my calculated T to check 
        # that in proper bounds.
        T_atl02 = self.atl02_store.laser_internal_temp

        E_total = self.E_total(sensor, S, T_atl02)

        return T, S, E_total

    def lrs_energy(self): 
        """ Calculated the LRS energy.
        Returns a 1x898 array.
        """
        # For selection from CAL-54.
        sensor = 'LRS'

        # T read from telemetry
        # S from equation 5-2, where each beam read from telemetry.
        ## Notes say APID1120, A_LRS_HK, ANALOGUE_HK_CHANNEL_33.
        T = self.atl01_store.raw_ldc_t.converted

        ## Average the temperature?
        print("LRS T: ", T)
        T_mean = np.mean(T)

        # Need add all the individual spot energies.    
        S = self.S_LRS()

        # Fill out an array of size S for T
        T = np.ones(len(S))*T_mean

        # Because ASAS does an interpolation of temperature to match
        # tempo of the energy read-outs, use the ATL02 temperature
        # to do calculation, but retain my calculated T to check 
        # that in proper bounds.
        T_atl02 = self.atl02_store.lrs_temp

        E_total = self.E_total(sensor, S, T_atl02)

        return T, S, E_total

    def E_total(self, sensor, S, T):
        """ Equation 5-1.
        Calculates the total energy for the given sensor.

        The coefficients a, b0, b1, c0, c1, d, and e are read from CAL-54.

        For the SPD energy montiors, S is the raw telemetry value + 32768.
        For other sensors, S is the raw telemetry value.

        Parameters
        ----------
        sensor : str
            spdA, spdB, laser1internal, laser2internal. lrs.
            This decides the CAL-54 coefficients and S. 
        S : float
            Sensor reading from telemetry for SPD and laser internal monitors.
            For LRS, need retrieve this value using Equation 5-2.
        T : float
            Sensor temperature from telemetry.

        ATL02 Location
        --------------
        /atlas/housekeeping/laser_energy_internal/e_tx
        /atlas/housekeeping/laser_energy_lrs/e_tx
        /atlas/housekeeping/laser_energy_spd/e_tx
        """
        T0 = self.cal54.T0 ## Note: T0 may become a column.
        coeffs= self.cal54.select_coeffs(sensor, laser=self.laser)

        if self.verbose:
            print("T: ", T)
            print("S: ", S)
            print("coeffs: ", coeffs)

        return coeffs['a'] + (coeffs['b0'] + coeffs['b1']*(T-T0))*S + (coeffs['c0'] + \
            coeffs['c1']*(T-T0))*S**2 + coeffs['d']*S**3 + coeffs['e']*S**4

    def S_LRS(self):
        """ Equation 5-2.
        For the LRS sensor, need sum all spot magnitudes to get sensor reading.
        """
        # Check for each of 6 spot's availability by ensuring the spot's 
        # column in raw_quality_f is all 0s.
        # Spots start at column index 4 (this is already taken care of in `DataSet.convert`.)
        # Transform the matrix so can index by column.
        sum_mags = np.zeros(len(self.atl01_store.raw_quality_f.raw[0]))
        j = 0
        for quality_f, cent_mag in zip(self.atl01_store.raw_quality_f.raw, self.atl01_store.raw_cent_mag.raw):
            if 1 not in quality_f:
                sum_mags += cent_mag 
            else:
                print("WARNING: Spot {} missing in raw_cent_mag.".format(j+1))
            j += 1
        return np.array(sum_mags)

    #######################################
    # Individual Transmitted Beam Energies
    #######################################

    def E_fractional(self, E_total):
        """ Equations 5-3 and 5-4. 
        Compute the fractional energy for each spot.

        Parameters
        ----------
        E_total : list
            The total energy.

        ATL02 Location
        --------------
        /atlas/housekeeping/laser_energy_internal/
        /atlas/housekeeping/laser_energy_lrs/
        /atlas/housekeeping/laser_energy_spd/

        For each, Beam 1 through Beam 6 are labeled pce1_s, pce1_w, pce2_s, pce2_w, pce3_s, and pce3_w.
        """
        # Multiply each CAL-45 value by total energy.
        E_spot1 = np.float64(E_total*self.cal45.select_value(1))
        E_spot2 = np.float64(E_total*self.cal45.select_value(2))
        E_spot3 = np.float64(E_total*self.cal45.select_value(3))
        E_spot4 = np.float64(E_total*self.cal45.select_value(4))
        E_spot5 = np.float64(E_total*self.cal45.select_value(5))
        E_spot6 = np.float64(E_total*self.cal45.select_value(6))

        return E_spot1, E_spot2, E_spot3, E_spot4, E_spot5, E_spot6

    ####################################
    # Small-signal Receiver Sensitivity
    ####################################

    def receiver_sensitivity_ratio(self, V, V_nom, b, c):
        """ Equation 5-7
        The ratio of receiver sensitivity at the current voltage V to 
        the receiver sensitivity at the nominal voltage.

        Parameters
        ----------
        V : list
            Voltages for the sensor.
        V_nom : float
            Nominal bias voltage from CAL-46.
        b : float
            Parameter with units 1/V from CAL-46.
        c : float
            Parameter with units 1/V^2 from CAL-46.
        """
        return (c*(V - V_nom)**2 + b*(V - V_nom) + 1)

    def S_max(self, m, S_over_S_nom):
        """ Equation 5-8. 
        Max sensitivity, computed for each of 6 spots on the active side.
        This is an intermediate calculation that is used to produce
        background sensitivity and return sensitivity. It itself is
        not reported in ATL02.

        Note all values except the aging correction are side, 
        spot-dependent.

        Parameters
        ----------
        m : float
            The slope, as read from CAL-30.
        S_over_S_nom : list
            The ratio of receiver sensitivity at the current voltage V
            to the receiver sensitivity at the nominal voltage.
        """
        return m*S_over_S_nom*0.813*self.anc27.select_value('cal46_aging')

    #########################
    # Background Sensitivity 
    #########################

    def S_BG(self, S_max):
        """ Equation 5-9.
        Background sensitivity is the receiver's response in events/sec
        per watt of continuous illumination in the receiver's passband 
        from a diffuse source larger than the receiver's field of view,
        in the absence of any dead time effects.

        Parameters
        ----------
        S_max : list
            The max sensitivity for a given sensor + spot.
        """
        return self.scaling_factor_bg*S_max

    #####################
    # Return Sensitivity
    #####################

    def calculate_theta_phi_misalignment(self, t, t_n, t_nplus1, angle_n, angle_nplus1):
        """ Equations 5-11 and 5-13.
        Calculates the misalignment for theta or phi from CAL-47 relative
        to zero misalignment.

        Parameters
        ----------
        t : float
            Time of observation.
        t_n : float
            Time of first of two consecutive AMCS calibrations.
        t_nplus1 : float
            Time of second of two consecutive AMCS calibrations.
        angle_n : float
            Either the theta or phi. First of two from ANC27.
        angle_nplus1 : float
            Either the theta or phi. Second of two from ANC27.
        """
        return ((t - t_n) / (t_nplus1 - t_n))*(angle_nplus1 - angle_n)

    def receiver_sensitivity_ratio_tuning(self, h, D_lam, D_lam_max):
        """ Equation 5-17
        The ratio of receiver sensitivity at the current state of 
        tuning to the receiver sensitivity at zero mistuning.

        Parameters
        ----------
        h : float
            Parameter read from CAL-61.
        D_lam : list
            The calibrated peak/edge wavelength deviation ratio.
        D_lam_max : float
            The value of D_lam when the laser is tuned to maximum
            transmittance of the OFMs. Read from CAL-61.
        """
        return np.float64(1 + h*(D_lam - D_lam_max)**2)

    def S_RET(self, S_max, S_over_S0, cal47_value):
        """ Equation 5-18.
        The return sensitivity, unique for each sensor + spot.

        Parameters
        ----------
        S_max : list
            The maximum sensitivity for the sensor + spot.
        S_over_S0 : list
            Ratio of receiver sensitivity at the current state of 
            tuning to the receiver sensitivity at zero mistuning.
        cal47_value : 
            The image cal-47 value for the appropriate misalignment
            angle. This is INTERPOLATED via Eq 5-13.
        """
        return self.scaling_factor_ret*S_max*cal47_value*S_over_S0

    #############
    # Misc Stuff
    #############

    def threshold_readbacks(self):
        """ 
        These are telemetry values indicating the actual SPD upper and lower 
        threshold voltages.  They are not used in any calculations in ATL02.
        As a diagnostic of ATLAS, they should be consistent with the commands
        that were sent to set the thresholds.  They are also necessary for 
        anyone trying to interpret the SPD upper and lower widths and skew
        that we include in ATL02.
        """
        if self.side == 'A':
            thrhi = self.atl01_store.raw_pri_thrhi_rdbk.converted
            thrlo = self.atl01_store.raw_pri_thrlo_rdbk.converted
        else:
            thrhi = self.atl01_store.raw_red_thrhi_rdbk.converted
            thrlo = self.atl01_store.raw_red_thrlo_rdbk.converted
        # Convert from mV to V
        return thrhi*1e-3, thrlo*1e-3
