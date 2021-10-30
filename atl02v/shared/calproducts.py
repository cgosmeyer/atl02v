""" Calibration product selection functions.

Author:

    C.M. Gosmeyer, May 2018

Notes:
    
    Should I make eventeh TOF calibrations classes, where ALL files are
    read in and stored as pandas df, saving i/o,

"""

import datetime
import glob
import numpy as np
import os
import pandas as pd

from atl02v.shared.tools import find_nearest, find_nearest_date, sips2atl02, str2datetime

#################
# Time of Flight
#################

class ReadANC13(object):
    """
    """
    def __init__(self, anc13_path, valid_date_range=[0,5.93e8]):
        self.anc13_file = glob.glob(os.path.join(anc13_path, 'ANC13*'))[0]
        self.valid_date_range = valid_date_range
        self.select_values()

    def select_values(self):
        """
        """
        # Wrtite to pandas dataframe 
        df = pd.read_table(self.anc13_file, sep=' ', comment='#', header=None)
        # Convert date column from string to float
        df[22] = np.array([np.float64(i.split('D')[0]) for i in df[22]])
        # Select only row that falls in valid date range.
        data = df[df[22] > self.valid_date_range[0]][df[22] < self.valid_date_range[1]]
        if len(data) > 1:
            # Select last.
            data = data[len(data)-1:len(data)]

        # To make parsing easier, remove from series to np array
        # and convert all values to strings.
        data = np.asarray(data.values[0], dtype=str)

        self.laser = int(data[0])
        self.side_spd = data[1].strip()
        self.side_pcd = data[2].strip()
        self.oscillator = data[3].strip()
        self.mode = int(data[4])
        self.spd_lower = np.float64(data[6].split('D')[0])
        self.spd_upper = np.float64(data[7].split('D')[0])
        self.cmd_laser_vbg_temp = np.float64(data[8].split('D')[0])
        self.lrs_params = data[9]
        self.cmd_wtom_temp = np.float64(data[11].split('D')[0])
        self.cmd_ofa1_temp = np.float64(data[12].split('D')[0])
        self.cmd_ofa2_temp = np.float64(data[13].split('D')[0])
        self.cmd_ofa3_temp = np.float64(data[14].split('D')[0])
        self.cmd_ofa4_temp = np.float64(data[15].split('D')[0])
        self.cmd_hvpc1_temp = np.float64(data[16].split('D')[0])
        self.cmd_hvpc2_temp = np.float64(data[17].split('D')[0])
        self.cmd_hvpc3_temp = np.float64(data[18].split('D')[0])
        self.cmd_hvpc4_temp = np.float64(data[19].split('D')[0])
        self.cmd_hvpc5_temp = np.float64(data[20].split('D')[0])
        self.cmd_hvpc6_temp = np.float64(data[21].split('D')[0])
        self.d_implement_time =np.float64(data[22])


class ReadANC27(object):
    __name__ = 'ReadANC27'
    """ The ANC27 is side-dependent.
    """
    def __init__(self, anc27_path, side, atl02_date, verbose=False):
        self.verbose = verbose
        self.side = side
        self.atl02_date = atl02_date
        self.anc27_file = glob.glob(os.path.join(anc27_path, 'ANC27*_{}*'.format(self.side.upper())))[0]
        print("Reading ANC27 file ", self.anc27_file)
        self.fields = []
        self.values = []
        self.app_dates = []
        self.build_dataframe()
        # Note that ASAS and ANC27 use different reference dates.
        self.anc27_ref_date = datetime.datetime(1970,1,1)
        self.asas_ref_date = datetime.datetime(2018,1,1)

    def parse4pd(self, arr):
        for item in arr:
            if item[0] != '#':
                item_split = item.split(',')
                if self.verbose:
                    print("item: ", item)
                    print("item_split: ", item_split)
                self.fields.append(item_split[0].split('=')[0].lower())
                self.values.append(np.float64(item_split[1].split('D')[0]))
                self.app_dates.append(str2datetime(sips2atl02(item_split[2].split('D')[0])))

    def build_dataframe(self):
        """
        Select the values that come closest to, but do not exceed, 
        the atl02 date.
        """
        uso_freqs = []
        cal_46_agings = []
        bias_offset_xs = []
        bias_offset_ys = []
        bias_rates = []
        wtom_tuning_flags = []
        wtom_lambda_offs = []
        wtom_alt_tune_corrs = []

        with open(self.anc27_file) as f: 
            for line in f:
                if 'USO_FREQ=' in line:
                    #uso_freq = line
                    uso_freqs.append(line)
                elif 'CAL46_AGING_{}='.format(self.side) in line:
                    #cal_46_aging = line
                    cal_46_agings.append(line.replace('CAL46_AGING_{}='.format(self.side), 'CAL46_AGING='))
                elif 'BIAS_OFFSET_X' in line:
                    #bias_offset_x = line
                    bias_offset_xs.append(line)
                elif 'BIAS_OFFSET_Y' in line:
                    #bias_offset_y = line
                    bias_offset_ys.append(line)
                elif 'BIAS_RATE' in line:
                    #bias_rate = line
                    bias_rates.append(line)
                elif 'WTOM_TUNING_FLAG' in line:
                    #wtom_tuning_flag = line
                    wtom_tuning_flags.append(line)
                elif 'WTOM_LAMBDA_OFF' in line:
                    #wtom_lambda_off = line
                    wtom_lambda_offs.append(line)
                elif 'WTOM_ALT_TUNE_CORR' in line:
                    #wtom_alt_tune_corr = line
                    wtom_alt_tune_corrs.append(line)

        for arr in [uso_freqs, cal_46_agings, bias_offset_xs, bias_offset_ys,
            bias_rates, wtom_tuning_flags, wtom_lambda_offs, wtom_alt_tune_corrs]:
            self.parse4pd(arr)

        # Create pandas dataframe
        if self.verbose:
            print("ANC27: ")
            print("fields: ", self.fields)
            print("values: ", self.values)
            print("applicable dates: ", self.app_dates)

        self.df = pd.DataFrame.from_records(np.array([self.fields, self.values, self.app_dates]).T, 
            columns=['Fields', 'Values', 'AppDates'])

    def select_value(self, field, straddle=False, return_dates=False):
        """
        Parameters
        ----------
        field : str
            Name of the ANC27 field.
        straddle : {True, False}
            Set to True to return both the lesser and the greater nearest
            value. Default is to return only lesser nearest.
        return_dates : {True, False}
            To return the file date, and the nearest upper and lower dates,
            all as DateTime seconds.
        """
        if return_dates:
            straddle = True

        field = field.lower()

        if field == 'bias_time':
            # First find appropriate bias rate.
            bias_rate = self.find_nearest('bias_rate')
            # The bias_time is the last value in each of the bias_offset_x, bias_offset_y, and bias rate lines
            ## To convert it from UTC to ATLAS epoch seconds, subtract 568036805.0
            bias_time = self.find_nearest('bias_rate', straddle=True, return_dates=True)
            print('bias_time: ', bias_time)

            # Diff the reference dates to find total seconds between them.
            # Then can subtract the seconds to convert from seconds from 1/1/1970 UTC 
            # to seconds from ATLAS start of epoch (1/1/2018).
            refdate_diff = (self.asas_ref_date - self.anc27_ref_date).total_seconds()
            if not straddle:
                return bias_time['t_n'] - refdate_diff
            else:
                
                return [bias_time['t_n'] - refdate_diff, float(bias_time['t_nplus1']) - refdate_diff]
        else:
            return self.find_nearest(field, straddle, return_dates)
        
    def find_nearest(self, field, straddle=False, return_dates=False):
        """
        Parameters
        ----------
        field : str
            Name of the ANC27 field.
        straddle : {True, False}
            Set to True to return both the lesser and the greater nearest
            value. Default is to return only lesser nearest.
        return_dates : {True, False}
            To return the file date, and the nearest upper and lower dates,
            all as DateTime seconds.
        """
        # First check how many dates are available to choose from.
        # If just one, return that value.
        if len(self.df['AppDates'][self.df['Fields'] == field]) == 1:
            nearest_value = np.float64(self.df['Values'][self.df['Fields'] == field].values[0])
            if not straddle:
                return nearest_value
            else:
                if not return_dates:
                    return [nearest_value, nearest_value]
                else:
                    # Equation 5-15 should just reduce to 1*{theta,phi}.
                    return {'t':2, 't_n':1, 't_nplus1':2}

        # If multiple values, return the nearest lessthan value (default).
        # Or if straddle True, return the nearest lessthan and greater than values.
        else:
            atl02_date_sec = (str2datetime(self.atl02_date)-self.anc27_ref_date).total_seconds()
            lessthan_dates = self.df['AppDates'][self.df['Fields'] == field][self.df['AppDates'] <= self.atl02_date].values
            lessthan_values = self.df['Values'][self.df['Fields'] == field][self.df['AppDates'] <= self.atl02_date].values
            lessthan_dates = [(str2datetime(x)-self.anc27_ref_date).total_seconds() for x in lessthan_dates]
            nearest_lessthan_date = find_nearest_date(lessthan_dates, atl02_date_sec)
            nearest_lessthan_idx = lessthan_dates.index(nearest_lessthan_date)
            nearest_lessthan_value = lessthan_values[nearest_lessthan_idx]
            if not straddle:
                return np.float64(nearest_lessthan_value)
            else:
                grtrthan_dates = self.df['AppDates'][self.df['Fields'] == field][self.df['AppDates'] >= self.atl02_date].values
                grtrthan_values = self.df['Values'][self.df['Fields'] == field][self.df['AppDates'] >= self.atl02_date].values
                grtrthan_dates = [(str2datetime(x)-self.anc27_ref_date).total_seconds() for x in grtrthan_dates]
                nearest_grtrthan_date = find_nearest_date(grtrthan_dates, atl02_date_sec)
                nearest_grtrthan_idx = grtrthan_dates.index(nearest_grtrthan_date)
                nearest_grtrthan_value = grtrthan_values[nearest_grtrthan_idx]
                if not return_dates:
                    return [np.float64(nearest_lessthan_value), np.float64(nearest_grtrthan_value)]
                else:
                    # See ATBD Equation 5-15.
                    return {'t':np.float64(atl02_date_sec), 't_n':np.float64(nearest_lessthan_date), 
                        't_nplus1':np.float64(nearest_grtrthan_date)}

class ReadCal17(object):
    def __init__(self, cal17_path, verbose=False):
        """Table 12.
        Calibration product for PCE Unit Cell Delay.

        RX is per-event and uses PCE, toggle, channel, and cell to index.
        TX is per-MF and uses PCE, LL/OT, and cell to index.

        Parameters
        ----------
        cal17_path : string
            Path to CAL-17 files.
        verbose : {True, False}
            False by default.
        """
        self.verbose = verbose
        self.cal17_path = cal17_path

        # First read in index, and find correct file via the index using PCE + Cal_rising/Cal_falling
        self.index_file = glob.glob(os.path.join(self.cal17_path, 'CAL_17_IDX*csv'))[0]   
        self.index_df = pd.read_csv(self.index_file, skiprows=7)
        self.index_df['master_key'] = np.arange(len(self.index_df))
        self.cal17_df = self.__build_table()
        
    def __build_table(self):
        """
        """
        # Read each product listed in index file.
        # Write table with columns of PCE, LL, OT, Cell, Cal_Rising, chan_[1-20]_R, Cal_Falling, chan_[1-20]_F
        cal_dfs = []
        for idx in self.index_df['master_key']:
            cal_df = pd.read_csv(os.path.join(self.cal17_path, self.index_df['File_Name'][idx]), skiprows=18)
            cal_df['PCE'] = self.index_df['PCE'][idx]
            cal_df['Temperature'] = self.index_df['Temperature'][idx]
            cal_df['Cal_Falling'] = self.index_df['Cal_Falling'][idx]
            cal_df['Cal_Rising'] = self.index_df['Cal_Rising'][idx]
            cal_df['File_Name'] = self.index_df['File_Name'][idx]
            cal_df.rename(columns={'LU':'OT', 'LU_std':'OT_std',
                                   'TU':'OT', 'TU_std':'OT_std',
                                   'TL':'OT', 'TL_std':'OT_std'}, inplace=True)
            cal_dfs.append(cal_df)

        # Concatenate the tables
        cal17_df = pd.concat(cal_dfs, ignore_index=True, sort=False)

        return cal17_df

    def select_values(self, pce, FC_TDC_rising, FC_TDC_falling, 
        raw_rx_channel_id, raw_rx_toggle_flg, rx_fc=[], tx_fc=[], 
        component=None):
        """
        """
        # First filter for PCE.
        pce_df = self.cal17_df[self.cal17_df['PCE'] == pce]

        # Find nearest rising, falling values.
        rising_nearest, idx_rising_nearest = find_nearest(pce_df['Cal_Rising'], FC_TDC_rising)
        falling_nearest, idx_falling_nearest = find_nearest(pce_df['Cal_Falling'], FC_TDC_falling)

        # Filter for rising
        rising_df = pce_df[pce_df['Cal_Rising'] == rising_nearest]

        # Filter for falling
        falling_df = pce_df[pce_df['Cal_Falling'] == falling_nearest]

        # Choose the calibrated Fine Count based on whether Rx or Tx.
        if len(rx_fc) != 0:
            # rising=1, falling=0
            # This only matters for RX, since TX must always reference rising.
            toggles = np.array(raw_rx_toggle_flg, dtype=str)
            toggles[toggles == '1'] = 'R'
            toggles[toggles == '0'] = 'F'

            if len(rx_fc) != len(raw_rx_toggle_flg):
                print("ERROR: Length of rx_fc doesn't match raw_rx_toggle_flg: {} != {}".format(len(rx_fc), len(raw_rx_toggle_flg)))
                return None

            # If Rx, select value based on tx_fc (row) and channel+rising/falling (column)
            # Calculate the correct column, using channel number and whether rising or falling.
            cal17_values = []

            # Loop over events in the major frame.
            for i in range(len(rx_fc)):
                # Find the rx_fc "cell" row. 
                if raw_rx_channel_id[i] == 0:
                    cal17_value = 0  
                elif toggles[i] == 'R':
                    cal17_value = rising_df['Chan_{}_{}'.format(raw_rx_channel_id[i],toggles[i].upper())][rising_df['Cell'] == rx_fc[i]].values[0]
                elif toggles[i] == 'F':
                    cal17_value = falling_df['Chan_{}_{}'.format(raw_rx_channel_id[i],toggles[i].upper())][falling_df['Cell'] == rx_fc[i]].values[0]           
                cal17_values.append(np.float32(cal17_value))  

        elif len(tx_fc) != 0:
            # If Tx, select value based on LL/Other
            component = component[:2].upper()

            cal17_values = []
            # Loop over events in the major frame.
            # TX always references the rising table because transmit times are always rising.
            for i in range(len(tx_fc)):
                cal17_value = rising_df[component][rising_df['Cell'] == tx_fc[i]].values[0]
                cal17_values.append(np.float32(cal17_value))
            
        return np.asarray(cal17_values)
        

class ReadCal44(object):
    __name__ = 'ReadCal44'
    def __init__(self, cal44_path, side, atl02, verbose=False):
        """ Start Timing Skews

        Parameters
        ----------
        cal44_path : string
            Path to CAL-44 files.
        side : string
            The SPD side. 'A' or 'B'.
        atl02 : 
            Open ATL02 file.
        verbose : {True, False}
            False by default.
        """
        self.verbose = verbose
        self.side = '{}XX'.format(side.upper())

        # First read in index, and find correct file via the side and temperature.
        self.index_file = glob.glob(os.path.join(cal44_path, 'CAL_44_IDX*csv'))[0]

        # Read the file.
        index_df = pd.read_csv(self.index_file, skiprows=7)

        # Find the temperature, taking the mean of the dataset
        temperatures = np.array(atl02['housekeeping/thermal/spd{}_therm_t'.format(side.lower())].value)
        self.temperature = np.mean(temperatures)

        # Filter out the sides and find the row with closest matching temperature.
        side_filtered = index_df['SPD_Temp'][index_df['Configuration'] == self.side]
        self.nearest_temp, ind_nearest = find_nearest(side_filtered, self.temperature)

        # Need add to the index if side B.
        if side == 'BXX':
            ind_add = len(index_df['Configuration'][index_df['Configuration'] == 'AXX'])
        else:
            ind_add = 0

        # Select file.
        self.cal_file = os.path.join(cal44_path, index_df['File_Name'][ind_add + ind_nearest])

        if self.verbose:
            print("nearest_temp: ", self.nearest_temp)
            print("ind_add: ", ind_add )
            print("cal_file: ", self.cal_file)   

        # Read the file.
        self.cal_df = pd.read_csv(self.cal_file, skiprows=15)

        # Read the LL1 value from header.
        with open(self.cal_file) as f:
            for line in f:
                if 'LL1' in line and '-LL1' not in line:
                    line = line.split(',')
                    self.ll1 = np.float64(line[1])

    def select_value(self, pce):
        """
        """
        valid_cols = {1:'LU-LL1', 2:'TU-LL2', 3:'TL-LL3'}

        # Find the cal-44 value for the PCE.
        cal44_value = self.cal_df[valid_cols[pce]][0]

        return np.float32(cal44_value)

class ReadCal49(object):
    __name__ = 'ReadCal49'
    """
    """
    def __init__(self, cal49_path, side, atl02, verbose=False):
        """Table 16. 
        Receiver Channel Skews

        Parameters
        ----------
        cal49_path : string
            Path to CAL-49 files.
        side : string
            The detector bank side. 'A' or 'B'.
        atl02 : 
            Open ATL02 file.
        verbose : {True, False}
            False by default.
        """
        self.verbose = verbose

        self.side = 'X{}X'.format(side.upper())

        # First read in index, and find correct file via the temperature and side.
        self.index_file = glob.glob(os.path.join(cal49_path, 'CAL_49_IDX*csv'))[0]

        index_df = pd.read_csv(self.index_file, skiprows=7)        

        # Find the temperature, taking the mean of the dataset.
        temperatures = np.array(atl02['housekeeping/thermal/hkt_dem1_t4_et_t'].value)  ## this may need be updated!
        self.temperature = np.mean(temperatures)

        # Filter out the sides and find the row with closest matching temperature.
        side_filtered = index_df['Temperature'][index_df['Configuration'] == self.side]
        self.nearest_temp, ind_nearest = find_nearest(side_filtered, self.temperature)

        # Need add to the index if side B.
        if self.side == 'XBX':
            ind_add = len(index_df['Configuration'][index_df['Configuration'] == 'XAX'])
        else:
            ind_add = 0

        # Select file.
        self.cal_file = os.path.join(cal49_path, index_df['File_Name'][ind_add + ind_nearest])

        if self.verbose:
            print("side_filtered: ", side_filtered)
            print("nearest_temp: ", self.nearest_temp)
            print("ind_add: ", ind_add)
            print("cal_file: ", self.cal_file)   

        # Read the file.
        self.cal_df = pd.read_csv(self.cal_file, skiprows=13)

    def select_value(self, event_edge, pce, channel):
        # Find the skew value from pce, spot, and event edge

        if channel == 0:
            return 0
        else:
            cal49_value = self.cal_df['Skew'][self.cal_df['Rising/Falling'] == event_edge][self.cal_df['PCE'] == pce][self.cal_df['Spot_Channel'] == channel].values[0]
            return np.float32(cal49_value)

class ReadKs(object):
    __name__ = 'ReadKs'
    def __init__(self, anc27_file):
        """ 
        ** depricated by ReadANC27? **

        Parameters
        ----------
        anc27_file : string
            Name of the ANC27.
        """
        self.anc27_file = anc27_file

        self.k_data = self.read_ks_into_matrix()

    def read_ks_into_matrix(self):
        """ Read from ANC27.
        
        Returns
        -------
        k_data : list of lists
            The k-values for each scenario read in from ANC27.        
        """
        k_data = []
        with open(self.anc27_file) as f:
            for line in f:
                if 'START_TIME_COEFF_' in line:
                    scenario = line.split(',')
                    # remove the first and the last values
                    scenario = scenario[1:-1]
                    # Convert to floats, convert the first value nanoseconds, and add to output list.
                    k_data.append([np.float32(scenario[i])*10**-9 if i==0 \
                        else np.float32(scenario[i]) for i in range(len(scenario))])
        return k_data

    def select_k(self, lu, tu, tl):
        """ Table 16.
        Select the scenario-specific coefficients, for calculating centroid time,
        T_center.

        Parameters
        ----------
        lu : float
            The Leading Upper value.
        tu : float
            The Trailing Upper value.
        tl : float
            The Trailing Lower value.
        """
        # Make an array of True/False.
        other_values = [np.isnan(lu), np.isnan(tu), np.isnan(tl)]

        # See which scenario it matches.
        # 'False' means a real number, 'True' means a NaN.
        scenarios = {1:[False, False, False],
                     2:[True, False, False],
                     3:[False, True, False],
                     4:[False, False, True],
                     5:[True, True, False],
                     6:[True, False, True],
                     7:[False, True, True],
                     8:[True, True, True]}

        # Based on whether the lu, tu, and tl are NANs, select the scenario.
        for scenario, ks in zip(scenarios.keys(), scenarios.values()):
            if ks == other_values:
                # Index into k_data.
                index = scenario - 1
                return self.k_data[index]
                
#############
# Radiometry
#############

class ReadCal30(object):
    __name__ = 'ReadCal30'
    def __init__(self, cal30_path, side, atl02):
        """ Nominal RX Sensitivity

        Parameters
        ----------
        cal30_path : string
            Path to CAL-30 files.
        side : string
            The detector bank side. 'A' or 'B'.
        atl02 : 
            Open ATL02 file.
        """
        self.side = 'X{}X'.format(side.upper())

        # First read in index, and find correct file via the temperature and side.
        self.index_file = glob.glob(os.path.join(cal30_path, 'CAL_30_IDX*csv'))[0]
        index_df = pd.read_csv(self.index_file, skiprows=7)   

        # Add a master key column, to make indexing later easier.
        index_df['master_key'] = np.arange(len(index_df))

        # Select the calibration file with matching side.
        master_idx = int(index_df['master_key'][index_df['Configuration'] == self.side])
        self.cal_file = os.path.join(cal30_path, index_df['File_Name'][master_idx])

        # Read the file.
        self.cal_df = pd.read_csv(self.cal_file, skiprows=14)

    def select_value(self, spot):
        """
        """
        slope = np.float64(self.cal_df['Slope'][self.cal_df['Spot'] == spot])

        # Convert slope from [counts/s/pW] to [counts/s/W]
        return slope*1e12

class ReadCal45(object):
    __name__ = 'ReadCal45'
    def __init__(self, cal45_path, side):
        """ Laser Energy Fraction

        Parameters
        ----------
        cal45_path : string
            Path to CAL-45 files.
        side : string
            The {...} side. 'A' or 'B'.
        """
        # There is only one,
        self.cal_file = glob.glob(os.path.join(cal45_path, 'CAL_45*csv'))[0]

        # Read the file.
        self.cal_df = pd.read_csv(self.cal_file, skiprows=14)
        
        # Read the optics throughputs.
        with open(self.cal_file) as f:
            for line in f:
                if 'Optics_Throughput' in line:
                    line = line.split(',')
                    self.e3 = np.float64(line[3])
                    self.e6 = np.float64(line[4])
                    self.e11 = np.float64(line[5])

    def select_value(self, spot):
        """
        """
        spot = str(spot)
        beam_fraction = np.float64(self.cal_df[spot])
        return beam_fraction

class ReadCal46(object):
    __name__ = 'ReadCal46'
    def __init__(self, cal46_path, side):
        """ HV Bias Receiver Radiometric Sensitivity

        Parameters
        ----------
        cal46_path : string
            Path to CAL-46 files.
        side : string
            The detector bank side. 'A' or 'B'.
        """
        # Select file.
        self.cal_file = glob.glob(os.path.join(cal46_path, 'CAL_46_X{}X*csv'.format(side)))[0]

        # Read the file.
        self.cal_df = pd.read_csv(self.cal_file, skiprows=11)
        
    def select_values(self, spot):
        """
        """
        V_nom = np.float64(self.cal_df['Vnom (V)'][self.cal_df['Spot'] == spot])
        b = np.float64(self.cal_df['b (1/V)'][self.cal_df['Spot'] == spot])
        c = np.float64(self.cal_df['c (1/V^2)'][self.cal_df['Spot'] == spot])

        return V_nom, b, c

class ReadCal47(object):
    __name__ = 'ReadCal47'
    def __init__(self, cal_path, atl01, atl02, laser, verbose=False):
        """
        """
        self.verbose = verbose
        self.temperature = np.mean(atl01.raw_hkt_beamx_t.converted)
        self.laser = laser
        self.configuration = 'XX{}'.format(self.laser)

        # First read in index, and find correct file via the temperature and side.
        self.index_file = glob.glob(os.path.join(cal_path, 'CAL_47_INDX*csv'))[0]
        index_df = pd.read_csv(self.index_file, skiprows=7)

        side_filtered = index_df['Temperature, degC'][index_df['Configuration']==self.configuration]

        # Select all the files (spots) with desired configuration and temperature.
        self.nearest_temp, ind_nearest = find_nearest(side_filtered, self.temperature)
        self.cal_files = index_df['File_Name'][index_df['Configuration']==self.configuration]\
            [index_df['Temperature, degC']==self.nearest_temp].values

        # Create dataframe to hold the 6 spots.
        self.cal_df = pd.DataFrame(columns=[1,2,3,4,5,6])

        # Read in all spots to a dataframe.
        for cal_file in self.cal_files:
            spot = int(cal_file.split('SPOT')[1][0])
            spot_df = pd.read_csv(os.path.join(cal_path, cal_file), skiprows=13)
            self.cal_df[spot] = np.array(spot_df.columns)

        # Set parameters
        self.azimuth_grid_range = [-70,70]
        self.elevation_grid_range = [-70,70]
        self.grid_spacing = 1.0

    def create_spot_matrix(self, spot):
        """ Create the matrix for a given spot.
        """
        # Select the spot column.
        spot_col = self.cal_df[spot]

        # Initialize 70x70 matrix
        matrix = []

        # Wrap the column up into a 140x140 matrix, theta x phi.
        folds = np.arange(0,19600,140) 
        cnts = np.arange(0,140)

        for fold in folds:
            new_row = []
            for cnt in cnts:
                new_row.append(np.float64(spot_col)[cnt+fold])
            matrix.append(np.array(new_row))

        return np.array(matrix)

    def select_value(self, spot, theta=0, phi=0):
        """ Theta and phi are zero by default, read eventually from ANC27 
            [should be calculated in Equation 5-15]

        ##Is theta the column and phi the row? 

        Parameters
        ----------
        theta : int
            Must be between 0 and 69. Zero by default.
        phi : int
            Must be between 0 and 69. Zero by default.
        """
        matrix = self.create_spot_matrix(spot)

        # Select the nearest theta (column) and phi (row) out of (-70,70)
        self.nearest_theta, theta_idx = find_nearest(np.arange(-70,70), theta)
        self.nearest_phi, phi_idx = find_nearest(np.arange(-70,70), phi)

        if self.verbose:
            print("theta: {}, nearest_theta: {}, theta_idx: {}".format(theta, self.nearest_theta, theta_idx))
            print("phi: {}, nearest_phi: {}, phi_idx: {}".format(phi, self.nearest_phi, phi_idx))

        # Now select the value corresponding to the theta, phi.
        cal47_value = np.float64(matrix[theta_idx][phi_idx])

        return cal47_value

class ReadCal54(object):
    __name__ = 'ReadCal54'
    def __init__(self, cal54_path, verbose=False):
        """ Laser Energy Conversion

        Parameters
        ----------
        cal54_path : string
            Path to CAL-54 files.
        verbose : {True, False}
            False by default.
        """
        self.verbose = verbose
        # Select file.
        self.cal_file = glob.glob(os.path.join(cal54_path, 'CAL_54*csv'))[0]

        # Read the file.
        self.cal_df = pd.read_csv(self.cal_file, skiprows=14)

        # Read the reference temperature T0
        with open(self.cal_file) as f:
            for line in f:
                if 'Temperature' in line and 'deg C' in line:
                    self.T0 = np.float64((line.split(',')[1]))

    def select_coeffs(self, sensor, laser):
        """
        """
        sensor_laser_filtered = self.cal_df[self.cal_df['Sensor'] == sensor][self.cal_df['Laser'] == laser]

        coeffs = {'a': np.float64(sensor_laser_filtered['a_(J)']),
                  'b0': np.float64(sensor_laser_filtered['b0_(J/count)']),
                  'b1': np.float64(sensor_laser_filtered['b1_(J/degC_count)']),
                  'c0': np.float64(sensor_laser_filtered['c0_(J/count^2)']),
                  'c1': np.float64(sensor_laser_filtered['c1_(J/degC_count^2)']),
                  'd': np.float64(sensor_laser_filtered['d_(J/count^3)']),
                  'e': np.float64(sensor_laser_filtered['e_(J/count^4)']),
                  'std': np.float64(sensor_laser_filtered['std_of_residuals'])
                  }

        if self.verbose:
            print("CAL54")
            print("sensor: ", sensor)
            print("coeffs: ", coeffs)

        return coeffs

class ReadCal61(object):
    __name__ = 'ReadCal61'
    def __init__(self, cal61_path, atl02, side, laser, mode, verbose=False):
        """  RX Sensitivity vs WTOM

        Parameters
        ----------
        cal61_path : string
            Path to CAL-61 files.
        atl02 : 
            Open ATL02 file.
        side : string
            The detector bank side. 'A' or 'B'.
        laser : int
            The laser number.
        mode : int
            The laser energy mode.
        verbose : {True, False}
            False by default.
        """
        # Find the temperature.
        # raw_hkt_dem1_t4_et_t
        self.temperature = np.mean(atl02.hkt_dem1_t4_et_t)
        self.laser = laser
        self.side = 'X{}{}'.format(side, self.laser)

        # Read in index, and find correct file via the temperature and side.
        index_file = glob.glob(os.path.join(cal61_path, 'CAL_61_IDX*csv'))[0]

        index_df = pd.read_csv(index_file, skiprows=7)

        # Add a master key column, to make indexing later easier.
        index_df['master_key'] = np.arange(len(index_df))

        # Check that mode is listed in index file. Otherwise choose the nearest available mode.
        self.mode, _ = find_nearest(list(set(index_df['Energy_Level'].values)), mode)
        print("mode: ", self.mode)

        # Filter out the sides and find the row with closest matching temperature.
        side_energy_filtered = index_df['Temperature'][index_df['Configuration'] == self.side]\
            [index_df['Energy_Level'] == self.mode]
        self.nearest_temp, ind_nearest = find_nearest(side_energy_filtered, self.temperature)

        # Select the calibration file.
        master_indx = int(index_df['master_key'][index_df['Configuration'] == self.side]\
            [index_df['Energy_Level'] == self.mode][index_df['Temperature'] == self.nearest_temp])
        self.cal_file = os.path.join(cal61_path, index_df['File_Name'][master_indx])

        if verbose:
            print("nearest CAL-61 mode: ", self.mode)
            print("nearest_temp: ", self.nearest_temp)
            print("master_indx: ", master_indx)
            print("cal_file: ", self.cal_file)

        self.cal_df = pd.read_csv(self.cal_file, skiprows=18)        

    def select_value(self, spot):
        """
        """
        h = np.float64(self.cal_df['h'][self.cal_df['Spot'] == spot])
        ## This may change. I'm guessing xpeak is D_lam_max, but could be mistaken.
        xpeak = np.float64(self.cal_df['xpeak'][self.cal_df['Spot'] == spot])
        return h, xpeak

#################################################################
# For Funzies (Stored in ATL02 but not used in any calculations)
#################################################################

class ReadCal19(object):
    __name__ = 'ReadCal19'
    def __init__(self, cal19_path, verbose=False):
        """ First Photon Bias

        Parameters
        ----------
        cal19_path : string
            Path to CAL-19 files.
        verbose : {True, False}
            False by default.   
        """
        self.verbose = verbose

        index_file = glob.glob(os.path.join(cal19_path, 'CAL_19_IDX*csv'))[0]

        self.index_df = pd.read_csv(index_file, skiprows=7)

        # Add a master key column, to make indexing later easier.
        self.index_df['master_key'] = np.arange(len(self.index_df))

        self.file_names = self.index_df['File_Name'].values
        self.dead_times = self.index_df['Dead_Time'].values

class ReadCal20(object):
    __name__ = 'ReadCal20'
    def __init__(self, cal20_path, atl02, laser, mode, side, verbose=False):
        """ Low Link Impulse Response

        ** laser currently hard-coded to 1 for side AA **

        Parameters
        ----------
        cal20_path : string
            Path to CAL-20 files.
        verbose : {True, False}
            False by default.   
        """
        self.verbose = verbose
        self.laser = laser
        self.mode = mode
        self.side = side
        ## Temporary:
        if self.side == 'A':
            self.laser = 1
        ##
        # Use same sensor as for CAL-61 (for now)
        self.temperature = np.mean(np.array(atl02['housekeeping/thermal/hkt_dem1_t4_et_t'].value))

        index_file = glob.glob(os.path.join(cal20_path, 'CAL_20_IDX*csv'))[0]

        self.index_df = pd.read_csv(index_file, skiprows=7)

        # Add a master key column, to make indexing later easier.
        self.index_df['master_key'] = np.arange(len(self.index_df))

        print("index_df: ", self.index_df)
        print("laser: ", self.laser)
        print("mode: ", self.mode)
        print("side: ", self.side)

        # Filter out the side, laser, and mode, and find the row with closest matching temperature.
        side_filtered = self.index_df['Temperature']\
            [self.index_df['Configuration'] == '{}{}{}'.format(self.side, self.side, self.laser)]\
            [self.index_df['Mode'] == self.mode].values

        self.nearest_temp, idx_nearest = find_nearest(side_filtered, self.temperature)

        # Select the calibration file.
        master_idx = int(self.index_df['master_key']\
            [self.index_df['Configuration'] == '{}{}{}'.format(self.side, self.side, self.laser)]\
            [self.index_df['Mode'] == self.mode]\
            [self.index_df['Temperature'] == self.nearest_temp])
        self.cal_file = os.path.join(cal20_path, self.index_df['File_Name'][master_idx])

        # Read the file.
        self.cal_df = pd.read_csv(self.cal_file, skiprows=18)

        # Find the number of bins.
        self.num_bins = len(list(self.cal_df)[5:])

        # Read the BinWidth and ReturnSource
        with open(self.cal_file) as f:
            for line in f:
                line = line.split(',')
                if 'BinWidth' in line:
                    self.bin_width = np.float64(line[1])
                elif 'ReturnSource' in line:
                    self.return_source = line[1].strip()

    def select_values(self, pce):
        """
        """
        # Filter for only desired PCE.
        pce_df = self.cal_df[self.cal_df['PCE']==pce]
        # Return columsn as np matrix.
        temp = pce_df.values
        print(temp.shape)
        # Remove first five columns so return only histogram values.
        hist_values = [t[5:] for t in temp]

        return np.asarray(hist_values)


class ReadCal34(object):
    __name__ = 'ReadCal34'
    def __init__(self, cal34_path, verbose=False):
        """ Dead Time Radiometric Signal Loss

        Parameters
        ----------
        cal34_path : string
            Path to CAL-34 files.
        verbose : {True, False}
            False by default.        
        """
        self.verbose = verbose

        # Read in index, and find correct file via the temperature and side.
        index_file = glob.glob(os.path.join(cal34_path, 'CAL_34_IDX*csv'))[0]

        self.index_df = pd.read_csv(index_file, skiprows=7)

        # Add a master key column, to make indexing later easier.
        self.index_df['master_key'] = np.arange(len(self.index_df))

        self.file_names = self.index_df['File_Name'].values
        self.dead_times = self.index_df['Dead_Time'].values

class ReadCal42(object):
    __name__ = 'ReadCal42'
    def __init__(self, cal42_path, atl02, side, verbose=False):
        """ Dead Time

        Parameters
        ----------
        cal42_path : string
            Path to CAL-42 files.
        atl02 : 
            Open ATL02 file.
        side : string
            The detector bank side. 'A' or 'B'.
        verbose : {True, False}
            False by default.
        """
        self.verbose = verbose
        self.side = side
        # Use same sensor as for CAL-61 (for now)
        self.temperature = np.mean(np.array(atl02['housekeeping/thermal/hkt_dem1_t4_et_t'].value))

        # Read in index, and find correct file via the temperature and side.
        index_file = glob.glob(os.path.join(cal42_path, 'CAL_42_IDX*csv'))[0]

        self.index_df = pd.read_csv(index_file, skiprows=7)

        # Add a master key column, to make indexing later easier.
        self.index_df['master_key'] = np.arange(len(self.index_df))

        # Filter out the sides and find the row with closest matching temperature.
        side_filtered = self.index_df['Temperature'][self.index_df['Side'] == self.side]
        self.nearest_temp, idx_nearest = find_nearest(side_filtered, self.temperature)

        # Select the calibration file.
        master_idx = int(self.index_df['master_key'][self.index_df['Side'] == self.side]\
            [self.index_df['Temperature']==self.nearest_temp])
        self.cal_file = os.path.join(cal42_path, self.index_df['File_Name'][master_idx])

        # Read the file.
        self.cal_df = pd.read_csv(self.cal_file, skiprows=16)

    def select_values(self, pce):
        """
        """
        deadtimes = self.cal_df['DeadTime'][self.cal_df['PCE'] == pce].values
        sigmas = self.cal_df['Sigma'][self.cal_df['PCE'] == pce].values

        return deadtimes, sigmas

