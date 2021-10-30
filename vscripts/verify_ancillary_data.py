#!/usr/bin/env python

""" Verifying datasets in /ancillary_data/

Author:

    C.M. Gosmeyer

"""

import argparse
import glob
import h5py
import numpy as np
import os
import pandas as pd
from pydl import uniq
from atl02v.shared.calproducts import ReadANC13, ReadCal17, ReadCal19,\
    ReadCal20, ReadCal34, ReadCal42
from atl02v.shared.paths import path_to_outputs
from atl02v.shared.tools import make_file_dir, make_timestamp_dir, pickle_out
from atl02v.verification.ancillary_data_verification import VerifyAncillaryData, \
    VerifyCalibrations, VerifyCalibrationsPCE
from atl02v.verification.verification_tools import VFileReader


def verify_ancillary_data(vfiles, path_out):
    """ Function for verifying datasets in /ancillary_data/

    Notes
    -----
    JLee 29 May 2019 on comparing start and end dates between ATL01 and 
    ATL02:
    I would think everything should match within 1 second. The ATL01 times 
    are based off CCSDS header times; the ATL02 times are based on 
    computed Time-of-Day. The ATL02 is more correct.
    """
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)
    atl02 = h5py.File(vfiles.atl02, 'r', driver=None)

    # Find the ATL02 ancillary_data labels 
    atl02_ancillary_data = list(atl02['ancillary_data/'].keys())

    # Because lambda functions in a for loop will keep referencing same parameter,
    # need embed them in a second function.
    # See https://stackoverflow.com/questions/19837486/python-lambda-in-a-loop
    def makeFunc(atl02_label):
        return (lambda : np.array(atl01['ancillary_data/{}'.format(atl02_label)].value))

    # Build verification dictionary.
    d = {}

    date_labels = ['end_gpssow', 'start_gpssow', 'data_end_utc', 'data_start_utc']

    nonverifiable_labels = ['calibrations', 'housekeeping', 'isf', 'tep', 'tod_tof', 'qa_at_interval']

    for atl02_label in atl02_ancillary_data:
        print("atl02_label: ", atl02_label)
        if atl02_label in date_labels and atl02_label not in nonverifiable_labels:
            func = makeFunc(atl02_label)
            d[atl02_label] = [1, func, 'passthrough']
        elif atl02_label not in nonverifiable_labels and atl02_label not in date_labels:
            func = makeFunc(atl02_label)
            d[atl02_label] = [0, func, 'passthrough']

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyAncillaryData(vfiles, tolerance, atl02_dataset, path_out).do_verify(custom_func)
    atl01.close()
    atl02.close()

def verify_calibrations(vfiles, path_out):
    """ Call on individual verification functions for each calibration.
    """
    tof = pickle_out(vfiles.tof)
    radiometry = pickle_out(vfiles.radiometry)

    # Verify the upper-level values.
    d = {'ds_channel': [0, (lambda : np.arange(1,21)), 'calculation'],
         'ds_fine_counts': [0, (lambda : np.arange(0,75)), 'calculation']}
    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyAncillaryData(vfiles, tolerance, 'calibrations/{}'.format(atl02_dataset), 
            path_out).do_verify(custom_func)

    # Verify the calibration sub-groups.

    # CAL-17: effective_cell_delay
    d_cal17, d_pce_cal17 = ready_cal17(vfiles)
    for atl02_dataset in d_pce_cal17.keys():
        tolerance, custom_func, vtype = d_pce_cal17[atl02_dataset]
        VerifyCalibrationsPCE(vfiles, tolerance, 'effective_cell_delay', 
            atl02_dataset, path_out).do_verify(custom_func)
    for atl02_dataset in d_cal17.keys():
        tolerance, custom_func, vtype = d_cal17[atl02_dataset]
        VerifyCalibrations(vfiles, tolerance, 'effective_cell_delay', 
            atl02_dataset, path_out).do_verify(custom_func)

    # CAL-19: first_photon_bias
    d_cal19 = ready_cal19(vfiles)
    for atl02_dataset in d_cal19.keys():
        tolerance, custom_func, vtype = d_cal19[atl02_dataset]
        VerifyCalibrations(vfiles, tolerance, 'first_photon_bias', 
            atl02_dataset, path_out).do_verify(custom_func)

    # CAL-20: low_link_impulse_response
    d_cal20, d_pce_cal20 = ready_cal20(vfiles)
    for atl02_dataset in d_pce_cal20.keys():
        tolerance, custom_func, vtype = d_pce_cal20[atl02_dataset]
        VerifyCalibrationsPCE(vfiles, tolerance, 'low_link_impulse_response', 
            atl02_dataset, path_out).do_verify(custom_func)
    for atl02_dataset in d_cal20.keys():
        tolerance, custom_func, vtype = d_cal20[atl02_dataset]
        VerifyCalibrations(vfiles, tolerance, 'low_link_impulse_response', 
            atl02_dataset, path_out).do_verify(custom_func)

    # CAL-30: nominal_rx_sensitivity
    d_cal30, d_pce_cal30 = ready_cal30(radiometry)
    for atl02_dataset in d_pce_cal30.keys():
        tolerance, custom_func, vtype = d_pce_cal30[atl02_dataset]
        VerifyCalibrationsPCE(vfiles, tolerance, 'nominal_rx_sensitivity', 
            atl02_dataset, path_out).do_verify(custom_func)
    for atl02_dataset in d_cal30.keys():
        tolerance, custom_func, vtype = d_cal30[atl02_dataset]
        VerifyCalibrations(vfiles, tolerance, 'nominal_rx_sensitivity', 
            atl02_dataset, path_out).do_verify(custom_func)

    # CAL-34: dead_time_radiometric_signal_loss
    d_cal34 = ready_cal34(vfiles)
    for atl02_dataset in d_cal34.keys():
        tolerance, custom_func, vtype = d_cal34[atl02_dataset]
        VerifyCalibrations(vfiles, tolerance, 'dead_time_radiometric_signal_loss', 
            atl02_dataset, path_out).do_verify(custom_func)

    # CAL-42: dead_time
    d_cal42, d_pce_cal42 = ready_cal42(vfiles, radiometry)
    for atl02_dataset in d_pce_cal42.keys():
        tolerance, custom_func, vtype = d_pce_cal42[atl02_dataset]
        VerifyCalibrationsPCE(vfiles, tolerance, 'dead_time', 
            atl02_dataset, path_out).do_verify(custom_func)
    for atl02_dataset in d_cal42.keys():
        tolerance, custom_func, vtype = d_cal42[atl02_dataset]
        VerifyCalibrations(vfiles, tolerance, 'dead_time', 
            atl02_dataset, path_out).do_verify(custom_func)

    # CAL-44: start_timing_skews
    d_cal44 = ready_cal44(tof)
    for atl02_dataset in d_cal44.keys():
        tolerance, custom_func, vtype = d_cal44[atl02_dataset]
        VerifyCalibrations(vfiles, tolerance, 'start_timing_skews', 
            atl02_dataset, path_out).do_verify(custom_func)

    # CAL-45: laser_energy_fraction
    d_cal45 = ready_cal45(radiometry)
    for atl02_dataset in d_cal45.keys():
        tolerance, custom_func, vtype = d_cal45[atl02_dataset]
        VerifyCalibrations(vfiles, tolerance, 'laser_energy_fraction', atl02_dataset, 
            path_out).do_verify(custom_func)

    # CAL-46: hv_bias_receiver_radiometric_sensitivity
    d_cal46, d_pce_cal46 = ready_cal46(radiometry)
    for atl02_dataset in d_pce_cal46.keys():
        tolerance, custom_func, vtype = d_pce_cal46[atl02_dataset]
        VerifyCalibrationsPCE(vfiles, tolerance, 'hv_bias_receiver_radiometric_sensitivity', 
           atl02_dataset, path_out).do_verify(custom_func)
    for atl02_dataset in d_cal46.keys():
        tolerance, custom_func, vtype = d_cal46[atl02_dataset]
        VerifyCalibrations(vfiles, tolerance, 'hv_bias_receiver_radiometric_sensitivity', 
           atl02_dataset, path_out).do_verify(custom_func)

    # CAL-47: rx_sensitivity_to_misalignment_img
    d_cal47, d_pce_cal47 = ready_cal47(radiometry)
    for atl02_dataset in d_pce_cal47.keys():
        tolerance, custom_func, vtype = d_pce_cal47[atl02_dataset]
        VerifyCalibrationsPCE(vfiles, tolerance, 'rx_sensitivity_to_misalignment', 
            atl02_dataset, path_out).do_verify(custom_func)
    for atl02_dataset in d_cal47.keys():
        tolerance, custom_func, vtype = d_cal47[atl02_dataset]
        VerifyCalibrations(vfiles, tolerance, 'rx_sensitivity_to_misalignment', 
            atl02_dataset, path_out).do_verify(custom_func)

    # CAL-49: receiver_channel_skews
    d_cal49 = ready_cal49(tof)
    for atl02_dataset in d_cal49.keys():
        tolerance, custom_func, vtype = d_cal49[atl02_dataset]
        VerifyCalibrationsPCE(vfiles, tolerance, 'receiver_channel_skews', 
            atl02_dataset, path_out).do_verify(custom_func)

    # CAL-54: laser_energy_conversion
    d_cal54 = ready_cal54(radiometry)
    for atl02_dataset in d_cal54.keys():
        tolerance, custom_func, vtype = d_cal54[atl02_dataset]
        VerifyCalibrations(vfiles, tolerance, 'laser_energy_conversion', atl02_dataset, 
           path_out).do_verify(custom_func)

    # CAL-61: rx_sensitivity_vs_wtom
    d_cal61, d_pce_cal61 = ready_cal61(radiometry)
    for atl02_dataset in d_pce_cal61.keys():
        tolerance, custom_func, vtype = d_pce_cal61[atl02_dataset]
        VerifyCalibrationsPCE(vfiles, tolerance, 'rx_sensitivity_vs_wtom', 
            atl02_dataset, path_out).do_verify(custom_func)
    for atl02_dataset in d_cal61.keys():
        tolerance, custom_func, vtype = d_cal61[atl02_dataset]
        VerifyCalibrations(vfiles, tolerance, 'rx_sensitivity_vs_wtom', 
            atl02_dataset, path_out).do_verify(custom_func)

def ready_cal17(vfiles):

    cal17_path = os.path.join('/'.join(vfiles.atl02.split('/')[:-1]), 'CAL17')
    cal17 = ReadCal17(cal17_path)

    def build_rise_fall(pce, toggle):
        """
        Loop through all files for given PCE, looping through each channel-toggle
        column and cell row. This channel-toggle column becomes a column in the 
        output array, and each cell its own new matrix. So one matrix per cell.
        Each file is its own row each matrix. 
        At end hava matrix 5x20x75 (5 files x 20 channels x 75 cells)
        """
        pce_files = cal17.index_df['File_Name'][cal17.index_df['PCE'] == pce].values
        print("pce_files: ", pce_files)
        cell_matrix = []
        for cell in np.arange(75):
            pce_matrix = []
            for pce_file in pce_files:
                # Read in file to DataFrame.
                pce_df = pd.read_csv(os.path.join(cal17_path, pce_file), skiprows=18)
                channel_array = []
                for channel in np.arange(1,21):
                    channel_value = pce_df['Chan_{}_{}'.format(channel, toggle)][pce_df['Cell']==cell].values
                    channel_array.append(channel_value[0])
                pce_matrix.append(np.array(channel_array))
            cell_matrix.append(np.array(pce_matrix).T)

        return np.array(cell_matrix).T

    def build_ll_ot(pce, component):
        """
        Loop through all files for given PCE, reading either 'LL' or 'OT' column.
        This column becomes a row in the output array.
        Therefore one row for each file.
        """
        component_matrix = []

        ot_dict = {1:'LU', 2:'TU', 3:'TL'}

        pce_files = cal17.index_df['File_Name'][cal17.index_df['PCE'] == pce].values
        print("pce_files: ", pce_files)
        for pce_file in pce_files:
            # Read in file to DataFrame.
            pce_df = pd.read_csv(os.path.join(cal17_path, pce_file), skiprows=18)
            if component == 'LL':
                component_array = np.asarray(pce_df[component.upper()].values[0:75])
            else:
                component_array = np.asarray(pce_df[ot_dict[pce]].values[0:75])
            component_matrix.append(component_array)

        print("component_matrix: ", component_matrix)
        return component_matrix
        
    def build_cal_product():
        raw_cal_products = glob.glob(os.path.join(cal17_path, 'CAL_17_XXX*csv'))
        cal_products = [product.split('/')[-1] for product in raw_cal_products]
        return cal_products

    d = {'cal17_product':  [0, (lambda : build_cal_product()), 'passthrough']}

    d_pce = {'cal_fall': [1e-5, (lambda pce : np.array(cal17.index_df['Cal_Falling'][cal17.index_df['PCE']==pce])), 'passthrough'], # Limit of ASAS precision
             'cal_rise': [1e-5, (lambda pce : np.array(cal17.index_df['Cal_Rising'][cal17.index_df['PCE']==pce])), 'passthrough'],  # Limit of ASAS precision
             'efc_fall': [1e-5, (lambda pce : build_rise_fall(pce, toggle='F')), 'passthrough'],
             'efc_ll': [1e-5, (lambda pce : build_ll_ot(pce, 'LL')), 'passthrough'],
             'efc_ot': [1e-5, (lambda pce : build_ll_ot(pce, 'OT')), 'passthrough'],
             'efc_rise': [1e-5, (lambda pce : build_rise_fall(pce, toggle='R')), 'passthrough'],
             'temperature': [0, (lambda pce : np.array(cal17.index_df['Temperature'][cal17.index_df['PCE']==pce])), 'passthrough']
            }

    return d, d_pce

def ready_cal19(vfiles):
    
    cal19_path = os.path.join('/'.join(vfiles.atl02.split('/')[:-1]), 'CAL19')
    cal19 = ReadCal19(cal19_path)

    def build_rad_corr():
        width_matrix = []
        df0 = pd.read_csv(os.path.join(cal19_path, cal19.file_names[0]), skiprows=18)
        for width in list(df0)[2:]:
            print("**width: ", width)
            file_matrix = []
            for file in cal19.file_names:
                print("*file: ", file)
                # Read in file to DataFrame.
                df = pd.read_csv(os.path.join(cal19_path, file), skiprows=18)
                # Add a master key column, to make indexing later easier.
                df['master_key'] = np.arange(len(df))
                radcorr_array = []
                for master_key in df['master_key'].values:
                    radcorr_value = df[width][df['master_key']==master_key].values
                    print("master_key, radcorr: ", master_key, radcorr_value)
                    radcorr_array.append(radcorr_value[0])
                file_matrix.append(np.array(radcorr_array))

            width_matrix.append(np.array(file_matrix).T)
        print('width_matrix: ', width_matrix)
        return np.array(width_matrix).T

    def build_strength(component):
        component_matrix = []

        for file in cal19.file_names:
            # Read in file to DataFrame.
            df = pd.read_csv(os.path.join(cal19_path, file), skiprows=18)
            component_array = np.asarray(df['Apparent Strength ({})'.format(component)].values, dtype=float)
            component_matrix.append(component_array)
        return component_matrix

    def build_width():
        width_matrix = []
        for file in cal19.file_names:
            df = pd.read_csv(os.path.join(cal19_path, file), skiprows=18)
            widths = np.asarray(list(df)[2:], dtype=float)
            width_matrix.append(widths)
        return width_matrix

    d = {'cal19_product': [0, (lambda : cal19.file_names), 'passthrough'], 
         'dead_time': [1e-7, (lambda : cal19.dead_times), 'passthrough'],
         #'ffb_corr': [0, (lambda : build_rad_corr()), 'passthrough'],
         'strength_strong': [0, (lambda : build_strength('strong')), 'passthrough'],
         'strength_weak': [0, (lambda : build_strength('weak')), 'passthrough'],
         'width': [0, (lambda : build_width()), 'passthrough']}

    return d

def ready_cal20(vfiles):

    side_dict = {'A':1, 'B':2}
    source_dict = {'NONE':0, 'TEP':1, 'MAAT':2, 'ECHO':3}

    atl02 = h5py.File(vfiles.atl02, 'r')
    anc13_path = '/'.join(vfiles.atl02.split('/')[:-1])
    anc13 = ReadANC13(anc13_path)
    cal20_path = os.path.join('/'.join(vfiles.atl02.split('/')[:-1]), 'CAL20')
    cal20 = ReadCal20(cal20_path=cal20_path, atl02=atl02, laser=anc13.laser, mode=anc13.mode, side=anc13.side_pcd)

    def build_hist_x(start, stop, step):
        return step * np.arange(start / step, stop / step)

    d = {'bin_width': [1e-12, (lambda : cal20.bin_width), 'passthrough'],
         'cal20_product': [0, (lambda : cal20.cal_file.split('/')[-1]), 'passthrough'],
         #'hist_x': [0, (lambda : build_hist_x(0, cal20.num_bins*cal20.bin_width, cal20.bin_width)), 'calculation'],
         'laser': [1e-12, (lambda : cal20.laser), 'passthrough'],
         'mode': [1e-12, (lambda : cal20.mode), 'passthrough'],
         'num_bins': [1e-12, (lambda : cal20.num_bins), 'passthrough'],
         'return_source': [1e-12, (lambda : source_dict[cal20.return_source.upper()]), 'passthrough'],
         'side': [1e-12, (lambda : side_dict[cal20.side.upper()]), 'passthrough'],
         'temperature': [1e-6, (lambda : cal20.nearest_temp), 'passthrough'] 
        }
    d_pce = {#'hist': [1e-12, (lambda pce : cal20.select_values(pce)), 'passthrough'],
             'total_events': [1e10, (lambda pce : cal20.cal_df['TotalEvents'][cal20.cal_df['PCE']==pce].values), 'passthrough']
            }

    return d, d_pce

def ready_cal30(radiometry):

    side_dict = {'A':1, 'B':2}

    pce_dict = {1: {'s':radiometry.cal30.cal_df[radiometry.cal30.cal_df['Spot'] == 1],
                    'w':radiometry.cal30.cal_df[radiometry.cal30.cal_df['Spot'] == 2]},
                2: {'s':radiometry.cal30.cal_df[radiometry.cal30.cal_df['Spot'] == 3],
                    'w':radiometry.cal30.cal_df[radiometry.cal30.cal_df['Spot'] == 4]},
                3: {'s':radiometry.cal30.cal_df[radiometry.cal30.cal_df['Spot'] == 5],
                    'w':radiometry.cal30.cal_df[radiometry.cal30.cal_df['Spot'] == 6]}
                }

    d = {'cal30_product':  [0, (lambda : radiometry.cal30.cal_file.split('/')[-1]), 'passthrough'],
         'side': [0, (lambda : side_dict[radiometry.side.upper()]), 'passthrough'],
         'temperature': [1e-7, (lambda : radiometry.cal30.nearest_temp), 'passthrough']
        }

    d_pce = {
         'rms_resid_strong': [1e-11, (lambda pce : float(pce_dict[pce]['s']['RMS_Residual_Frac'])), 'passthrough'],
         'rms_resid_weak': [1e-11, (lambda pce : float(pce_dict[pce]['w']['RMS_Residual_Frac'])), 'passthrough'],
         'sdev_strong': [1e-5, (lambda pce : float(pce_dict[pce]['s']['StdDev(slope)'])), 'passthrough'],
         'sdev_weak': [1e-5, (lambda pce : float(pce_dict[pce]['w']['StdDev(slope)'])), 'passthrough'],
         'slope_strong': [1e-6, (lambda pce : float(pce_dict[pce]['s']['Slope'])), 'passthrough'],
         'slope_weak': [1e-6, (lambda pce : float(pce_dict[pce]['w']['Slope'])), 'passthrough']
        }

    return d, d_pce

def ready_cal34(vfiles):
    
    cal34_path = os.path.join('/'.join(vfiles.atl02.split('/')[:-1]), 'CAL34')
    cal34 = ReadCal34(cal34_path)

    def build_rad_corr():
        width_matrix = []
        df0 = pd.read_csv(os.path.join(cal34_path, cal34.file_names[0]), skiprows=18)
        for width in list(df0)[2:]:
            print("**width: ", width)
            file_matrix = []
            for file in cal34.file_names:
                print("*file: ", file)
                # Read in file to DataFrame.
                df = pd.read_csv(os.path.join(cal34_path, file), skiprows=18)
                # Add a master key column, to make indexing later easier.
                df['master_key'] = np.arange(len(df))
                radcorr_array = []
                for master_key in df['master_key'].values:
                    radcorr_value = df[width][df['master_key']==master_key].values
                    print("master_key, radcorr: ", master_key, radcorr_value)
                    radcorr_array.append(radcorr_value[0])
                file_matrix.append(np.array(radcorr_array))

            width_matrix.append(np.array(file_matrix).T)

        print('width_matrix: ', width_matrix)
        return np.array(width_matrix).T


    def build_strength(component):
        component_matrix = []

        for file in cal34.file_names:
            # Read in file to DataFrame.
            df = pd.read_csv(os.path.join(cal34_path, file), skiprows=18)
            component_array = np.asarray(df['Apparent Strength ({})'.format(component)].values, dtype=float)
            component_matrix.append(component_array)
        return component_matrix

    def build_width():
        width_matrix = []
        for file in cal34.file_names:
            df = pd.read_csv(os.path.join(cal34_path, file), skiprows=18)
            widths = np.asarray(list(df)[2:], dtype=float)
            width_matrix.append(widths)
        return width_matrix

    d = {'cal34_product': [0, (lambda : cal34.file_names), 'passthrough'], 
         'dead_time': [1e-7, (lambda : cal34.dead_times), 'passthrough'], # Limit of ASAS precision
         #'rad_corr': [0, (lambda : build_rad_corr()), 'passthrough'],
         'strength_strong': [0, (lambda : build_strength('strong')), 'passthrough'],
         'strength_weak': [0, (lambda : build_strength('weak')), 'passthrough'],
         'width': [0, (lambda : build_width()), 'passthrough']}

    return d

def ready_cal42(vfiles, radiometry):

    side_dict = {'A':1, 'B':2}

    # Since not part of either TOF or Radiometry, read CAL-42 here.
    atl02 = h5py.File(vfiles.atl02, 'r')
    cal42_path = os.path.join('/'.join(vfiles.atl02.split('/')[:-1]), 'CAL42')
    print("cal42_path: ", cal42_path)
    side = radiometry.side.upper()

    cal42 = ReadCal42(cal42_path, atl02, side)

    d = {'cal42_product': [0, (lambda : cal42.cal_file.split('/')[-1]), 'passthrough'],
         'side': [0, (lambda : side_dict[cal42.side.upper()]), 'passthrough'],
         'temperature': [1e-6, (lambda : cal42.nearest_temp), 'passthrough'] # where is this temperature specified?
        }

    d_pce = {'dead_time': [1e-10, (lambda pce : cal42.select_values(pce)[0]), 'passthrough'], # Limit of ASAS precision
             'sigma': [1e-12, (lambda pce : cal42.select_values(pce)[1]), 'passthrough'] # Limit of ASAS precision
            }

    return d, d_pce

def ready_cal44(tof):

    side_dict = {'A':1, 'B':2}

    d = {'cal44_product': [0, (lambda : tof.cal44.cal_file.split('/')[-1]), 'passthrough'],
         'll1': [1e-9, (lambda : tof.cal44.ll1), 'passthrough'],
         'll2_ll1': [1e-9, (lambda : tof.cal44.cal_df['LL2-LL1'][0]), 'passthrough'],
         'll3_ll1': [1e-9, (lambda : tof.cal44.cal_df['LL3-LL1'][0]), 'passthrough'],
         'lu_ll1': [1e-9, (lambda : tof.cal44.cal_df['LU-LL1'][0]), 'passthrough'],
         'side': [0, (lambda : side_dict[tof.cal44.side[0].upper()]), 'passthrough'],
         'spd_temp': [1e-6, (lambda : tof.cal44.nearest_temp), 'passthrough'],
         'tl_ll3': [1e-9, (lambda : tof.cal44.cal_df['TU-LL2'][0]), 'passthrough'],
         'tu_ll2': [1e-9, (lambda : tof.cal44.cal_df['TL-LL3'][0]), 'passthrough']
        }

    return d

def ready_cal45(radiometry):

    d = {'cal45_product': [0, (lambda : radiometry.cal45.cal_file.split('/')[-1]), 'passthrough'],
         'energy_fract': [1e-7, (lambda : np.array(radiometry.cal45.cal_df.iloc[[0]])[0][1:]), 'passthrough'],
         'optics_throughput': [1e-7, (lambda : [radiometry.cal45.e3, radiometry.cal45.e6, radiometry.cal45.e11]), 'passthrough']
        }

    return d

def ready_cal46(radiometry):
    """
    """
    side_dict = {'A':1, 'B':2}

    pce_dict = {1: {'s':radiometry.cal46.cal_df[radiometry.cal46.cal_df['Spot'] == 1],
                    'w':radiometry.cal46.cal_df[radiometry.cal46.cal_df['Spot'] == 2]},
                2: {'s':radiometry.cal46.cal_df[radiometry.cal46.cal_df['Spot'] == 3],
                    'w':radiometry.cal46.cal_df[radiometry.cal46.cal_df['Spot'] == 4]},
                3: {'s':radiometry.cal46.cal_df[radiometry.cal46.cal_df['Spot'] == 5],
                    'w':radiometry.cal46.cal_df[radiometry.cal46.cal_df['Spot'] == 6]}
                }

    d = {'cal46_product':  [0, (lambda : radiometry.cal46.cal_file.split('/')[-1]), 'passthrough'],
         'side': [0, (lambda : side_dict[radiometry.side.upper()]), 'passthrough']
        }

    d_pce = {'b_strong' : [1e-10, (lambda pce : float(pce_dict[pce]['s']['b (1/V)'])), 'passthrough'],
         'b_weak' : [1e-10, (lambda pce : float(pce_dict[pce]['w']['b (1/V)'])), 'passthrough'],
         'c_strong' : [1e-10, (lambda pce : float(pce_dict[pce]['s']['c (1/V^2)'])), 'passthrough'],
         'c_weak' : [1e-10, (lambda pce : float(pce_dict[pce]['w']['c (1/V^2)'])), 'passthrough'],         
         'npoints_strong' : [0, (lambda pce : float(pce_dict[pce]['s']['N Points'])), 'passthrough'],
         'npoints_weak' : [0, (lambda pce : float(pce_dict[pce]['w']['N Points'])), 'passthrough'],
         'rnom_strong' : [1e-5, (lambda pce : float(pce_dict[pce]['s']['Rnom(unitless)'])), 'passthrough'],
         'rnom_weak' : [1e-5, (lambda pce : float(pce_dict[pce]['w']['Rnom(unitless)'])), 'passthrough'],
         'sigma_b_strong' : [1e-9, (lambda pce : float(pce_dict[pce]['s']['sigma(b) (1/V)'])), 'passthrough'],
         'sigma_b_weak' : [1e-9, (lambda pce : float(pce_dict[pce]['w']['sigma(b) (1/V)'])), 'passthrough'],
         'sigma_c_strong' : [1e-9, (lambda pce : float(pce_dict[pce]['s']['sigma(c) (1/V^2)'])), 'passthrough'],
         'sigma_c_weak' : [1e-9, (lambda pce : float(pce_dict[pce]['w']['sigma(c) (1/V^2)'])), 'passthrough'],
         'sigma_fit_strong' : [1e-9, (lambda pce : float(pce_dict[pce]['s']['sigma(fit) (unitless)'])), 'passthrough'],
         'sigma_fit_weak' : [1e-9, (lambda pce : float(pce_dict[pce]['w']['sigma(fit) (unitless)'])), 'passthrough'],
         'vnom_strong' : [0, (lambda pce : float(pce_dict[pce]['s']['Vnom (V)'])), 'passthrough'],
         'vnom_weak' : [0, (lambda pce : float(pce_dict[pce]['w']['Vnom (V)'])), 'passthrough']
        }

    return d, d_pce

def ready_cal47(radiometry):

    print("entered cal47")

    pce_dict = {1: {'s':radiometry.cal47.create_spot_matrix(1)[0], 
                    'w':radiometry.cal47.create_spot_matrix(2)[0]},
                2: {'s':radiometry.cal47.create_spot_matrix(3)[0], 
                    'w':radiometry.cal47.create_spot_matrix(4)[0]},
                3: {'s':radiometry.cal47.create_spot_matrix(5)[0], 
                    'w':radiometry.cal47.create_spot_matrix(6)[0]}
                }

    d = {'azimuth': [0, (lambda : np.arange(radiometry.cal47.azimuth_grid_range[0], 
            radiometry.cal47.azimuth_grid_range[1], radiometry.cal47.grid_spacing)), 'calculation'],
         'azimuth_grid_range': [0, (lambda : radiometry.cal47.azimuth_grid_range), 'passthrough'],
         'cal47_product': [0, (lambda : radiometry.cal47.cal_files), 'passthrough'],
         'elevation': [0, (lambda : np.arange(radiometry.cal47.elevation_grid_range[0], 
            radiometry.cal47.elevation_grid_range[1], radiometry.cal47.grid_spacing)), 'calculation'],
         'elevation_grid_range': [0, (lambda : radiometry.cal47.elevation_grid_range), 'passthrough'],
         'grid_spacing': [0, (lambda : radiometry.cal47.grid_spacing), 'passthrough'],
         'temperature': [1e-5, (lambda : radiometry.cal47.temperature), 'passthrough'],
         }

    print("reached d_pce")

    d_pce = {#'rel_intensity_strong': [1e-11, (lambda pce : pce_dict[pce]['s']), 'passthrough'],
             #'rel_intensity_weak': [1e-11, (lambda pce : pce_dict[pce]['w']), 'passthrough']
             }

    return d, d_pce

def ready_cal49(tof):

    side_dict = {'A':1, 'B':2}

    d = {'cal49_product': [0, (lambda pce : tof.cal49.cal_file.split('/')[-1]), 'passthrough'],
         'side': [0, (lambda pce : side_dict[tof.side.upper()]), 'passthrough'],
         'skew_fall': [1e-12, (lambda pce : tof.cal49.cal_df['Skew'][tof.cal49.cal_df['Rising/Falling'] == 'F'][tof.cal49.cal_df['PCE'] == pce].values), 'passthrough'],
         'skew_fall_stderr': [1e-12, (lambda pce : tof.cal49.cal_df['Skew_Stderr'][tof.cal49.cal_df['Rising/Falling'] == 'F'][tof.cal49.cal_df['PCE'] == pce].values), 'passthrough'],
         'skew_rise': [1e-12, (lambda pce : tof.cal49.cal_df['Skew'][tof.cal49.cal_df['Rising/Falling'] == 'R'][tof.cal49.cal_df['PCE'] == pce].values), 'passthrough'],
         'skew_rise_stderr': [1e-12, (lambda pce : tof.cal49.cal_df['Skew_Stderr'][tof.cal49.cal_df['Rising/Falling'] == 'R'][tof.cal49.cal_df['PCE'] == pce].values), 'passthrough'],
         'temperature': [0, (lambda pce : tof.cal49.nearest_temp), 'passthrough']
        }

    return d

def ready_cal54(radiometry):
    
    d = {'base_temp': [1e-7, (lambda : float(radiometry.cal54.T0)), 'passthrough'],
         'cal54_product': [0, (lambda : radiometry.cal54.cal_file.split('/')[-1]), 'passthrough'],
         'internal': [1e-12, (lambda : np.array(list(radiometry.cal54.select_coeffs('Laser_{}'.format(radiometry.laser), laser=radiometry.laser).values()))), 'passthrough'],
         'lrs': [1e-12, (lambda : np.array(list(radiometry.cal54.select_coeffs('LRS', laser=radiometry.laser).values()))), 'passthrough'],
         'spd': [1e-12, (lambda : np.array(list(radiometry.cal54.select_coeffs('SPD_{}'.format(radiometry.side), laser=radiometry.laser).values()))), 'passthrough']
         }

    return d

def ready_cal61(radiometry):

    side_dict = {'A':1, 'B':2}

    pce_dict = {1: {'s':radiometry.cal61.cal_df[radiometry.cal61.cal_df['Spot'] == 1],
                    'w':radiometry.cal61.cal_df[radiometry.cal61.cal_df['Spot'] == 2]},
                2: {'s':radiometry.cal61.cal_df[radiometry.cal61.cal_df['Spot'] == 3],
                    'w':radiometry.cal61.cal_df[radiometry.cal61.cal_df['Spot'] == 4]},
                3: {'s':radiometry.cal61.cal_df[radiometry.cal61.cal_df['Spot'] == 5],
                    'w':radiometry.cal61.cal_df[radiometry.cal61.cal_df['Spot'] == 6]}
                }

    d = {'cal61_product': [0, (lambda : radiometry.cal61.cal_file.split('/')[-1]), 'passthrough'],
         'laser': [0, (lambda : radiometry.laser), 'passthrough'],  # radiometry.cal.laser
         'mode': [0, (lambda : radiometry.mode), 'passthrough'],
         'side': [0, (lambda : side_dict[radiometry.cal61.side[1].upper()]), 'passthrough'],
         'temperature': [1e-6, (lambda : radiometry.cal61.nearest_temp), 'passthrough']
        }

    d_pce = {'h_strong': [1e-7, (lambda pce : float(pce_dict[pce]['s']['h'])), 'passthrough'],
             'h_weak': [1e-7, (lambda pce : float(pce_dict[pce]['w']['h'])), 'passthrough'],
             'rms_of_fit_strong': [1e-7, (lambda pce : float(pce_dict[pce]['s']['RMS of fit'])), 'passthrough'],
             'rms_of_fit_weak': [1e-7, (lambda pce : float(pce_dict[pce]['w']['RMS of fit'])), 'passthrough'],
             'sdev_h_strong': [1e-7, (lambda pce : float(pce_dict[pce]['s']['StdDev(h)'])), 'passthrough'],
             'sdev_h_weak': [1e-7, (lambda pce : float(pce_dict[pce]['w']['StdDev(h)'])), 'passthrough'],
             'sdev_xpeak_strong': [1e-7, (lambda pce : float(pce_dict[pce]['s']['StdDev(xpeak)'])), 'passthrough'],
             'sdev_xpeak_weak': [1e-7, (lambda pce : float(pce_dict[pce]['w']['StdDev(xpeak)'])), 'passthrough'],
             'sdev_ypeak_strong': [1e-5, (lambda pce : float(pce_dict[pce]['s']['StdDev(ypeak) (counts/s)'])), 'passthrough'],
             'sdev_ypeak_weak': [1e-5, (lambda pce : float(pce_dict[pce]['w']['StdDev(ypeak) (counts/s)'])), 'passthrough'],
             'xpeak_strong': [1e-7, (lambda pce : float(pce_dict[pce]['s']['xpeak'])), 'passthrough'],
             'xpeak_weak': [1e-7, (lambda pce : float(pce_dict[pce]['w']['xpeak'])), 'passthrough'],
             'ypeak_strong': [1e-6, (lambda pce : float(pce_dict[pce]['s']['ypeak (counts/s)'])), 'passthrough'],
             'ypeak_weak': [1e-6, (lambda pce : float(pce_dict[pce]['w']['ypeak (counts/s)'])), 'passthrough']
            }

    return d, d_pce


def verify_housekeeping(vfiles, path_out):
    """ Function for verifying datasets in ancillary_data/housekeeping/

    Notes
    -----
    JLee 29 May 2019 on pdu_ab_flag: It likely is redundant since I don't 
    think the hvpc (power supply) a/b can be cross-strapped with the PDU. 
    I think the reason  for this is that something was sensitive to the 
    PDU side and the power supply status was the only way of telling which 
    PDU was active.
    """
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)

    def check_status(arr_a):
        # ATL01: 0=a, 1=b
        # ATL02: 1=a, 2=b
        if arr_a[0] == 0:
            # If array for side A is all 0s, then side A must be on.
            # Therefore return '1' for side A.
            return[1]
        elif arr_a[0] == 1:
            # If array for side A is all 0s, then side B must be on.  
            # Therefore return '2' for side B.
            return[2]        

    d = {'det_ab_flag' :    [0, (lambda : check_status(atl01['atlas/a_hkt_status_1065/pdua_det_ps'].value)), 'passthrough'],
         'hvpc_ab_flag' :   [0, (lambda : check_status(atl01['atlas/a_hkt_status_1065/pdua_hvpc'].value)), 'passthrough'],
         'laser_12_flag' :  [0, (lambda : atl01['atlas/a_sla_hk_1032/raw_laser_select_cfg'].value), 'passthrough'],
         'lrs_ab_flag' :    [0, (lambda : check_status(atl01['atlas/a_hkt_status_1065/pdua_lrs'].value)), 'passthrough'],
         'pdu_ab_flag' :    [0, (lambda : check_status(atl01['atlas/a_hkt_status_1065/pdua_hvpc'].value)), 'passthrough'], # This is redundant with hvpc_ab_flag. See JLee's comment
         'spd_ab_flag' :    [0, (lambda : check_status(atl01['atlas/a_hkt_status_1065/pdua_spd_ps'].value)), 'passthrough'], 
         'tams_ab_flag' :   [0, (lambda : check_status(atl01['atlas/a_hkt_status_1065/pdua_tams_ls'].value)), 'passthrough']
        }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyAncillaryData(vfiles, tolerance, 'housekeeping/{}'.format(atl02_dataset), path_out).do_verify(custom_func)
    atl01.close()

def verify_isf(vfiles, path_out):
    """ Function for verifying datasets in ancillary_data/isf/

    Notes:
    ------
    bias_offset_x, bias_offset_y, and bias_time will need be adjusted post-launch
    with actual before and after values, instead of repeating same value twice.
    """
    radiometry = pickle_out(vfiles.radiometry)
    tof = pickle_out(vfiles.tof)


    def build_k_matrix():
        """ Removes the ns decimal places from the first column,
        to return it to its ANC27 state.
        """
        k_data = tof.ks.k_data
        new_k_matrix = []
        for k in k_data:
            row = []
            for i in range(len(k)):
                if i==0:
                    row.append(k[i]*10**9)
                else:
                    row.append(k[i])
            new_k_matrix.append(np.asarray(row))
        return np.asarray(new_k_matrix)

    d = {'bias_offset_x' :      [0, (lambda : radiometry.anc27.select_value('bias_offset_x', straddle=True)), 'passthrough'],
         'bias_offset_y' :      [0, (lambda : radiometry.anc27.select_value('bias_offset_y', straddle=True)), 'passthrough'],
         'bias_rate' :          [0, (lambda : [radiometry.anc27.select_value('bias_rate')]), 'passthrough'],
         'bias_time' :          [1e-6, (lambda : radiometry.anc27.select_value('bias_time', straddle=True)), 'conversion'],
         'cal46_aging' :        [0, (lambda : [radiometry.anc27.select_value('cal46_aging')]), 'passthrough'],
         'start_time_coeff' :   [1e-12, (lambda : build_k_matrix()), 'passthrough'],
         'uso_freq_dev' :       [0, (lambda : [radiometry.anc27.select_value('uso_freq')]), 'passthrough'],
         'wtom_alt_tune_corr' : [0, (lambda : [radiometry.anc27.select_value('wtom_alt_tune_corr')]), 'passthrough'], 
         'wtom_lambda_off' :    [0, (lambda : [radiometry.anc27.select_value('wtom_lambda_off')]), 'passthrough'],
         'wtom_tune_flag' :     [0, (lambda : [radiometry.anc27.select_value('wtom_tuning_flag')]), 'passthrough']
        }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyAncillaryData(vfiles, tolerance, 'isf/{}'.format(atl02_dataset), path_out).do_verify(custom_func)

def verify_tep(vfiles, path_out):
    """ Function for verifying datasets in /ancillary_data/tep/
    """
    tep = pickle_out(vfiles.tep)

    def check_for_tep(pce):
        """ If tep data available, return 1.
        """
        if len(tep.map_pce(pce).filtered.TOF_TEP) != 0:
            return [1]
        else:
            return [0]

    d = {'tep_check_pce1':  [0,  (lambda : check_for_tep(pce=1)), 'calculation'],
         'tep_check_pce2':  [0,  (lambda : check_for_tep(pce=2)), 'calculation'],
         'tep_check_pce3':  [0,  (lambda : [0]), 'calculation'], # ideally this field should be removed
         'thres_tep_max':   [0,  (lambda : [1e-7]), 'passthrough'],
         'thres_tep_min':   [0,  (lambda : [1e-9]), 'passthrough']
        }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyAncillaryData(vfiles, tolerance, 'tep/{}'.format(atl02_dataset), path_out).do_verify(custom_func)

def verify_tod_tof(vfiles, path_out):
    """ Function for verifying datasets in /ancillary_data/tod_tof/
    """
    tod = pickle_out(vfiles.tod)
    tof = pickle_out(vfiles.tof)
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)

    d = {'cal_risefall_box_int':    [0,  (lambda : [6000]), 'constant'],
         'cal_uso_scale':           [0,  (lambda : [tof.SF_USO]), 'constant'],
         'corr_rx_coarse_pce1':     [0,  (lambda : np.ones(20)*tof.map_pce(pce=1).rx_cal_ccoffset), 'constant'],
         'corr_rx_coarse_pce2':     [0,  (lambda : np.ones(20)*tof.map_pce(pce=2).rx_cal_ccoffset), 'constant'],
         'corr_rx_coarse_pce3':     [0,  (lambda : np.ones(20)*tof.map_pce(pce=3).rx_cal_ccoffset), 'constant'],
         'corr_tx_coarse_pce1':     [0,  (lambda : [tof.map_pce(pce=1).tx_cal_ccoffset]), 'constant'],
         'corr_tx_coarse_pce2':     [0,  (lambda : [tof.map_pce(pce=2).tx_cal_ccoffset]), 'constant'],
         'corr_tx_coarse_pce3':     [0,  (lambda : [tof.map_pce(pce=3).tx_cal_ccoffset]), 'constant'],
         'dt_imet':                 [0,  (lambda : [tof.dt_imet]), 'constant'],
         'dt_t0':                   [0,  (lambda : [tof.dt_t0]), 'passthrough'],
         'dt_uso':                  [0,  (lambda : [tof.d_USO]), 'constant'],
         'lrs_clock':               [0,  (lambda : [tof.lrs_clock]), 'constant']
        }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyAncillaryData(vfiles, tolerance, 'tod_tof/{}'.format(atl02_dataset), path_out).do_verify(custom_func)
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
        default=['ancillary_data', 'calibrations', 'housekeeping', 'isf', 'tep', 'tod_tof'],
        action='store', type=str, nargs='+', required=False, 
        help="Select groups in pcex to verify ['ancillary_data', 'calibrations', 'housekeeping', 'isf', 'tep', 'tod_tof']. By default all.")
    
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
   
    if 'ancillary_data' in args.fields:
        verify_ancillary_data(vfiles=vfiles, path_out=path_out)
    if 'calibrations' in args.fields:
        verify_calibrations(vfiles=vfiles, path_out=path_out)        
    if 'housekeeping' in args.fields:
        verify_housekeeping(vfiles=vfiles, path_out=path_out)
    if 'isf' in args.fields:
        verify_isf(vfiles=vfiles, path_out=path_out)
    if 'tep' in args.fields:
        verify_tep(vfiles=vfiles, path_out=path_out)
    if 'tod_tof' in args.fields:
        verify_tod_tof(vfiles=vfiles, path_out=path_out)
