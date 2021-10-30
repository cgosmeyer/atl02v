#!/usr/bin/env python

""" Verifying datasets in housekeeping

Author:

    C.M. Gosmeyer

Notes:

    The 'delta_time' fields are verified separately in `verify_delta_time.py`.
"""

import argparse
import h5py
import numpy as np
import os
import pandas as pd
from atl02v.conversion.convert import Converter
from atl02v.shared.paths import path_to_outputs
from atl02v.shared.tools import make_file_dir, make_timestamp_dir, pickle_out
from atl02v.verification.atlas_housekeeping_verification import VerifyHousekeeping, \
    match_fields_02to01, return_values
from atl02v.verification.verification_tools import VFileReader


def verify_radiometry(vfiles, path_out):
    radiometry = pickle_out(vfiles.radiometry)

    d = {'bg_sensitivity_pce1_s' : [1e15, (lambda : radiometry.spot1.S_BG), 'calculation'],
         'bg_sensitivity_pce1_w' : [1e15, (lambda : radiometry.spot2.S_BG), 'calculation'],
         'bg_sensitivity_pce2_s' : [1e15, (lambda : radiometry.spot3.S_BG), 'calculation'],
         'bg_sensitivity_pce2_w' : [1e15, (lambda : radiometry.spot4.S_BG), 'calculation'],
         'bg_sensitivity_pce3_s' : [1e15, (lambda : radiometry.spot5.S_BG), 'calculation'],
         'bg_sensitivity_pce3_w' : [1e15, (lambda : radiometry.spot6.S_BG), 'calculation'],
         'ret_sensitivity_pce1_s' : [1e15, (lambda : radiometry.spot1.S_RET), 'calculation'],
         'ret_sensitivity_pce1_w' : [1e15, (lambda : radiometry.spot2.S_RET), 'calculation'],
         'ret_sensitivity_pce2_s' : [1e15, (lambda : radiometry.spot3.S_RET), 'calculation'],
         'ret_sensitivity_pce2_w' : [1e15, (lambda : radiometry.spot4.S_RET), 'calculation'],
         'ret_sensitivity_pce3_s' : [1e15, (lambda : radiometry.spot5.S_RET), 'calculation'],
         'ret_sensitivity_pce3_w' : [1e15, (lambda : radiometry.spot6.S_RET), 'calculation']    
          }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyHousekeeping(vfiles, tolerance, 'radiometry/{}'.format(atl02_dataset), path_out).do_verify(custom_func)

def verify_laser_energy_internal(vfiles, path_out):
    radiometry = pickle_out(vfiles.radiometry)

    d = {'e_tx' : [1e-3, (lambda : radiometry.map_sensor('laser_internal').E_total), 'calculation'],
         'e_tx_pce1_s' : [1e-3, (lambda : radiometry.map_sensor('laser_internal').E_fract.spot1), 'calculation'],
         'e_tx_pce1_w' : [1e-3, (lambda : radiometry.map_sensor('laser_internal').E_fract.spot2), 'calculation'],
         'e_tx_pce2_s' : [1e-3, (lambda : radiometry.map_sensor('laser_internal').E_fract.spot3), 'calculation'],
         'e_tx_pce2_w' : [1e-3, (lambda : radiometry.map_sensor('laser_internal').E_fract.spot4), 'calculation'],        
         'e_tx_pce3_s' : [1e-3, (lambda : radiometry.map_sensor('laser_internal').E_fract.spot5), 'calculation'],
         'e_tx_pce3_w' : [1e-3, (lambda : radiometry.map_sensor('laser_internal').E_fract.spot6), 'calculation'],
         'laser_mode' : [0, (lambda : radiometry.mode*np.ones(len(radiometry.map_sensor('laser_internal').E_total))), 'passthrough'],
         'laser_temp' : [2e-3, (lambda : radiometry.map_sensor('laser_internal').T), 'conversion']
        }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyHousekeeping(vfiles, tolerance, 'laser_energy_internal/{}'.format(atl02_dataset), path_out).do_verify(custom_func)


def verify_laser_energy_lrs(vfiles, path_out):
    radiometry = pickle_out(vfiles.radiometry)

    d = {'e_tx' : [1e-3, (lambda : radiometry.map_sensor('lrs').E_total), 'calculation'],
         'e_tx_pce1_s' : [1e-3, (lambda : radiometry.map_sensor('lrs').E_fract.spot1), 'calculation'],
         'e_tx_pce1_w' : [1e-3, (lambda : radiometry.map_sensor('lrs').E_fract.spot2), 'calculation'],
         'e_tx_pce2_s' : [1e-3, (lambda : radiometry.map_sensor('lrs').E_fract.spot3), 'calculation'],
         'e_tx_pce2_w' : [1e-3, (lambda : radiometry.map_sensor('lrs').E_fract.spot4), 'calculation'],
         'e_tx_pce3_s' : [1e-3, (lambda : radiometry.map_sensor('lrs').E_fract.spot5), 'calculation'],
         'e_tx_pce3_w' : [1e-3, (lambda : radiometry.map_sensor('lrs').E_fract.spot6), 'calculation'],
         'lrs_temp' : [6e-3, (lambda : radiometry.map_sensor('lrs').T), 'conversion']
        }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyHousekeeping(vfiles, tolerance, 'laser_energy_lrs/{}'.format(atl02_dataset), path_out).do_verify(custom_func)


def verify_laser_energy_spd(vfiles, path_out):
    """

    Notes
    -----
        Need to transpose the SPD energy arrays so can properly diff with ATL02.
        (so '.T' in the energy arrays is NOT temperature)
    """
    radiometry = pickle_out(vfiles.radiometry)


    d = {'ds_10' : [0, (lambda : np.arange(1,11)), 'passthrough'],
         'e_tx' : [1e-3, (lambda : radiometry.map_sensor('spd').E_total.T), 'calculation'],
         'e_tx_pce1_s' : [1e-3, (lambda : radiometry.map_sensor('spd').E_fract.spot1.T), 'calculation'],
         'e_tx_pce1_w' : [1e-3, (lambda : radiometry.map_sensor('spd').E_fract.spot2.T), 'calculation'],
         'e_tx_pce2_s' : [1e-3, (lambda : radiometry.map_sensor('spd').E_fract.spot3.T), 'calculation'],
         'e_tx_pce2_w' : [1e-3, (lambda : radiometry.map_sensor('spd').E_fract.spot4.T), 'calculation'],        
         'e_tx_pce3_s' : [1e-3, (lambda : radiometry.map_sensor('spd').E_fract.spot5.T), 'calculation'],
         'e_tx_pce3_w' : [1e-3, (lambda : radiometry.map_sensor('spd').E_fract.spot6.T), 'calculation'],
         'edge_xmtnc' : [1e-3, (lambda : radiometry.atl01_store.raw_edge_xmtnc.converted), 'conversion'],
         'laser_temp' : [1e-3, (lambda : radiometry.map_sensor('spd').T), 'conversion'],
         'peak_xmtnc' : [1e-3, (lambda : radiometry.atl01_store.raw_peak_xmtnc.converted), 'conversion'],
         'thrhi_rdbk' : [3e-6, (lambda : radiometry.thrhi_rdbk), 'conversion'],
         'thrlo_rdbk' : [3e-6, (lambda : radiometry.thrlo_rdbk), 'conversion']       
        }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyHousekeeping(vfiles, tolerance, 'laser_energy_spd/{}'.format(atl02_dataset), path_out).do_verify(custom_func)


def verify_mce_position(vfiles, path_out):
    """

    Notes
    -----
    JLee on mce_az and mce_el: Code was converted from IDL script: 
    converttomceengineeringunits.pro (provided via email 2/Jul/2019)
    """
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)

    def get_bsm_temp():
        """
        """
        bsm_temp1Counts = np.array(atl01['a_mce_hk_1056/bsm1_t'].value, dtype=np.float64)
        bsm_temp2Counts = np.array(atl01['a_mce_hk_1056/bsm2_t'].value, dtype=np.float64)

        bsm1K = np.array([-7.3242E+01, 1.3780E-02, -1.0905E-06, 4.7518E-11, -1.0846E-15, 1.2340E-20, -5.5054E-26], dtype=np.float64)
        bsm2K = np.array([-7.3638E+01, 1.4021E-02, -1.1196E-06, 4.9137E-11, -1.1290E-15, 1.2925E-20, -5.8013E-26], dtype=np.float64)

        # Convert BSM to Volts and take average
        bsm_temp1Volts = bsm1K[0] + bsm1K[1]*bsm_temp1Counts + bsm1K[2]*bsm_temp1Counts**2 + \
            bsm1K[3]*bsm_temp1Counts**3 + bsm1K[4]*bsm_temp1Counts**4 + \
            bsm1K[5]*bsm_temp1Counts**5 + bsm1K[6]*bsm_temp1Counts**6

        bsm_temp2Volts = bsm2K[0] + bsm2K[1]*bsm_temp2Counts + bsm2K[2]*bsm_temp2Counts**2 + \
            bsm2K[3]*bsm_temp2Counts**3 + bsm2K[4]*bsm_temp2Counts**4 + \
            bsm2K[5]*bsm_temp2Counts**5 + bsm2K[6]*bsm_temp2Counts**6

        bsm_temp = np.mean(bsm_temp1Volts + bsm_temp2Volts)*0.5 - 22.0

        print("bsm_temp: ", bsm_temp)

        return bsm_temp

    def convert_mce_az(bsm_temp):
        """
        """
        azCounts = np.array(atl01['a_mce_pos_1057/raw_az'].value, dtype=np.float64)

        # Polynomials
        azK = np.array([-8.3044, 2.5348e-04], dtype=np.float64)

        # Sensor cal values
        ktAz = 0.00040468867198012
        bAz = 0.005984325320480
        mAz = 1.012576196358776

        # Convert counts to volts
        azVolts = azK[0] + (azCounts*azK[1])

        # Convert volts to mils
        azMils = ((ktAz*bsm_temp) + mAz)*azVolts + bAz

        # Convert mils to mechanical microradians
        azMech = np.arctan(azMils/1875.0)*1e6

        return azMech

    def convert_mce_el(bsm_temp):
        """
        """
        elCounts = np.array(atl01['a_mce_pos_1057/raw_el'].value, dtype=np.float64)

        # Polynomials
        elK = np.array([-8.3147, 2.5381E-04], dtype=np.float64)

        # Sensor cal values
        ktEl = 0.00044997551399642
        bEl = -0.006250885036312
        mEl = 1.011470070600611

        # Convert counts to volts
        elVolts = elK[0] + (elCounts*elK[1])

        # Convert volts to mils
        elMils = ((ktEl*bsm_temp) + mEl)*elVolts + bEl

        # Convert mils to mechanical microradians
        elMech = np.arctan(elMils/1875.0)*1e6

        return elMech

    def convert_mce():
        """
        """
        # BSM cal values
        CalAzA = 1.015604931900874
        CalAzE = -0.001577018120419
        CalAzOffset = -0.694169364349398
        CalElA = -0.000970812200351
        CalElE = 1.009235507623523
        CalElOffset = 2.045500789736499

        bsm_temp = get_bsm_temp()

        # Do all conversions and apply BSM calibration
        azMech = convert_mce_az(bsm_temp)
        elMech = convert_mce_el(bsm_temp)

        azCal = (CalAzA*azMech) + (CalAzE*elMech) + CalAzOffset
        elCal = (CalElA*azMech) + (CalElE*elMech) + CalElOffset

        # Finally convert mechnical angle to output beam angle
        azBeam = np.sqrt(2.0)*azCal
        elBeam = 2.0*elCal

        return azBeam, elBeam

    azBeam, elBeam = convert_mce()
    print("azBeam: ", azBeam)
    print("elBeam: ", elBeam)

    # 
    d = {'ds_50' :          [0, (lambda : np.arange(1,51)), 'passthrough'],
         'mce_az' :         [1e-3, (lambda : azBeam), 'calculation'], # 32 bits, xx.xxxx
         'mce_el' :         [1e-2, (lambda : elBeam), 'calculation'], # 32 bits, xxx.xxxx
         'mce_total_cycles' :   [0, (lambda : atl01['a_mce_pos_1057/raw_total_cycles'].value), 'passthrough']
        }

    # Call Verify
    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyHousekeeping(vfiles, tolerance, 'mce_position/{}'.format(atl02_dataset), path_out).do_verify(custom_func)
    atl01.close()


def verify_meb_pdu_thermal(vfiles, path_out, atl02_group):
    """
    """
    # Read ATL01 and ATL02 H5 files.
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)
    atl02 = h5py.File(vfiles.atl02, 'r', driver=None)

    # Dictionary of ATL02 label : ATL01 label
    atl02_to_atl01_dict = match_fields_02to01(atl01, atl02, atl02_group)
    print("atl02_to_atl01_dict: ", atl02_to_atl01_dict)

    # List only ATL01 labels.
    atl01_labels = atl02_to_atl01_dict.values()
    print("atl01_labels: ", atl01_labels)

    atl01_values = return_values(atl01, atl01_labels)
    print("atl01_values: ", atl01_values)

    # Dictionary of ATL01 label : converted parameters
    converted_dict, itos_dict = Converter(atl01_labels, atl01_values, verbose=True).convert_all()
    print("converted_dict: ", converted_dict)
    print("itos_dict: ", itos_dict)

    # Because lambda functions in a for loop will keep referencing same parameter,
    # need embed them in a second function.
    # See https://stackoverflow.com/questions/19837486/python-lambda-in-a-loop
    def makeFunc(atl02_label):
        return (lambda : np.array(converted_dict[atl02_to_atl01_dict[atl02_label]]))
    # Build verification dictionary.
    d = {}
    for atl02_label in atl02_to_atl01_dict.keys():
        if atl02_label != None and atl02_to_atl01_dict[atl02_label] != None:
            func = makeFunc(atl02_label)
            # The following tolerances for temperature, voltage, and current
            # are based on observing many runs of the verification software.
            # There appears to be fundamental limits to which I can match ASAS
            # due to differences in read-in and conversion expression precision,
            # and for temperatures in particular, due to our differences in 
            # interpolation.
            if '_t' in atl02_label:
                d[atl02_label] = [5e-6, func, 'conversion']
            elif '_v' in atl02_label:
                d[atl02_label] = [5e-7, func, 'conversion']
            elif '_i' in atl02_label:
                d[atl02_label] = [5e-7, func, 'conversion']
            elif 'hvpc_' in atl02_label:
                d[atl02_label] = [1e-4, func, 'conversion']
            else:
                d[atl02_label] = [1e-11, func, 'conversion']                

    print("d: ", d)

    # Call Verify
    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyHousekeeping(vfiles, tolerance, '{}/{}'.format(atl02_group, atl02_dataset), path_out).do_verify(custom_func)

    # Make report of all the expressions
    atl01s = []
    atl02s = []
    hkts = []
    channels = []
    exprs = []
    for atl02_label in atl02_to_atl01_dict.keys():
        if atl02_label != None and atl02_to_atl01_dict[atl02_label] != None:
            atl01_label = atl02_to_atl01_dict[atl02_label]
            atl01s.append(atl01_label)
            atl02s.append(atl02_label)
            hkts.append(itos_dict[atl01_label][0])
            channels.append(itos_dict[atl01_label][1])
            exprs.append(itos_dict[atl01_label][2])

    df = pd.DataFrame({
        'ALT01_Label':atl01s, 'ATL02_Label':atl02s, 
        'HKT':hkts, "Channel":channels, 'Expr':exprs})
    df.to_csv(os.path.join(path_out, '{}.csv'.format(atl02_group)))
    atl01.close()
    atl02.close()

def verify_pointing(vfiles, path_out):
    """
    """
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)

    d = {'q_sc_i2b_1' :         [1e-10, (lambda : atl01['a_sc_pon_1138/raw_q_sc_i2b'].value.T[0]*1e-9), 'conversion'],
         'q_sc_i2b_2' :         [1e-10, (lambda : atl01['a_sc_pon_1138/raw_q_sc_i2b'].value.T[1]*1e-9), 'conversion'],
         'q_sc_i2b_3' :         [1e-10, (lambda : atl01['a_sc_pon_1138/raw_q_sc_i2b'].value.T[2]*1e-9), 'conversion'],
         'q_sc_i2b_4' :         [1e-10, (lambda : atl01['a_sc_pon_1138/raw_q_sc_i2b'].value.T[3]*1e-9), 'conversion'],
         'sc_solution_sec' :    [0, (lambda : atl01['a_sc_pon_1138/raw_sc_solution_sec'].value), 'passthrough'],
         'sc_solution_subsec' : [0, (lambda : atl01['a_sc_pon_1138/raw_sc_solution_subsec'].value), 'passthrough'],
         'x_sc_body_rate' :     [1e-10, (lambda : atl01['a_sc_pon_1138/raw_x_sc_body_rate'].value*1e-9), 'conversion'],
         'y_sc_body_rate' :     [1e-10, (lambda : atl01['a_sc_pon_1138/raw_y_sc_body_rate'].value*1e-9), 'conversion'],
         'z_sc_body_rate' :     [1e-10, (lambda : atl01['a_sc_pon_1138/raw_z_sc_body_rate'].value*1e-9), 'conversion']
        }

    # Call Verify
    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyHousekeeping(vfiles, tolerance, 'pointing/{}'.format(atl02_dataset), path_out).do_verify(custom_func)

    atl01.close()

def verify_position_velocity(vfiles, path_out):
    """
    """
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)

    d = {'sc_solution_sec' :    [0, (lambda : atl01['a_sc_pos_1137/raw_sc_solution_sec'].value), 'passthrough'],
         'sc_solution_subsec' : [0, (lambda : atl01['a_sc_pos_1137/raw_sc_solution_subsec'].value), 'passthrough'],
         'x_sc_eci_pos' :       [1e-3, (lambda : atl01['a_sc_pos_1137/raw_x_sc_eci_pos'].value*0.01), 'conversion'], # to m
         'x_sc_eci_vel' :       [1e-5, (lambda : atl01['a_sc_pos_1137/raw_x_sc_eci_vel'].value*1e-4), 'conversion'], # to m/s
         'y_sc_eci_pos' :       [1e-3, (lambda : atl01['a_sc_pos_1137/raw_y_sc_eci_pos'].value*0.01), 'conversion'], # to m
         'y_sc_eci_vel' :       [1e-5, (lambda : atl01['a_sc_pos_1137/raw_y_sc_eci_vel'].value*1e-4), 'conversion'], # to m/s
         'z_sc_eci_pos' :       [1e-3, (lambda : atl01['a_sc_pos_1137/raw_z_sc_eci_pos'].value*0.01), 'conversion'], # to m
         'z_sc_eci_vel' :       [1e-5, (lambda : atl01['a_sc_pos_1137/raw_z_sc_eci_vel'].value*1e-4), 'conversion'] # to m/s
        }

    # Call Verify
    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyHousekeeping(vfiles, tolerance, 'position_velocity/{}'.format(atl02_dataset), path_out).do_verify(custom_func)

    atl01.close()

def verify_status(vfiles, path_out):
    """
    """
    # Read ATL01 and ATL02 H5 files.
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)
    atl02 = h5py.File(vfiles.atl02, 'r', driver=None)

    # Dictionary of ATL02 label : ATL01 label
    atl02_to_atl01_dict = match_fields_02to01(atl01, atl02, atl02_group='status')
    print("atl02_to_atl01_dict: ", atl02_to_atl01_dict)

    atl01_values = return_values(atl01, atl02_to_atl01_dict.values())
    print("atl01_values: ", atl01_values)

    # Because lambda functions in a for loop will keep referencing same parameter,
    # need embed them in a second function.
    # See https://stackoverflow.com/questions/19837486/python-lambda-in-a-loop
    def makeFunc(atl01_value):
        return (lambda : np.array(atl01_value))

    # Build verification dictionary.
    d = {}
    for atl02_label, atl01_value in zip(atl02_to_atl01_dict.keys(), atl01_values):
        func = makeFunc(atl01_value)
        d[atl02_label] = [0, func, 'passthrough']

    print("d: ", d)

    # Call Verify
    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyHousekeeping(vfiles, tolerance, 'status/{}'.format(atl02_dataset), path_out).do_verify(custom_func)
    atl01.close()
    atl02.close()

def verify_time_at_the_tone(vfiles, path_out):
    atl01 = h5py.File(vfiles.atl01, 'r', driver=None)

    d = {'gps_1pps_sec' :       [0, (lambda : atl01['atlas/a_sc_tat_1136/raw_gps_1pps_sec'].value), 'passthrough'],
         'gps_1pps_subsec' :    [0, (lambda : atl01['atlas/a_sc_tat_1136/raw_gps_1pps_subsec'].value), 'passthrough'],
         'sc_time_1pps_sec' :   [0, (lambda : atl01['atlas/a_sc_tat_1136/raw_sc_time_1pps_sec'].value), 'passthrough'],
         'sc_time_1pps_subsec' : [0, (lambda : atl01['atlas/a_sc_tat_1136/raw_sc_time_1pps_subsec'].value), 'passthrough']
        }

    for atl02_dataset in d.keys():
        tolerance, custom_func, vtype = d[atl02_dataset]
        VerifyHousekeeping(vfiles, tolerance, 'time_at_the_tone/{}'.format(atl02_dataset), path_out).do_verify(custom_func)
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
        default=['laser_energy_internal', 'laser_energy_lrs', 'laser_energy_spd',
            'meb', 'pdu','radiometry', 'pointing', 'position_velocity', 'status', 
            'thermal', 'time_at_the_tone'],
        action='store', type=str, nargs='+', required=False, 
        help="Select groups in pcex to verify ['laser_energy_internal', 'laser_energy_lrs', \
            'laser_energy_spd', 'mce_position', 'meb', 'pdu','radiometry', 'pointing', \
            'position_velocity', 'status', 'thermal', 'time_at_the_tone']. \
            By default all, EXCEPT 'mce_position'.")
    
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
   
    if 'laser_energy_internal' in args.fields:
        verify_laser_energy_internal(vfiles=vfiles, path_out=path_out)
    if 'laser_energy_lrs' in args.fields:
        verify_laser_energy_lrs(vfiles=vfiles, path_out=path_out)
    if 'laser_energy_spd' in args.fields:
        verify_laser_energy_spd(vfiles=vfiles, path_out=path_out)
    if 'mce_position' in args.fields:
        verify_mce_position(vfiles=vfiles, path_out=path_out)
    if 'meb' in args.fields:
        verify_meb_pdu_thermal(vfiles=vfiles, path_out=path_out, atl02_group='meb')
    if 'pdu' in args.fields:
        verify_meb_pdu_thermal(vfiles=vfiles, path_out=path_out, atl02_group='pdu')
    if 'pointing' in args.fields:
        verify_pointing(vfiles=vfiles, path_out=path_out)
    if 'position_velocity' in args.fields:
        verify_position_velocity(vfiles=vfiles, path_out=path_out)
    if 'radiometry' in args.fields:
        verify_radiometry(vfiles=vfiles, path_out=path_out)
    if 'status' in args.fields:
        verify_status(vfiles=vfiles, path_out=path_out)
    if 'thermal' in args.fields:
        verify_meb_pdu_thermal(vfiles=vfiles, path_out=path_out, atl02_group='thermal')
    if 'time_at_the_tone' in args.fields:
        verify_time_at_the_tone(vfiles=vfiles, path_out=path_out)

