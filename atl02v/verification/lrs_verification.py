
""" Module for lrs/ verification class.

Author:

    C.M. Gosmeyer
"""

import h5py
import numpy as np
import pylab as plt
from atl02qa.conversion.convert import Converter
from atl02qa.verification.verification_tools import Verify

class VerifyLRS(Verify):
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        atl02_dataset = 'lrs/{}'.format(atl02_dataset)
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)

    def __verify_single(self, custom_func):
        """
        """
        # Read the values from ATL02.
        atl02_values = self.atl02[self.atl02_dataset].value
        print("atl02_values: ", atl02_values)

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func()
        print("verifier_values: ", verifier_values)

        # Diff the arrays, plot diff, and record statistics.
        self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        try:
            self.__verify_single(custom_func)
        except Exception as error:
            self.record_exception(error)

class VerifyImageWindowNN(Verify):
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)

    def __verify_single(self, custom_func, cat, nn):
        """
        """
        # Create the base out name.
        self.base_filename = 'lrs.{}_image.window{}.{}'.format(cat, nn, self.atl02_dataset)

        # Read the values from ATL02.
        atl02_values = self.atl02['lrs/{}_image/window{}/{}'.format(cat, nn, self.atl02_dataset)].value
        print("atl02_values: ", atl02_values)

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func(nn)
        print("verifier_values: ", verifier_values)

        # Diff the arrays, plot diff, and record statistics.
        self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func, cat):
        """
        Notes
        -----
        JLee on 29 May 2019:
        These are 'fixed' in the telemetry. 
        Stellar and Laser are not consistent (and not supposed to be)
        """
        if cat == 'laser':
            nns = ['00', '01', '02', '03', '10']
        if cat == 'stellar':
            nns = ['01', '02', '03', '04', '05']

        for nn in nns:
            try:
                self.__verify_single(custom_func, cat, nn)
            except Exception as error:
                self.record_exception(error)

def convert_lrs_temp(atl01_label_t, atl01_value_t):
    """ Converts raw counts to LRS temperature. 
    This equation is NOT in ITOS. It originates from 
    ICESat-2-LRS-IFACE-1794, pg 172.

    Parameters
    ----------
    atl01_label_t : str
        The ATL01 label of the temperature field to be converted.
    atl01_value_t : array
        The ATL01 values of the temperature field to be converted.
    """
    # First convert to voltage.
    # Use a 4-V mnemonic to instigate the voltage conversion.
    temp_converted_dict, _ = Converter([atl01_label_t], [atl01_value_t], 
            mnemonic='A_LRS_HK_GROUND1_V', verbose=False).convert_all()
    v = np.array(temp_converted_dict[atl01_label_t])

    # Then convert to resistance.
    r = (4000.0 - (11000.0*v))/(v - 4.0)

    # Convert to temperature in K.
    t_K = 1.0/(0.00102839 + 0.00023926*np.log(r) + 0.000000156159*np.log(r)**3)

    # Finally convert to temperature in degrees C.
    t_C = t_K - 273.15

    return list(t_C)

def match_lrs_fields_02to01(atl01, atl02, remove=['delta_time', 'srate_x', 'srate_y', 'srate_z', 'chkstat_s_ld_df']):
    """ Matches fields in given ATL02 group to fields in the ATL01.

    Parameters
    ----------
    atl01 : h5py._hl.files.File
        Open ATL01 H5 file.
    atl02 : h5py._hl.files.File
        Open ATL02 H5 file.
    remove : list of str
        The datasets to be removed from the ATL02 field.

    Returns
    -------
    atl02_to_atl01_dict : dict
        Keys of ATL02 field, values of ATL01 field.
    """

    atl01_lrs = list(atl01['lrs/hk_1120'].keys())
    print('atl01_lrs: ', atl01_lrs)

    atl02_lrs = list(atl02['lrs/hk_1120'].keys())
    print('atl02_lrs: ', atl02_lrs)
    # Remove given fields from list
    for item in remove:
        atl02_lrs.remove(item)

    def match_02to01(atl02_item):

        for atl01_item in atl01_lrs:
            # First reduce the ATL01 item name to look like an ATL02 name
            atl01_compare = atl01_item.replace('raw_', '')

            # Then compare the two names.
            if atl01_compare == atl02_item:
                return 'lrs/hk_1120/{}'.format(atl01_item)

        print("WARNING: No ATL01 match found for ATL02 item {}".format(atl02_item))

        return None

    atl02_to_atl01_dict = {}

    # For each item in atl02 housekeeping, find its ATL01 source
    for atl02_item in atl02_lrs:
        atl01_item = match_02to01(atl02_item)
        atl02_to_atl01_dict[atl02_item] = atl01_item

    return atl02_to_atl01_dict
