
""" Module for atlas/housekeeping verification classes.

Author:

    C.M. Gosmeyer
    
"""

import h5py
import numpy as np
import pylab as plt
from atl02qa.verification.verification_tools import Verify
from atl02qa.shared.constants import pces


class VerifyHousekeeping(Verify):
    """
    For groups in Housekeeping/.
    """
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        atl02_dataset = 'housekeeping/{}'.format(atl02_dataset)
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)

    def __verify_single(self, custom_func):
        """
        """
        # Read the values from ATL02.
        atl02_values = self.atl02[self.atl02_dataset].value

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func()

        print("shape: ", verifier_values.shape)
        # Diff the arrays, plot diff, and record statistics.
        self.compare_arrays(atl02_values, verifier_values)        

    def do_verify(self, custom_func):
        try:
            self.__verify_single(custom_func)
        except Exception as error:
            self.record_exception(error)

def return_values(atl01, atl01_labels):
    """
    """
    atl01_values = []
    for label in atl01_labels:
        if label != None:
            atl01_value = np.array(atl01[label].value)
        else:
            print("WARNING: ATL01 label is None".format(label))
            atl01_value = None
        atl01_values.append(atl01_value)
    return atl01_values


def match_fields_02to01(atl01, atl02, atl02_group, remove=['delta_time']):
    """ Matches fields in given ATL02 group to fields in the ATL01.

    Parameters
    ----------
    atl01 : h5py._hl.files.File
        Open ATL01 H5 file.
    atl02 : h5py._hl.files.File
        Open ATL02 H5 file.
    atl02_group : str
        The name of the ATL02 group in atlas/housekeeping/ such as 'meb',
        'pdu', or 'thermal'.
    remove : list of str
        The datasets to be removed from the ATL02 field.

    Returns
    -------
    atl02_to_atl01_dict : dict
        Keys of ATL02 field, values of ATL01 field.
    """
    # Find side
    side = int(atl02['housekeeping/pdu_ab_flag'].value)
    if side == 1:
        side = 'a'
    else:
        side = 'b'

    atl01_atlas = list(atl01['atlas/'].keys())
    # Remove pce1-3
    for item in ['pce1', 'pce2', 'pce3']:
        atl01_atlas.remove(item)

    atl02_housekeeping = list(atl02['housekeeping/{}'.format(atl02_group)].keys())
    # Remove given fields from list
    for item in remove:
        atl02_housekeeping.remove(item)

    def match_02to01(atl02_item):

        for atl01_group in atl01_atlas:
            atl01_items = list(atl01['{}'.format(atl01_group)].keys())
            atl01_items.remove('delta_time')
            atl01_items.remove('ccsds')

            for atl01_item in atl01_items:
                # First reduce the ATL01 item name to look like an ATL02 name
                if atl02_group == 'status':
                    atl01_compare = atl01_item
                elif 'hvpc_mod_' in atl02_item:
                    side_dict = {'a':'pri', 'b':'red'}
                    atl01_compare = atl01_item.replace('raw_{}_'.format((side_dict[side])), '')
                elif 'pdu' in atl02_item and atl02_group == 'thermal':
                    atl01_compare = atl01_item.replace('_raw_', 'temp').replace('raw_', '').replace('temp', '_raw_')
                else:
                    atl01_compare = atl01_item.replace('raw_', '').replace('pdu{}'.format(side), 'pdu')

                # Then compare the two names.
                if atl01_compare == atl02_item:
                    return 'atlas/{}/{}'.format(atl01_group, atl01_item)

        print("WARNING: No ATL01 match found for ATL02 item {}".format(atl02_item))
        return None

    atl02_to_atl01_dict = {}

    # For each item in atl02 housekeeping, find its ATL01 source
    for atl02_item in atl02_housekeeping:
        atl01_item = match_02to01(atl02_item)
        atl02_to_atl01_dict[atl02_item] = atl01_item

    return atl02_to_atl01_dict

