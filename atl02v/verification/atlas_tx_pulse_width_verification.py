
""" Module for atlas/tx_pulse_width verification class.

Author:

    C.M. Gosmeyer
    
"""

import h5py
import numpy as np
import pylab as plt
from atl02qa.verification.verification_tools import Verify


class VerifyTxPulseWidth(Verify):
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        atl02_dataset = 'tx_pulse_width/{}'.format(atl02_dataset)
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)        

    def __verify_single(self, custom_func):
        """
        """
        # Read the values from ATL02.
        atl02_values = self.atl02[self.atl02_dataset].value

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func()

        # Diff the arrays, plot diff, and record statistics.
        self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        self.__verify_single(custom_func)