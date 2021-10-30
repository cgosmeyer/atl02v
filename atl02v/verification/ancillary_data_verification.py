
""" Module for ancillary_data/ verification classes.

Author:

    C.M. Gosmeyer
    
"""

import h5py
import numpy as np
import pylab as plt
from atl02qa.shared.constants import pces
from atl02qa.shared.tools import str2datetime
from atl02qa.verification.verification_tools import Verify, smart_decode


class VerifyAncillaryData(Verify):
    """
    For groups in ancillary_data/.
    """
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        atl02_dataset = 'ancillary_data/{}'.format(atl02_dataset)
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)

    def __verify_single(self, custom_func):
        """
        """
        # Read the values from ATL02.
        atl02_values = self.atl02[self.atl02_dataset].value

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func()

        print('atl02_values: ', atl02_values, type(atl02_values))
        print('verifier_values: ', verifier_values, type(verifier_values))

        # Special case for comparing dates that should be within +/- 1 second
        if 'data_start_utc' in self.atl02_dataset or 'data_end_utc' in self.atl02_dataset:
            atl02_values = str2datetime(smart_decode(atl02_values[0]).strip())
            verifier_values = str2datetime(smart_decode(verifier_values[0]).strip())
            print('atl02_values: ', atl02_values, type(atl02_values))
            print('verifier_values: ', verifier_values, type(verifier_values))   
            self.compare_arrays(np.array([atl02_values]), np.array([verifier_values]))

        # Diff the arrays, plot diff, and record statistics.
        # Unless the arrays contain strings! Then just compare them.
        elif isinstance(atl02_values[0], bytes) or isinstance(verifier_values[0], bytes):
            print("**option1")
            self.compare_strings(atl02_values, verifier_values)
        elif type(verifier_values[0]) != str and (type(atl02_values[0]) != list and type(atl02_values[0]) is not np.ndarray):
            print("**option2")
            self.compare_arrays(atl02_values, verifier_values)
        else:
            if (type(atl02_values[0]) == list or type(atl02_values[0]) is np.ndarray):
                print("**option3a")
                self.compare_array_of_arrays(atl02_values, verifier_values)
            else:
                print("**option3b")
                self.compare_arrays(atl02_values, verifier_values)
                self.compare_strings(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        try:
            self.__verify_single(custom_func)
        except Exception as error:
            self.record_exception(error)

class VerifyCalibrationsPCE(Verify):
    """
    For groups in ancillary_data/calibrations/ that have pce subgroups.

    """
    def __init__(self, vfiles, tolerance, calgroup, atl02_dataset, path_out=''):
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)
        self.calgroup = calgroup

    def __verify_single(self, pce, custom_func):
        """ Verify the 'pce' group parameters.
        """
        self.base_filename = 'ancillary_data.calibrations.{}.pce{}.{}'.format(self.calgroup, pce, self.atl02_dataset)

        # Read the values from ATL02.
        atl02_values = self.atl02['calibrations/{}/pce{}/{}'\
            .format(self.calgroup, pce, self.atl02_dataset)].value

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func(pce)
        if len(atl02_values) == 1 and ((type(verifier_values) is not np.ndarray) and \
            type(verifier_values) != list):
            verifier_values = [verifier_values]

        print('atl02_values: ', atl02_values, type(atl02_values))
        print('verifier_values: ', verifier_values, type(verifier_values))

        if (type(verifier_values[0]) == list or type(verifier_values[0]) is np.ndarray) and \
            (type(atl02_values[0]) == list or type(atl02_values[0]) is np.ndarray) and \
            (len(verifier_values) == 1 and len(atl02_values) == 1):
            print("**option1")
            self.compare_arrays(atl02_values[0], verifier_values[0])
        elif type(verifier_values[0]) != str and ((type(verifier_values[0]) is not np.ndarray) or \
            len(verifier_values) > 1):
            print("**option2")
            self.compare_arrays(atl02_values, verifier_values)
        else:
            print("**option3")
            self.compare_strings(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        for pce in pces:
            try:
                self.__verify_single(pce, custom_func)
            except Exception as error:
                self.record_exception(error)

class VerifyCalibrations(Verify):
    """
    For groups in ancillary_data/calibrations.
    """
    def __init__(self, vfiles, tolerance, calgroup, atl02_dataset, path_out=''):
        atl02_dataset = 'calibrations/{}/{}'.format(calgroup, atl02_dataset)
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)
        self.calgroup = calgroup

    def __verify_single(self, custom_func):
        """
        """
        # Read the values from ATL02.
        atl02_values = self.atl02[self.atl02_dataset].value

        print('atl02_values: ', atl02_values, type(atl02_values))

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func()
        print('verifier_values: ', verifier_values, type(verifier_values))

        if len(atl02_values) == 1 and type(verifier_values) != str:
            print("**option1")
            verifier_values = [verifier_values]
            self.compare_arrays(atl02_values, verifier_values)

        elif type(verifier_values[0]) != str and (type(verifier_values[0]) != np.ndarray and type(verifier_values[0]) != list):
            print("**option2")            
            self.compare_arrays(atl02_values, verifier_values)

        elif type(verifier_values[0]) != str and (type(verifier_values[0]) == np.ndarray or type(verifier_values[0]) == list):
            print("**option3")
            if type(verifier_values[0][0]) != str:          
                self.compare_array_of_arrays(atl02_values, verifier_values)
            else:
                self.compare_strings(np.asarray(atl02_values).flatten(), np.asarray(verifier_values).flatten())

        else:
            print("**option4")
            if type(verifier_values) != list and type(verifier_values) != np.ndarray:
                verifier_values = [verifier_values]
            self.compare_strings(atl02_values, verifier_values)


    def do_verify(self, custom_func): 
        try:
            self.__verify_single(custom_func)
        except Exception as error:
            self.record_exception(error)
