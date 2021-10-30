""" Module for orbit_info/ verification classes.

Author:

    C.M. Gosmeyer

"""

import h5py
import numpy as np
import pylab as plt
from atl02qa.verification.verification_tools import Verify
from atl02qa.shared.constants import pces


class VerifyQualityAssessment(Verify):
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        atl02_dataset = 'quality_assessment/{}'.format(atl02_dataset)
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
        #self.diff_arrs(atl02_values, verifier_values)
        self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        try:
            self.__verify_single(custom_func)
        except Exception as error:
            self.record_exception(error)

class VerifyAlongTrack(Verify):
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)
        self.atl02_dataset = atl02_dataset

    def __verify_single(self, custom_func):
        """
        """
        # Read the values from ATL02.
        atl02_values = self.atl02['quality_assessment/along_track/{}'.format(self.atl02_dataset)].value
        print("atl02_values: ", atl02_values)

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func()
        print("verifier_values: ", verifier_values)

        # Diff the arrays, plot diff, and record statistics.
        #self.diff_arrs(atl02_values, verifier_values)
        self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        try:
            self.__verify_single(custom_func)
        except Exception as error:
            self.record_exception(error)

class VerifyRecordCounts(Verify):
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)
        self.atl02_dataset = atl02_dataset

    def __verify_single(self, custom_func):
        """
        """
        # Read the values from ATL02.
        atl02_values = self.atl02['quality_assessment/record_counts/{}'.format(self.atl02_dataset)].value
        print("atl02_values: ", atl02_values)

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func()
        print("verifier_values: ", verifier_values)

        # Diff the arrays, plot diff, and record statistics.
        #self.diff_arrs(atl02_values, verifier_values)
        self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        try:
            self.__verify_single(custom_func)
        except Exception as error:
            self.record_exception(error)

class VerifySummary(Verify):
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out=''):
        Verify.__init__(self, vfiles, tolerance, atl02_dataset, path_out)
        self.atl02_dataset = atl02_dataset

    def __verify_single(self, custom_func):
        """
        """
        # Read the values from ATL02.
        atl02_values = self.atl02['quality_assessment/summary/{}'.format(self.atl02_dataset)].value
        print("atl02_values: ", atl02_values)

        # Read the dataset to be compared against ATL02.
        verifier_values = custom_func()
        print("verifier_values: ", verifier_values)

        # Diff the arrays, plot diff, and record statistics.
        #self.diff_arrs(atl02_values, verifier_values)
        self.compare_arrays(atl02_values, verifier_values)

    def do_verify(self, custom_func):
        try:
            self.__verify_single(custom_func)
        except Exception as error:
            self.record_exception(error)

