""" Module containing tools for verifying the housekeeping delta_times.

Author:

    C.M. Gosmeyer, Sep 2018

References:

    On recursive searching in H5: https://github.com/h5py/h5py/issues/406
"""

import h5py
import numpy as np
import os
import pandas as pd
from pydl import uniq


def seek_delta_times(atl02):
    """ Crawls through ATL02 file to find all 'delta_time' fields.
    """
    temp_file = 'temp_delta_time.txt'
    
    # Make sure won't be appending an already-existing temp file.
    if os.path.isfile(temp_file):
        os.remove(temp_file)

    atl02.visititems(print_attrs2file)

    # Read temp file into list.
    delta_time_ls = [line.strip() for line in open(temp_file, 'r')]
    print(delta_time_ls)

    os.remove(temp_file)

    return delta_time_ls


def print_attrs2file(name, obj):
    """
    """
    if 'delta_time' in name:
        print(name)
        # Hack! Since can't find a way to return a list from 
        # this recursive loop, just write all found delta_times
        # to a file to be read into a list later.
        with open('temp_delta_time.txt', 'a') as f:
            f.write(name + '\n')


class DeltaTime(object):
    __name__ = 'DeltaTime'
    def __init__(self, delta_time_name, atl02, verbose=True):
        """ DeltaTime object.

        Parameters
        ----------
        delta_time_name : str
            The H5 path name to the delta_time field.
        atl02 : 
            Open H5 file of ATL02.
        """
        self.verbose = verbose
        self.atl02 = atl02
        self.delta_time_name = delta_time_name
        self.delta_times = self.atl02[delta_time_name].value

        self.start = self.delta_times[0]
        self.end = self.delta_times[-1]

    def check_limits(self, limit_tolerance=0.5):
        """ Checks that limits of delta_time are within granule.
        
        For setting the bounds of the granule, use ATL02's

            quality_assessment/summary/delta_time_start
            quality_assessment/summary/delta_time_end
        """
        # Find bounds of the granule.
        self.min_start = self.atl02['quality_assessment/summary/delta_time_start'].value[0]
        self.max_end = self.atl02['quality_assessment/summary/delta_time_end'].value[0]

        if self.verbose:
            print(" ")
            print("field: ", self.delta_time_name)
            print("min_start: ", self.min_start)
            print("max_end: ", self.max_end)
            print("delta_time start: ", self.start)
            print("delta_time end: ", self.end)

        # Check that delta_time falls within granule
        if self.start >= self.min_start-limit_tolerance and self.end <= self.max_end+limit_tolerance:
            return True, True
        elif self.start >= self.min_start-limit_tolerance and self.end > self.max_end+limit_tolerance:
            return True, False
        elif self.start < self.min_start-limit_tolerance and self.end <= self.max_end+limit_tolerance:
            return False, True
        else:
            return False, False

    def check_rate(self, rate_check=1, rate_tolerance=0.5):
        """ Checks that data rate is within 1 Hz, ie 1 second
        (or as given).

        Parameters
        ----------
        rate_check : float
            The rate to check against, in units of Hz.
        rate_tolerance : float
            The tolerance within to check rate, in units of Hz.
        """
        # First unique the delta_times so not over-estimating the rate.
        uniq_delta_times = uniq(self.delta_times)

        # Calculate the rate of the data.
        self.rate = 1./abs((self.start - self.end) / len(uniq_delta_times))
        
        if self.verbose:
            print("rate: ", self.rate)

        # Check that rate is near 1 +/- 0.5 Hz (ie, 1 second)
        if self.rate >= rate_check-rate_tolerance and self.rate <= rate_check+rate_tolerance:
            return True
        else:
            return False
