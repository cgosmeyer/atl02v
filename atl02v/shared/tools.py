""" Tools shared across TOD, TOF, and TEP.

Author:

    C.M. Gosmeyer, Apr 2018
"""

import datetime
import h5py
import numpy as np
import os
import pandas as pd
import pickle
import time

from atl02v.shared.constants import pces


def sips2atl02(sips_sec):
    """ Converts SIPS UTC date (seconds since Jan 1 2000 12:00, J2000)
    to ATL02 UTC date (yyyy-mm-ddThh:mm:ss).
    """
    # First convert date from str to int.
    sips_sec = int(str(sips_sec).split('D')[0])

    # Create datetime object for the SIPS start date 
    sips_start = datetime.datetime(month=1, year=2000, day=1, hour=12, minute=0, second=0)

    # Add the SIPS date in seconds to the SIPS start date
    sips_date = sips_start + datetime.timedelta(seconds=int(sips_sec))

    # Return format expected by ATL02.
    atl02_date = sips_date.strftime('%Y-%m-%dT%H:%M:%S')

    return atl02_date


def diff20(n):
    """ Return the channel 1-20, without regard to side or PCE.
    In other words, reduces (I think) a 'super-channel' value to 
    a 'normal' per-PCE channel value.

    Parameters
    ----------
    n : int
        Channel number.
    """
    if n < 21:
        return n
    else:
        return(diff20(n-20))

def find_nearest_date(arr, search_value):
    """
    https://stackoverflow.com/questions/32237862/find-the-closest-date-to-a-given-date
    """
    return min(arr, key=lambda x: abs(x - search_value))

def find_nearest(array, search_value, flt=False):
    """ Find the value in 'array' that is nearest to 'search_value',

    Parameters
    ----------
    array : list or np.array
        The array to be searched over.
    search_value : float or int
        The value for which you wish to find the closest match.

    Returns
    -------
    nearest_value : float or int
        The value from 'array' nearest to 'search_value'.
    idx : int
        Index of 'nearest_value' in 'array'.
    flt : {True, False}
        Switch on in particular if need compare out to bit-64
        precision.
    """
    if not flt:
        array = np.array(array, dtype=float)
        search_value = float(search_value)
        idx = pd.Series((np.abs(array-search_value))).idxmin()
        nearest_value = array[idx]
        return nearest_value, int(idx)
    else:
        # Need puff the floats out to enough sig figs to 
        # do a meaningful comparision
        array = np.array(array)*1e3
        search_value = float(search_value)*1e3
        idx = pd.Series((np.abs(array-search_value))).idxmin()
        nearest_value = array[idx]
        return nearest_value/1e3 , int(idx) 

def find_nearest_bracketing(array, search_idx, bracket_value):
    """ Returns the indices of the entries nearest before and after the search
    index whose values match bracket_value

    Parameters
    ----------
    array : list or np.array
        The array to be searched over.
    search_idx : int
        The index around which to find nearest two indices
        whose value matches 'bracket_value'.
    bracket_value : int or float
        The value of which to search for the nearest before
        after the 'search_idx'.

    Returns
    -------
    idx_before : int
        Closest 'before' index to search_idx with value matching 'bracket_value'.
    idx_after : int
        Closest 'after' index to search_idx with value matching 'bracket_value'.

    Notes
    -----
    There is a bug. When the bracket group is very large, sometimes the two "closest"
    brackets aren't necessary the start and stop of the group the search_idx is in. For
    example, it could be finding the start of its current bracket and the next closest
    is the start of the previous bracket.
    """
    array = np.array(array, dtype=float)
    bracket_value = float(bracket_value)

    # First find indicies of all values that could possibly be a bracket.
    possible_brackets = np.where(np.abs(array-bracket_value)==0)[0]

    # Next find the index brackets that are nearest to the search index.
    sorted_brackets = sorted(np.abs(possible_brackets - search_idx))

    # The first two items in the sorted list should be the indices
    # of the nearest before and after brackets.
    # Because absolute values, need to check which is really the before
    # and which is really the after.
    indices = []
    if array[int(search_idx + sorted_brackets[0])] == bracket_value:
        indices.append(int(search_idx + sorted_brackets[0]))
    elif array[int(search_idx - sorted_brackets[0])] == bracket_value:
        indices.append(int(search_idx - sorted_brackets[0]))
    if array[int(search_idx + sorted_brackets[1])] == bracket_value:
        indices.append(int(search_idx + sorted_brackets[1]))
    elif array[int(search_idx - sorted_brackets[1])] == bracket_value:
        indices.append(int(search_idx - sorted_brackets[1]))

    # Account for corner case where at the very beginning of array where
    # no "before" bracket yet.
    print("indices: ", indices)
    if len(indices) > 1:
        idx_before = sorted(indices)[0]
        idx_after = sorted(indices)[1]
    else:
        idx_before = 0
        idx_after = indices[0]

    return idx_before, idx_after

def flatten_mf_arrays(mf_array_of_arrays):
    """ Flattens an array of major frame arrays to an array of
    just events.

    Parameters
    ----------
    mf_array_of_arrays : list or array
        An array containing arrays for each major frame.

    ** MAY BE DEPRICATED **
    """
    flat = [i for arr in mf_array_of_arrays for i in arr]
    return np.array(flat).flatten()
    #return np.array(mf_array_of_arrays).flatten()

def make_file_dir(dest, atl02_filename):
    """ Creates directory named for the ATL02 file, 
    YYYYMMDDHHMMSS_nnnnccnn_rrr

    nnn = RGT number
    cc = cycle number
    nn = span
    rrr = release number

    Ideally this function will create subfolders to 

        data/
        outputs/
            data/
            reports/
            vfiles/

    The time-stemped directories will come below this directory.

    Parameters
    ----------
    dest : string
        Path to where the directory should be created.

    Returns
    -------
    path_to_dir : string
        Path to and including the ATL02-derived directory.

    Outputs
    -------
    Directory at 'dest' with the ATL02-derived name.
    """
    basename = '_'.join(atl02_filename.split('/')[-1].split('_')[-4:-1])
    path_to_dir = os.path.join(dest, basename)

    # If one does not exist for today, create the time-stamp dir.
    if not os.path.isdir(path_to_dir):
        os.mkdir(path_to_dir)
        return path_to_dir
    else:
        return path_to_dir


def make_timestamp_dir(dest):
    """Creates time-stamped directory. YYYY-MM-DD
    If already exists, creates directory with an underscore integer, 
    1-50.

    Parameters
    ----------
    dest : string
        Path to where the time-stamp directory should be created.

    Returns
    -------
    path_to_time_dir : string
        Path to and including the time-stamped directory.

    Outputs
    -------
    Directory at 'dest' with a time-stamped name.
    """
    tt = time.localtime()
    year = str(tt.tm_year)
    month = str(tt.tm_mon)
    day = str(tt.tm_mday)
    hour = str(tt.tm_hour)
    minute = str(tt.tm_min)
    second = str(tt.tm_sec)

    if len(month) == 1:
        month = '0' + month
    if len(day) == 1:
        day = '0' + day        
        
    time_dir = '-'.join([year,month,day]) + '_' + '-'.join([hour,minute,second])
    path_to_time_dir = os.path.join(dest, time_dir)
    
    # If one does not exist for today, create the time-stamp dir.
    if not os.path.isdir(path_to_time_dir):
        os.mkdir(path_to_time_dir)
        return path_to_time_dir
    else:
        return path_to_time_dir

def map_channel2spot(channel):
    """ Maps channel 1-16 to strong ('s') and 17-20 to weak ('w').

    For use on ATL01's `raw_rx_channel_id` only.

    Parameters
    ----------
    channel : int
        Channel number of 1-20. If 0, returns 0.
    """
    # First reduce channel to an integer in range [1,20]
    channel = diff20(channel)
    # Map channel to strong or weak spot.
    if channel == 0:
        return 0
    elif channel in np.arange(1,17):
        return 's'
    elif channel in np.arange(17,21):
        return 'w'

def map_detector2channels(side, toggle):
    """ Table 11.
    """
    side = side.upper()
    toggle = toggle.lower()

    if side == 'A':
        if toggle == 'rising':
            channels = {1:np.arange(1,21),
                        2:np.arange(21,41),
                        3:np.arange(41,61)}
        elif toggle == 'falling':
            channels = {1:np.arange(61,81),
                        2:np.arange(81,101),
                        3:np.arange(101,121)}
    elif side == 'B':
        if toggle == 'rising':
            channels = {1:np.arange(121,141),
                        2:np.arange(141,161),
                        3:np.arange(161,181)}
        elif toggle == 'falling':
            channels = {1:np.arange(181,201),
                        2:np.arange(201,221),
                        3:np.arange(221,241)}
    return channels     

def map_mf2rx(mf_array, raw_pce_mframe_cnt_ph):
    """ Maps a major frame-space array to rx-space.

    Parameters
    ----------
    mf_array : array
        Array that is in major-frame space.
    raw_pce_mframe_cnt_ph : array
        Read from ATL01.
    """
    mf_array = np.array(mf_array)

    # So that major frames start count from 0.
    mframes_rx = raw_pce_mframe_cnt_ph - raw_pce_mframe_cnt_ph[0]

    # Unique set of major frame number.
    mframes_set = np.array(list(set(mframes_rx)), dtype=int)

    # Initialize the new rx-mapped array.
    rx_array = np.zeros(len(mframes_rx))

    # Loop over major frames.
    for mframe in mframes_set:
        # Obtain indicies of where the major frame exists in rx-space 
        mframe_indx = np.where(mframes_rx == mframe)[0]
        # Apply indicies to transform the major frame-indexed
        # array to rx-space.
        rx_array[mframe_indx] = mf_array[mframe]

    return rx_array

def nan2zero(x):
    """ If a NaN, change to 0.
    """
    if np.isnan(x):
        return 0
    else:
        return x

def pickle_in(obj, out_location=''):
    """ Pickle instance of a TOD, TOF, or TEP object.

    Downside is that if the class code for any of these is changed,
    will have trouble unpickling. So only plan to pickle imediate
    flow from TOD-TOF-TEP, but don't use for long-term storage.

    Use
    ---
        tod = TOD(args**)
        pickle_in(tod)

    Parameters
    ----------
    obj : object
        Instance of either TOD, TOF, or TEP object.
    out_location : string
        Out location of the pickle file.

    Returns
    -------
    pickle_file : str
        Name of the pickle file.

    References
    ----------
        https://wiki.python.org/moin/UsingPickle
    """
    timenow = datetime.datetime.now()
    timenow = timenow.strftime('%Y-%jT%H-%M-%S')
    pickle_file = os.path.join(out_location, '{}_{}.pkl'.format(obj.__name__, timenow))
    pickle.dump(obj, open(pickle_file, "wb"))
    print("pickled {}".format(pickle_file))

    return pickle_file

def pickle_out(pickle_file):
    """ Un-pickle instance of a TOD, TOF, or TEP object.

    Parameters
    ----------
    pickle_file : str
        Name of the pickle file.
    """
    return pickle.load(open(pickle_file, "rb"))

def str2datetime(datestr):
    """ Converts a date in format 2003-01-16T00:00:00 to seconds.
    """
    # Remove any decimal seconds, otherwise below will explode.
    datestr = str(datestr).split('.')[0]
    return datetime.datetime.strptime(datestr, '%Y-%m-%dT%H:%M:%S')
