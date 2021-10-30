""" Class to trace each photon by unique ID from ATL02 back to ATL01.

Author:

    C.M. Gosmeyer

"""

import datetime
import h5py
import numpy as np
import os
import pandas as pd

from atl02v.shared.constants import pces
from hdf2pandas import hdf2pandas as hdf

class PhotonID(object):
    __name__ = 'PhotonID'   
    def __init__(self, pce, atl01_file, atl02_file):
        """ Matches each unique ID of ATL02 back to ATL01, for
        strong, weak, TEP, and all events.
        """
        self.pce = pce
        self.end_row = 4000000
        self.atl01_file = atl01_file
        self.atl02_file = atl02_file

        self.df1 = self.read_atl01()

    def columns01(self):
        """ Returns ATL01 columns.
        """
        return {'raw_pce_mframe_cnt':'raw_pce_mframe_cnt', 
                'ph_id_pulse':'ph_id_pulse',  
                'ph_id_count':'ph_id_count', 
                'ph_id_channel':'ph_id_channel'}

    def columns02(self):
        """ Returns ATL02 columns.
        """
        return {'pce_mframe_cnt':'pce_mframe_cnt', 
                'ph_id_pulse':'ph_id_pulse',  
                'ph_id_count':'ph_id_count', 
                'ph_id_channel':'ph_id_channel'}

    def read_atl01(self):
        """  Reads the ATL01 raw_pce_mframe_cnt, ph_id_pulse, ph_id_channel,
        and ph_id_count arrays from given /atlas/pcex/a_alt_science_ph
        group and writes them into a DataFrame.
        """
        # Read desired ATL01 columns for given PCE into a dataframe.
        with hdf(self.atl01_file, mode='r') as hdf5:
            hdf5.epoch = hdf5['atlas_sdp_gps_epoch'][0]
            datagrp = 'pce{}/a_alt_science_ph/'.format(self.pce)

            atl01_df, _ = hdf5.dg2pandas(datagrp, datasets=self.columns01().keys(), columns=self.columns01(), end_row=2*self.end_row)  

        # Create an immutable master id
        atl01_df['master_id'] = np.arange(len(atl01_df))

        # Create a single index for this combination of columns.
        df1 = atl01_df.set_index(['raw_pce_mframe_cnt', 'ph_id_pulse', 'ph_id_channel', 'ph_id_count'])

        return df1

    def read_atl02(self, spot):
        """ Reads the ATL02 pce_mframe_cnt, ph_id_pulse, ph_id_channel, and
        ph_id_count arrays from given /atlas/pcex/altimetry/spot/photons 
        group and writes them into a DataFrame.

        Parameters
        ----------
        spot : str
            Either 'strong' or 'weak'.
        """

        # Read desired ATL02 columns for given PCE and spot (strong/weak) into a dataframe.
        with hdf(self.atl02_file, mode='r') as hdf5:
            hdf5.epoch = hdf5['atlas_sdp_gps_epoch'][0]
            datagrp = 'pce{}/altimetry/{}/photons/'.format(self.pce, spot)
            atl02_df, _ = hdf5.dg2pandas(datagrp, datasets=self.columns02().keys(), columns=self.columns02(), end_row=2*self.end_row)  

        # Create an immutable atl02 id
        atl02_df['atl02_id'] = np.arange(len(atl02_df))

        # Create a single index for this combination of columns.
        df2 = atl02_df.set_index(['pce_mframe_cnt', 'ph_id_pulse', 'ph_id_channel', 'ph_id_count'])

        return df2

    def read_tep(self):
        """ Reads the ATL02 pce_mframe_cnt, ph_id_pulse, ph_id_channel, and
        ph_id_count arrays from given /atlas/pcex/tep group and writes them
        into a DataFrame.
        """

        # Read desired ATL02 columns for given PCE and spot (strong/weak) into a dataframe.
        with hdf(self.atl02_file, mode='r') as hdf5:
            hdf5.epoch = hdf5['atlas_sdp_gps_epoch'][0]
            datagrp = 'pce{}/tep/'.format(self.pce)
            atl02_df, _ = hdf5.dg2pandas(datagrp, datasets=self.columns02().keys(), columns=self.columns02(), end_row=2*self.end_row)  

        # Create an immutable atl02 id
        atl02_df['atl02_id'] = np.arange(len(atl02_df))

        # Create a single index for this combination of columns.
        df2 = atl02_df.set_index(['pce_mframe_cnt', 'ph_id_pulse', 'ph_id_channel', 'ph_id_count'])

        return df2

    def map_ids(self, df2):
        """ Map the ATL02 events back to the 'Master' index of ATL01
        events. Each ATL02 event will have a unique ATL01 ID.

        Parameters
        ----------
        df2 : pandas.DataFrame
            The ATL02 DataFrame.
        """
        concat_df = pd.concat((self.df1, df2), axis=1)

        # Sort by Master ID
        concat_df = concat_df.sort_values(by=['master_id'])

        concat_df.reset_index(inplace=True)
        concat_df.rename(columns={'level_0':'mf', 'level_1':'pulse', 'level_2':'chan', 'level_3':'cnt'}, inplace=True)
        # Replace all NaN's with -1's in atl02_id column.
        concat_df['atl02_id'].fillna(-1, inplace=True)

        # Filter out the atl02_id rows with -1's. 
        # The remaining master IDs are the IDs that map directly from ATL02 back to ATL01.
        mapped_ids = concat_df['master_id'][concat_df['atl02_id'] != -1].values

        return mapped_ids

    def all_ids(self):
        """ Returns master IDs from ATL01 for all strong+weak ATL02 photons.
        """
        strong_df = self.read_atl02('strong')
        weak_df = self.read_atl02('weak')
        concat_df = pd.concat((strong_df, weak_df))
        return self.map_ids(concat_df)

    def strong_ids(self):
        """ Returns master IDs from ATL01 for all strong spot ATL02 photons.
        """
        df2 = self.read_atl02('strong')
        return self.map_ids(df2)

    def weak_ids(self):
        """ Returns master IDs from ATL01 for all weak spot ATL02 photons.
        """
        df2 = self.read_atl02('weak')
        return self.map_ids(df2)

    def tep_ids(self):
        """ Returns master IDs from ATL01 for all TEP ATL02 photons
        (PCE1+PCE2, strong only!)
        """
        df2 = self.read_tep()
        return self.map_ids(df2)


######################################
# For saving the IDs in an HDF5 file.
######################################

def save_ids(atl01_file, atl02_file, path_out=''):
    """ Saves the PhotonID outputs for strong, weak, all, and tep, for
    each PCE, in an HDF5 file.

    Parameters
    ----------
    atl01_file : str
        The path and filename of the ATL01 file.
    atl02_file : str
        The path and filename of the ATL02 file.
    path_out : str
        Path under which to save output file.
    """
    start_time = datetime.datetime.now()
    print("PhotonID start time: ", start_time)

    timenow = start_time.strftime('%Y-%jT%H-%M-%S')

    filename = os.path.join(path_out, "photon_ids_{}.hdf5".format(timenow))

    f = h5py.File(filename, "w")

    for pce in pces:
        print("PCE:", pce)

        pi = PhotonID(pce, atl01_file, atl02_file)

        strong_ids = pi.strong_ids()
        print("strong_ids: ", len(strong_ids), strong_ids)
        weak_ids = pi.weak_ids()
        print("weak_ids: ", len(weak_ids), weak_ids)
        all_ids = pi.all_ids()
        print("all_ids: ", len(all_ids), all_ids)

        if pce != 3:
            try:
                tep_ids = pi.tep_ids()
                print("tep_ids: ", len(tep_ids), tep_ids)
            except:
                print("No pce{}/tep group found.".format(pce))

        grp = f.create_group("pce{}".format(pce))
        dset_strong = grp.create_dataset("strong", data=strong_ids)
        dset_weak = grp.create_dataset("weak", data=weak_ids)
        dset_all = grp.create_dataset("all", data=all_ids)

        if pce != 3:
            try:
                dset_tep = grp.create_dataset("tep", data=tep_ids)
            except:
                print("No pce{}/tep for dataset to be created.".format(pce))

    print("Finished writing {}".format(filename))

    f.close()

    print("--PhotonID is complete.--")
    run_time = datetime.datetime.now() - start_time
    print("Run-time: ", run_time)

    return filename


