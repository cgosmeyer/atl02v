""" Module containing class to calculate the following ATL02 fields.

    atlas/pcex/altimetry/
        ch_mask_s
        ch_mask_w
    atlas/pcex/atltimetry/strong-weak
        band1_offset
        band1_width
        band2_offset
        band2_width

Author:

    C.M. Gosmeyer, Feb 2019
"""

import h5py
import numpy as np
import os
import pandas as pd
from datetime import datetime
from atl02v.shared.constants import d_USO, SF_USO
from atl02v.shared.paths import path_to_data, path_to_outputs
from atl02v.shared.tools import pickle_out


class SpotContainer(object):
    def __init__(self, band1_offset=None, band2_offset=None,
        band1_width=None, band2_width=None):
        self.band1_offset = band1_offset
        self.band2_offset = band2_offset
        self.band1_width = band1_width
        self.band2_width = band2_width

class PCEVariables(object):
    def __init__(self, ch_mask_s, ch_mask_w, 
        band1_offset_s, band1_offset_w, band2_offset_s, band2_offset_w,
        band1_width_s, band1_width_w, band2_width_s, band2_width_w):
        self.ch_mask_s = ch_mask_s
        self.ch_mask_w = ch_mask_w
        self.strong = SpotContainer(band1_offset_s, band2_offset_s, band1_width_s, band2_width_s)
        self.weak = SpotContainer(band1_offset_w, band2_offset_w, band1_width_w, band2_width_w)

class ReadATL01(object):
    """ Read all needed ATL01 datasets.
    """
    def __init__(self, atl01, pce):
        self.raw_alt_band_offset = np.array(atl01['pce{}/a_alt_science/raw_alt_band_offset'.format(pce)].value, dtype=np.float64)
        self.raw_alt_band_width = np.array(atl01['pce{}/a_alt_science/raw_alt_band_width'.format(pce)].value, dtype=np.float64)
        self.raw_alt_ch_mask = np.array(atl01['pce{}/a_alt_science/raw_alt_ch_mask'.format(pce)].value, dtype=np.float64)

class DownlinkBands(object):
    __name__ = 'DownlinkBands'
    def __init__(self, atl01_file, tof_pickle=None, verbose=False, very_verbose=False):
        """
        Parameters
        ----------
        atl01_file : str
            The name of the ATL01 H5 file.
        tof_pickle : str
            [Optional/Recommended] Name of the pickled TOF instance. If this
            parameter is filled, will obtain constants from constants.py.
        verbose : {True, False}
            On by default. Print read-in and calculated final arrays.
        very_verbose : {True, False}
            Off by default. Print in-between calculations in for loops.
        """
        # Begin timer.
        start_time = datetime.now()
        print("DownlinkBands start time: ", start_time)

        self.verbose = verbose
        self.very_verbose = very_verbose

        # Unpickle TOF
        if tof_pickle != None:
            self.tof = pickle_out(tof_pickle)
            # Assign TOF attributes to DownlinkBands
            self.d_USO = self.tof.d_USO
            self.SF_USO = self.tof.SF_USO
        else:
            self.d_USO = np.float64(d_USO) 
            self.SF_USO = np.float64(1.0)  # SF_USO(-2.5)

        # Open the ATL01 dataframe.
        self.atl01_file = atl01_file
        self.atl01 = h5py.File(self.atl01_file, 'r', driver=None)

        # Read the ATL01 dataframe.
        self.atl01_dict = {}
        for i in range(1, 4):
            self.atl01_dict[i] = ReadATL01(self.atl01, pce=i)

        # Close ATL01 and set to None, since can't pickle an H5 file.
        self.atl01.close()
        self.atl01 = None

        # Calculate the channel masks and the band offsets and widths.
        self.pce1 = self.calculate_bands(pce=1)
        self.pce2 = self.calculate_bands(pce=2)
        self.pce3 = self.calculate_bands(pce=3)

        print("--DownlinkBands is complete.--")
        self.run_time = datetime.now() - start_time
        print("Run-time: ", self.run_time)

    def map_pce(self, pce):
        """ If you need a way to map PCE number to PCE attribute.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        if pce == 1:
            return self.pce1
        elif pce == 2:
            return self.pce2
        elif pce == 3:
            return self.pce3

    def map_spot(self, pce, spot):
        """ To map a PCE number and spot to an attribute.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        spot : string
            's' or 'w',
        """
        if pce == 1:
            if 's' in spot:
                return self.pce1.strong
            elif 'w' in spot:
                return self.pce1.weak
        elif pce == 2:
            if 's' in spot:
                return self.pce2.strong
            elif 'w' in spot:
                return self.pce2.weak
        elif pce == 3:
            if 's' in spot:
                return self.pce3.strong
            elif 'w' in spot:
                return self.pce3.weak    

    def calculate_bands(self, pce):
        """ Perform for givne PCE the calculations for the downlink band
        channel masks, the band offsets, and the band widths.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        """
        if self.verbose:
            print("")
            print("PCE: ", pce)

        ch_mask_s, ch_mask_w = self.build_ch_masks(pce)

        if self.verbose:
            print("ch_mask_s: ", ch_mask_s.shape, ch_mask_s)
            print("ch_mask_w: ", ch_mask_w.shape, ch_mask_w)

        band1_offset_s, band2_offset_s, band1_offset_w, band2_offset_w = self.create_band_arrays(pce, param='offset')
        band1_width_s, band2_width_s, band1_width_w, band2_width_w = self.create_band_arrays(pce, param='width')

        if self.verbose:
            print("band1_offset_s: ", len(band1_offset_s), band1_offset_s)
            print("band2_offset_s: ", len(band2_offset_s), band2_offset_s)
            print("band1_offset_w: ", len(band1_offset_w), band1_offset_w)
            print("band2_offset_w: ", len(band2_offset_w), band2_offset_w)
            print("band1_width_s: ", len(band1_width_s), band1_width_s)
            print("band2_width_s: ", len(band2_width_s), band2_width_s)
            print("band1_width_w: ", len(band1_width_w), band1_width_w)
            print("band2_width_w: ", len(band2_width_w), band2_width_w)

        pce_variables = PCEVariables(
            ch_mask_s = ch_mask_s,
            ch_mask_w = ch_mask_w,
            band1_offset_s = band1_offset_s,
            band2_offset_s = band2_offset_s,
            band1_offset_w = band1_offset_w,
            band2_offset_w = band2_offset_w,
            band1_width_s = band1_width_s,
            band2_width_s = band2_width_s,
            band1_width_w = band1_width_w,
            band2_width_w = band2_width_w
            )

        return pce_variables

    ###############
    # Channel mask 
    ###############

    def return_valid_bands(self, row):
        """ Returns two length-4 arrays, where each index represents bands 1-4.
        
        If strong bits are 0s, the band is open to strong events.
        Likewise if weak bits are 0s, the band is open to weak events.

        Parameters
        ----------
        row : array
            Row from the raw channel mask.

        Returns
        -------
        return_row_s : array
            Length-4 array designating which bands are valid for strong.
        return_row_w : array
            Length-4 array designating which bands are valid for weak.            
        """
        # Assume at first no events in any of four bands.
        return_row_s = [False]*4       
        return_row_w = [False]*4       
        
        # If there is an event, the band will have a boolean value of 0. A logical AND
        # will return True if both bands in the pair have values of 0, signifying there
        # are events in the band pair.
        b12_s = np.logical_and(int.from_bytes(bytes(row[1:3]), 'big')==0, int.from_bytes(bytes(row[4:6]), 'big')==0)
        b34_s = np.logical_and(int.from_bytes(bytes(row[7:9]), 'big')==0, int.from_bytes(bytes(row[10:12]), 'big')==0)
        b14_s = np.logical_and(int.from_bytes(bytes(row[1:3]), 'big')==0, int.from_bytes(bytes(row[10:12]), 'big')==0)
        b23_s = np.logical_and(int.from_bytes(bytes(row[4:6]), 'big')==0, int.from_bytes(bytes(row[7:9]), 'big')==0)     
        
        b12_w = np.logical_and(int.from_bytes(bytes(row[0:1]), 'big')==0, int.from_bytes(bytes(row[3:4]), 'big')==0)
        b34_w = np.logical_and(int.from_bytes(bytes(row[6:7]), 'big')==0, int.from_bytes(bytes(row[9:10]), 'big')==0)
        b14_w = np.logical_and(int.from_bytes(bytes(row[0:1]), 'big')==0, int.from_bytes(bytes(row[9:10]), 'big')==0)
        b23_w = np.logical_and(int.from_bytes(bytes(row[3:4]), 'big')==0, int.from_bytes(bytes(row[6:7]), 'big')==0)
        
        # For every band in which events are found (True), set the position 
        # in the return array to True.
        # This seems heavy-handed but I am too fed up to be clever.
        if b12_s and b34_w:
            if int.from_bytes(bytes(row[0:1]), 'big') != 0:
                return_row_s[0] = True
            if int.from_bytes(bytes(row[3:4]), 'big') != 0:
                return_row_s[1] = True
            if int.from_bytes(bytes(row[7:9]), 'big') != 0:
                return_row_w[2] = True
            if int.from_bytes(bytes(row[10:12]), 'big') != 0:
                return_row_w[3] = True
                
        if b34_s and b12_w:
            if int.from_bytes(bytes(row[6:7]), 'big') != 0:
                return_row_s[2] = True
            if int.from_bytes(bytes(row[9:10]), 'big') != 0:
                return_row_s[3] = True
            if int.from_bytes(bytes(row[1:3]), 'big') != 0:
                return_row_w[0] = True
            if int.from_bytes(bytes(row[4:6]), 'big') != 0:        
                return_row_w[1] = True        
                
        if b14_s and b23_w:
            if int.from_bytes(bytes(row[0:1]), 'big') != 0:
                return_row_s[0] = True
            if int.from_bytes(bytes(row[9:10]), 'big') != 0:
                return_row_s[3] = True
            if int.from_bytes(bytes(row[4:6]), 'big') != 0:  
                return_row_w[1] = True
            if int.from_bytes(bytes(row[7:9]), 'big') != 0:
                return_row_w[2] = True      
                
        if b23_s and b14_w:
            if int.from_bytes(bytes(row[3:4]), 'big') != 0:        
                return_row_s[1] = True
            if int.from_bytes(bytes(row[6:7]), 'big') != 0:
                return_row_s[2] = True
            if int.from_bytes(bytes(row[1:3]), 'big') != 0:
                return_row_w[0] = True
            if int.from_bytes(bytes(row[10:12]), 'big') != 0:
                return_row_w[3] = True             
            
        if self.very_verbose:
            print("return_row_s: {}, return_row_w: {}".format(return_row_s, return_row_w))

        # Check that there are not more than two 'True's per spot.
        if return_row_s.count(True) <= 2 and return_row_w.count(True) <= 2:
            return return_row_s, return_row_w
        else:
            print("ERROR: More than two bands selected in row {} for spot {}.".format(row, spot))
            return [None]*4, [None]*4

    def mask_row(self, s_band_row, w_band_row):
        """ Use the valid bands to build masks.
        
        If there are two active bands of W and two active bands of S, does
        S take precedence over W? (so in that case, S are 0s and W are 1s)
        
        In this mask, 0 = transparent = an event is present.
                      1 = opaque = no events.

        Parameters
        ----------
        s_band_row : array
            Length-4 array designating which bands are valid for strong.
        w_band_row : array
            Length-4 array designating which bands are valid for weak.

        Returns
        -------
        s_mask_row : array
            Length-16 array mask for strong.
        w_mask_row : array
            Length-4 array mask for weak.
        """
        # Initialize mask rows for strong and weak spots.
        s_mask_row = np.array([0]*16)
        w_mask_row = np.array([0]*4)
        
        if s_band_row.count(True) == 2 and w_band_row.count(True) == 2:
            # S bands take precedence. Set W slots to one.
            w_mask_row[:] = 1
          
        if s_band_row.count(True) == 0:
            # If no S events, whole row should be set to one.
            s_mask_row[:] = 1
            
        if w_band_row.count(True) == 0:
            # Likewise, if no W events, whole row should be set to one.
            w_mask_row[:] = 1

        if self.very_verbose:
            print("s_mask_row: {}, w_mask_row: {}".format(s_mask_row, w_mask_row))

        ## What other cases would set the row to 1?
        ## Can just a subset of the row be 1?
            
        return s_mask_row, w_mask_row

    def build_ch_masks(self, pce):
        """ Builds channel strong and weak mask matrices.
        
        Parameters
        ----------
        pce : int
            1, 2, or 3.

        Returns
        -------
        ch_mask_s : array of arrays 
            The 16xnMF strong channel mask matrix.
        ch_mask_w : array of arrays 
            The 4xnMF weak channel mask matrix.
        """
        # Initialize the s and w channel masks.
        ch_mask_s = []
        ch_mask_w = []

        # Loop over rows in raw channel mask.
        # The number of rows equals the number of major frames.
        for row in self.atl01_dict[pce].raw_alt_ch_mask:
            # Find the valid strong and weak bands.
            s_band_row, w_band_row = self.return_valid_bands(row)

            # Build the s and w mask rows.
            s_mask_row, w_mask_row = self.mask_row(s_band_row, w_band_row)

            # Append to the mask matrices.
            ch_mask_s.append(s_mask_row)
            ch_mask_w.append(w_mask_row)

        return np.array(ch_mask_s), np.array(ch_mask_w)

    ########################
    # Band offset and width
    ########################

    def select_band_row(self, raw_alt_band_row, band_mask_row, spot):
        """ Use return_valid_bands over raw_alt_band_offset and _width to determine
        which bands should be used.
        
        Returns a cell (row) of band1_{offset/width} and band2_{offset_width}.
        
        This should work exactly the same for either strong or weak, and
        therefore specifying the spot is not required.
        
        Parameters
        ----------
        raw_alt_band_row : list
            Either raw_alt_band_offset or _width.
        band_mask_row : list
            Length-4 list of True (band in use) and False (band empty) 
            for strong/weak spot.
        spot : string
            Either 's' or 'w'.

        Returns
        -------
        band1_cell : float

        band2_cell : float
        """
        
        if self.very_verbose:
            print("spot: ", spot)
            print("raw_alt_band_row: ", len(raw_alt_band_row), raw_alt_band_row)
            print("band_mask_row: ", len(band_mask_row), band_mask_row)
        
        # Use the boolean band_row to decide what goes to band1 and what goes to band2.
        # And at same time convert to seconds.
        try:
            if spot == 's':
                band1_cell, band2_cell = raw_alt_band_row[band_mask_row]*self.d_USO*self.SF_USO
            elif spot == 'w':
                if band_mask_row[1] == True:
                    band2_cell, band1_cell = raw_alt_band_row[band_mask_row]*self.d_USO*self.SF_USO
                else:
                    band1_cell, band2_cell = raw_alt_band_row[band_mask_row]*self.d_USO*self.SF_USO
        except:
            # In the case where there is only one valid band, set the cell for the second band to 0.
            if spot == 's':
                band1_cell = (raw_alt_band_row[band_mask_row]*self.d_USO*self.SF_USO)[0]
                band2_cell = 0
            elif spot == 'w':
                if band_mask_row[1] == True:
                    band2_cell = (raw_alt_band_row[band_mask_row]*self.d_USO*self.SF_USO)[0]
                    band1_cell = 0
                elif band_mask_row[0] == True:
                    band1_cell = (raw_alt_band_row[band_mask_row]*self.d_USO*self.SF_USO)[0]
                    band2_cell = 0
                else:
                    # Might be making this up.
                    try:
                        band1_cell = (raw_alt_band_row[band_mask_row]*self.d_USO*self.SF_USO)[0]
                        band2_cell = 0
                    except:
                        band1_cell = 0
                        band2_cell = 0

        if self.very_verbose:    
            print("band1_cell: ", band1_cell)
            print("band2_cell: ", band2_cell)
    
        return band1_cell, band2_cell

    def create_band_arrays(self, pce, param):
        """ Creates arrays for band1 and band2, strong and weak,
        for given PCE and param (offset or width).

        Parameters
        ----------
        pce : int
            1, 2, or 3
        param : string
            'offset' or 'width'
        """
        if param == 'offset':
            raw_alt_band_matrix = self.atl01_dict[pce].raw_alt_band_offset
        elif param == 'width':
            raw_alt_band_matrix = self.atl01_dict[pce].raw_alt_band_width

        raw_alt_ch_mask = self.atl01_dict[pce].raw_alt_ch_mask
        
        band1_array_s = []
        band2_array_s = []
        band1_array_w = []
        band2_array_w = []
        
        for raw_alt_band_row, band_mask_row in zip(raw_alt_band_matrix, raw_alt_ch_mask):
            
            if self.very_verbose:
                print("raw_alt_band_row: ", raw_alt_band_row)
                print("band_mask_row: ", band_mask_row)

            s_band_row, w_band_row = self.return_valid_bands(band_mask_row)

            band1_cell_s, band2_cell_s = self.select_band_row(raw_alt_band_row, s_band_row, spot='s')
            band1_cell_w, band2_cell_w = self.select_band_row(raw_alt_band_row, w_band_row, spot='w')
            
            band1_array_s.append(band1_cell_s)
            band2_array_s.append(band2_cell_s)
            
            band1_array_w.append(band1_cell_w)
            band2_array_w.append(band2_cell_w)
            
        return np.array(band1_array_s), np.array(band2_array_s), np.array(band1_array_w), np.array(band2_array_w)

