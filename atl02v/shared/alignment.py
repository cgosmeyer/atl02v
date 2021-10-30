""" Aligner class and tools.

Author:

    C.M. Gosmeyer, Apr 2018

"""

import numpy as np
from pydl import uniq
from atl02v.shared.constants import pces
from atl02v.shared.tools import find_nearest

class PCEContainer(object):
    def __init__(self, timetags=None, uniq=None, start_pad=None, end_pad=None):
        self.timetags = timetags
        self.uniq = uniq
        self.start_pad = start_pad
        self.end_pad = end_pad

class Aligner(object):
    __name__ = 'Aligner'
    def __init__(self, pce1_timetags, pce2_timetags, pce3_timetags, 
        verbose=False, very_verbose=False):
        """ Aligner class. For aligning the three PCEs and mapping
        RX to TX and conversely TX to RX.

        Parameters
        ----------
        pce1_timetags : array
            The timetags (DeltaTime_ll) for PCE1.
        pce2_timetags : array
            The timetags (DeltaTime_ll) for PCE2.
        pce3_timetags : array
            The timetags (DeltaTime_ll) for PCE3.
        verbose : {True, False}
            Off by default. Print read-in and calculated final arrays.
        very_verbose : {True, False}
            Off by default. Print in-between calculations in for loops.      
        """
        self.verbose = verbose
        self.very_verbose = very_verbose

        self.pce1_timetags = pce1_timetags
        self.pce2_timetags = pce2_timetags
        self.pce3_timetags = pce3_timetags

        # Get the indices of the unique items in each array.
        # So one entry per Tx.
        ## 'uniq' returns the last of each of the sequential unique items
        ## and it assumes the values are sorted so that all identical items are clustered
        self.uniq1 = uniq(self.pce1_timetags)
        self.uniq2 = uniq(self.pce2_timetags)
        self.uniq3 = uniq(self.pce3_timetags)
        
        if self.verbose:
            print("self.uniq1: ", len(self.uniq1), self.uniq1)
            print("self.uniq2: ", len(self.uniq2), self.uniq2)
            print("self.uniq3: ", len(self.uniq3), self.uniq3)

        self.start_pads, self.end_pads = self.__align_pces()

        self.pce1 = PCEContainer(timetags=self.pce1_timetags, uniq=self.uniq1, 
            start_pad=self.start_pads[0], end_pad=self.end_pads[0]) 
        self.pce2 = PCEContainer(timetags=self.pce2_timetags, uniq=self.uniq2, 
            start_pad=self.start_pads[1], end_pad=self.end_pads[1]) 
        self.pce3 = PCEContainer(timetags=self.pce3_timetags, uniq=self.uniq3, 
            start_pad=self.start_pads[2], end_pad=self.end_pads[2]) 

    def map_pce(self, pce):
        """ If you need a way to map PCE number to PCE attribute.
        """
        if pce == 1:
            return self.pce1
        if pce == 2:
            return self.pce2
        if pce == 3:
            return self.pce3

    def __find_pad_length(self, timetag_array, value):
        """ Private function for finding length of start padding
        needed to make lengths of the three PCE arrays equal.

        Parameters
        ----------
        timetag_array : array
            The unique timetags of the PCE in question.
        value : float
            The first timetag value the uniqued timetag array.
        """
        nearest_value, nearest_index = find_nearest(timetag_array, value) 
        pad_length = nearest_index
        pad = [np.nan for j in range(pad_length)]

        return pad

    def __align_pces(self):
        """ Private function that aligns three PCEs to the same start 
        time and pads their arrays so all are the same length.

        Notes
        -----
        For now, assume, post-starting match, all the values
        match and nothing is missing. 
        """
        # Filter so only unique values remain
        uniq_pce1_timetags = self.pce1_timetags[self.uniq1]
        uniq_pce2_timetags = self.pce2_timetags[self.uniq2]
        uniq_pce3_timetags = self.pce3_timetags[self.uniq3]

        if self.verbose:
            print("uniq_pce1_timetags: ", len(uniq_pce1_timetags), uniq_pce1_timetags)
            print("uniq_pce2_timetags: ", len(uniq_pce2_timetags), uniq_pce2_timetags)
            print("uniq_pce3_timetags: ", len(uniq_pce3_timetags), uniq_pce3_timetags)

        timetag_arrays = [uniq_pce1_timetags, uniq_pce2_timetags, uniq_pce3_timetags]

        # Create an array of the first values
        first_vals = [uniq_pce1_timetags[0], uniq_pce2_timetags[0], uniq_pce3_timetags[0]]
        
        # Create an array of the last values
        last_vals = [uniq_pce1_timetags[-1], uniq_pce2_timetags[-1], uniq_pce3_timetags[-1]]
        
        # Find the minimum start 
        min_val = np.min(first_vals)
        
        min_index = first_vals.index(min_val)
        min_pce = min_index + 1

        if self.verbose:
            print("first_vals: ", first_vals)
            print("last_vals: ", last_vals)
            print("min_val: ", min_val)

        # Buffer each PCE so all "start" at same value.
        start_pad1 = self.__find_pad_length(timetag_arrays[min_index], timetag_arrays[0][0])
        start_pad2 = self.__find_pad_length(timetag_arrays[min_index], timetag_arrays[1][0])
        start_pad3 = self.__find_pad_length(timetag_arrays[min_index], timetag_arrays[2][0])

        if self.verbose:
            print("start_pad1: ", len(start_pad1), start_pad1)
            print("start_pad2: ", len(start_pad2), start_pad2)        
            print("start_pad3: ", len(start_pad3), start_pad3)        

        # This will store indices from uniqed timetag arrays and NaNs where no matches to other PCEs exist.
        aligned_timetag_masks = [start_pad1+list(np.arange(len(uniq_pce1_timetags))), 
                                 start_pad2+list(np.arange(len(uniq_pce2_timetags))),
                                 start_pad3+list(np.arange(len(uniq_pce3_timetags)))]

        # Find maximum length, including the start pads.
        lengths = [len(aligned_timetag_masks[0]), 
                         len(aligned_timetag_masks[1]), 
                         len(aligned_timetag_masks[2])]
        max_length = np.max(lengths)
        max_length_index = lengths.index(max_length)

        if self.verbose:
            print("lengths: ", lengths)
            print("max_length: ", max_length)
        
        # Find lenghts of the end pads.
        end_pad_length1 = max_length - lengths[0]
        end_pad_length2 = max_length - lengths[1]
        end_pad_length3 = max_length - lengths[2]

        if self.verbose:
            print("end_pad_length1: ", end_pad_length1)
            print("end_pad_length2: ", end_pad_length2)
            print("end_pad_length3: ", end_pad_length3)

        # Create the end pads.
        end_pad1 = [np.nan for j in range(end_pad_length1)]
        end_pad2 = [np.nan for j in range(end_pad_length2)]
        end_pad3 = [np.nan for j in range(end_pad_length3)]

        if self.verbose:
            print("end_pad1: ", len(end_pad1), end_pad1)
            print("end_pad2: ", len(end_pad2), end_pad2)
            print("end_pad3: ", len(end_pad3), end_pad3)

        # Package the start and ends pads for return.
        start_pads = [start_pad1, start_pad2, start_pad3]
        end_pads = [end_pad1, end_pad2, end_pad3]
        
        return start_pads, end_pads

    def align2timetags_all(self, pce1_data, pce2_data, pce3_data):
        """ Wrapper on `align2timetags`, mapping in one go all three 
        PCE arrays (in TX-space) to the aligned timetags and adding
        appropriate padding.

        Parameters
        ----------
        pce1_data : array
            Any array in TX-space of PCE 1.
        pce2_data : array
            Any array in TX-space of PCE 2.
        pce3_data : array
            Any array in TX-space of PCE 3.

        Use
        ---
        aligner = Aligner(pce1_timetags, pce2_timetags, pce3_timetags)
        aligned_TX_T = alinger.match2timetags(TX_T_1, TX_T_2, TX_T_3)

        Notes
        -----
        Only requirement is that pce{}_data is indexed and sized exactly as pce{}_timetags.
        (that is, they reference the same TXs)
        """
        pce1_aligned = self.align2timetags(pce=1, data=pce1_data)
        pce2_aligned = self.align2timetags(pce=2, data=pce2_data)
        pce3_aligned = self.align2timetags(pce=3, data=pce3_data)

        return pce1_aligned, pce2_aligned, pce3_aligned

    def align2timetags(self, pce, data, limit=None):
        """ Maps the PCE array (must be in TX-space) to the aligned timetags
        and adds appropriate padding.

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        data : array
            Any array in TX-space of the given PCE.
        limit : int
            Index at which to limit mapping. (Use in cases where mf_limit != None
            in TEP.)
        """
        if limit == None:
            uniqued_data = data[self.map_pce(pce).uniq]
        else:
            limited_uniq = np.array([i for i in self.map_pce(pce).uniq if i < limit])
            uniqued_data = data[limited_uniq]
        aligned_data = np.asarray(self.map_pce(pce).start_pad + list(uniqued_data) + self.map_pce(pce).end_pad)

        return aligned_data

    def maptx2rx_all(self, pce1_aligned, pce2_aligned, pce3_aligned):
        """ Wrapper on `maptx2rx`, in one go mapping the aligned TX 
        arrays out to filling the lengths of the RX arrays for all PCEs.

        Parameters
        ----------
        pce1_aligned : array
            Any array in aligned TX-space of PCE 1.
        pce2_aligned : array
            Any array in aligned TX-space of PCE 2.
        pce3_aligned : array
            Any array in aligned TX-space of PCE 3.
        """
        aligned_data = [pce1_aligned, pce2_aligned, pce3_aligned]
        uniqs = [self.uniq1, self.uniq2, self.uniq3]
        timetag_data = [self.pce1_timetags, self.pce2_timetags, self.pce3_timetags]

        filled_data = []
        
        # Loop over PCEs.
        for p in np.arange(3):
            pce_aligned = aligned_data[p]
            pce_timetags = timetag_data[p]
            uniq = uniqs[p]
            
            uniq_pce_timetags = pce_timetags[uniq]

            start_pad = self.start_pads[p]
            end_pad = self.end_pads[p]
            
            # Remove start and end pad.
            pce_aligned_padless = pce_aligned[len(start_pad):len(pce_aligned)-len(end_pad)]
 
            if self.verbose:
                print("pce_aligned: ", len(pce_aligned), pce_aligned)
                print("pce_timetags: ", len(pce_timetags), pce_timetags)
                print("uniq: ", len(uniq), uniq)
                print("uniq_pce_timetags: ", len(uniq_pce_timetags), uniq_pce_timetags)
                print("start_pad: ", len(start_pad), start_pad)
                print("end_pad: ", len(end_pad), end_pad)
                print("pce_aligned_padless: ", len(pce_aligned_padless), pce_aligned_padless)
                # Now the aligned, padless array should be equal in slength to the unique timetags
                print("DOES len(pce_aligned_padless) == len(uniq_pce_timetags)?", len(pce_aligned_padless) == len(uniq_pce_timetags))

            # Loop over each index listed in the uniq, and fill the new event-length array
            # with the unique value from the aligned transmit-length array.
            pce_filled = np.zeros(len(pce_timetags))
            print(pce_filled)
            for tx in range(len(uniq)):
                if tx == 0:
                    pce_filled[:uniq[tx]+1] = pce_aligned_padless[tx]
                    if self.very_verbose:
                        print("pce_filled[:uniq[i]+1]: ", pce_filled[:uniq[i]+1])
                else:
                    pce_filled[uniq[tx-1]+1:uniq[tx]+1] = pce_aligned_padless[tx]
                    if self.very_verbose:
                        print("pce_filled[uniq[i-1]+1:uniq[i]+1]: ", pce_filled[uniq[i-1]+1:uniq[i]+1])
            
            if self.verbose:
                print("pce_filled: ", len(pce_filled), pce_filled)

            filled_data.append(np.array(pce_filled, dtype=np.float64))

        return filled_data

    def maptx2rx(self, pce, aligned_data):
        """ Maps the aligned TX array out to fill the length of an event (RX)
        array. Each transmit has a one-to-many relationship with RX events. 

        Parameters
        ----------
        pce : int
            1, 2, or 3.
        aligned_data : array
            Any array in aligned TX-space of the given PCE.
        """
        timetags = self.map_pce(pce).timetags
        uniq = self.map_pce(pce).uniq
    
        uniq_timetags = timetags[uniq]

        start_pad = self.map_pce(pce).start_pad
        end_pad = self.map_pce(pce).end_pad

        # Remove start and end pad.
        aligned_padless_data = aligned_data[len(start_pad):len(aligned_data)-len(end_pad)]

        self.verbose=False
        self.very_verbose=False
        if self.verbose:
            print("aligned_data: ", len(aligned_data), aligned_data)
            print("timetags: ", len(timetags), timetags)
            print("uniq: ", len(uniq), uniq)
            print("uniq_timetags: ", len(uniq_timetags), uniq_timetags)
            print("start_pad: ", len(start_pad), start_pad)
            print("end_pad: ", len(end_pad), end_pad)
            print("aligned_padless_data: ", len(aligned_padless_data), aligned_padless_data)
            # Now the aligned, padless array should be equal in slength to the unique timetags
            print("DOES len(aligned_padless_data) == len(uniq_timetags)?", len(aligned_padless_data) == len(uniq_timetags))

        # Loop over each index listed in the uniq, and fill the new event-length array
        # with the unique value from the aligned transmit-length array.
        filled_data = np.zeros(len(timetags))
        for tx in range(len(uniq)):
            if tx == 0:
                filled_data[:uniq[tx]+1] = aligned_padless_data[tx]
                if self.very_verbose:
                    print("pce_filled[:uniq[i]+1]: ", pce_filled[:uniq[i]+1])
            else:
                filled_data[uniq[tx-1]+1:uniq[tx]+1] = aligned_padless_data[tx]
                if self.very_verbose:
                    print("pce_filled[uniq[i-1]+1:uniq[i]+1]: ", pce_filled[uniq[i-1]+1:uniq[i]+1])

        return np.array(filled_data, dtype=np.float64)
