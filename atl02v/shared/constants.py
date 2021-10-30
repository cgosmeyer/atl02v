""" Constants shared across TOD, TOF, and TEP.

Author:

	C.M. Gosmeyer, Apr 2018
"""

import numpy as np

d_USO = np.float64(10e-9)  # seconds  ## Is this value something other than 1?
dt_imet = 40e-9 # Defined in Section 1.2
dt_t0 = 1e-4 # Internal to ASAS (not defined in ATL02 ATBD).
lrs_clock = 1.1851851851851852E-6 # Used for LRS data (defined in LRS ATBD?).

pces = [1,2,3]
spots = [1,2,3,4,5,6]
downlink_bands = [1,2,3,4]
channels = np.ndarray(20)

def SF_USO(f_cal):
    """ Equation 3-5.
    """
    return (1./d_USO) / ((1./d_USO) + f_cal)