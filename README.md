# `atl02v` 

## About

Partial examples of the code used to verify the Level-1B ("ATL02") data products of the NASA science satellite ICESat-2's ATLAS laser altimeter. This includes indepenedent calculations of photon times of flight, radiometry, and housekeeping data conversions.

## Installation

1. Activate your Python 3 conda environment.

2. Cd to the location you wish to store the package.

3. Type the following

    ```
    git clone git@github.com:cgosmeyer/I2L1Bv.git
    ```

4. Cd to `atl02v`.

5. Run `setup.py` to install the package.

    ```
    python setup.py develop
    ```
   You should now be able to import `atl02v` from anywhere, as long as you are in the same conda environment.

6. Run `init_paths.py` to generate the output directory structure.

    ```
    python init_paths.py
    ```
   You should now see, in the path you specified, a directory structure as follows

    ```
    verification/
        data/
        docs/
        outputs/
            data/
            reports/
            vfiles/
    ```
   It is your responsibility to add sub-directories to `data/` containing each associated batch of ATl01, ATL02, and calibration files that you wish to verify. The `outputs/data` sub-directory will fill with the h5 and pickle files generated to compare against the ATL02 and the `outputs/reports` sub-directory will hold any plots or text files created with verification testing. 
   You will also find a new file, `atl02v/shared/paths.py` containing the paths to outputs you specified. Do NOT git commit this file (it's already listed in the `.gitignore`, so you should never need think about it). 

7. Unzip `data/ITOS_db.zip`. You will find the files necessary to perform the counts to physical units conversions in `atl02v/tof/tof.py`, `atl02v/radiometry/radiometry.py`, `vfiles/verify_ancillary_data.py`, and `vfiles/verify_lrs.py`. You can point your "path_to_itos" parameter in `atl02v/shared/paths.py` here, or move the files.

    > **WARNING:** Do NOT commit the unzipped `ITOS_db`. It is included in the repo for your convenience. Committing it unzipped will add 40 Mb to the repo, which though easy to fix, will make you and @cgosmeyer temporarily unhappy.

8. If you find a dependency is missing, install it by conda or by pip. (If it is missing from conda, it will be available from pip.)

    ```
    conda install missing_package
    ```
    or
    ```
    pip install missing_package
    ```

## How to Run Verification Suite

The following commands assume you are running the generation and verification scripts from the primary level of `atl02v`.


### Generate Test Files

Follow these steps to generate the test file one at a time. Once you have a good grasp of what each step is doing, you can instead run the wrapper script `gscripts/gen_all.py` that will take care of steps 2-7. 

1. Place ATL01, ATL02, ANC13, ANC27, and CAL files in a subfolder (preferably named after the granule, e.g., `20190501072836_05040313_203_01`) in the `verification/data` folder.

2. Create a verification file in `verification/outputs/vfiles`, using as a template `examples/vfile_example.csv`. Fill in the ATL01 and ATL02 fields, and fill in the rest of the fields as you proceed with the following steps. A single verification file should list only outputs generated under same conditions, for same source ATL01 file, under same version of the software.

3. Generate a photonid hdf5 file (creates map of the ATL02 photons to ATL01).
    
    ```
    python> gscripts/gen_photonid.py
    ```
 
4. Generate a Time of Day pickle file.

    ```
    python> gscripts/gen_tod.py
    ```

5. Generate a Time of Flight pickle file.

    ```
    python> gscripts/gen_tof.py
    ```
    
    > **WARNING:** This is the most time- and memory-intensive script. If you have a large ATL01, you may want to limit the number of majorframes and do the file in chunks; otherwise, the output may be too large to be pickled. See the docstrings of `gscripts/gen_tof.py` and `atl02v/tof/tof.py` for more information.

6. Generate a Radiometry pickle file.

    ```
    python> gscripts/gen_radiometry.py
    ```

7. Generate a duplicate mask (marks on a '1' matrix as '0' all ATL01 indices not present in ATL02).

    ```
    python gscrips/gen_duplicatemask.py
    ```

8. Make sure your verification file is completely filled in. You are now ready to start verifying ATL02!


### Verify ATL02 with Test Files

Each higher level group in ALT02 will have its own verification script. The following examples assume you wish to run them one at a time. You can also run all of the verification scripts together using the wrapper `vscripts/verify_all.py`.

The ATL02 groups to be verified by their own scripts are

    ancillary_data
    atlas
        housekeeping
        pcex
        tx_pulse_width
    gpsr
    lrs
    orbit_info
    sc

In addition, `delta_time` fields, which are present in every group, has its own verification script, `vscripts/verify_delta_time.py`

1. EXAMPLE: I want to verify the fields in the atlas/pcex/altimetry group with verification file `11-11-2018.csv`

    ```
    python> vscripts/atlas_pcex_verification.py --f altimetry --v 11-11-2018.csv
    ```

2. EXAMPLE: I want to verify in one go the fields in all subgroups of atlas/pcex/ with verification file `11-11-2018.csv`

    ```
    python> vscripts/atlas_pcex_verification.py --v 11-11-2018.csv
    ```

3. EXAMPLE: I want to verify the fields in both the atlas/pcex/altimetry and atlas/pcex/tep groups in one go with verification file `11-11-2018.csv`

    ```
    python> vscripts/atlas_pcex_verification.py --f altimetry tep --v 11-11-2018.csv
    ```

### Verification Outputs

For all examples, check `outputs/reports` for a directory named for the ATL01 basefile (e.g., `20190501072836_05040313_203_01`), and in that directory, a time-stamped directory specific to each run of the verification scripts. These time-stamped directories will contain the following:

* Field-specific verification reports.  For example, `ancillary_data.atlas_sdp_gps_epoch.csv`
* Field-specific diff plots. For example, `ancillary_data.atlas_sdp_gps_epoch.png`
* Record of ATL01, ATL02, and pickle files used in the run of the software, `vfiles.csv`
* Abridged summary of all the verification reports in the group, `vsummary.csv`
* Plotted pass/fail and completed/errored summary of all verification reports in the group, `vsummary.png`

If the group under verification contains fields that were converted from counts in ATL01 to physical units in ATL02, the conversion expressions will be recorded in a summary file. For example, in the group "atlas/housekeeping/meb" a report will be generated named `meb.csv`.

