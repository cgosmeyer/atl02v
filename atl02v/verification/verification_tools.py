""" Contains tools for verification classes and scripts.

    Author:

        C.M. Gosmeyer, June 2018
"""

import glob
import h5py
import numpy as np
import pandas as pd
import os
import pylab as plt
import matplotlib.ticker as ticker
from collections import namedtuple
from datetime import datetime
from atl02qa.shared.paths import path_to_data, path_to_outputs
from atl02qa.shared.tools import make_timestamp_dir, pickle_out

Stats = namedtuple('Stats', ['data_max', 'data_min', 
    'data_mean', 'data_stdv', 'diff_max', 'diff_min', 
    'diff_mean', 'diff_stdv'], verbose=True)

class SummarizeGroup(object):
    __name__ = 'SummarizeGroup'
    def __init__(self, path_out, group_name):
        """
        """
        self.path_out = path_out
        self.group_name = group_name
        self.df = self.read_csvs()

    def read_csvs(self):
        """ Reads all the verification results CSV files in path 
        'path_out' and writes them to a single Pandas dataframe
        with the following columns. 

            label | status | match? | mean_diff 
        """
        # Create empty pandas dataframe
        df = pd.DataFrame()

        # Read in each CSV file and store as entry in a pandas dataframe
        csv_files = glob.glob(os.path.join(self.path_out, '*csv'))

        # Remove vfiles and other reports.
        for remove in ['vfiles', 'meb', 'pdu', 'thermal', 'lrs_hkt_1120', 'limits']:
            try:
                csv_files.remove(os.path.join(self.path_out, '{}.csv'.format(remove)))
            except:
                pass

        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            # rename diff_within_tol? to match?
            try:
                data = data.rename(index=str, columns={"diff_within_tol?": "match?"})
            except:
                pass
 
            # Add a column for the label name.
            data['label'] = pd.Series(csv_file.split('/')[-1].split('.csv')[0], index=data.index)
            # Append to dataframe
            df = df.append(data, ignore_index=True)

        return df

    def write_summary(self):
        """ Writes columns 'label', 'status', and 'match?' from dataframe 
        to a CSV file named 'vsummary.csv'.
        """
        columns = ['label', 'status', 'match?']
        csv_name = '{}.csv'.format(os.path.join(self.path_out, 'vsummary'))
        # Sort alphebatically
        self.df = self.df.sort_values(by='label')
        self.df.to_csv(csv_name, columns=columns)

    def plot_summary(self):
        """ Creates a plot of passes and fails from dataframe to a
        graphic named 'vsummary.png'.

        References
        ----------
        https://pythonspot.com/matplotlib-bar-chart/
        """
        passes = len(self.df[self.df['match?']==True].values)
        fails = len(self.df[self.df['match?']==False].values)

        completed = len(self.df[self.df['status']=='COMPLETED'].values)
        errored = len(self.df[self.df['status']=='ERROR'].values)

        # create plot
        fig, ax = plt.subplots()
        index = np.arange(2)
        bar_width = 0.35
        opacity = 0.75
         
        for i in range((completed//10)+1):
            plt.axhline(i*10, color='gray')

        good_news = plt.bar(index, [passes, completed], bar_width,
                         alpha=opacity,
                         color='b')
         
        bad_news = plt.bar(index + bar_width, [fails, errored], bar_width,
                         alpha=opacity,
                         color='r')
         
        plt.ylabel('# ATL02 Fields')
        plt.title(self.group_name)
        plt.xticks(index + bar_width*.5, ('Passed/Failed', 'Completed/Errored'))
        plt.minorticks_on()
        plt.tick_params(axis='x',which='minor',bottom='off')
        plt.tight_layout()

        plot_name = '{}.png'.format(os.path.join(self.path_out, 'vsummary'))
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close()

class VerificationContainer(object):
    __name__ = "VerificationContainer"
    def __init__(self, filename=None, descr=None, atl01=None, atl02=None, 
        tod=None, tof=None, tep=None, radiometry=None, dupmask=None, 
        photonids=None, dlbs=None):
        """ 
        """
        self.filename = filename
        self.descr = descr
        self.atl01 = atl01
        self.atl02 = atl02
        self.photonids = photonids
        self.tod = tod
        self.tof = tof
        self.tep = tep
        self.dupmask = dupmask
        self.radiometry = radiometry
        self.dlbs = dlbs

    def construct_filename(self):
        now = datetime.now()
        vfile_name = '{}_{}_{}_{}_{}_{}.csv'.format(self.prepend0(now.day), 
            self.prepend0(now.month), now.year, self.prepend0(now.hour), 
            self.prepend0(now.minute), self.prepend0(now.second))
        vfile = os.path.join(path_to_outputs, 'vfiles', vfile_name)

        return vfile

    def prepend0(self, x):
        if len(str(x)) == 1:
            return '0{}'.format(x)
        else:
            return x

    def write(self):
        vfile = self.construct_filename()

        f = open(vfile, 'w')
        f.write("descr, {}\n".format(self.descr))
        f.write("atl01, path_to_data, {}\n".format(self.atl01))
        f.write("atl02, path_to_data, {}\n".format(self.atl02))
        f.write("tod, path_to_outputs, {}\n".format(self.tod))
        f.write("tof, path_to_outputs, {}\n".format(self.tof))
        f.write("tep, path_to_outputs, {}\n".format(self.tep))
        f.write("radiometry, path_to_outputs, {}\n".format(self.radiometry))
        f.write("photonids, path_to_outputs, {}\n".format(self.photonids))
        f.write("dupmask, path_to_outputs, {}\n".format(self.dupmask))
        f.write("dlbs, path_to_outputs, {}\n".format(self.dlbs))
        f.close()

        return vfile

class VFileReader(object):
    def __init__(self, vfile_name):
        """ Reads vfile into VerificationContainer object.

        Parameters
        ----------
        vfile_name : string
            Name of the verification CSV file. Should contain rows named
            as the attributes in `VerificationContainer`.
        """
        self.vfile_name = vfile_name
        self.vfile = os.path.join(path_to_outputs, 'vfiles', self.vfile_name)

    def read(self):
        """
        """
        path_dict = {'path_to_data': path_to_data,
                    'path_to_outputs': path_to_outputs}

        # Loop through the vfile and match rows to VC attribbutes.
        with open(self.vfile) as f:
            for line in f.readlines():
                line = line.split(',')
                print("line: ", line)
                if line[0].strip() == 'descr':
                    descr = line[1].replace('\n', '')
                elif line[0].strip() == 'atl01':
                    atl01_path = line[1].strip()
                    atl01_file = line[2].replace('\n', '').strip()
                elif line[0].strip() == 'atl02':
                    atl02_path = line[1].strip()
                    atl02_file = line[2].replace('\n', '').strip()
                elif line[0].strip() == 'tod':
                    tod_path = line[1].strip()
                    tod_file = line[2].replace('\n', '').strip()
                elif line[0].strip() == 'tof':
                    tof_path = line[1].strip()
                    tof_file = line[2].replace('\n', '').strip()
                elif line[0].strip() == 'tep':
                    tep_path = line[1].strip()
                    tep_file = line[2].replace('\n', '').strip()
                elif line[0].strip() == 'radiometry':
                    radiometry_path = line[1].strip()
                    radiometry_file = line[2].replace('\n', '').strip()
                elif line[0].strip() == 'dupmask':
                    dupmask_path = line[1].strip()
                    dupmask_file = line[2].replace('\n', '').strip()
                elif line[0].strip() == 'photonids':
                    photonids_path = line[1].strip()
                    photonids_file = line[2].replace('\n', '').strip()
                elif line[0].strip() == 'dlbs':
                    dlbs_path = line[1].strip()
                    dlbs_file = line[2].replace('\n', '').strip()

        # Create VC with parameters from vfile.
        vc = VerificationContainer(
            filename=self.vfile_name,
            descr=descr,
            atl01=os.path.join(path_dict[atl01_path], atl01_file),
            atl02=os.path.join(path_dict[atl02_path], atl02_file),
            tod=os.path.join(path_dict[tod_path], tod_file), 
            tof=os.path.join(path_dict[tof_path], tof_file),
            tep=os.path.join(path_dict[tep_path], tep_file),
            radiometry=os.path.join(path_dict[radiometry_path], radiometry_file),
            dupmask=os.path.join(path_dict[dupmask_path], dupmask_file), 
            photonids=os.path.join(path_dict[photonids_path], photonids_file),
            dlbs=os.path.join(path_dict[dlbs_path], dlbs_file)
            )

        return vc

class Verify(object):
    __name__ = 'Verify'
    """ A super class for verification classes.
    """
    def __init__(self, vfiles, tolerance, atl02_dataset, path_out):
        self.vfiles = vfiles
        self.tolerance = tolerance
        self.atl02_dataset = atl02_dataset
        self.path_out = path_out
        
        # Read from ATL01 hdf5
        self.atl01 = h5py.File(self.vfiles.atl01, 'r', driver=None)        
        # Read from ATL02 hdf5
        self.atl02 = h5py.File(self.vfiles.atl02, 'r', driver=None)
        # Read PhotonIDs from hdf5
        self.photonids = h5py.File(self.vfiles.photonids, 'r', driver=None)

        # Create base filename for all output files.
        self.base_filename = '{}'.format(self.atl02_dataset.replace('/', '.'))

        # Report the verification files used
        self.report_vfiles()

        print(" ")
        print(atl02_dataset)

    def report_vfiles(self):
        """ Write the verification files to a txt file.
        """
        vfile_attrs = vars(self.vfiles)
        print(vfile_attrs)
        
        df = pd.DataFrame.from_dict(vfile_attrs, orient='index')
        df.to_csv('{}.csv'.format(os.path.join(self.path_out, 'vfiles')))             

    def record_exception(self, error_msg):
        """ Records in a CSV file the error message captured by an 
        exception. This effectively takes the place of the report that
        would have been generated in one of the 'compare_*' functions
        had an exception not occurred. The report has the following columns.

            status | error_msg | match?
        """
        print("ERROR in {}".format(self.atl02_dataset))
        print(error_msg)
        columns = ['status', 'error_msg', 'match?']
        vals = [('ERROR', error_msg, ''),]
        df = pd.DataFrame.from_records(vals, columns=columns)
        csv_name = '{}.csv'.format(os.path.join(self.path_out, self.base_filename))
        df.to_csv(csv_name, index=False)
        print("Created {}".format(csv_name))

    def get_stats(self, atl02_values, values_diff):
        """ Calculate the max, min, mean and standard dev of the
        ATL02 data and the diffed data.

        Parameters
        ----------
        atl02_values : list
            The ATL02 values to be verified.

        Returns
        -------
        stats : Stats collections.namedtuple
            Contains calculated stats.
        """
        data_max = np.max(atl02_values)
        data_min = np.min(atl02_values)
        try:
            data_mean = np.mean(atl02_values)
        except:
            data_mean = 0
        try:
            data_stdv = np.std(atl02_values)
        except:
            data_stdv = 0

        diff_max = np.max(values_diff)
        diff_min = np.min(values_diff)
        diff_mean = np.mean(values_diff)
        diff_stdv = np.std(values_diff)

        stats = Stats(data_max, data_min, data_mean, data_stdv,
            diff_max, diff_min, diff_mean, diff_stdv)

        return stats

    def compare_array_of_arrays(self, atl02_values, verifier_values):
        """ Compares ATL02 and verificier values if they are arrays of arrays.
        Outputs PNG diff plot and CSV report.

        Parameters
        ----------
        atl02_values : list
            The ATL02 values to be verified.
        verifier_values : list
            The values to be compared against ATL02 values.
        """
        values_diffs = []
        percent_diffs = []
        for i in range(len(atl02_values)):
            values_diff = atl02_values[i] - verifier_values[i]
            percent_diff = (values_diff / atl02_values[i])*100
            values_diffs.append(np.asarray(values_diff))

        values_diffs = np.array(values_diffs)
        percent_diffs = np.array(percent_diffs)

        print("values_diffs: ", values_diffs)
        self.plot_diff(values_diffs.flatten(), percent_diffs.flatten())

        for i in np.arange(values_diffs.shape[0]):
            stats = self.get_stats(atl02_values[i], values_diffs[i])
            print("values_diff[i]: ", values_diffs[i])
            print("atl02_values[i]: ", atl02_values[i])
            ind_outside_tol, frac_outside_tol = self.check_tolerance(values_diffs[i])

            if frac_outside_tol == 0:
                diff_within_tol = True
            else:
                diff_within_tol = False

            columns = ['status', 'tolerance', 'frac_outside_tol', 'diff_within_tol?',
                'data_dim', 'data_max', 'data_min', 'data_mean', 
                'data_stdv', 'diff_max', 'diff_min', 'diff_mean', 'diff_stdv']
            vals = [('COMPLETED', self.tolerance, frac_outside_tol, diff_within_tol,
                '{}'.format(atl02_values.shape), stats.data_max, 
                stats.data_min, stats.data_mean, stats.data_stdv, stats.diff_max, 
                stats.diff_min, stats.diff_mean, stats.diff_stdv),]
            df = pd.DataFrame.from_records(vals, columns=columns)

            csv_name = '{}_matrix{}.csv'.format(os.path.join(self.path_out, self.base_filename), i)
            df.to_csv(csv_name)
            print("Created {}".format(csv_name))

    def compare_arrays(self, atl02_values, verifier_values):
        """ Compares ATL02 and verificier values if they are arrays.
        Outputs PNG diff plot and CSV report.

        Parameters
        ----------
        atl02_values : list
            The ATL02 values to be verified.
        verifier_values : list
            The values to be compared against ATL02 values.
        """
        values_diff = atl02_values - verifier_values

        try:
            percent_diff = (values_diff / atl02_values)*100
        except:
            print("Could not divide to obtain percent diff.")
            values_diff = np.array([v.total_seconds() for v in values_diff])
            percent_diff = np.zeros(len(values_diff))

        print("values_diff: ", values_diff)
        self.plot_diff(values_diff.flatten(), percent_diff.flatten())

        stats = self.get_stats(atl02_values, values_diff)
        ind_outside_tol, frac_outside_tol = self.check_tolerance(values_diff)

        if frac_outside_tol == 0:
            diff_within_tol = True
        else:
            diff_within_tol = False

        columns = ['status', 'tolerance', 'frac_outside_tol', 'diff_within_tol?',
            'data_dim', 'data_max', 'data_min', 'data_mean', 
            'data_stdv', 'diff_max', 'diff_min', 'diff_mean', 'diff_stdv']
        vals = [('COMPLETED', self.tolerance, frac_outside_tol, diff_within_tol, 
            '{}'.format(atl02_values.shape), stats.data_max, 
            stats.data_min, stats.data_mean, stats.data_stdv, stats.diff_max, 
            stats.diff_min, stats.diff_mean, stats.diff_stdv),]
        df = pd.DataFrame.from_records(vals, columns=columns)

        csv_name = '{}.csv'.format(os.path.join(self.path_out, self.base_filename))
        df.to_csv(csv_name, index=False)
        print("Created {}".format(csv_name))

    def compare_strings(self, atl02_values, verifier_values):
        """ Compares ATL02 and verifier values if they are strings
        and a diff is not possible. Outputs CSV report.

        Parameters
        ----------
        atl02_values : list
            The ATL02 values to be verified.
        verifier_values : list
            The values to be compared against ATL02 values.
        """
        # First sort alphabetically.
        atl02_values_sorted = np.sort(atl02_values)
        verifier_values_sorted = np.sort(verifier_values)

        # Then create an array of True-False for whether they match element-wise.
        matches = [smart_decode(atl02_values_sorted[i]).strip() == smart_decode(verifier_values_sorted[i]).strip() \
            for i in range(len(atl02_values_sorted))]

        # If a single False found, match flag must be False
        if False in matches:
            match = False
        else:
            match = True

        columns = ['status', 'match?', 'atl01/verifier', 'atl02']
        vals = [('COMPLETED', match, verifier_values, atl02_values),]
        df = pd.DataFrame.from_records(vals, columns=columns)
        csv_name = '{}.csv'.format(os.path.join(self.path_out, self.base_filename))
        df.to_csv(csv_name)
        print("Created {}".format(csv_name))        

    def check_tolerance(self, values_diff):
        """ Checks whether diffed values are within tolerance bounds.

        Parameters
        ----------
        values_diff : array
            Diff of the ATL02 and verifiier values.
        """
        ind_outside_tol = np.where(abs(values_diff) > self.tolerance)[0]
        diff_outside_tol = values_diff[ind_outside_tol]
        print("diff_outside_tol: ", diff_outside_tol)
        num_outside_tol = len(diff_outside_tol)
        frac_outside_tol = num_outside_tol / len(values_diff)

        return ind_outside_tol, frac_outside_tol

    def plot_diff(self, diff, percent_diff):
        """ Plots the diff and percent diff of the ATL02 and verifier 
        values, producing a two-plot PNG.

        Parameters
        ----------
        diff : list
            The differenced ATL02 and verifier values.
        percent_diff : list
            (ATL02 - verifier) / ATL02 * 100.
        """
        print("shape of diff: ", diff.shape)
        print("shape of percent_diff: ", percent_diff.shape)

        if len(diff.shape) < 2 or 'ch_mask' in self.atl02_dataset:
            fig, (ax1, ax2) = plt.subplots(2, figsize=(10,12))
            ax1.plot(diff, color='blue', marker='x')
            ax1.set_title('{}'.format(self.base_filename.split('/')[-1]), size=20)
            ax1.set_ylabel('Diff', size=14)
            ax1.tick_params(labelsize=12)
            # Draw lines at tolerance level
            ax1.axhline(y=self.tolerance, color='red')
            ax1.axhline(y=-self.tolerance, color='red')

            ax2.plot(percent_diff, color='orange', marker='x')
            ax2.axhline(y=0.0, color='grey', linestyle='--')
            ax2.set_xlabel('Event', size=14)
            ax2.set_ylabel('% Diff of ATL02', size=14)
            ax2.tick_params(labelsize=12)

            png_name = '{}.png'.format(os.path.join(self.path_out, self.base_filename))
            plt.savefig(png_name, bbox_inches='tight')
            print('Created {}'.format(png_name))

        else:
            for i in np.arange(diff.shape[0]):
                fig, (ax1, ax2) = plt.subplots(2, figsize=(10,12)) 
                ax1.plot(diff[i], color='blue', marker='x')
                ax1.set_title('{}'.format(self.base_filename.split('/')[-1]), size=20)
                ax1.set_ylabel('Diff', size=14)
                ax1.tick_params(labelsize=12)
                # Draw lines at tolerance level
                ax1.axhline(y=self.tolerance, color='red')
                ax1.axhline(y=-self.tolerance, color='red')

                if percent_diff.shape[0] != 0:
                    ax2.plot(percent_diff[i], color='orange', marker='x')
                    ax2.axhline(y=0.0, color='grey', linestyle='--')
                    ax2.set_xlabel('Event', size=14)
                    ax2.set_ylabel('% Diff of ATL02', size=14)
                    ax2.tick_params(labelsize=12)

                png_name = '{}_matrix{}.png'.format(os.path.join(self.path_out, self.base_filename), i)
                plt.savefig(png_name, bbox_inches='tight')
                print('Created {}'.format(png_name))                


def smart_decode(item):
    if isinstance(item, bytes):
        return str(item.decode())
    else:
        return str(item)

class Precision(object):
    def __init__(self, nfltpnts, interval=1, path_out=''):
        """

        References
        ----------
        https://ipfs.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/IEEE_floating_point.html
        """
        self.nfltpnts = nfltpnts
        self.interval = interval
        self.path_out = path_out

        self.b32_precision = 24*np.log10(2) # ~7
        self.b64_precision = 53*np.log10(2) # ~16

        self.fltpnt_powers = np.arange(-self.nfltpnts, self.nfltpnts+1, self.interval)
        self.fltpnt_values = self.calc_floating_point_values()

        self.b32_fltpnt_precisions = self.calc_floating_point_precisions(self.b32_precision)
        self.b64_fltpnt_precisions = self.calc_floating_point_precisions(self.b64_precision)

    def calc_floating_point_values(self):
        """ Creates array of 10^-nfltpnts to 10^nfltpnts, inclusive, in steps
        of 10^interval.
        """
        return np.array([10**float(n) for n in self.fltpnt_powers])

    def calc_floating_point_precisions(self, bit_precision):
        """
        Know that at 10^0, precision is 10^bit_precision and at 10^-10, 10^-26 precision.
        Therefore equation for the floating point precision is 
            10^(-bit_precision + fltpnt_power)
        """
        #return np.array([(10**float(-bit_precision+n)) for n in self.fltpnt_powers])
        return np.array([(10**float(-bit_precision+n)) for n in self.fltpnt_powers])

    def plot(self):
        """
        """
        fig, ax = plt.subplots(figsize=(16,10))
 
        # Make grid
        for j in self.fltpnt_values:
            ax.axvline(x=j, color='grey', linestyle='--', linewidth=1)
        for i in np.array([10**float(-self.b64_precision+n) for n in np.arange(-self.nfltpnts-1, self.nfltpnts+11)]):
            ax.axhline(y=i, color='grey', linestyle='--', linewidth=1)

        ax.plot(self.fltpnt_values, self.b32_fltpnt_precisions, color='teal', linewidth=4, label='bit-32 precision')
        ax.plot(self.fltpnt_values, self.b64_fltpnt_precisions, color='blue', linewidth=4, label='bit-64 precision')  
        ax.loglog()

        # Plot the precision for TOF
        #ax.axhline(y=10**-16, color='red')
        #ax.axvline(x=10**0, color='red')

        plt.xlim([self.fltpnt_values[0],self.fltpnt_values[-1]])
        plt.ylim([self.b64_fltpnt_precisions[0], self.b32_fltpnt_precisions[-1]])

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor')

        ax.grid(True, which='major', linestyle='-', linewidth=1.5, color='black')
        ax.legend(loc='upper left', fontsize=16)

        ax.set_xlabel('Floating Point Value', size=18)
        ax.set_ylabel('Floating Point Precision', size=18)

        png_name = os.path.join(self.path_out, 'precision.png')
        plt.savefig(png_name, bbox_inches='tight')


