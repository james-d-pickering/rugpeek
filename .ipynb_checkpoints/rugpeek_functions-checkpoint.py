import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pn
import lmfit as lf
import scipy as sp
import matplotlib.colors as colors
import os 
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, Cursor
import random
from datetime import date
import glob
import re
import mplcursors
from time import gmtime, strftime
import time
from random import randrange
from zipfile import ZipFile
import pandas as pd
import sys
import re
import pprint
import fnmatch




class RugTools:
    """
    RugTools contains functions for generic tools to help data processing
    
    
    """
   # class for generic tools to help data processing

    def read_harpia_matrix(file, extension):
        """
        Reads in file
    
        Parameters
        -----------
    
            file : string
                 Harpia matrix is the overall TA matrix from a measurement not divided into individual scans
        
            extension : string
                The file extension
    
    
        Returns
        ---------
    
            wavelengths_nm : np.ndarray
                Wavelenths data
        
            delays_ps : np.ndarray
                Time delay data
                
            delta_OD : np.ndarray
                Absorption data
                
        """
       
    # Harpia matrix is the overall TA matrix from a measurement
    # not divided into individual scans
        count = 0
        with open(file+extension, encoding="utf-8") as f:
            for line in f:

                line = line.strip()
                if np.char.isnumeric(line[0]):
                    break

                count = count + 1
                
        data = np.genfromtxt(file+extension, skip_header=count)
        wavelengths_nm = data[0,1:]
        delays_ps = data[1:,0]
        delta_OD = data[1:, 1:]
        
        return wavelengths_nm, delays_ps, delta_OD
    
    
    


    def indep_roll(arr, shifts, axis=0):
        """Apply an independent roll for each dimensions of a single axis.

        Nicked from stackoverflow, clever implementation. 

        Parameters
         ----------
        arr : np.ndarray
            Array of any shape.

        shifts : np.ndarray
            How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.

        axis : int
            Axis along which elements are shifted. 
        """
        
        #JDP swaps the axis you want to shift with the last axis (so makes the shifted axis -1)
        arr = np.swapaxes(arr,axis,-1)

        #JDP get all the indices for any dimensionality 
        all_idcs = np.ogrid[[slice(0,n) for n in arr.shape]]

        # Convert to a positive shift 
        shifts[shifts < 0] += arr.shape[-1]

        #JDP shift the last index by the shift amount (semi witchcraft)
        all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

        #JDP index the array to the new indices (fully witchcraft) 
        result = arr[tuple(all_idcs)]

        #JDP swap the axes back to how they originally were
        arr = np.swapaxes(result,-1,axis)
        return arr

    def nm_to_THz(rug):
        """
        Overwrites the wavelength in nm to an optical freq in THz
        
        """
        rug.wavelengths = (2.997E8/(rug.wavelengths*1.0E-9))/1E12
        return 
    
       
    def find_nearest(array, value):
        """
        Finds nearest real value from input value
        
        Parameters
        ----------
            array : np.ndarray
                Array of all the real values
            value : float
                User input value
                
        Returns
        ----------
            idx : integer
                Index of the real value
            
            array[idx] : float
                Real value
                
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]
    
    
    
 #############################

    def find_missing(lst):
        """
        Return missing index values in a list.
        
        Parameters
        ----------
            lst : list
                List of all the known indexes
                
        Returns
        ----------
            list1 : list
                List of missing indexes
        
        """
        max = lst[0]
        for i in lst:
            if i > max:
                max = i

        min = lst[0]
        for i in lst:
            if i < min:
                min = i

        list1 = []

        for num in range(min + 1, max):
            if num not in lst:
                list1.append(num)

        return list1

    def get_metadata(file):
        """
        Extracts metadata from a file
        
        Parameters
        ----------
            file : string
                Data file
                
        Returns
        ----------
            dictionary : dictionary
                Dictionary of all the metadata from the data file. Includes 'Sample', 'Measurement', 'Run', 'Pump_wavelength', 'Concentration', 'WLC_source', 'Other'. 
        
        """
        
        
        file = file[:-7]
        file_metadata = {}


        metadata = file.split('_')

        limit = len(metadata)


        indexes = []
        keys = []
        items = []
        file_metadata.update({'sample':metadata[0]})
        indexes.append(0)
        items.append(metadata[0])
        keys.append('Sample')

        if len(metadata) == 1:
            print('Missing information -  filename assumed to be sample')
            
        if any([item == 'TA' or item == 'chirp' for item in metadata])  == True:
            measure = ([item == 'TA' or item == 'chirp' for item in metadata])
            measureindex = measure.index(True)
            indexes.append(measureindex)
            items.append(metadata[measureindex])
            keys.append('Measurement')
        
        else:
            print('Missing: Measurement')
            keys.append('Measurement')
            items.append('None')

        if any([item == 'TA' or item == 'chirp' for item in metadata])  == True:
            run = ([item == 'TA' or item == 'chirp' for item in metadata])
            runindex = run.index(True)
            indexes.append(runindex+1)
            items.append(metadata[runindex+1])
            keys.append('Run')
             
        else:
            print('Missing: Run')
            keys.append('Run')
            items.append('None')

        if any([item.endswith("nm") for item in metadata]) == True:
            wave = ([item.endswith('nm') for item in metadata])
            waveindex = wave.index(True)
            indexes.append(waveindex)
            items.append(metadata[waveindex])
            keys.append('Pump_wavelength')    

        else:
            print('Missing: Pump Wavelength')
            keys.append('Pump_wavelength')
            items.append('None')

        if any([item.endswith("M") for item in metadata]) == True:
            conc = ([item.endswith('M') for item in metadata])
            concindex = conc.index(True)
            indexes.append(concindex)
            items.append(metadata[concindex])
            keys.append('Concentration')


        else:
            keys.append('Concentration')
            items.append('None')
            print('Missing: Concentration')


        if any([item == 'SHG' or item =='FUN' for item in metadata])  == True:
            laser = ([item == 'SHG' or item =='FUN' for item in metadata])
            laserindex = laser.index(True)
            indexes.append(laserindex)
            items.append(metadata[laserindex])
            keys.append('WLC_source')

        else:
            keys.append('WLC_source')
            items.append('None')
            print('Missing: WLC source')

        dictionary = dict(zip(keys, items))
        
        missing_indexes = RugTools.find_missing(indexes)

        for i in missing_indexes:
                items.append(metadata[i])
                keys.append('Other'+str(i-2))
                dictionary.setdefault('Other', []).append(metadata[i])
        else:
            dictionary.setdefault('Other', []).append('None')
            
        return dictionary

        
    
    def findfiles():
        """
        Finds files with a specific wavelength and probe inputed by the user.
        
        Returns
        ----------
            cleaned_files : list
                All the files that meet the user requirements, excluding processed files.
        
        """
        
        filelist = glob.glob('*.dat')
        
        with ZipFile('aarhus_TA_jan24.zip', 'r') as zipObj:
            for entry in zipObj.infolist():
                filelist.append(entry.filename)
                

        print(filelist)
        
        
        wave = str(input('Enter wavelength: '))
        probe = str(input('Enter probe type: '))

        chirpfinder = r'\b' + wave + r'\b' + '.*' + probe + r'\b' + '.*' + "chirp" + r'\b'
        everything = r'\b' + wave + r'\b' + '.*' + probe + r'\b'


        dictionary = {}
        emptyl = []
        place = []
        count = 0
        for file in filelist:
 
            splitting = file.split('_')
            newstring = ' '.join(map(str, splitting))

            if re.search(str(chirpfinder), newstring):
                place.append(-count)
                emptyl.append(file)

            elif re.search(str(everything), newstring):
                place.append(count)
                emptyl.append(file)
                
            count += 1


        dictionary = dict(zip(place, emptyl))
        
        if len(dictionary) == 0:
            raise NameError('Invalid probe and/or wavelength entered. Suffix of wavelength must be nm')

        solvent_file = []
        sample_files = []

        for key, value in dictionary.items():
            if key < 0:
                solvent_file.append(value)
            else:
                sample_files.append(value)
            
        solventfile = solvent_file[0]
        files = [solventfile] + sample_files
        cleaned_files = [file for file in files if not file.endswith('processed.dat')]
        
        return cleaned_files

    def average_rugs(rugs):
        #JDP could add a check here in case people average nonsensical stuff
        averaged_rug = rugs[0]
        averaged_rug.matrix = np.mean(np.array([i.matrix for i in rugs]), axis=0)
        averaged_rug.filename + averaged_rug.filename.removesuffix('_matrix')
        averaged_rug.filename = averaged_rug.filename + '_averaged'
        return averaged_rug


    def interpolate_time_axis(rug, mintime=None, maxtime=None, timestep=None, method='nearest',
                             undo=True, verbose=False):

        if mintime:
            interp_min = mintime
        else:
            interp_min = np.min(rug.times)
        
        if maxtime:
            interp_max = maxtime
        else:
            interp_max = np.max(rug.times)

        if timestep:
            dt = timestep
        else:
            dt = np.min(np.ediff1d(rug.times)) 
            
        #JDP set the time resolution to the smallest step
        if verbose:
            print("interpolation time step is", dt)
            print("interpolation limits are", interp_min, interp_max)
        
        # number of interpolation points is latest time - earliest time / smallest step

        samples = int((interp_max-interp_min)/dt)
        if verbose:
            print("number of interpolated time points is", samples)
        
        #JDP create a new axis to get the new time points on
        x = np.linspace(interp_min, interp_max, samples)

        #JDP interpolate from the old time axis to the new one
        t_interp = np.interp(x, rug.times, rug.times)


        #JDP get a meshgrid with the new interpolated time axis so we can evaluate interpolated TA data
        wavelength_2D, times_2D = np.meshgrid(rug.wavelengths, t_interp, indexing="ij")
        
        #JDP interpolate the actual TA data from the matrix (or, create a function to do it)
        interp2D = sp.interpolate.RegularGridInterpolator((rug.times, rug.wavelengths), rug.matrix,
                bounds_error=False, fill_value=-9999, method=method)

        data_interp = interp2D((times_2D, wavelength_2D)).T 
        
        if undo:
            time_indices = []
            for time in rug.times:
                time_idx, _ = RugTools.find_nearest(t_interp, time)
                time_indices.append(time_idx)

            rug.matrix = data_interp[time_indices, :]
            rug.times = t_interp[time_indices]
        else:
            rug.matrix = data_interp
            rug.raw_times = rug.times
            rug.times = t_interp
        return t_interp

    def undo_time_interpolation(rug, new_axis):
        time_indices = []
        for time in new_axis:
            time_idx, _ = RugTools.find_nearest(rug.times, time)
            time_indices.append(time_idx)

        rug.matrix = rug.matrix[time_indices, :]
        rug.times = rug.times[time_indices]
        return
    
    def combine_rugs_wavelengths(rugs, fname):
        #JDP assume first file is 'origin'
        #JDP if you want to combine more than 2 needs editing.
        #JDP assumes that you want to trim overlapping wavelengths from specturm 1, also needs changing

        #JDP need to interpolate everything onto the same time grid for this to be reliable
        #JDP get the min and max times for the interpolation
        mintime = np.min(np.array([i.times for i in rugs]))
        maxtime = np.max([i.times for i in rugs])
        original_time = rugs[0].times
        #JDP interpolate the data onto a common time axis
        for rug in rugs:
            RugTools.interpolate_time_axis(rug, mintime=mintime, maxtime=maxtime, 
                                           undo=False, verbose=False)

        
        start_wl = rugs[-1].wavelengths[0] #first wl of the most red spectrum
        end_wl = rugs[0].wavelengths[-1] #last wl of the most blue spectrum
        overlap_point_blue, _ = RugTools.find_nearest(rugs[0].wavelengths, start_wl)
        overlap_point_red, _ = RugTools.find_nearest(rugs[-1].wavelengths, end_wl)
        overlap_points = [[0, overlap_point_blue], [0, -1]]
        
        
        combined_rug = Rug()
        combined_rug.matrix = np.concatenate([i.matrix[:, overlap_points[idx][0]:overlap_points[idx][1]] for idx, i in enumerate(rugs)], axis=1)
        combined_rug.wavelengths = np.concatenate([i.wavelengths[overlap_points[idx][0]:overlap_points[idx][1]] for idx, i in enumerate(rugs)], axis=0)
        combined_rug.times = rugs[0].times #assume same times from interpolation
        combined_rug.filename = fname     

        #stick it back onto the original axis for speed
        RugTools.undo_time_interpolation(combined_rug, original_time)
        combined_rug.peek()

        return combined_rug
     
       # fig = plt.figure()
       # ax = fig.gca()
       # ax.imshow(combined_rug.matrix, vmin=-10, cmap='PuOr')
       # sys.exit()
       # ax.set_title('Click on reference baseline')
    # refx, refy = plt.ginput(1)[0]
       # ax.axhline(y=refy)
       # ax.set_title('Click on second baseline')
       # plt.draw()
       # secx, secy = plt.ginput(1)[0]
       # ax.axhline(y=secy)

        #difference = refy - secy
        #return combined_rug
        
    def cut_wavelengths_spectrum(wavelengths, spectrum, wlmin, wlmax, fill=np.nan):
        wlmin_idx, _ = RugTools.find_nearest(wavelengths, wlmin)
        wlmax_idx, _ = RugTools.find_nearest(wavelengths, wlmax)
        print(wlmin_idx, wlmax_idx)
        spectrum[wlmin_idx:wlmax_idx] = fill
        return 
    
    def file_sort(filename):

        Log = pd.read_excel(filename, sheet_name='Log', skiprows=2)

        filenames = Log['Filename']

        list_of_files = []


        for f in filenames:
            splitted = f.split('_')
            list_of_files.append(splitted)

        allfiles = []

        matchedSHG = []
        matchedFUN = []


        listofstrings = []
        for file in filenames:
            splitting = file.split('_')
            newstring = ' '.join(map(str, splitting))
            listofstrings.append(newstring)
        #removes the underscores and adds gaps instead

        FUN_testing = 'FUN'
        for file in listofstrings:
            if ('FUN' in file):
                matchedFUN.append(file)
        #collates all the files with FUN probe and a wavelength    


        for file in listofstrings:
            if ('SHG' in file):
                matchedSHG.append(file)
        #collates all the files with SHG probe and a wavelength 

        newSHG_list = []
        newFUN_list = []

        for strings in matchedSHG:
            s = strings.replace(' ', '_')
            newSHG_list.append(s)

        for strings in matchedFUN:
            s = strings.replace(' ', '_')
            newFUN_list.append(s)

        SHGlist_spaces = []
        FUNlist_spaces = []

        for s in newSHG_list:
            splitted = s.split('_')
            SHGlist_spaces.append(splitted)

        for f in newFUN_list:
            splitted = f.split('_')
            FUNlist_spaces.append(splitted)

        every_single_waveSHG = []
        every_single_waveFUN = []
        re.containsZ = re.compile(r'.*nm.*')

        #finds all the possible wavelengths in SHG files
        for file in SHGlist_spaces:
            for words in file:
                if re.containsZ.match(words):
                    #print(re.containsZ)
                    every_single_waveSHG.append(words)

        #finds all the possible wavelengths in FUN files
        for file2 in FUNlist_spaces:
            for words in file2:
                if re.containsZ.match(words):
                    every_single_waveFUN.append(words)

        wavesSHG = []
        wavesFUN = []

        #Getting rid of the duplicates
        for wave in every_single_waveSHG:
            if wave not in wavesSHG:
                wavesSHG.append(wave)

        #Getting rid of the duplicates
        for wave in every_single_waveFUN:
            if wave not in wavesFUN:
                wavesFUN.append(wave)

        matchedSHG = {w: [] for w in wavesSHG}
        matchedFUN = {w: [] for w in wavesFUN}

        #Making the dictionaries with wavelengths as keys and files as items
        for wave in wavesSHG:
            for file in SHGlist_spaces:
                if (wave in file):
                    string = ""
                    for item in file:
                        string = string + item+"_"

                    string = string + "matrix"
                    matchedSHG[wave].append(string)

        for wave2 in wavesFUN:
            for file2 in FUNlist_spaces:
                if (wave2 in file2):
                    string2 = ""
                    for item2 in file2:
                        string2 = string2 + item2+"_"

                    string2 = string2 + "matrix"
                    matchedFUN[wave2].append(string2)


        #combined dictionary with outer key of the probe, inner key of the wavlength
        properdict = {'SHG' : matchedSHG,
                     'FUN' : matchedFUN}

        #*print('proper dict:', properdict)
        ###prints combined dictionary
        pprint.pprint(properdict)

        FUNkeys = matchedFUN.keys()
        FUNitems = matchedFUN.items()

        dictvalues = []
        for key, value in matchedFUN.items():
            dictvalues.append(value)
        for key2, value2 in matchedSHG.items():
            dictvalues.append(value2)

        biglist = []
        dict2 = {}

        #Rearranging the files so that the chirp file is the first element

        for files in dictvalues:
            chirpfinder2 = r'\bchirp\b'
            listofstrings = []
            solventfiles = []
            samplefiles = []
            groupedlist = []

            for f in files:
                splitting = f.split('_')
                #print(splitting)
                newstring2 = ' '.join(map(str, splitting))
                #print(newstring2)
                listofstrings.append(newstring2)

                if re.search(chirpfinder2, newstring2):
                    solventfiles.append(f)
                else:
                    samplefiles.append(f)

            if len(solventfiles) == 0:
                solventfiles.append('None')

            solvent_file = solventfiles[0]
            groupedlist = [solvent_file] + samplefiles

            #Gets rid of processed files
            cleaned_grouplist = [file for file in groupedlist if not file.endswith('processed.dat')]

            biglist.append(cleaned_grouplist)
            
        #print('Biglist', biglist)
        
        return biglist

##############################
    
class RugFits:
    """
    RugModels is a class for models that can be used to fit to - work in progress
    
    """
    def convolve(arr, kernel):
        out = np.convolve(arr, kernel, mode='full')
        return out
    
    def gaussian(x, t0g, FWHM, Ag):
        sigma = FWHM/2.35482
        A = Ag
        B = ((x-t0g)/sigma)**2
        gauss = A*np.exp(-B/2)
        return gauss
    
    def exponential(x, tau, t0e, A):
        expon = A*np.exp(-((x-t0e)/tau))
        return expon
        
    def heaviside_step(x, t0h):
        heaviside = np.zeros(x.size)
        idxmid = max(np.where(x <= t0h)[0])
        heaviside[idxmid:] = 1.0
        return heaviside
    
    
    def exp_mod_gauss(x, t0, FWHM, tau, A):

        sigma = FWHM/2.35482
        B = ((sigma**2)/(2*tau**2))-((x-t0)/tau)
        z = (((x-t0)/sigma) - (sigma/tau))/(np.sqrt(2))
        y = ((t0/sigma)+(sigma/tau))/(np.sqrt(2))
        #C = sp.special.erf(y) + sp.special.erf(z)
        C = sp.special.erfc(z)
        exp_mod_gaussian = A*np.exp(B)*C
        return exp_mod_gaussian

    def residual( params, t, y):
        t01 = params['t01']
        FWHM1 = params['FWHM1']
        tau1 = params['tau1']
        A1 = params['A1']
        t02 = params['t02']
        FWHM2 = params['FWHM2']
        tau2 = params['tau2']
        A2 = params['A2']

        res = RugFits.exp_mod_gauss(t, t01, FWHM1, tau1, A1) + RugFits.exp_mod_gauss(t, t02, FWHM2, tau2, A2) - y
        #print(res, res.shape)
        return res
        
    


class Rug:
    """
    Rug is a class that holds the TA data "rug" and associated metadata and functions
    
    """

    def __init__(self, fname=None, extension=None):
        """
        Initialise the matrix from file and stores data with wavelength axis, time axis and also the metadata.
        
        Parameters
        ----------
        
            fname : string
                The file name
            extension: string
                Extension of the file

        """
        if fname and extension:
            self.wavelengths, self.times, self.matrix = RugTools.read_harpia_matrix(fname, extension)
            self.filename = os.path.basename(fname)
        
        #############
        
            self.metadata = RugTools.get_metadata(self.filename)
            self.sample = self.metadata['Sample']
            self.concentration = self.metadata['Concentration']
            self.pump_wavelength = self.metadata['Pump_wavelength']
            self.WLC_source = self.metadata['WLC_source']
            self.measurement = self.metadata['Measurement']
            self.run = self.metadata['Run']
            self.other = self.metadata['Other']
        else:
            self.wavelengths = None
            self.times = None
            self.filename = None
            self.matrix = None
            
        
        
        
        #############

        #consistent filename convention: split with underscores. define a function to split it up
        #and get the relevant bits, so the metadata is stored so it can be correlated with the dispersion fit
        


 
    def get_t0_positions(self, degree):
        """
        Allows the user to clicks points on the graph and stores the callibration points for dispersion correction
        
        Parameters
        ----------
            degree : interger
                Number of terms in the line equation which defines the line.

        """
        if self.axis:
            self.axis.set_title(f"Click t0 at {degree+1} or more points, then press enter")
            plt.draw()
            self.calib_points = plt.ginput(n=-1, timeout=-1, show_clicks=True)
            #print(self.calib_points)
        else:
            raise Exception("need to have plotted the data to enable t0 point selection, run .peek() first")
        return

    def fit_dispersion_curve(self, degree):
        """
        Produces coefficents from the calibration points
        
        Parameters
        ----------
            degree : interger
                Number of terms in the line equation which defines the line.
        
        """
        wavelengths = [point[0] for point in self.calib_points]
        times = [point[1] for point in self.calib_points]
        fit = pn.Polynomial.fit(wavelengths, times, deg=degree)
        coefs = fit.convert().coef
        self.corrected_time = -pn.polyval(self.wavelengths, coefs)
        # corrected_time are the time corrections we need to use
        self.dispersion_coefs = coefs
        
        return 


    def peek(self, ax=None, cmap='PuOr', min_max=None,aspect='equal',
             interpolation='none',norm=None,scale='log',
             title=None,colourbar=False, xlabel=True, 
             ylabel=True, yticks=True, xticks=True, show=False,
             plot_dispersion=False, plotted_matrix='matrix'):
        """
        Plots a figure of time delay vs wavelengths
        
        Parameters
        ----------
            ax : AxesSubplot, optional
                Axis to plot the figure on.
            cmap : string
                Colourmap of the figure.
            min_max : list
                List containing the max and min of the matrix.
            aspect : string
                Aspect of the axis scaling. Allowable values are 'auto' and 'equal'.
            interpolation : string
                The interpolation parameter determines how the image is displayed or rendered on the screen.
            scale : string
                The scale of the axis.
            title : string
                Title of the graph obtained.
            colourbar : boolean, optional
                Colourbar of the colourmap.
            xlabel : boolean
                Choice of x-axis label being displayed.
            ylabel : boolean 
                Choice of y-axis label being displayed.
            yticks : boolean
                Choice of y-axis ticks being displayed.
            xticks : boolean
                Choice of x-axis ticks being displayed.
            show : boolean
                Displays the figure.
            raw : boolean
                Displays raw figure.
            plot_dispersion : boolean
                Applies dispersion on the figure.
                
        
        """
        if not ax:
            fig = plt.figure(figsize=(6,6))
            ax0 = fig.gca()
            self.axis = ax0
        else:
            ax0 = ax
            self.axis = ax0
        
        if min_max:
            vmin = min_max[0]
            vmax = min_max[1]
        else:
            vmin = np.nanmin(self.matrix)
            vmax = np.nanmax(self.matrix)
            
        title = self.filename
        title = self.filename

        #print(vmin, vmax)
        if plotted_matrix == 'matrix':
            nanmask = np.ma.masked_where(np.isnan(self.matrix), self.matrix)
            im = ax0.pcolormesh(self.wavelengths, self.times, nanmask,
            cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        elif plotted_matrix == 'chirped':
            nanmask = np.ma.masked_where(np.isnan(self.raw_matrix), self.raw_matrix)
            im = ax0.pcolormesh(self.wavelengths, self.times, nanmask,
            cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        elif plotted_matrix == 'unsubtracted':
            nanmask = np.ma.masked_where(np.isnan(self.unsubtracted_matrix), self.unsubtracted_matrix)
            im = ax0.pcolormesh(self.wavelengths, self.times, nanmask,
            cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        elif plotted_matrix == 'uncut':
            nanmask = np.ma.masked_where(np.isnan(self.uncut_matrix), self.uncut_matrix)
            im = ax0.pcolormesh(self.wavelengths, self.times, nanmask,
            cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
            
        if plot_dispersion:
            if hasattr(self, "corrected_time"):
                pass
            elif hasattr(self, "dispersion_coefs"):
                self.corrected_time = -pn.polyval(self.wavelengths, self.dispersion_coefs)
            else:
                raise Exception("can't plot dispersion correction curve when no correction given.")

            self.axis.plot(self.wavelengths, -self.corrected_time, color='k')
         
        if scale == 'log':
            ax0.set_yscale('symlog')
        elif scale == 'linear':
            ax0.set_yscale('linear')
        else:
            raise NameError

        if title:
            ax0.set_title(title)

        if colourbar:
            if not ax:
                #fig = .figure(figsize=(7,6))
                fig = plt.gcf()
                fig.set_size_inches(8, 7)
                colorb = fig.colorbar(im, ticks=[vmin, 0, vmax], shrink=1)
                colorb.set_label('ΔOD')
            else:
                raise NameError
        if xlabel:
            ax0.set_xlabel('Wavelength [nm]')
        if ylabel:
            ax0.set_ylabel('Time delay [ps]')

        if not yticks:
            ax0.tick_params(axis="y", left=False, labelleft=False)
        elif yticks == True:
            pass
        else:
            ax0.set_yticks(yticks)
            ax0.set_yticklabels([str(i) for i in yticks]) 
        
        if not xticks:
            ax0.tick_params(axis="x", bottom=False, labelbottom=False)
        elif xticks == True:
            pass
        else:
            ax0.set_xticks(xticks)
            ax0.set_xticklabels([str(i) for i in xticks])
        if show:
            plt.show()
        return 

    def peek_3D(self, cmap='inferno', stride=10):
        """
        Displays a 3D figure
        
        Parameters
        ----------
            cmap : string
                Colourmap of the figure.
            stride : integer
                Resolution of the 3D plot.
        
        """
        
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d', box_aspect=(1,1,1))

        wavelength_3d, times_3d = np.meshgrid(self.wavelengths, self.times)
        ax.plot_surface(wavelength_3d, times_3d, self.matrix, cmap=cmap, vmin=-10, vmax=10,
                       rstride=stride, cstride=stride, antialiased=False)
        ax.view_init(90,0,90)
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Time delay [ps]')
        ax.set_zlabel(r'$\Delta mOD$')
        ax.set_aspect('equalxy')

    ###########################################################

    
    def peekexplore(self, cmap='PuOr', min_max=None, aspect='equal', interpolation='none',
                norm=None,scale='log', title=None, colourbar=False, xlabel=True, ylabel=True,
                yticks=True, xticks=True, show=False, raw=False, plot_dispersion=False):
        """
        A widget displaying a timeslice and a wavelength slice alongside a 'peek' plot.
        
        The user can interact with the widget.
        
        Parameters
        ----------
            cmap : string
                Colourmap of the figure.
            min_max : list
                List containing the max and min of the matrix.
            aspect : string
                Aspect of the axis scaling. Allowable values are 'auto' and 'equal'.
            interpolation : string
                The interpolation parameter determines how the image is displayed or rendered on the screen.
            scale : string
                The scale of the axis.
            title : string
                Title of the graph obtained.
            colourbar : boolean, optional
                Colourbar of the colourmap.
            xlabel : boolean
                Choice of x-axis label being displayed.
            ylabel : boolean 
                Choice of y-axis label being displayed.
            yticks : boolean
                Choice of y-axis ticks being displayed.
            xticks : boolean
                Choice of x-axis ticks being displayed.
            show : boolean
                Displays the figure.
            raw : boolean
                Displays raw figure.
            plot_dispersion : boolean
                Applies dispersion on the figure.
        
        """
        timeslice = self.get_time_trace(450)
        wlslice_g = self.get_wavelength_trace(0)
        
        # Define initial parameters
        W_init = self.wavelengths[0] 
        T_init = self.times[0]
        
        fig = plt.figure(figsize=(10,10))
        grid = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], 
                        wspace=0.3, hspace=0.3)
        ax1 = fig.add_subplot(grid[0]) #rug
        ax2 = fig.add_subplot(grid[1]) #decay slice
        ax3 = fig.add_subplot(grid[2]) #spectrum slice
    
        ax1.set_ylabel('Time delay / ps')
        ax1.set_xlabel('Wavelength / nm')
        ax1.set_yscale('symlog')
        line_vertical = ax1.axvline(x=W_init, color= 'red',
                                    label= 'axvline - full height')
        line_horizontal = ax1.axhline(y=T_init, color='red', 
                                      label= 'axhline - full height')
            
        line2, = ax2.plot(self.times, timeslice, color='hotpink')
        ax2.set_xscale("symlog")
        ax2.set_xlabel('Time / s')
        ax2.set_ylabel('ΔA')
        ax2.set_title('Timeslice')
            
        line3, = ax3.plot(self.wavelengths, wlslice_g, color='hotpink')
        ax3.set_xlabel('Wavelength / λ')
        ax3.set_ylabel('ΔA')
        ax3.set_title('Wavelength slice')

        title = self.filename
        
        self.peek(ax=ax1)
        # Sliders
        axtime = plt.axes([0.60, 0.40, 0.30, 0.02])
        time_slider = Slider(
            ax=axtime,
            label='Wavelength',
            valmin=self.wavelengths[0],
            valmax=self.wavelengths[-1],
            valinit=self.wavelengths[0],
            color='hotpink'
        )
        
        axwave = plt.axes([0.60, 0.35, 0.30, 0.02])
        wavelength_slider = Slider(
            ax=axwave,
            label='Time',
            valmin=self.times[0],
            valmax=self.times[-1],
            valinit=self.times[0],
            color='hotpink'
        )
        
        def update(val):
            """
            Updates the graphs when the user interacts with the widget.
            
            """
            
            time_ydata = self.get_time_trace(np.log(time_slider.val))
            line2.set_ydata(time_ydata)
            try:
                ax2.set_ylim(np.nanmin(time_ydata)-1.5, np.nanmax(time_ydata)+1.5)
            except:
                pass
                
            wl_ydata = self.get_wavelength_trace(wavelength_slider.val)
            line3.set_ydata(wl_ydata)
            try:
                ax3.set_ylim(np.nanmin(wl_ydata)-2, np.nanmax(wl_ydata)+2)
            except:
                pass
                
            fig.canvas.draw_idle()
        
            ax1.axvline(x=time_slider.val, color= 'red', 
                        label= 'axvline - full height')
            ax1.lines[0].remove()
        
            ax1.axhline(y=wavelength_slider.val, color='red',
                        label= 'axhline - full height')
            ax1.lines[0].remove()
            
            
        time_slider.on_changed(update)
        wavelength_slider.on_changed(update)
        
        # Buttons
        resetax = plt.axes([0.1, 0.925, 0.1, 0.04])
        buttonax = plt.axes([0.25, 0.925, 0.1, 0.04])
        annotateax = plt.axes([0.40, 0.925, 0.1, 0.04])
        button = Button(resetax, 'Reset', color='pink', hovercolor='0.9')
        button2 = Button(buttonax, 'Colour', color='pink', hovercolor='0.9')
        button3 = Button(annotateax, 'Annotate', color='pink', hovercolor='0.9')

        def colour_change(event):
            """
            Randomly changes the colour of the line graphs.
            
            """
            color = '#'+"%06x" % random.randint(0, 0xFFFFFF)
            line2.set_color(color)
            line3.set_color(color)
            
            fig.canvas.draw_idle()
            
        def annotate(event):
            """
            Click a point to annotate.
            
            """
            mplcursors.cursor(line2)
            mplcursors.cursor(line3)
        
        def reset(event):
            """
            Pressing reset button reverts all graphs back to original presets.
            
            """
            time_slider.reset()
            wavelength_slider.reset()
            line2.set_color('hotpink')
            line3.set_color('hotpink')
            
        button.on_clicked(reset)
        button2.on_clicked(colour_change)
        button3.on_clicked(annotate)

        resetax._button = button
        buttonax._button = button2
        annotateax._button = button3
        
        plt.show()

        return 


    #############################################################
 

    def get_row_background(self):
        """
        Analyses the  horizontal row of the matrix at the coordinate clicked by the user. The average value from each row is calulated and is then subtracted from the entire matrix across all the rows vertically.
        """
        if self.axis:
            self.axis.set_title(f"Click pre-t0 at 1 point, then press enter")
            plt.draw()
            self.scalar_point = plt.ginput(n=-1, timeout=-1, show_clicks=True)
            coord = self.scalar_point[0]


            ypoint_real = coord[1]
       

            yclick_index, point2_idx = RugTools.find_nearest(self.times, ypoint_real)
          
            row_matrix = len(self.times) - yclick_index
         

            subtract = self.matrix[yclick_index, :]
            
            subtract_matrix = np.zeros_like(self.matrix) #returns an array of zeros with the same shape as data matrix
            for i in range(len(self.times)):
                subtract_matrix[i, :] = subtract_matrix[i, :] + subtract

            self.bg_matrix = self.matrix.copy()

            matrix = self.matrix - subtract_matrix
            self.matrix = matrix


            fig = plt.figure(figsize=(7,6))
            ax0 = fig.gca()
            self.axis = ax0
            vmin = np.min(matrix)
            vmax = np.max(matrix)
            ax0.set_yscale('symlog')
            ax0.set_xlabel('Wavelength [nm]')
            ax0.set_ylabel('Time delay [ps]')

            im = ax0.pcolormesh(self.wavelengths, self.times, matrix,
            cmap='PuOr', norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
            
            fig.colorbar(im, ax=ax0)

        else:
            raise Exception("need to have plotted the data to enable point selection, run .peek() first")
        return

    def get_background_correction_row(self):
        self.peek(show=False)
        self.get_row_background()
        return


    def background_subtract(self):
        """
        User defines an area for background removal by clicking two lines on the plot. An average value is calulated from the defined area and is then removed from the entire matrix. 
        
        """
        times = []
        count = 0
        if not hasattr(self, 'ax'):
            self.peek(show=False)
            
        while count < 2:
            self.axis.set_title(f"Click 2 lines to define area for background removal")
            plt.draw()
            pts=plt.ginput(1)
            time=pts[0][1]  
            times.append(time)
            self.axis.axhline(y=time)
            count += 1
            plt.pause(0.05)
        else:
            print('\nTaken average background')
          

        maximum = np.max(times)
        minimum = np.min(times)
        
        max_index, actualmax = RugTools.find_nearest(self.times, maximum)
        min_index, actualmin = RugTools.find_nearest(self.times, minimum)
      
        area_indexes = []
        for i in range(min_index, max_index+1):
            area_indexes.append(i)

  
        background_region = self.matrix[[area_indexes], :].T


        mean_background = np.mean(background_region, axis =1)
          
        background_matrix = np.zeros_like(self.matrix)
        
        background_matrix = np.stack([mean_background for i in range(len(self.times))], axis=0)[:,:,0]
        print(self.matrix.shape, background_matrix.shape)
        
        if not hasattr(self, 'unsubtracted_matrix'):
            self.unsubtracted_matrix = self.matrix.copy()

        matrix = self.matrix - background_matrix
        self.matrix = matrix
        self.peek(plotted_matrix='unsubtracted')
        self.peek()
        return

    def get_background_correction_average(self):
        self.peek(show=False)
        self.get_average_background()
        return
    
    def savefile(self, coefc, dispersion, offset):
        """
        Saves processed data as a new file which includes its metadata, date, time and all the processing applied to it.
        
        Parameters
        ----------
            coefc : string
                Describes how the dispersion correction was calculated. Either by eye, user input or not calcuated at all.
            dispersion : string
                Describes how the dispersion correction was applied. Either by row or by removing an average value from the whole matrix.
            offset : string
                Describes whether an offset was applied to the plot.
        
        """
        empty = np.array([])
        lent = len(self.times)
        lenm = len(self.wavelengths)

        array = np.zeros((lent+1, lenm+1))

        array_wl = self.wavelengths
        array_times = self.times[:,None].reshape((lent,))
        array_matrix = self.matrix

        array[0,1:] = array_wl
        array[1:,0] = array_times
        array[1:, 1:] = array_matrix

        metadata = str(self.metadata)
        today = date.today()

        current_time = time.localtime()
        timestring = time.strftime("%d/%m/%Y %H:%M:%S")
      
        file = np.savetxt(self.filename+'_processed'+'.dat', array,
                          header='XAxisTitleWavelength(nm)\nYAxisTitle Delay (ps)\n'+metadata+'\n'+'Date&time: '+timestring+'\n'+coefc+'\n'+dispersion+'\n'+offset, comments='')
        print("\033[1m" + 'Saved processed file of' + "\033[0;0m", self.filename)
       
        fig = self.peek(colourbar=True)
        plt.savefig(self.filename+"_processed.png")
        #plt.savefig('myfig')
       
        return
    
    
    
    def offset(self, coef):
        """
        Calculates the offset and adjusts the figure accordingly
        
        Parameters
        ----------
            coef : np.ndarray
                Either inputed by the user or obtained from the get_t0_positions function.
        
        """
        times = []
        count = 0
        
        if not hasattr(self, "axis"):
            self.peek()

        while count < 1:
  
                
            self.axis.set_title('Indicate where baseline is')
            plt.draw()
            pts=plt.ginput(1)
            time=pts[0][1]  
            times.append(time)
            self.axis.axhline(y=time)
            count += 1
            plt.pause(0.05)


        offset = coef - times
        #print(offset)
        newtimes = self.times + offset
        self.times = newtimes

        fig = plt.figure(figsize=(6,6))
        ax0 = fig.gca()
        self.axis = ax0
        vmin = np.min(self.matrix)
        vmax = np.max(self.matrix)
        ax0.set_yscale('symlog')

        im = ax0.pcolormesh(self.wavelengths, self.times, self.matrix,
        cmap='PuOr', norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        title = self.filename
        ax0.set_xlabel('Wavelength [nm]')
        ax0.set_ylabel('Time delay [ps]')
        ax0.set_title(title)
        print('Offset corrected')
   

     
    
    def MatplotlibClearMemory(self):
        """
        Clears all Matplotib memory.
        
        """
        allfignums = plt.get_fignums()
        for i in allfignums:
            fig = plt.figure(i)
            fig.clear()
            plt.close(fig)
        return


    def processing(self, solventdata):
        """
        Basic processing function which only requires the user to define the data file to process. Solvent data is obtained through the findfiles function which is called here. Dispersion correction is calculated by eye here.

        """
        self.peek(show=False)
       
        solventdata.get_dispersion_correction_eye()
        coefc = 'Dispersion correction by eye'
        scoefs = solventdata.dispersion_coefs
        self.apply_dispersion_correction(coefs=scoefs)
        dispersion = ''
        self.get_background_correction_average()
        self.peek()
        self.offset(coef=scoefs[-1])
        offset = 'Offset added'
        self.MatplotlibClearMemory()
        self.savefile(coefc, dispersion, offset)
        return
    
        
    def auto_processing(files, background='area', correct_dispersion='eye', offset=True):
        
        """
        Contains functions which can apply dispersion correction, apply an offset to the peak and resave the entire file as a processed file.
        
        Parameters
        ----------
            files : list
                List of all the files containing the specific wavelength and probe requested by the user.
            
            background: string
                Determines what sort of dispersion correction is conducted. Allowable values are 'area' and 'row'. The deafault is 'area'.
         
            correct_dispersion: string
                It figures out whether the dispersion coeffients are calculated by eye by the user or entered in as an input. Allowable values are 'eye' and 'coefs'. The deafault is 'eye'.
            
            offset : boolean
                If true then the user performs an offset on the graph and offset correction is applied. The deafault is 'True'.

        """
       
        allowed_methods = ('row', 'area')
        
        allowed_dispersion = ('eye', 'coefs')
        
        if background not in allowed_methods:
            raise ValueError('Invalid background removal method - allowed methods are "row" and "area"')
            
        if correct_dispersion not in allowed_dispersion:
            raise ValueError('Invalid dispersion method - allowed methods are "eye" and "coefs"')
            
        solventfile = files[0][:-4]
        solventdata = Rug(solventfile, ".dat")
        #solventdata.peek()
        print('type of solventdata:', type(solventdata), solventdata)
        files.pop(0)
        
        
        
        if correct_dispersion == 'eye':
            #solventdata.peek(show=False)
            solventdata.get_dispersion_correction_eye()
            scoefs = solventdata.dispersion_coefs
            coefc = 'Coefs calculated by eye'
                
        elif correct_dispersion == 'coefs':
            usercoefs = str(input('Enter coef:'))
            arraycoefs = np.array(usercoefs, ndmin=1)
            string = ""
            for elem in arraycoefs:
                string += elem
            listofcoefs = list(map(float, string.split(',')))
            scoefs = np.asarray(listofcoefs)
            coefc = 'Coefs entered by user'
            
        else:
            dispersionc = 'No dispersion correction applied'
            print('No dispersion correction applied')     
                
        for file in files:
            file = file[:-4]
            sampledata = Rug(file, ".dat")
            
            sampledata.apply_dispersion_correction(coefs=scoefs)
            
            if background == 'row':
                sampledata.get_background_correction_row()
                dispersionc = 'Dispersion correction by row applied'
            if background == 'area':
                sampledata.get_background_correction_average()
                dispersionc = 'Dispersion correction average applied'
                
            sampledata.peek()
            
            if offset == True:
                sampledata.offset(scoefs[-1])
                offsetc = 'Offset applied'
                print('Offset applied')
                sampledata.savefile(coefc, dispersionc, offsetc)
                sampledata.MatplotlibClearMemory()
                
            else:
                offsetc = 'No offset applied'
                print('No offset applied')
                sampledata.savefile(coefc, dispersionc, offsetc)
                sampledata.MatplotlibClearMemory()
                
            #sys.exit()
                
                #sampledata.MatplotlibClearMemory()
        print("\033[1m" + '\nProcessing completed' + "\033[0;0m")
        return
    
    def batch_processing(filename, background='area', correct_dispersion='eye', offset=True):
        """
        Batch processing which contains functions which can apply dispersion correction, apply an offset to the peak and resave the entire file as a processed file.
        
        Parameters
        ----------
            filename: string 
                Name of the file of the excel sheet containing the names of the files.
            
            background: string
                Determines what sort of dispersion correction is conducted. Allowable values are 'area' and 'row'. The deafault is 'area'.
         
            correct_dispersion: string
                It figures out whether the dispersion coeffients are calculated by eye by the user or entered in as an input. Allowable values are 'eye' and 'coefs'. The deafault is 'eye'.
            
            offset : boolean
                If true then the user performs an offset on the graph and offset correction is applied. The deafault is 'True'.

        """
        
        allowed_methods = ('row', 'area')
        allowed_dispersion = ('eye', 'coefs')
        
        if background not in allowed_methods:
            raise ValueError('Invalid background removal method - allowed methods are "row" and "area"')
        
        if correct_dispersion not in allowed_dispersion:
            raise ValueError('Invalid dispersion method - allowed methods are "eye" and "coefs"')
       
        batch = filename.copy()
        newbatch = filename.copy()
       
        #Excludes the chirpless files from the processing
        count = 0
        removalcount = 0
        
        for i in batch:
            count += 1
            if i[0] == 'None':
                removalcount += 1
                print('\nRemoved:', i, 'Number', removalcount)
                newbatch.remove(i)
                
        print('\nLen of old batch:', count)
        print('\nHow many lists removed:', removalcount)
        print('\nLen of the newbatch:', len(newbatch))
        #sys.exit()
        
        
        cleanedbatch = newbatch
       
        
        #allfiles = glob.glob('*.dat')
        allfiles = []
        
        with ZipFile('aarhus_TA_jan24.zip', 'r') as zipObj:
            for entry in zipObj.infolist():
                allfiles.append(entry.filename)
        
        for groupedfiles in cleanedbatch:
            
            solventfile = groupedfiles[0]
            groupedfiles.remove(solventfile)
            samplefiles = groupedfiles
            print('Solvent file:', solventfile)
            print('Sample files:', samplefiles)
            
        
            for files in allfiles:
                if solventfile in files:
                    solvent = files
                    solvent = solvent[:-4]
                    exactsolvent = Rug(solvent, ".dat")

                    if correct_dispersion == 'eye':
                        exactsolvent.get_dispersion_correction_eye()
                        scoefs = exactsolvent.dispersion_coefs
                        coefc = 'Coefs calculated by eye'

                    elif correct_dispersion == 'coefs':
                        usercoefs = str(input('Enter coef:'))
                        arraycoefs = np.array(usercoefs, ndmin=1)
                        string = ""
                        for elem in arraycoefs:
                            string += elem
                        listofcoefs = list(map(float, string.split(',')))
                        scoefs = np.asarray(listofcoefs)
                        coefc = 'Coefs entered by user'

                    else:
                        dispersionc = 'No dispersion correction applied'
                        print('No dispersion correction applied')   
                        
                for sample in samplefiles:
                    if sample in files:
                        exact = files[:-4]
                        samplefile = Rug(exact, ".dat")
            
                        samplefile.apply_dispersion_correction(coefs=scoefs)

                        if background == 'row':
                            samplefile.get_background_correction_row()
                            dispersionc = 'Dispersion correction by row applied'
                        if background == 'area':
                            samplefile.get_background_correction_average()
                            dispersionc = 'Dispersion correction average applied'

                        samplefile.peek()

                        if offset == True:
                            samplefile.offset(scoefs[-1])
                            offsetc = 'Offset applied'
                            print('Offset applied')
                            samplefile.savefile(coefc, dispersionc, offsetc)
                            samplefile.MatplotlibClearMemory()

                        else:
                            offsetc = 'No offset applied'
                            print('No offset applied')
                            samplefile.savefile(coefc, dispersionc, offsetc)
                            samplefile.MatplotlibClearMemory()

        
        print("\033[1m" + '\nBatch Processing completed' + "\033[0;0m")
        
        
        return
    

    def compute_SVD(self, threshold=100, tol=1.0E-10):

        try:

            U, S, V = np.linalg.svd(self.matrix)
            S[np.abs(S) < tol] = 0
        except:

            raise Exception('SVD did not converge - correct data loaded?')



        self.singular_values = S

        self.principal_spectra = V

        self.principal_kinetics = U.T

        singular_threshold = np.min(S[np.nonzero(S)])*threshold
        print(f'Taking threshold for singular value significance as {threshold}x larger than smallest singular value ({np.round(singular_threshold, 9)}).')
        singular_differences = np.abs(np.ediff1d(self.singular_values))
        Stotal = np.sum(self.singular_values**2)
        self.singular_fractions = self.singular_values**2 / Stotal
        self.relevant_sing = np.array([ i for i in singular_differences if i > singular_threshold])
        print(f'Found {len(self.relevant_sing)} significant singular values')

        cpt_sum = 0
        for i in range(len(self.relevant_sing)):
            cpt_sum = cpt_sum + np.round(self.singular_fractions[i]*100,2)
            print(f'{np.round(self.singular_fractions[i]*100,2)} % of data variance described by component {i+1}.')

        print(f'Overall, {np.round(cpt_sum,2)} % of data variance described by the first {len(self.relevant_sing)} components.')
        print(f'Leaving {np.round((100-cpt_sum),2)} % of data variance described by the the remaining {np.round(len(self.singular_values) -len(self.relevant_sing),2)} components which are below threshold (roughly {np.round((100-cpt_sum)/(len(self.singular_values) -len(self.relevant_sing)),2)} % per component).')

        
        self.relevant_spectra = self.principal_spectra[0:len(self.relevant_sing)]
        self.relevant_kinetics = self.principal_kinetics[0:len(self.relevant_sing)]

        self.relevant_matrices = np.array([ self.relevant_sing[i]*(np.outer(self.relevant_kinetics[i],self.relevant_spectra[i])) for i in range(len(self.relevant_sing))])

        self.reconstructed_TA = np.sum(self.relevant_matrices, axis=0)



        
        return

    def plot_SVD(self):
        if hasattr(self, 'relevant_sing'):
            plt.figure()
        pass

    def SVD_explorer(self):

        fig = plt.figure(figsize=(10,10))
        grid = GridSpec(2, 2, width_ratios=[3, 3], 
                        height_ratios=[3, 3], wspace=0.3, hspace=0.3)
        ax1 = fig.add_subplot(grid[0])
        ax1.set_ylabel('Time delay (ps)')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_yscale('symlog')
        ax1.set_title(self.filename)
       

        ax2 = fig.add_subplot(grid[1])
        ax2.set_ylabel('Amplitude (a.u)')
        ax2.set_xlabel('Time delay (ps)')


        ax3 = fig.add_subplot(grid[2])
        ax3.set_ylabel('Amplitude (a.u)')
        ax3.set_xlabel('Wavelength (nm) ')

        ax4 = fig.add_subplot(grid[3])
        ax4.set_ylabel('Amplitude (a.u)')
        ax4.set_xlabel('Componenet number')

        ax2.plot(self.times, self.principal_kinetics[0], label='1', color='tab:blue')
        ax2.plot(self.times, self.principal_kinetics[1], label='2', color='orange')
        ax2.plot(self.times, self.principal_kinetics[2], label='3', color='green')
                
        ax3.plot(self.wavelengths, self.principal_spectra[0])
        ax3.plot(self.wavelengths, self.principal_spectra[1])
        ax3.plot(self.wavelengths, self.principal_spectra[2])


        h, l= ax2.get_legend_handles_labels()
        ax2.legend(h, l)
        
        resetax = plt.axes([0.1, 0.925, 0.1, 0.04])
        button = Button(resetax, 'Reset', color='#B0FC38', hovercolor='0.9')

        ax4.scatter(range(0, len(self.singular_values)), self.singular_values)
        vmin = np.min(self.matrix)
        vmax = np.max(self.matrix)
        ax1.set_yscale('symlog')
        im = ax1.pcolormesh(self.wavelengths, self.times, self.matrix,
                cmap='PuOr', norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))

        plt.show()

        self.scalar_point = plt.ginput(n=-1, timeout=-1, show_clicks=True)
        coord = self.scalar_point
        
        indexes = []
        
        for i in coord:
            real = i[1]
            index, value = RugTools.find_nearest(self.singular_values, real)
            indexes.append(index)

        print(indexes)
        
        print('Cleared worked')
        ax2.clear()
        ax3.clear()
        plt.draw()
        
        for i in indexes:
            line1, = ax2.plot(self.times, self.principal_kinetics[i], label=i+1 )
            line2, = ax3.plot(self.wavelengths, self.principal_spectra[i])
            ax2.set_ylabel('Amplitude (a.u)')
            ax2.set_xlabel('Time delay (ps)')
            ax3.set_ylabel('Amplitude (a.u)')
            ax3.set_xlabel('Wavelength (nm) ')
            h, l= ax2.get_legend_handles_labels()
            ax2.legend(h, l)
            
            if i == 0:
                line1.set_color('tab:blue')
                line2.set_color('tab:blue')
            if i == 1:
                line1.set_color('orange')
                line2.set_color('orange')
            if i == 2:
                line1.set_color('green')
                line2.set_color('green')
            if i == 3:
                line1.set_color('red')
                line2.set_color('red')

        def update(self):
            indexes = []
            indexes.clear()
            self.scalar_point = plt.ginput(n=-1, timeout=-1, show_clicks=True)
            coord = self.scalar_point

          

            for i in coord:
                real = i[1]
                index, value = RugTools.find_nearest(self.singular_values, real)
                indexes.append(index)

            print(indexes)

            print('Cleared worked')
            ax2.clear()
            ax3.clear()
            plt.draw()
            ax2.set_ylabel('Amplitude (a.u)')
            ax2.set_xlabel('Time delay (ps)')
            ax3.set_ylabel('Amplitude (a.u)')
            ax3.set_xlabel('Wavelength (nm) ')

            for i in indexes:
                line1, = ax2.plot(self.times, self.principal_kinetics[i], label=i+1 )
                line2, = ax3.plot(self.wavelengths, self.principal_spectra[i], label=i+1)
                ax2.set_ylabel('Amplitude (a.u)')
                ax2.set_xlabel('Time delay (ps)')
                ax3.set_ylabel('Amplitude (a.u)')
                ax3.set_xlabel('Wavelength (nm) ')
                h, l= ax2.get_legend_handles_labels()
                ax2.legend(h, l)
                
                if i == 0:
                    line1.set_color('tab:blue')
                    line2.set_color('tab:blue')
                if i == 1:
                    line1.set_color('orange')
                    line2.set_color('orange')
                if i == 2:
                    line1.set_color('green')
                    line2.set_color('green')
                if i == 3:
                    line1.set_color('red')
                    line2.set_color('red')
          

        
        
        def reset(event):
            ax2.clear()
            ax3.clear()
            plt.draw()  
            ax2.plot(self.times, self.principal_kinetics[0], label='1', color='tab:blue')
            ax2.plot(self.times, self.principal_kinetics[1], label='2', color='orange')
            ax2.plot(self.times, self.principal_kinetics[2], label='3', color='green')

            ax3.plot(self.wavelengths, self.principal_spectra[0])
            ax3.plot(self.wavelengths, self.principal_spectra[1])
            ax3.plot(self.wavelengths, self.principal_spectra[2])
            ax2.set_ylabel('Amplitude (a.u)')
            ax2.set_xlabel('Time delay (ps)')
            ax3.set_ylabel('Amplitude (a.u)')
            ax3.set_xlabel('Wavelength (nm) ')
            h, l= ax2.get_legend_handles_labels()
            ax2.legend(h, l)
            
            update(self)
            h, l= ax2.get_legend_handles_labels()
            ax2.legend(h, l)
            
                    
        button.on_clicked(reset)
        resetax._button = button
        plt.show()

        return

    ###############################################################
 
    def apply_dispersion_correction(self, coefs=[None]):
        """
        Applies a dispersion correction to the figure depending on what the user has selected.
        
        Parameters
        ----------
        
            Coefs : np.ndarray
                Dispersion coefficeints inputed by the user. If not given it checks to see if any coefficents have been saved from the get_t0_positions function.
                
        

        """
        
        #JDP if user inputs coefficients, prioritise using these
        if all(coefs):
            #JDP note that you need to use polyval from the Polynomial class (pn), not base numpy...
            #JDP need to edit so that it will enter this if block if there is an array coefs.
            coefs = np.asarray(coefs)
            self.corrected_time = -pn.polyval(self.wavelengths, coefs)
            self.dispersion_coefs = coefs #save them
            #print(coefs)
        #JDP otherwise, if the coefs have not been given, check if they exist from an "by eye" fit and use that
        
        
        elif hasattr(self, "dispersion_coefs"):
            self.corrected_time = -pn.polyval(self.wavelengths, self.dispersion_coefs)
        else:
            raise Exception("need to find the dispersion correction before applying it, obvs")

        interp_min = np.min(self.times)+np.min(self.corrected_time)
        interp_max = np.max(self.times)+np.max(self.corrected_time)
        

        #JDP interpolate the time axis so we can apply the chirp correction
        
        dt = np.min(np.ediff1d(self.times)) #JDP set the time resolution to the smallest step
        #print("interpolation time step is", dt)
        #print("interpolation limits are", interp_min, interp_max)
        
        # number of interpolation points is latest time - earliest time / smallest step
        samples = int((interp_max-interp_min)/dt)
        #print("number of interpolated time points is", samples)
        
        #JDP create a new axis to get the new time points on
        x = np.linspace(interp_min, interp_max, samples)

        #JDP interpolate from the old time axis to the new one
        t_interp = np.interp(x, self.times, self.times)

        #JDP save it as an attribute of the rug
        self.interpolated_time = t_interp

        #JDP get a meshgrid with the new interpolated time axis so we can evaluate interpolated TA data
        wavelength_2D, times_2D = np.meshgrid(self.wavelengths, self.interpolated_time, indexing="ij")
        
        #JDP interpolate the actual TA data from the matrix (or, create a function to do it)
        interp2D = sp.interpolate.RegularGridInterpolator((self.times, self.wavelengths), self.matrix,
                bounds_error=False, fill_value=-9999, method='nearest')

        #JDP evaluate the TA interpolation function over the whole grid
        data_interp = interp2D((times_2D, wavelength_2D)).T 
        #JDP no understanding of why this needs transposing, some subtlety with the interpolator?


        #JDP note NB pcolormesh with nearest shading centers each time point  on a grid cell, and extends the cell
        # to edges defined by the distance to the next time point. this can make it look like the plot is 
        # making fake data at different interpolation levels, but it isn't. explicitly limit the plot 
        # to the limits of the time axis to avoid confusion
    

        #JDP corrected time is the time by which each wavelength needs to be shifted
        #JDP  turn this into a number of elements to shift given the interpolation scale
        time_shifts = np.rint(self.corrected_time/dt).astype(np.int64) 
        
        #JDP get the maximum shift, as we'll pad the interpolated matrix by this to ensure we don't roll over edges
        maxshift = int(np.max(np.abs(time_shifts)))
        
        #JDP pad the matrix with an unphysical value
        matrix_padded = np.pad(data_interp, ((maxshift, maxshift), (0,0)), mode='constant',
            constant_values=-99)
        
        #JDP save the raw (uncorrected) matrix
        self.raw_matrix = self.matrix.copy()

        #JDP apply the chirp correction by rolling each wavelength's time axis by the necessary amount
        shifted_matrix = RugTools.indep_roll(data_interp, time_shifts, axis=0)


        #JDP now need to "undo" the interpolation. should just be:
        time_indices = []
        for time in self.times:
            time_idx, _ = RugTools.find_nearest(self.interpolated_time, time)
            time_indices.append(time_idx)

        self.matrix = shifted_matrix[time_indices, :]

        return 

   

    def get_dispersion_correction_eye(self, degree=3):
        """
        Corrects the plot using the dispersion coeficients determined by the get_t0_positions fuction.
        
        """
        self.peek(show=False)
        self.get_t0_positions(degree)
        self.fit_dispersion_curve(degree=degree)
        return
    
    def apply_dispersion_correction_auto(self, file):
        """
        Applies a dispersion correction to the plot using the user inputed coefficients. ##############might delete this as there's now an auto processing function ###########
        """
        self.get_dispersion_correction_eye()
        chirpcoefs = self.dispersion_coefs
        print(chirpcoefs)
        #self.apply_dispersion_correction(coefs=chirpcoefs)
        newf = self.apply_dispersion_correction(coefs=chirpcoefs)
        #newf = file.apply_dispersion_correction(coefs = chirpcoefs)
        return newf


        
    def get_time_trace(self, wavelength, plot=False):
        """
        Returns an array of the time slice at a specific wavelength
        
        Parameters
        ----------
            wavelength : float
                The wavelength
            plot : boolean
                Plot of the time slice
        
        Returns
        -----------
            time_trace : np.ndarray
                Array of the time slice
       
        
        """
        
        wavelength_idx, wavelength_real = RugTools.find_nearest(self.wavelengths, wavelength)
        time_trace = self.matrix[:, wavelength_idx]
        if plot:
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(self.times, time_trace)
            ax.set_xscale("symlog")
        return time_trace

    def get_wavelength_trace(self, time, plot=False):
        """
        Returns an array of the wavelength slice at a specific time
        
        Parameters
        ----------
            time : float
                The time
            plot : boolean
                Plot of the wavelength slice
        
        Returns
        -----------
            wavelength_trace : np.ndarray
                Array of the wavelength slice
        
        """
        delay_idx, delay_real = RugTools.find_nearest(self.times, time)
        wavelength_trace= self.matrix[delay_idx, :]
        if plot:
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(self.wavelengths, wavelength_trace)
        return wavelength_trace
    
    def fit_time_trace(self, time_trace, model, params, method='least_squares'):
        parameters = model.make_params()
        for i, param in enumerate(params):
            parameters.add(name=param[0], value=param[1], vary=param[2],
                          min=param[3], max=param[4], expr=param[6])
        parameters.pretty_print()
        result = model.fit(time_trace, params=parameters, x=self.times, method=method )
        return result

    def reset_matrix(self):
        if hasattr(self, 'uncut_matrix'):
            self.matrix = np.copy(self.uncut_matrix)
        if hasattr(self, 'uncut_times'):
            self.times = np.copy(self.uncut_times)
        if hasattr(self, 'uncut_wavelengths'):
            self.wavelengths = np.copy(self.uncut_wavelengths)
        else:
            pass
        return
        
    def limit_times(self, tmin=None, tmax=None):
        if not hasattr(self, 'uncut_matrix'):
            self.uncut_matrix = np.copy(self.matrix)
        if not hasattr(self, 'uncut_times'):
            self.uncut_times = np.copy(self.times)
            
        if tmax:
            tidx, _= RugTools.find_nearest(self.times, tmax)
            self.matrix = self.matrix[:tidx, :]
            self.times = self.times[:tidx]
        if tmin:
            tidx, _ = RugTools.find_nearest(self.times, tmin)
            self.matrix=self.matrix[tidx:, :]
            self.times = self.times[tidx:]
        return

    def cut_wavelengths(self, wlmin, wlmax, fill=0):
        wlmin_idx, _ = RugTools.find_nearest(self.wavelengths, wlmin)
        wlmax_idx, _ = RugTools.find_nearest(self.wavelengths, wlmax)

        if hasattr(self, 'uncut_matrix'):
            pass
        else:
            self.uncut_matrix = np.copy(self.matrix)
            
        self.matrix[:, wlmin_idx:wlmax_idx] = fill
        return

    
    
    
    def limit_wavelengths(self, wlmin=None, wlmax=None):
        
        if not hasattr(self, 'uncut_matrix'):
            self.uncut_matrix = np.copy(self.matrix)
        if not hasattr(self, 'uncut_wavelengths'):
            self.uncut_wavelengths = np.copy(self.wavelengths)
            
        if wlmin:
            wlix, _ = RugTools.find_nearest(self.wavelengths, wlmin)
            self.matrix = self.matrix[:, wlix:]
            self.wavelengths = self.wavelengths[wlix:]
        if wlmax:
            wlix, _ = RugTools.find_nearest(self.wavelengths, wlmax)
            self.matrix = self.matrix[:, :wlix]
            self.wavelengths = self.wavelengths[:wlix]

        return


