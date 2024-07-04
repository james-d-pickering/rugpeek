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


class RugTools:
   # class for generic tools to help data processing

    def read_harpia_matrix(file, extension):
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
        #overwrites the wavelength in nm to an optical freq in THz
        rug.wavelengths = (2.997E8/(rug.wavelengths*1.0E-9))/1E12
        return 
    
       
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]
    
    
    
 #############################

    def find_missing(lst):
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
            #items.append(metadata[runindex])
            #indexes.append(runindex)
            #keys.append('Measurement')
     
             
        else:
            print('Missing: Run')
            #print('Missing: Measurement')
        
            keys.append('Run')
            #keys.append('Measurement')
            items.append('None')
            #items.append('None')
            
        

        if any([item.endswith("nm") for item in metadata]) == True:
            wave = ([item.endswith('nm') for item in metadata])
            waveindex = wave.index(True)
            indexes.append(waveindex)
            items.append(metadata[waveindex])
            keys.append('pump_wavelength')    

        else:
            print('Missing: Pump Wavelength')
            keys.append('pump_wavelength')
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
        filelist = glob.glob('*.dat')
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
         #   print(newstring)
            
            randomn = randrange(100) ################ fix this

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
               # print(key, value)
                solvent_file.append(value)
            else:
                sample_files.append(value)
            
        #dictionary.values('chirp')
        #print(dictionary)
       # print('\nSample files:', sample_files)
       # print('\nSolvent files:', solvent_file)
       # print('\nSolvent file:', solvent_file[0]) #only need the one here cause they're all copies
        solventfile = solvent_file[0]
        #solventfile = solventfile[:-4]
        #solventdata = Rug(solventfile, ".dat")
       # print('Solvent file:', solventfile)
        #print('Sample file(s):', sample_files)
        files = [solventfile] + sample_files
        cleaned_file = [file for file in files if not file.endswith('processed.dat')]
        #print('cleanedfile', cleaned_file)
        
        
        return cleaned_file
##############################
    
class RugModels:
   # class for models that can be used to fit to - very work in progress 
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


class Rug:
    # class that holds the TA data "rug" and associated metadata and functions (stuff that only works on data)

    def __init__(self, fname, extension):
        # initialise the matrix from file. note matrix file only stores the data with wavelength 
        # and time axis. no metadata. at the moment we need to get the metadata from he filename.
        self.wavelengths, self.times, self.matrix = RugTools.read_harpia_matrix(fname, extension)
        self.filename = os.path.basename(fname)
        
        #############
        
        self.metadata = RugTools.get_metadata(self.filename)
        self.sample = self.metadata['Sample']
        self.concentration = self.metadata['Concentration']
        self.pump_wavelength = self.metadata['pump_wavelength']
        self.WLC_source = self.metadata['WLC_source']
        self.measurement = self.metadata['Measurement']
        self.run = self.metadata['Run']
        self.other = self.metadata['Other']
        
        
        
        #############

        #consistent filename convention: split with underscores. define a function to split it up
        #and get the relevant bits, so the metadata is stored so it can be correlated with the dispersion fit
        


 
    def get_t0_positions(self, degree):
        if self.axis:
            self.axis.set_title(f"Click t0 at {degree+1} or more points, then press enter")
            plt.draw()
            self.calib_points = plt.ginput(n=-1, timeout=-1, show_clicks=True)
            print(self.calib_points)
        else:
            raise Exception("need to have plotted the data to enable t0 point selection, run .peek() first")
        return

    def fit_dispersion_curve(self, degree):
        wavelengths = [point[0] for point in self.calib_points]
        times = [point[1] for point in self.calib_points]
        fit = pn.Polynomial.fit(wavelengths, times, deg=degree)
        coefs = fit.convert().coef
        self.corrected_time = -pn.polyval(self.wavelengths, coefs)
        # corrected_time are the time corrections we need to use
        self.dispersion_coefs = coefs
        return 


    def peek(self, ax=None, cmap='PuOr', min_max=None, aspect='equal', interpolation='none',
                norm=None,scale='log', title=None, colourbar=False, xlabel=True, ylabel=True,
                yticks=True, xticks=True, show=False, raw=False, plot_dispersion=False):
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
            vmin = np.min(self.matrix)
            vmax = np.max(self.matrix)
            
        title = self.filename
        title = self.filename

        #print(vmin, vmax)
        if raw and hasattr(self, "raw_matrix"):
            im = ax0.pcolormesh(self.wavelengths, self.times, self.raw_matrix,
            cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        elif raw and not hasattr(self, "raw_matrix"):
            raise Exception("Can't plot the raw matrix when dispersion correction not applied")
        else:
            im = ax0.pcolormesh(self.wavelengths, self.times, self.matrix,
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
                fig.colorbar(im)
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
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d', box_aspect=(1,1,1))

        wavelength_3d, times_3d = np.meshgrid(self.wavelengths, self.times)
        ax.plot_surface(wavelength_3d, times_3d, self.matrix, cmap=cmap, vmin=-10, vmax=10,
                       rstride=stride, cstride=stride, antialiased=False)
        ax.view_init(90,0,90)
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Time delay [ps]')
        ax.set_zlabel('$\Delta mOD$')
        ax.set_aspect('equalxy')

    ###########################################################

    
    def peekexplore(self, cmap='PuOr', min_max=None, aspect='equal', interpolation='none',
                norm=None,scale='log', title=None, colourbar=False, xlabel=True, ylabel=True,
                yticks=True, xticks=True, show=False, raw=False, plot_dispersion=False):
        
        timeslice = self.get_time_trace(450)
        wlslice_g = self.get_wavelength_trace(0)
        
        # Define initial parameters
        W_init = 450 
        T_init = 0
        
        fig = plt.figure(figsize=(10,10))
        grid = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], wspace=0.3, hspace=0.3)
        ax1 = fig.add_subplot(grid[0]) 
        ax2 = fig.add_subplot(grid[1])
        ax3 = fig.add_subplot(grid[2])
    
        ax1.set_ylabel('Time delay / ps')
        ax1.set_xlabel('Wavelength / nm')
        ax1.set_yscale('symlog')
        line_vertical = ax1.axvline(x=W_init, color= 'red', label= 'axvline - full height')
        line_horizontal = ax1.axhline(y=T_init, color='red', label= 'axhline - full height')
            
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


        

        if min_max:
            vmin = min_max[0]
            vmax = min_max[1]
        else:
            vmin = np.min(self.matrix)
            vmax = np.max(self.matrix)

        if raw and hasattr(self, "raw_matrix"):
            im = ax1.pcolormesh(self.wavelengths, self.times, self.raw_matrix,
            cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        elif raw and not hasattr(self, "raw_matrix"):
            raise Exception("Can't plot the raw matrix when dispersion correction not applied")
        else:
            im = ax1.pcolormesh(self.wavelengths, self.times, self.matrix,
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
            ax1.set_yscale('symlog')
        elif scale == 'linear':
            ax1.set_yscale('linear')
        else:
            raise NameError

        if title:
            ax1.set_title(title)

        if colourbar:
            if not ax:
                fig.colorbar(im)
            else:
                raise NameError
        if xlabel:
            ax1.set_xlabel('Wavelength [nm]')
        if ylabel:
            ax1.set_ylabel('Time delay [ps]')

        if not yticks:
            ax1.tick_params(axis="y", left=False, labelleft=False)
        elif yticks == True:
            pass
        else:
            ax1.set_yticks(yticks)
            ax1.set_yticklabels([str(i) for i in yticks]) 
        
        if not xticks:
            ax1.tick_params(axis="x", bottom=False, labelbottom=False)
        elif xticks == True:
            pass
        else:
            ax1.set_xticks(xticks)
            ax1.set_xticklabels([str(i) for i in xticks])
        if show:
            plt.show()

        # Sliders
        axtime = plt.axes([0.60, 0.40, 0.30, 0.02])
        time_slider = Slider(
            ax=axtime,
            label='Wavelength',
            valmin=346.587463,
            valmax=553.152954,
            valinit=W_init,
            color='hotpink'
        )
        
        axwave = plt.axes([0.60, 0.35, 0.30, 0.02])
        wavelength_slider = Slider(
            ax=axwave,
            label='Time',
            valmin=-10,
            valmax=1000,
            valinit=T_init,
            color='hotpink'
        )
        
        def update(event):
            time_ydata = self.get_time_trace(time_slider.val)
            line2.set_ydata(time_ydata)
            ax2.set_ylim(min(time_ydata)-1.5, max(time_ydata)+1.5)
        
            wl_ydata = self.get_wavelength_trace(wavelength_slider.val)
            line3.set_ydata(wl_ydata)
            ax3.set_ylim(min(wl_ydata)-2, max(wl_ydata)+2)
            fig.canvas.draw_idle()
        
            ax1.axvline(x=time_slider.val, color= 'red', label= 'axvline - full height')
            ax1.lines[0].remove()
        
            ax1.axhline(y=wavelength_slider.val, color='red', label= 'axhline - full height')
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
            color = '#'+"%06x" % random.randint(0, 0xFFFFFF)
            line2.set_color(color)
            line3.set_color(color)
            
            fig.canvas.draw_idle()
            
        def annotate(event):
            mplcursors.cursor(line2)
            mplcursors.cursor(line3)
        
        def reset(event):
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






    

    def get_average_background(self):
        times = []
        count = 0
     
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
          

        maximum = max(times)
        minimum = min(times)
        
        max_index, actualmax = RugTools.find_nearest(self.times, maximum)
        min_index, actualmin = RugTools.find_nearest(self.times, minimum)
      
        area_indexes = []
        for i in range(min_index, max_index+1):
            area_indexes.append(i)

  
        test = self.matrix[[area_indexes], :]

        change = test.transpose()

        mean_values = np.mean(change, axis =1)

        reverse = mean_values.transpose()
  
        reverse_matrix = np.zeros_like(self.matrix)
        for i in range(len(self.times)):
            reverse_matrix[i, :] = reverse_matrix[i, :] + reverse
            
        self.bg_matrix = self.matrix.copy()

        matrix = self.matrix - reverse_matrix
        self.matrix = matrix



        fig = plt.figure(figsize=(7,6))
        ax0 = fig.gca()
        self.axis = ax0
        vmin = np.min(matrix)
        vmax = np.max(matrix)
        ax0.set_yscale('symlog')

        im = ax0.pcolormesh(self.wavelengths, self.times, matrix,
        cmap='PuOr', norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        ax0.set_xlabel('Wavelength [nm]')
        ax0.set_ylabel('Time delay [ps]')
        fig.colorbar(im, ax=ax0)

        return

    def get_background_correction_average(self):
        self.peek(show=False)
        self.get_average_background()
        return
    
    def save(self, coefc, dispersion, offset):
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
        print('Saved processed file of', self.filename)
        return file
    
    
    
    def offset(self, coef):
        times = []
        count = 0

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
        allfignums = plt.get_fignums()
        for i in allfignums:
            fig = plt.figure(i)
            fig.clear()
            plt.close(fig)
        return


    def processing(self):
        self.peek(show=False)
        solventdata,sample_files = RugTools.findfiles(self)
        solventdata.get_dispersion_correction_eye()
        scoefs = solventdata.dispersion_coefs
        self.apply_dispersion_correction(coefs=scoefs)
        self.get_background_correction_average()
        self.peek()
        self.offset()
        self.MatplotlibClearMemory()
        return
    
        
    def auto_processing(files, background='area', correct_dispersion='eye', offset=True):
       
        allowed_methods = ('row', 'area')
        
        allowed_dispersion = ('eye', 'coefs')
        
        if background not in allowed_methods:
            raise ValueError('Invalid background removal method - allowed methods are "row" and "area"')
            
        if correct_dispersion not in allowed_dispersion:
            raise ValueError('Invalid dispersion method - allowed methods are "eye" and "coefs"')
            
            
        solventfile = files[0][:-4]
        solventdata = Rug(solventfile, ".dat")
         
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
            #sampledata.apply_dispersion_correction(coefs=scoefs)

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
                sampledata.save(coefc, dispersionc, offsetc)
                sampledata.MatplotlibClearMemory()
                
            else:
                offsetc = 'No offset applied'
                print('No offset applied')
                sampledata.save(coefc, dispersionc, offsetc)
                sampledata.MatplotlibClearMemory()
                
            #sys.exit()
                
            #sampledata.MatplotlibClearMemory()
  
    
   


    ###############################################################
 
    def apply_dispersion_correction(self, coefs=None):
        
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
            pass
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
        self.peek(show=False)
        self.get_t0_positions(degree)
        self.fit_dispersion_curve(degree=degree)
        return
    
    def apply_dispersion_correction_auto(self, file):
        self.get_dispersion_correction_eye()
        chirpcoefs = self.dispersion_coefs
        print(chirpcoefs)
        #self.apply_dispersion_correction(coefs=chirpcoefs)
        newf = self.apply_dispersion_correction(coefs=chirpcoefs)
        #newf = file.apply_dispersion_correction(coefs = chirpcoefs)
        return newf


        
    def get_time_trace(self, wavelength, plot=False):
        wavelength_idx, wavelength_real = RugTools.find_nearest(self.wavelengths, wavelength)
        time_trace = self.matrix[:, wavelength_idx]
        if plot:
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(self.times, time_trace)
            ax.set_xscale("symlog")
        return time_trace

    def get_wavelength_trace(self, time, plot=False):
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
        
    def limit_times(self, tmin=None, tmax=None):
        if tmax:
            tidx, _= RugTools.find_nearest(self.times, tmax)
            self.matrix = self.matrix[:tidx, :]
            self.times = self.times[:tidx]
        if tmin:
            tidx, _ = RugTools.find_nearest(self.times, tmin)
            self.matrix=self.matrix[tidx:, :]
            self.times = self.times[tidx:]

        return
    
    def limit_wavelengths(self, wlmin=None, wlmax=None):
        if wlmin:
            wlix, _ = RugTools.find_nearest(self.wavelengths, wlmin)
            self.matrix = self.matrix[:, wlix:]
            self.wavelengths = self.wavelengths[wlix:]
        if wlmax:
            wlix, _ = RugTools.find_nearest(self.wavelengths, wlmax)
            self.matrix = self.matrix[:, :wlix]
            self.wavelengths = self.wavelengths[:wlix]

        return

