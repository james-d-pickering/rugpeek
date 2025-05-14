#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import re
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections.abc
import numpy.polynomial.polynomial as pn
import scipy as sp
import os
from datetime import date
import time
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from matplotlib import cm

from ipywidgets import interact ###
import ipywidgets as widgets ###

from matplotlib.widgets import Slider, Button, TextBox, Cursor, SpanSelector
from pymcr.mcr import McrAR
from pymcr.regressors import NNLS
from pymcr.constraints import ConstraintNonneg, ConstraintNorm
import kennard_stone as ks
import lmfit as lf
import string
from scipy.optimize import curve_fit

"""
Load a matrix '.dat' file
"""
def load_file(file, extension):
    count = 0
    with open(file+extension, encoding="utf-8") as f:
        for line in f:
            
            line = line.strip()
            if np.char.isnumeric(line[0]):
                break
    
            count = count + 1
            # opens the file as f then for each line in f removes whitespace characters and increases the value of the variable count by 
            # +1 until it reaches a line that begins with a numeric character
    
    data = np.genfromtxt(file+extension, skip_header=count)
    
    wavelengths_nm = data[0,1:]
    delays_ps = data[1:,0]
    delta_OD = data[1:, 1:]

    return wavelengths_nm, delays_ps, delta_OD

"""
Load a '.dat' file
"""
def load_dat_file(file, extension):
    """
    Opens a HARPIA '.dat' file and reads in each line contaning numeric characters and stores this in a list 
    Currently have to enter data file and press enter after wavelength: to avoid skipping the line containing the wavelength data

    Needs to be generalized to allow user to open any '_.dat' file without making specific adjustments to the code for each dataset.
    ^ currently works for sample/solvent_630_SHG
    
    adjust accordingly depending on the number of uninterrupted experimental 'runs' stored in the '.dat. file
    """
    
    # retrieve the delay time points
    """
    number_of_lines = []

    with open(file+extension, encoding="utf-8") as f:
        for line in f:
            number_of_lines.append(line)
    """     
    sort = []
    
    with open(file+extension, encoding="utf-8") as f:
        for line in f:
            sort.append(line)

    
    delays_ = sort[10:len(sort):6] # [11:len(sort):6] 
    ########################################################################################################################################################
    
    # skips over the first 12 lines to reach 'Measurement..... x' (x = delay time) then iterates over every 6th element 
    # (there are 5 lines separating the lines that contain sequential delay time point information) in the list 'sort' 
    # and stores this in the list 'delays_' (containing the delay time associated with every sample measurement incl. repeats from each 'run'


    # extract the delay point from each line adjusting for changes in line length due to scan number and delay point polarity (+/-)
    delays_list = []
    
    for i in delays_:     
        for character in i:
            delays_list.append(i[((len(i) - len(delays_[0]) )+27):((len(i) - len(delays_[0]) )+43)]) 
            break

    
    # remove empty spaces
    delays_ps = []
    
    for i in delays_list:
        j = i.replace(' ','') 
        delays_ps.append(j)


    # save the unique delay points to a new list
    output = []
    
    for x in delays_ps:
        if x not in output:
            output.append(x)
            
    delays_ps = list(map(float, output))
    



    
    # retrieve the wavelength data
    wavelengths = []
    
    with open(file+extension, encoding="utf-8") as f:
        for line in f:
            if 'Wavelength: ' in line:
                _w = line[line.rindex('Wavelength: '):]
                for v in _w[_w.rindex(' '):][1:-1]:
                    wavelengths.append(v)
                
                
    del wavelengths[15:len(wavelengths):16]

    wavelengths_nm = []
    
    for i in range(0, len(wavelengths), 15):
        wavelengths_nm.append(float(''.join(wavelengths[i:i+15])))    
    ######################################################################################################################################################


    
    # retrieve all the measured intensity data   
    data = []
        
    with open(file+extension, encoding="utf-8") as f:
        for line in f:
            if np.char.isnumeric(line.strip()[0]) == True:
                for i in line:
                    data.append(i) # stores each character from the line in the list 'data'
            else:
                pass
                
    del data[15:len(data):16] # remove the '\t' character at every 16th element
    
    data_ = []
    
    for i in range(0, len(data), 15):
        data_.append(''.join(data[i:i+15])) # combine every element with no spaces into groups of 15 (number of characters for each value)
        
    
    # wavelengths_nm = list(map(float, data_[0:256]))



    # retrieve the absorption data for both the pumped and unpumped, background and sample spectra 
    
    data_p_per_run = ((len(wavelengths_nm)*4) + (len(delays_ps)*len(wavelengths_nm)*4))
    # total number of intensity data points per 'run'
    # (4*len(wavelengths_nm) for one background spectrum per run
    # (len(delays_ps)*len(wavelengths_nm)*4) = len(delays_ps) sample spectra per run

    # calculate the number of complete experiment runs (so the incomplete runs can be ignored when loading data)
    # number_of_runs = int((len(data_[256:]) / data_p_per_run))
    number_of_runs = int((len(data_) / data_p_per_run))
    
    ######################################################################################################################################################
    run_indicies = []

    for i in range(number_of_runs):
        run_indicies.append(i)
        

    ######################################################################################################################################################
    # seems to work ok but not sure why I need to add 256 at the end.... ?
    
    # int_data = data_[256:(number_of_runs*data_p_per_run)+256] 
    int_data = data_[0:(number_of_runs*data_p_per_run)+256] 
    
    # store the 'unpumped' background absorption data from each run in an array
    background_up_int = np.zeros(shape=(len(run_indicies), len(wavelengths_nm)))
    # One background spectrum is measured for each 'run', len(run_indicies) 'runs' in total

    for i in run_indicies:
        background_up_int[i] = int_data[((i*data_p_per_run) + (len(wavelengths_nm)*3)) : ((i*data_p_per_run) + (len(wavelengths_nm)*4))]

    
    mean_bckg_up_int = np.zeros_like(wavelengths_nm)
    bckg_up_int_st_dev = np.zeros_like(wavelengths_nm)

    for i, v in enumerate(background_up_int.T):
        mean_bckg_up_int[i] = np.cumsum(v, axis=0)[-1]/number_of_runs
        bckg_up_int_st_dev[i] = np.std(v)
        

    # store the 'pumped' background absorption data from each run in an array
    background_p_int = np.zeros(shape=(len(run_indicies), len(wavelengths_nm)))
    
    for i in run_indicies:
        background_p_int[i] = int_data[(i*data_p_per_run) : ((i*data_p_per_run) + len(wavelengths_nm))]

    mean_bckg_p_int = np.zeros_like(wavelengths_nm)
    bckg_p_int_st_dev = np.zeros_like(wavelengths_nm)

    for i, v in enumerate(background_p_int.T):
        mean_bckg_p_int[i] = np.cumsum(v, axis=0)[-1]/number_of_runs
        bckg_p_int_st_dev[i] = np.std(v)

    

    data_p_per_run = len(delays_ps)*len(wavelengths_nm)*4 
    # update this variable so now it only incl. the number of data points for the measurement scans (exlude background data which we removed)

    # remove the background intensity data from the original array to make it easier to iterate over to extract the sample intensity data
    del int_data[0 : (len(wavelengths_nm)*4)]  
    
    for i in run_indicies:
        if i <=8:
            del int_data[((i+1)*data_p_per_run) : (((i+1)*data_p_per_run) + (len(wavelengths_nm)*4))]    



    
    # store the 'unpumped' sample intensity data from each run in an array
    sample_up_int = np.zeros(shape=(len(run_indicies), len(delays_ps), len(wavelengths_nm))) # , dtype=np.longdouble)
    
    for i in run_indicies:
        for idx, val in enumerate(delays_ps):
            sample_up_int[i, idx] = int_data[((i*data_p_per_run) + (len(wavelengths_nm)*2 + (4*len(wavelengths_nm)*idx)) ) : ( (i*data_p_per_run)+(len(wavelengths_nm)*3 + (4*len(wavelengths_nm)*idx)) )]

    
    # store the mean unpumped intensity value in an array
    mean_sample_up_int = np.zeros(shape=(len(delays_ps), len(wavelengths_nm)))
    mean_sample_up_int = np.cumsum(sample_up_int, axis=0)[-1]/number_of_runs

    # store the standard deviation of the distribution of the unpumped intensity values in an array
    sample_up_int_st_dev = np.zeros(shape=(len(delays_ps), len(wavelengths_nm)))
    sample_up_int_st_dev = np.std(sample_up_int, axis=0)

 

        
    # store the 'pumped' sample intensity data from each run in an array
    sample_p_int = np.zeros(shape=(len(run_indicies), len(delays_ps), len(wavelengths_nm)))
        
    for i in run_indicies:
        for idx, val in enumerate(delays_ps):
            sample_p_int[i, idx] = int_data[( (i*data_p_per_run) + (len(wavelengths_nm) + (4*len(wavelengths_nm)*idx))) : ( (i*data_p_per_run)+(len(wavelengths_nm)*2 + ((4*len(wavelengths_nm)*idx))))]

    # store the mean pumped intensity value in an array
    mean_sample_p_int = np.zeros(shape=(len(delays_ps), len(wavelengths_nm)))
    mean_sample_p_int = np.cumsum(sample_p_int, axis=0)[-1]/number_of_runs
    
    # store the standard deviation of the distribution of the pumped intensity values in an array
    sample_p_int_st_dev = np.zeros(shape=(len(delays_ps), len(wavelengths_nm)))
    sample_p_int_st_dev = np.std(sample_up_int, axis=0)



    
    """
    # propagate the uncertainty in each measurement to the difference absorption values
    A = I_up - I_bup
    B = I_p - I_bup
    
    difference_absorption(x) = np.log10(A/B)

    

    sig_A = np.sqrt(sig_I_up**2 + sig_I_bup**2) 
    sig_B = np.sqrt(sig_I_p**2 + sig_I_bp**2) 
    

    Y = A/B
    sig_Y = Y * np.sqrt( (sig_A/A)**2 + (sig_B/B)**2 )

    x = np.log10(Y)

    sig_x = sig_Y / (Y * np.log(10))

    """

    # create an array containing the unpumped background standard deviation values sharing the same dimensions as the array of unpumped signal st devs
    bckg_up_int_st_dev_ = np.zeros_like(sample_up_int_st_dev)
    
    for i, v in enumerate(bckg_up_int_st_dev_):
        bckg_up_int_st_dev_[i] = bckg_up_int_st_dev
        
    # propagate the standard deviation with respect to A
    sig_A = np.sqrt((sample_up_int_st_dev**2) + (bckg_up_int_st_dev_)**2)


    
    # create an array containing the pumped background standard deviation values sharing the same dimensions as the array of pumped signal st devs
    bckg_p_int_st_dev_ = np.zeros_like(sample_p_int_st_dev)
    
    for i, v in enumerate(bckg_p_int_st_dev_):
        bckg_p_int_st_dev_[i] = bckg_p_int_st_dev
        
    # propagate the error with respect to B
    sig_B = np.sqrt((sample_p_int_st_dev**2) + (bckg_p_int_st_dev_)**2)


    
    # create an array containing the background subtract unpumped intensity data
    A = np.zeros_like(sample_up_int)
    
    for i in run_indicies:
        for idx, val in enumerate(delays_ps):
            A = sample_up_int[i, idx] - background_up_int[i] 
            # only one background spectrum is taken per run so the same background spectrum is subtracted from the spectrum at each delay point

    # create an array containing the background subtracted pumped intensity data
    B = np.zeros_like(sample_p_int)
    
    for i in run_indicies:
        for idx, val in enumerate(delays_ps):
            B = sample_p_int[i, idx] - background_p_int[i] 
            # only one background spectrum is taken per run so the same background spectrum is subtracted from the spectrum at each delay point

    Y = A/B

    sig_Y = Y * np.sqrt( ((sig_A / A)**2) + ((sig_B / B)**2) )

    sig_x = sig_Y / (Y * np.log(10))





    
    
    # subtract the 'unpumped' background spectrum intensity from the corresponding 'unpumped' sample spectra intensity
    for i in run_indicies:
        for idx, val in enumerate(delays_ps):
            sample_up_int[i, idx] - background_up_int[i] 
            # only one background spectrum is taken per run so the same background spectrum is subtracted from the spectrum at each delay point
            


    
    # subtract the 'pumped' background spectrum intensity from the corresponding 'pumped' sample spectra intensity
    for i in run_indicies:
        for idx, val in enumerate(delays_ps):
            sample_p_int[i, idx] - background_p_int[i]
            # only one background spectrum is taken per run so the same background spectrum is subtracted from the spectrum at each delay point
    


    
    # calculate the difference absorption spectra for each run
    raw_delta_OD = np.zeros_like(sample_up_int)
    
    for i, v in enumerate(raw_delta_OD):
        raw_delta_OD[i] = np.log10(sample_up_int[i]/sample_p_int[i])*1000

    
    
    # average the 'unpumped' sample spectra intensity for each run
    """
    avg_sample_up_delta_OD = np.zeros(shape=(len(delays_ps), len(wavelengths_nm)))

    for idx, val in enumerate(delays_ps):
        for w_idx, w_val in enumerate(wavelengths_nm):
            avg_sample_up_delta_OD[idx, w_idx] = ( (sample_up_delta_OD[0][idx][w_idx] + sample_up_delta_OD[1][idx][w_idx] + sample_up_delta_OD[2][idx][w_idx] + sample_up_delta_OD[3][idx][w_idx]) / len(run_indicies) )
    """
    
    avg_sample_up_int = np.cumsum(sample_up_int, axis=0)[-1]/number_of_runs


    
    # average the 'pumped' sample absorption spectrum intensity for each run
    """
    avg_sample_p_delta_OD = np.zeros(shape=(len(delays_ps), len(wavelengths_nm)))

    for idx, val in enumerate(delays_ps):
        for w_idx, w_val in enumerate(wavelengths_nm):
            avg_sample_p_delta_OD[idx, w_idx] = ( (sample_p_delta_OD[0][idx][w_idx] + sample_p_delta_OD[1][idx][w_idx] + sample_p_delta_OD[2][idx][w_idx] + sample_p_delta_OD[3][idx][w_idx]) / len(run_indicies) )
    """
        
    avg_sample_p_int = np.cumsum(sample_p_int, axis=0)[-1]/number_of_runs
    




    

    
    # retrieve the difference absorption spectra by taking the log base 10 of the intensity of the 'unpumped' spectra from the intensity of the 'pumped' spectra 
    
    # delta_OD = np.zeros(shape=(len(delays_ps), len(wavelengths_nm)))
    """
    for idx, val in enumerate(delays_ps):
        for w_idx, w_val in enumerate(wavelengths_nm):
            delta_OD[idx, w_idx] = np.log10(avg_sample_up_delta_OD[idx, w_idx] / avg_sample_p_delta_OD[idx, w_idx])
    """
    # delta_OD = (np.log10(avg_sample_up_int / avg_sample_p_int)) * 1000
    # np.cumsum(raw_delta_OD, axis=0)[-1]/number_of_runs


    # produces difference absorption values closest to those in the matrix file (<1e-6 difference)
    delta_OD = np.zeros_like(sample_p_int)
    
    for idx, val in enumerate(delta_OD):
        for jdx, wal in enumerate(val):
            
            A = sample_up_int[idx, jdx] - background_up_int[idx]
            B = sample_p_int[idx, jdx] - background_p_int[idx]
            
            delta_OD[idx, jdx] = np.log10(A/B)*1000


    delta_OD = np.cumsum(delta_OD, axis=0)[-1]/number_of_runs

    
    """
     ^ the values in the array calculated here are different to those stored in the '_matrix' file 
       perhaps this is due to differences in averaging 
      (i.e. matrix may contain the average of the values from each scan and the number of independent runs whilst the data file contains the mean values for          each run averaged from all the repeats in that run which are then average together in this function)
    """





    


    
    # calculate the 'sum of squared total' np.sum((y_i - y{hat}_i)**2)
    
    y_dif = np.zeros_like(raw_delta_OD)

    for i in range(number_of_runs):
        y_dif[i] = (raw_delta_OD[i] - delta_OD)**2

    
    SST = np.zeros_like(delta_OD)
    SST = np.cumsum(y_dif, axis=0)[-1]

    # this isn't quite correct, basically want to use it to generate a metric for the quality of the fit of the kinetic model to the raw data
    # where each value is weighted by the standard deviation of the distribution of values surrounding the mean 
    # (so that values with large standard deviations are less important to the fit metric than those with smaller standard deviations)
    
        
        

    
        
    
    return np.array(wavelengths_nm), np.array(delays_ps), background_up_int, background_p_int, sample_up_int, sample_p_int, avg_sample_up_int, avg_sample_p_int, delta_OD, sig_x, SST
    
    # sample_up_int and sample_p_int are arrays containing the raw probe 'intensity' detected at the camera from each run
    # avg_p_" " and avg_up_" " are arrays containing the averaged 'pumped' and 'unpumped' probe 'intensity' data 
    # delta_OD is the difference absorption spectrum 



"""
load a 'stats.dat' file
"""

def load_stats_file(file, extension):
    """
    loads a 'stats.dat' file and returns the standard deviations of the pumped and unpumped signals and the background signals for each measurement
    """

    PS_SD = []
    NPS_SD = []
    delays = []
    

    
    with open(file+extension, encoding="utf-8") as f:
        for line in f:
            
            if ' PumpedSignalDeviation=' in line:
                _p = line[line.rindex(' PumpedSignalDeviation='):]
                PS_SD.append(float(_p[_p.rindex('='):][1:-1]))

            if 'NotPumpedSignalDeviation=' in line:
                n_p = line[line.rindex('NotPumpedSignalDeviation='):]
                NPS_SD.append(float(n_p[n_p.rindex('='):][1:-1]))

            if ' Delay ' in line:
                _d = line[line.rindex(' Dela'):]
                delays.append(float(_d[_d.rindex('y '):][1:18]))
                    
    PS_SD = PS_SD[0:len(PS_SD):2]
    NPS_SD = NPS_SD[0:len(NPS_SD):2]
    
    unique_delays = []
    
    for x in delays:
        if x not in unique_delays:
            unique_delays.append(x)
    
    B_SD = PS_SD[0:len(PS_SD):len(unique_delays)+1]
    
    del PS_SD[0:len(PS_SD):len(unique_delays)+1]
    del NPS_SD[0:len(PS_SD):len(unique_delays)+1]


    return np.array(PS_SD), np.array(NPS_SD), np.array(B_SD)


"""
Create a new class 'Rug' and initialise that class with a set of 'attributes'

The attributes 'wavelengths', 'delays' and 'abs' are arrays containing the wavelength, delay and difference absorption data, respectively.
"""

class Rug:
    def __init__(self, fname=None, extension=None, type=None):    # will need to add 'type' in the function for joining SHG/FUN data
        
        if fname and extension:
            if type == False:
                self.wavelengths, self.delays, self.abs = load_file(fname, extension)
                self.filename = os.path.basename(fname)
                
            elif type == True:
                self.wavelengths, self.delays, self.bup_int, self.bp_int, self.raw_sup_int, self.raw_sp_int, self.avg_sup_int, self.avg_sp_int, self.delta_OD, self.sig_x, self.SST = load_dat_file(fname, extension) 
                self.filename = os.path.basename(fname)
                # ^ self.abs = self.avg_sup_int, select the desired array by changing the attribute "._' '" to '.abs'

            elif type == 'stats':
                self.p_sd, self.up_sd, self.b_sd = load_stats_file(fname, extension) 
            
            
            return
            
        """
        if fname and extension: 
            # need an if statement so that a rug object can be created without providing an input for fname and extention 
            # as seen in the function combine_rugs_wavelengths
            self.wavelengths, self.delays, self.abs = load_file(fname, extension)
            self.filename = os.path.basename(fname)
        """   
        
    def extract_UV_VIS(arr1, arr2): 
        # currently need to have the 'unpumped' array of difference absorption data set as self.abs
        # extracts the 'unpumped' difference absorption data from the solvent and sample '.dat' files and generates a Static UV-VIS absorption spectrum
        # arr1 should be the array of unpumped difference absorption data from the solvent 
        # arr2 should be the array of unpumped difference absortion data from the sample

        # need to select difference absorption data at the same delay point
        
        sample_idx = Rug.find_nearest(arr2.delays, arr1.delays[95]) # arr1.delays[10]
        
        print(sample_idx, '\n')
        
        print(f'The spectrum at',arr1.delays[95], f'ps was selected from the array of solvent data')
        print(f'The spectrum at',arr2.delays[sample_idx[0]], f'ps was selected from the array of sample data')
        
        abs_spectrum = np.log(arr1.abs[95]/arr2.abs[sample_idx[0]]) 
        
        plt.figure()
        plt.plot(arr2.wavelengths, arr1.abs[95], color='black', label='Raw Solvent')
        plt.plot(arr2.wavelengths, arr2.abs[sample_idx[0]], color='blue', label='Raw Sample')
        
        plt.tick_params(axis="both", labelsize = 15)
        plt.xlabel('Wavelength (nm)', fontsize=20)
        plt.ylabel('Absorption (O.D)', fontsize=20)
        plt.xlim(arr2.wavelengths[0], arr2.wavelengths[-1])
        plt.legend()
        plt.title('Unpumped UV-VIS Absorption Spectra', fontsize=20)
        plt.show()

        plt.figure()
        plt.plot(arr2.wavelengths, abs_spectrum, color='black')
        
        plt.tick_params(axis="both", labelsize = 15)
        plt.xlabel('Wavelength (nm)', fontsize=20)
        plt.ylabel('Absorption (O.D)', fontsize=20)
        plt.xlim(arr2.wavelengths[0], arr2.wavelengths[-1])
        plt.title(arr2.filename+' Static UV-VIS Spectrum', fontsize=20)
        plt.show()

        w_array = np.zeros(shape=(len(abs_spectrum)))
        w_array = arr2.wavelengths
        
        abs_array = np.zeros(shape=(len(abs_spectrum)))
        abs_array = abs_spectrum
        
        data = np.column_stack([w_array, abs_array])
        
        np.savetxt(arr2.filename+'_chirp_corrected_Static_UV-VIS_Absorption_Spectrum'+'.dat', data)
        plt.savefig(arr2.filename+"__chirp_corrected_Static_UV-VIS_Absorption_Spectrum.png")
        
        """
        lent = len(arr2.delays)
        lenm = len(arr2.wavelengths)

        array = np.zeros((lent+1, lenm+1))

        array_wl = arr2.wavelengths
        array_times = arr2.delays[:, None].reshape((lent,))
        array_matrix = arr2.abs

        array[0, 1:] = array_wl
        array[1:, 0] = array_times
        array[1:, 1:] = array_matrix

        today = date.today()
        current_time = time.localtime()
        timestring = time.strftime("%d/%m/%Y %H:%M:%S")
      
        np.savetxt(arr2.filename+'_chirp_corrected_deltaOD_dat'+'.dat', array,
                   header='XAxisTitleWavelength(nm)\nYAxisTitle Delay (ps)\n'+'Date&time: '+ timestring)
        
        plt.savefig(arr2.filename+"_processed_dat.png")
        """
        return


    def reference_dat(arr1, arr2):
        """
        generates referenced difference spectra by referencing the 'pumped' and 'unpumped' absorption spectra prior to subtraction
        
        arr1 = solvent 
        arr2 = sample

        note the arr.abs may be set to be equal to either arr.avg_sup_abs or arr.avg_sp_abs so check what these are set to when the '.dat' file is 
        initialized as a rug
        """
        
        arr1.abs
        

    
        
    def find_nearest(array, value):
        """
        general function for finding the value in an array closest to a user specified input
        returns the index and the value in the array closest to the input value
        """
    
        array = np.asarray(array)              # converts the input data to an array
        
        idx = (np.abs(array - value)).argmin() 
        """
        creates an array of positive values by taking np.abs(the differences between the input value and each value in the array)
        then .argmin() returns the index of the value in the original array that the smallest value in the array of differences is associated with.
        np.abs() converts any negative value in the array into a positive value
        """
        return idx, array[idx]  # returns the index and the value associated with array[idx]

    
    def peek(self, contour=False):
        """
        plot a colourmap of the 'Rug'
        
        In RugPeek-Main the peek() function can be used to plot:
        the original Chirped dataset,
        the dataset after correcting for temporal dispersion,
        the dataset before subtracting the background signal,
        the dataset prior to removing the regions contaminated the pump and probe, 
        and the polynomial used to fit the temporal dispersion

        edit this so it can be used to generate contour plots that look nice for figures
        """
        
        vmin = np.nanmin(self.abs)
        vmax = np.nanmax(self.abs)
        #vcenter = (vmax - vmin) / 2 # v centre for '_.dat' files, set vcenter equal to 0 for '_matrix.dat' files
        
        
        fig, ax = plt.subplots()
        # create a figure so I can adjust the position of the labels using coordinates on the figure axis
        
        im = ax.pcolormesh(self.wavelengths, self.delays, self.abs, cmap='PuOr_r', 
                      norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))  
        
        # vcenter=0, vmin=vmin, vmax=vmax (for 'matrix.dat')
        # vmin=vmin, vcenter=vcenter, vmax=vmax (for viewing absorption (not difference absorption) data from HARPIA '_.dat' files)
        # # use a 'diverging' cmap (e.i. RdBu) so that the cmap can be set using colors.TwoSlopNorm so that zero array values are blank

        tick_size = 25
        axis_fontsize = 30
        title_fontsize = 30
        
        ax.tick_params(axis="both", labelsize = tick_size)
        ax.xaxis.set_label_coords(0, -0.08) # (0, -0.05)
        ax.yaxis.set_label_coords(-0.1, 0.5) # (-0.08, 0.5)
        #ax.title.set_label_coords(0.5, 0.5)
        
        ax.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize)
        ax.set_ylabel('Delay (ps)', fontsize = axis_fontsize, rotation = 360)
        ax.set_yscale('symlog')
        #ax.set_ylim(-1, 10)
        ax.set_title(self.filename, fontsize = title_fontsize, y=1.02) # fontweight='bold'
        
        tick_range = np.linspace(vmin, vmax, 10)
        cbar = fig.colorbar(im, ticks=tick_range) # change the fontsize ? [vmin, 0, vmax]
        cbar.set_label(label = 'Δ mO.D', fontsize = axis_fontsize, y = 0.55, labelpad = 40, rotation=360)
        # (label = 'Δ O.D', fontsize = 30, y = 0.52, labelpad = 30, rotation=360)
        cbar.ax.tick_params(labelsize = tick_size)


        ## contour plot ##
        n_pos = []
        n_neg = []

        for idx, ival in enumerate(self.abs):
            for jdx, jval in enumerate(self.abs[idx]):
                if jval > 0:
                    n_pos.append(jval)
                else:
                    n_neg.append(jval)
                    
        n_pos = len(n_pos) 
        n_neg = len(n_neg)

        n_points = []
        
        if n_pos >= n_neg:
            n_points = n_pos
        else:
            n_points = n_neg
            
        
        #n_pos = 110 # 124
        #n_neg = 146 # 132
        
        """
        pos = cm.get_cmap('YlOrRd', n_pos)
        neg = cm.get_cmap('Blues_r', n_neg)
        new_colours = np.vstack((neg(np.linspace(0, 1, n_neg)),
                                 pos(np.linspace(0, 1, n_pos))))
        """
        
        pos = cm.get_cmap('YlOrRd', n_points)
        neg = cm.get_cmap('Blues_r', n_points)
        
        new_colours = np.vstack((neg(np.linspace(0, 1, n_points)),
                                 pos(np.linspace(0, 1, n_points))))

        new_cmap = colors.ListedColormap(new_colours, name='RedBlue') # this colour scheme is ideal but needs modifying to make 0 = white, consistently.

        cmap = new_cmap # 'PuOr_r'
        
        
        # diverging colourmaps (blue/red,  0 = white):
        # 'seismic', 'bwr', 'coolwarm','RdBu_r'
        #cmap = 'RdBu_r'
        
        class MidpointNormalize(colors.Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                colors.Normalize.__init__(self, vmin, vmax, clip)
        
            def __call__(self, value, clip=None):
                # I'm ignoring masked values and all kinds of edge cases to make a
                # simple example...
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

        
        if contour == True:
            # impose 0 values on the background to avoid 'messy' contour lines
            pretty_array = np.zeros_like(self.abs) # np.full(shape=(len(self.delays), len(self.wavelengths)), fill_value=-1)
            
            for i, v in enumerate(self.delays[0:]):
                # ferric_630_total: [0:]//[20:125] levels = 24
                # ferric_630_SHG: [0:]//[20:125] levels = 30
                # ferric_630_FUN: [0:]//[0:] levels = 10

                # ferric_500_SHG: [0:55][25:135] levels = 15
                
                # ferrous_556_SHG: [0:]//[62:] levels = 30
                
                # oxy_581_SHG: [0:]//[55:130] levels = 15

                # oxy_542_SHG (10, 5, 1kHz): [0:]//[0:] levels = 17
                # oxy_542_SHG (2kHz): [0:]//[0:] levels = 10
                # oxy_542_SHG (500Hz): [0:]//[0:] levels = 5/10
                
                # oxy_416_FUN: [0:]//[0:] levels = 15
                # oxy_416_SHG: [0:]//[31:99] levels = 20

                # haem_350_SHG: [0:]//[0:] levels = 20
                # haem_387_SHG: [0:]//[25:112] levels = 16

                # Haem-Cys_400_SHG: 
                
                pretty_array[i, 0:] = self.abs[i, 0:]

            
            fig_1 = plt.figure(figsize=(10,10))
            grid_1 = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], 
                        wspace=0.3, hspace=0.3)
            
            ax_1 = fig_1.add_subplot(grid_1[0:7]) 
            
            levels = np.linspace(vmin, vmax, 20) 

            pump = 409 # add a vertical line to the plot at the pump wavelength
        
            # restrict the wavelengths and delay axis within a specified range 
            # ferric_630_total: 10, 136, 0, 90
            # ferric_630_SHG: 15, 130, 0, 90 
            # ferric_630_FUN: 5, 65, 0, 8

            # ferric_500_SHG: 15, 135, 0, 70
            
            # ferrous_556_SHG: 60, -1, 0, 90 
            
            # hskmMb-Fe[II]O2_581_SHG: 55, 130, 0, -1/90 
            # hskmMb-Fe[II]O2_542_SHG (incl. 10, 5, 2, 1kHz, 500Hz): 0:/[87, 124], 0:
            # hskmMb-Fe[II]O2_416_SHG: 25, 108, 0, -1

            # Free_haem_350_SHG: 0:, 0:
            # Free_haem_387_SHG: 20, 125, 0:

            # Haem-Cys_400_SHG: 
            
            wl_lim_init = Rug.find_nearest(self.wavelengths, self.wavelengths[0])[0] 
            wu_lim_init = Rug.find_nearest(self.wavelengths, self.wavelengths[-1])[0] 
            
            dl_lim_init = Rug.find_nearest(self.delays, self.delays[0])[0] 
            du_lim_init = Rug.find_nearest(self.delays, self.delays[-1])[0] 
            
            """
            test = ax_1.contourf(self.wavelengths[wl_lim_init:wu_lim_init], self.delays[dl_lim_init:du_lim_init], pretty_array[dl_lim_init:du_lim_init, wl_lim_init:wu_lim_init], levels, cmap=cmap, norm=MidpointNormalize(vmin = vmin, vmax = vmax, midpoint = 0)) # colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
            """
            # ^ use with existing colormap
            
            test = ax_1.contourf(self.wavelengths[wl_lim_init:wu_lim_init], self.delays[dl_lim_init:du_lim_init], pretty_array[dl_lim_init:du_lim_init, wl_lim_init:wu_lim_init], levels, cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
            # ^ use with 'RedBlue'
            
            contour_1 = ax_1.contour(self.wavelengths[wl_lim_init:wu_lim_init], self.delays[dl_lim_init:du_lim_init], pretty_array[dl_lim_init:du_lim_init, wl_lim_init:wu_lim_init], levels, colors=('black',), linewidths=(1,))
            
            # ax_1.axvline(x=pump, ymin=0, ymax=1, color='black', ls='--', lw=5, label=r'$\{pump}$')
            
            ax_1.tick_params(axis="both", labelsize = tick_size)
            ax_1.set_yscale('symlog')
            ax_1.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize)
            ax_1.set_ylabel('Delay (ps)', fontsize = axis_fontsize, rotation = 360)
            ax_1.xaxis.set_label_coords(0.5, -0.07) 
            
            ax_1.yaxis.set_label_coords(-0.1, 0.6) # sym-log # (-0.1, 0.58)
            # ax_1.yaxis.set_label_coords(-0.1, 0.45) # not scaled
            
            cbar_1 = fig_1.colorbar(test, ax=ax_1)
            cbar_1.set_label(label = 'Δ O.D', fontsize = axis_fontsize, y = 0.51, labelpad = 50, rotation=360)
            cbar_1.ax.tick_params(labelsize = tick_size)
            #ax_1.clabel(contour_1, fmt='%2.1f', colors='black', fontsize=10) # add labels to the levels directly on the plot
            ax_1.set_title(self.filename, fontsize = title_fontsize, y=1.02)

            # Sliders
            global levels_slider
            
            axwave = plt.axes([0.1, 0.005, 0.8, 0.01]) # ([0.1, 0.01, 0.8, 0.015]) 
            levels_slider = Slider(
                ax=axwave,
                label='Contours',
                valmin=0,
                valmax=40,
                valinit=20,
                valstep=1,
                color='lightsteelblue',
                handle_style={'facecolor': 'white', 'edgecolor': '.04', 'size': 10} # .75
            )
            levels_slider.label.set_size(20)

            # TextBoxes for adjusting axis limits
            ax_wl_min = fig_1.add_axes([0.2, 0.94, 0.04, 0.05]) 
            ax_wl_max = fig_1.add_axes([0.35, 0.94, 0.04, 0.05])
            
            ax_dl_min = fig_1.add_axes([0.55, 0.94, 0.04, 0.05])
            ax_dl_max = fig_1.add_axes([0.7, 0.94, 0.04, 0.05])
            
            wl_min_box = TextBox(ax=ax_wl_min, label=r'$\lambda_{min}$ (nm)', initial=str(np.round(self.wavelengths[0], 2)), label_pad=0.15)
            wl_max_box = TextBox(ax=ax_wl_max, label=r'$\lambda_{max}$ (nm)', initial=str(np.round(self.wavelengths[-1], 2)), label_pad=0.15)

            dl_min_box = TextBox(ax=ax_dl_min, label=r'$delay_{min}$ (ps)', initial=str(np.round(self.delays[0], 2)), label_pad=0.15)
            dl_max_box = TextBox(ax=ax_dl_max, label=r'$delay_{max}$ (ps)', initial=str(np.round(self.delays[-1], 2)), label_pad=0.15)
            
            wl_min_box.label.set_size(20)
            wl_max_box.label.set_size(20)

            dl_min_box.label.set_size(20)
            dl_max_box.label.set_size(20)
            
            def update(val):
                #Updates the plot 'levels' when the user interacts with the widget.
                ax_1.cla()
                
                wl_lim = Rug.find_nearest(self.wavelengths, float(wl_min_box.text))[0]
                wu_lim = Rug.find_nearest(self.wavelengths, float(wl_max_box.text))[0]
                
                dl_lim = Rug.find_nearest(self.delays, float(dl_min_box.text))[0]
                du_lim = Rug.find_nearest(self.delays, float(dl_max_box.text))[0]

                s_levels = np.linspace(vmin, vmax, levels_slider.val) 
                
                """
                test = ax_1.contourf(self.wavelengths[wl_lim:wu_lim], self.delays[dl_lim:du_lim], pretty_array[dl_lim:du_lim, wl_lim:wu_lim], levels_slider.val, cmap=cmap, norm=MidpointNormalize(vmin = vmin, vmax = vmax, midpoint = 0)) # colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
                """
                # ^ use with existing colormap
                
                test_1 = ax_1.contourf(self.wavelengths[wl_lim:wu_lim], self.delays[dl_lim:du_lim], pretty_array[dl_lim:du_lim, wl_lim:wu_lim], s_levels, cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
                # ^ use with 'RedBlue'
                
                contour_1 = ax_1.contour(self.wavelengths[wl_lim:wu_lim], self.delays[dl_lim:du_lim], pretty_array[dl_lim:du_lim, wl_lim:wu_lim], s_levels, colors=('black',), linewidths=(1,))
                
                # ax_1.axvline(x=pump, ymin=0, ymax=1, color='black', ls='--', lw=5, label=r'$\{pump}$')

                ax_1.tick_params(axis="both", labelsize = tick_size)
                ax_1.set_yscale('symlog')
                ax_1.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize)
                ax_1.set_ylabel('Delay (ps)', fontsize = axis_fontsize, rotation = 360)
                ax_1.xaxis.set_label_coords(0.5, -0.07) 
                
                ax_1.yaxis.set_label_coords(-0.1, 0.6) # sym-log (-0.1, 0.58)
                # ax_1.yaxis.set_label_coords(-0.1, 0.52) # not scaled
                
                ax_1.set_title(self.filename, loc='center', fontsize=title_fontsize, y=1.02)
                cbar_1 = fig_1.colorbar(test_1, cax=cax)
                cbar_1.set_label(label = 'Δ O.D', fontsize = axis_fontsize, y = 0.51, labelpad = 50, rotation=360)
                cbar_1.ax.tick_params(labelsize = tick_size)
                #ax_1.clabel(contour_1, fmt='%2.1f', colors='black', fontsize=10) # add labels to the levels directly on the plot
                
                
                fig_1.canvas.draw_idle()
                
            levels_slider.on_changed(update)
            wl_min_box.on_text_change(update)
            wl_max_box.on_text_change(update)
            dl_min_box.on_text_change(update)
            dl_max_box.on_text_change(update)

        else:
            pass

        
        
        #plt.figure()
        #plt.pcolormesh(self.wavelengths, self.delays, self.abs, cmap='RdBu', 
                       #norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        # # use a 'diverging' cmap so that the cmap can be set using colors.TwoSlopNorm so that zero values are blank
        
        #plt.tick_params(axis="both", labelsize = 15)
        #plt.set_xlabel_coords(0.5, 0.5)
        #plt.xlabel('Wavelength (nm)', fontsize = 20)
        #plt.ylabel('Delay (ps)', fontsize = 20, rotation = 360)
        #plt.yscale('symlog')
        #plt.title(self.filename, fontsize = 20)
        
        #tick_range = np.linspace(vmin, vmax, 10)
        #cbar = plt.colorbar(ticks=tick_range) # change the fontsize ? [vmin, 0, vmax]
        #cbar.set_label(label = 'Δ O.D', fontsize = 20, rotation=360)
        #plt.show()        
        return

    def get_spectrum(self, time, plot=False): # func called by 'explore_spectra'
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
        
        spectra = []
        
        try:
            for t in time:
                delay_idx, delay_real = Rug.find_nearest(self.delays, t)
                spectra.append(self.abs[delay_idx, :])
        except:
            delay_idx, delay_real = Rug.find_nearest(self.delays, time)
            spectra.append(self.abs[delay_idx, :])
            time = [time]

        if plot:
            fig = plt.figure()
            ax = fig.gca()
            for i, spectrum in enumerate(spectra):
                ax.plot(self.wavelengths, spectrum, label=str(time[i])+'ps')
            ax.set_xlabel('Wavelength [nm]')
            ax.set_ylabel(r'$\Delta$OD [mOD]')
            ax.legend()
        
        return spectra

    

    def get_trace(self, wavelength, plot=False):
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
        traces = []
        
        try:
            for wl in wavelength:
                wavelength_idx, wavelength_real = Rug.find_nearest(self.wavelengths, wl)
                traces.append(self.abs[:, wavelength_idx])

        except:
            wavelength_idx, wavelength_real = Rug.find_nearest(self.wavelengths, wavelength)
            traces.append(self.abs[:, wavelength_idx])
            wavelength = [wavelength] #JDP to save a faff with plotting
            

        if plot:
            fig = plt.figure()
            ax = fig.gca()
            for i, trace in enumerate(traces):
                ax.plot(self.delays, trace, label=str(wavelength[i])+'nm')
            ax.set_xscale("symlog")
            ax.set_xlabel('Delay [ps]')
            ax.set_ylabel(r'$\Delta$OD [mOD]')
            ax.legend()
        
        return traces


    def explore_spectra(self, cmap='PuOr', min_max=None, aspect='equal', interpolation='none',
                norm=None,scale='log', title=None, colourbar=False, xlabel=True, ylabel=True,
                yticks=True, xticks=True, show=False, raw=False, plot_dispersion=False, sl_min=0, sl_max=-1):
        """
        Interactive widget with a slider to inspect absorption spectra at a given delay
        """
        
        spectrum_init = self.get_spectrum(self.delays[sl_min]) # [0]

        fig = plt.figure(figsize=(10,10))
        grid = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], 
                        wspace=0.3, hspace=0.3)
        tick_size = 25 # 25
        axis_fontsize = 30 # 30
        title_fontsize = 25 # 25
        
        ax1 = fig.add_subplot(grid[0:7]) # grid[0:7]
        
        #line1, = ax1.plot(self.wavelengths, spectrum_init[0], color='darkslateblue', linewidth=2)
        ax1.plot(self.wavelengths, spectrum_init[0], color='darkslateblue', linewidth=2, label=str(np.round(self.delays[sl_min], 2))+" ps")
        
        ax1.tick_params(axis="both", labelsize = tick_size)
        ax1.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize)
        ax1.set_ylabel(r'$\Delta$OD [mOD]', fontsize = axis_fontsize) # 'Difference Absorption (mOD)'
        ax1.set_xlim(self.wavelengths[0], self.wavelengths[-1])
        #ax1.set_ylim(-55, 15)
        ax1.legend(fontsize=tick_size, loc='upper right') # loc='upper right'
        ax1.set_title('Transient Absorption Difference Spectrum at '+str(np.round(self.delays[sl_min], 2))+' ps', fontsize = title_fontsize, y=1.02)
        ax1.grid(visible=True)

        title = self.filename
        
        # Sliders
        global delay_slider
        
        axwave = plt.axes([0.1, 0.01, 0.8, 0.02]) # ([0.1, 0.03, 0.8, 0.03])
        delay_slider = Slider(
            ax=axwave,
            label='Delay',
            valmin=self.delays[sl_min],
            valmax=self.delays[sl_max], # self.delays[100], # seems to have a minimum increment depending on the total number of points so adjust as appropriate
            valinit=self.delays[0],
            valstep=self.delays, #[0:],
            color='lightsteelblue',
            #initcolor='none',
            #track_color='lightsteelblue',
            handle_style={'facecolor': 'white', 'edgecolor': '.05', 'size': 20} # .75
        )
        delay_slider.label.set_size(20)

        def update(val):
            
            #Updates the graphs when the user interacts with the widget.
            
            spectrum_ydata = self.get_spectrum(delay_slider.val)[0]
            #line1.set_ydata(spectrum_ydata) # /np.max(np.abs(raw_ydata)))  
            
            ax1.cla()
            
            ax1.plot(self.wavelengths, spectrum_ydata, color='darkslateblue', linewidth=2, label = str(np.round(delay_slider.val, 2))+' ps')

            ax1.tick_params(axis="both", labelsize = tick_size)
            ax1.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize)
            ax1.set_ylabel(r'$\Delta$OD [mOD]', fontsize = axis_fontsize) # 'Difference Absorption (mOD)'
            ax1.set_xlim(self.wavelengths[0], self.wavelengths[-1])
            #ax1.set_ylim(-60, 15)
            ax1.legend(fontsize=tick_size, loc='upper right')
            ax1.set_title('Transient Absorption Difference Spectrum at '+str(np.round(delay_slider.val, 2))+' ps', fontsize = title_fontsize, y=1.02)
            ax1.grid(visible=True)
            
            fig.canvas.draw_idle()
            
            
            try:
                
                ax1.set_ylim(np.nanmin(spectrum_ydata) - (np.abs(np.nanmin(spectrum_ydata))/4), np.nanmax(spectrum_ydata) + (np.abs(np.nanmax(spectrum_ydata))/4))
                
                
            except:
                pass
             
        delay_slider.on_changed(update)
        
        
        # Buttons 
        resetax = fig.add_axes([0.05, 0.945, 0.1, 0.04])  # 0.15
        add_clickax = fig.add_axes([0.25, 0.945, 0.1, 0.04])
        
        reset_button = Button(resetax, 'Reset', color = 'lightsteelblue', hovercolor='gainsboro')
        reset_button.label.set_fontsize(20)
        add_click_button = Button(add_clickax, 'Add', color = 'lightsteelblue', hovercolor = 'gainsboro')
        add_click_button.label.set_fontsize(20)

        mem_values = []
        
        def add_click(event):
            # stores the current delay value selected with the slider in a list
            mem_values.append(delay_slider.val)
            
            cmap_ = plt.get_cmap('viridis')
            colors_ = [cmap_(i) for i in np.linspace(0, 1, len(mem_values))]
            
            fig_1 = plt.figure(figsize=(10,10))
            grid_1 = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], 
                        wspace=0.3, hspace=0.3)
        
            ax2 = fig_1.add_subplot(grid_1[0:7])
            ax2.cla()
            # plt.style.use('seaborn-v0_8')
            
            for i, color in enumerate(colors_, start = 0):
                #line1, = ax1.plot(self.wavelengths, self.get_spectrum(i)[0], linewidth = 2, label = str(np.round(i, 2))+' ps')
                
                ax2.plot(self.wavelengths, self.get_spectrum(mem_values[i])[0], color=color, linewidth = 2, label = str(np.round(mem_values[i], 2))+' ps')
                
                ax2.tick_params(axis="both", labelsize = tick_size)
                """
                ax2.tick_params(axis="y",
                                which='both',
                                left=False,
                                labelleft=False)
                """
                ax2.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize) # , labelpad=20)
                ax2.set_ylabel(r'$\Delta$OD [mOD]', fontsize = axis_fontsize, labelpad=20)
                ax2.set_xlim(self.wavelengths[0], self.wavelengths[-1])
                #ax2.set_ylim(-55, 15)
                ax2.legend(fontsize=tick_size, loc='upper right')
                ax2.set_title(self.filename+' Transient Absorption Difference Spectra', fontsize = title_fontsize, y=1.02)
                ax2.grid(visible=True)
                
                fig_1.canvas.draw_idle()
                
            return
            
        
        def reset(event):
            
            mem_values.clear()
            delay_slider.reset()
            
            ax1.cla()
            
            ax1.plot(self.wavelengths, spectrum_init[0], color='darkslateblue', linewidth = 2, label = str(np.round(self.delays[0], 2))+' ps')
                
            ax1.tick_params(axis="both", labelsize = tick_size)
            ax1.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize)
            ax1.set_ylabel(r'$\Delta$OD [mOD]', fontsize = axis_fontsize) # 'Difference Absorption (mOD)'
            ax1.set_xlim(self.wavelengths[0], self.wavelengths[-1])
            #ax1.set_ylim(-60, 15)
            ax1.legend(fontsize=tick_size, loc='upper right')
            ax1.set_title('Transient Absorption Difference Spectrum at '+str(np.round(self.delays[0], 2))+' ps', fontsize = title_fontsize, y=1.02)
            ax1.grid(visible=True)
            
            fig.canvas.draw_idle()


        reset_button.on_clicked(reset)
        resetax._button = reset_button
        
        add_click_button.on_clicked(add_click)
        add_clickax._button = add_click_button

        # TextBox to ajust axis limits
        """
        global xlim_tbox
        
        def update_xlim(lower):#, upper):
            bound_1 = find_nearest(self.wavelengths, lower)
            bound_2 = find_nearest(self.wavelengths, upper)
            #ax1.set_xlim(bound_1, bound_2)
            ax1.set_title(upper)
            fig.canvas.draw_idle()

        
        xlim_tbox_ax = fig.add_axes([0.45, 0.945, 0.1, 0.04])
        xlim_tbox = TextBox(xlim_tbox_ax, "xlims", textalignment="center")
        
        xlim_tbox.label.set_fontsize(20)
        xlim_tbox.on_submit(update_xlim)
        
        #xlim_tbox.set_val(np.round(self.wavelengths[0], 2))#, self.wavelengths[-1])
        """
        plt.show()
        
        return
        
####################################################################################
    
    def explore_spectra_trial(self):
        
        #def btn_reset_click(self, dummy):
        #    for i in range(1,len(self.indexlist)):
        #        self.indexlist.remove(self.indexlist[-1])
        
        self.indexlist = list([0]) 
        
        def btn_add_click(self):
            self.indexlist.append(self.indexlist[0])
        
            
        def plot_lines(self, repeat_nb):
            
            self.indexlist[0] = repeat_nb
            print(self.indexlist[0])
            #plt.clf()
            
            for index in self.indexlist:
                plt.figure()
                plt.plot(self.wavelengths, self.get_spectrum(index)[0])
                plt.show()
        
        interact(plot_lines(self=self, repeat_nb = widgets.FloatSlider(min = self.delays[0], max = self.delays[-1], value = self.delays[0]) )) 
        
        """
        add_clickax = plt.axes([0.25, 0.945, 0.1, 0.04])
        add_click_button = Button(add_clickax, 'Add', color = 'lightsteelblue', hovercolor = 'gainsboro')
        add_click_button.label.set_fontsize(20)

        add_click_button.on_clicked(btn_add_click)
        add_clickax._button = add_click_button
        """
        btn_add = widgets.Button(description='memorize', disabled=False, button_style='')
        btn_add.on_click(btn_add_click)
        display(btn_add)
                                 

        plt.show()
                


        return
                
 ####################################################################################   
        

    def explore_traces(self, cmap='PuOr', min_max=None, aspect='equal', interpolation='none',
                norm=None,scale='log', title=None, colourbar=False, xlabel=True, ylabel=True,
                yticks=True, xticks=True, show=False, raw=False, plot_dispersion=False, ax_min=-1, ax_max=-1):
        """
        Interactive widget with a slider to inspect traces at a given wavelength
        """

        trace_init = self.get_trace(self.wavelengths[0])
        

        fig = plt.figure(figsize=(10,10))
        grid = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], 
                        wspace=0.3, hspace=0.3)

        tick_size = 25
        axis_fontsize = 30
        title_fontsize = 25
        
        ax1 = fig.add_subplot(grid[0:7]) # grid[0:7]
        
        #line1, = ax1.plot(self.delays, trace_init[0], color='darkslateblue', linewidth=2)
        ax1.plot(self.delays, trace_init[0], color='darkslateblue', linewidth=2)
        
        ax1.tick_params(axis="both", labelsize = tick_size)
        ax1.set_xlabel('Delay (ps)', fontsize = axis_fontsize)
        ax1.set_ylabel(r'$\Delta$OD [mOD]', fontsize = axis_fontsize) # 'Difference Absorption (mOD)'
        ax1.set_xlim(ax_min, self.delays[ax_max])
        ax1.set_title('Trace at '+str(np.round(self.wavelengths[0], 2))+' nm', fontsize = title_fontsize, y=1.02)
        ax1.grid(visible=True)
        
        # Sliders
        global spectrum_slider
        
        axwave = plt.axes([0.1, 0.01, 0.8, 0.02])
        spectrum_slider = Slider(
            ax=axwave,
            label='Delay',
            valmin=self.wavelengths[0],
            valmax=self.wavelengths[-1],
            valinit=self.wavelengths[0],
            valstep=self.wavelengths[0:],
            color='lightsteelblue',
            #initcolor='none',
            #track_color='lightsteelblue',
            handle_style={'facecolor': 'white', 'edgecolor': '.05', 'size': 20} # .75
        )
        spectrum_slider.label.set_size(20)

        def update(val):
            
            #Updates the graphs when the user interacts with the widget.

            trace_ydata = self.get_trace(spectrum_slider.val)[0]
            #line1.set_ydata(trace_ydata) # /np.max(np.abs(raw_ydata)))    

            ax1.cla()
            
            ax1.plot(self.delays, trace_ydata, color='darkslateblue', linewidth=2, label = str(np.round(spectrum_slider.val, 2))+' nm')

            ax1.tick_params(axis="both", labelsize = tick_size)
            ax1.set_xlabel('Delay (ps)', fontsize = axis_fontsize)
            ax1.set_ylabel(r'$\Delta$OD [mOD]', fontsize = axis_fontsize) # 'Difference Absorption (mOD)'
            ax1.set_xlim(ax_min, self.delays[ax_max])
            ax1.legend(fontsize=tick_size, loc='upper right')
            ax1.set_title('Trace at '+str(np.round(spectrum_slider.val, 2))+' nm', fontsize = title_fontsize, y=1.02)
            ax1.grid(visible=True)
            
            fig.canvas.draw_idle()
            
            try:
                #ax1.set_ylim(np.nanmin(trace_ydata) - (np.abs(np.nanmin(trace_ydata))/2) , np.nanmax(trace_ydata) + (np.abs(np.nanmax(trace_ydata))/2))

                ax1.set_ylim(np.nanmin(trace_ydata) - (np.abs(np.nanmin(trace_ydata))/4), np.nanmax(trace_ydata) + (np.abs(np.nanmax(trace_ydata))/4))
            except:
                pass
                
        spectrum_slider.on_changed(update)

        
        # Buttons 
        resetax = fig.add_axes([0.05, 0.945, 0.1, 0.04])  # 0.15
        add_clickax = fig.add_axes([0.25, 0.945, 0.1, 0.04])
        
        reset_button = Button(resetax, 'Reset', color = 'lightsteelblue', hovercolor='gainsboro')
        reset_button.label.set_fontsize(20)
        add_click_button = Button(add_clickax, 'Add', color = 'lightsteelblue', hovercolor = 'gainsboro')
        add_click_button.label.set_fontsize(20)

        mem_values = []
        
        def add_click(event):
            # stores the current delay value selected with the slider in a list
            mem_values.append(spectrum_slider.val)
            
            cmap_ = plt.get_cmap('viridis')
            colors_ = [cmap_(i) for i in np.linspace(0, 0.5, len(mem_values))]
            
            fig_1 = plt.figure(figsize=(10,10))
            grid_1 = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], 
                        wspace=0.3, hspace=0.3)
        
            ax2 = fig_1.add_subplot(grid_1[0:7])
            ax2.cla()
            
            for i, color in enumerate(colors_, start = 0):
                #line1, = ax1.plot(self.wavelengths, self.get_spectrum(i)[0], linewidth = 2, label = str(np.round(i, 2))+' ps')
                
                ax2.plot(self.delays, self.get_trace(mem_values[i])[0], color=color, linewidth = 2, label = str(np.round(mem_values[i], 2))+' nm')
                
                ax2.tick_params(axis="both", labelsize = tick_size)
                ax2.set_xlabel('Delay (ps)', fontsize = axis_fontsize)
                ax2.set_ylabel(r'$\Delta$OD [mOD]', fontsize = axis_fontsize) # 'Difference Absorption (mOD)'
                ax2.set_xlim(ax_min, self.delays[ax_max])
                ax2.legend(fontsize=tick_size, loc='upper right')
                ax2.set_title(self.filename+' Traces', fontsize = title_fontsize, y=1.02)
                ax2.grid(visible=True)
                
                fig_1.canvas.draw_idle()
                
            return
            
        
        def reset(event):
            
            mem_values.clear()
            spectrum_slider.reset()
            
            ax1.cla()
            
            ax1.plot(self.delays, trace_init[0], color='darkslateblue', linewidth = 2, label = str(np.round(self.wavelengths[0], 2))+' nm')
                
            ax1.tick_params(axis="both", labelsize = tick_size)
            ax1.set_xlabel('Delay (ps)', fontsize = axis_fontsize)
            ax1.set_ylabel(r'$\Delta$OD [mOD]', fontsize = axis_fontsize) # 'Difference Absorption (mOD)'
            ax1.set_xlim(ax_min, self.delays[ax_max])
            ax1.legend(fontsize=tick_size, loc='upper right')
            ax1.set_title('Trace at '+str(np.round(self.wavelengths[0], 2))+' nm', fontsize = title_fontsize, y=1.02)
            ax1.grid(visible=True)
            
            fig.canvas.draw_idle()


        reset_button.on_clicked(reset)
        resetax._button = reset_button
        
        add_click_button.on_clicked(add_click)
        add_clickax._button = add_click_button

        # TextBox to ajust axis limits
        """
        global xlim_tbox
        
        def update_xlim(lower):#, upper):
            bound_1 = find_nearest(self.wavelengths, lower)
            bound_2 = find_nearest(self.wavelengths, upper)
            #ax1.set_xlim(bound_1, bound_2)
            ax1.set_title(upper)
            fig.canvas.draw_idle()

        
        xlim_tbox_ax = fig.add_axes([0.45, 0.945, 0.1, 0.04])
        xlim_tbox = TextBox(xlim_tbox_ax, "xlims", textalignment="center")
        
        xlim_tbox.label.set_fontsize(20)
        xlim_tbox.on_submit(update_xlim)
        
        #xlim_tbox.set_val(np.round(self.wavelengths[0], 2))#, self.wavelengths[-1])
        """
        plt.show()
        
        return
        

    def explore_fit(raw_data, fit_data, residual_data, cmap='PuOr', min_max=None, aspect='equal', interpolation='none',
                norm=None,scale='log', title=None, colourbar=False, xlabel=True, ylabel=True,
                yticks=True, xticks=True, show=False, raw=False, plot_dispersion=False):
        """
        Interactive widget with a slider to inspect the global fit at each input wavelength 
        using the array of data generated from a global fit 'fit_data' loaded in as an instance of the 'Rug' class
        """
        
        wavelengths_indicies = []
        wavelengths_values = []

        for idx, val in enumerate(fit_data.wavelengths):
            wavelengths_indicies.append(idx)
            wavelengths_values.append(val)

        
        wavelengths_series = pd.Series(wavelengths_indicies, index=wavelengths_values)
        
        raw_wl_init = raw_data.get_trace(fit_data.wavelengths[0])
        fit_data_wl_init = fit_data.abs.T[0]
        residual_init = residual_data.abs.T[0]
        
        """
        fig = plt.figure(figsize=(10,10))
        grid = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], 
                        wspace=0.3, hspace=0.3)
        
        ax = fig.add_subplot(grid[0:7])
        """
        
        fig = plt.figure(layout="constrained") # layout="constrained"
        gs = GridSpec(3, 1, figure=fig, wspace=0.2, hspace=0.15) #(2, 3)

        xlim = [raw_data.delays[5], raw_data.delays[150]] # set the value for the upper and lower bounds for the x axis
        
        ax = fig.add_subplot(gs[1:, :])
        
        line1, = ax.plot(fit_data.delays, fit_data_wl_init, color='lightsteelblue', linewidth=3, label = 'fit')
        line, = ax.plot(raw_data.delays, raw_wl_init[0]/np.max(np.abs(raw_wl_init[0])), '.C0', label = 'raw') # color='lightsteelblue', linewidth=2, alpha = 0.3,
        ax.tick_params(axis="both", labelsize = 15)
        ax.set_xlabel('Delay (ps)', fontsize = 20)
        ax.set_ylabel(r'$\Delta$OD [mOD]', fontsize = 20) # 'Difference Absorption (mOD)'
        ax.set_xlim(xlim[0], xlim[1])
        ax.legend(loc='lower right', fontsize = 20)
        #ax.set_title('Fit to the data at '+str(np.round(wavelengths_values[0], 2))+'nm', fontsize = 25, y=1.02)
        ax.grid(visible=True)
        
        ax1 = fig.add_subplot(gs[0, :])

        line2, = ax1.plot(fit_data.delays, residual_init, '.C3', label='Residual')

        ax1.tick_params(axis="both", labelsize = 15)
        ax1.set_ylabel(r'$\Delta$OD [mOD]', fontsize = 20) # 'Difference Absorption (mOD)'
        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(np.nanmin(residual_init)-0.1, np.nanmax(residual_init)+0.1)
        #ax1.set_title('Residual at '+str(np.round(wavelengths_values[0], 2))+'nm', fontsize = 25, y=1.04)
        ax1.legend(loc='lower right', fontsize = 20)
        ax1.grid(visible=True)
        
        # Sliders
        global wavelength_slider
        
        axwave = plt.axes([0.18, 0.655, 0.75, 0.015]) 
        # [0.15, 0.03, 0.72, 0.015] # slider along the bottom (hspace=0.3)
        # [0.15, 0.62, 0.72, 0.015] # slider located between ax and ax1 (hspace=0.3)
        wavelength_slider = Slider(
            ax=axwave,
            label='Wavelength (nm)',
            valmin=wavelengths_values[0],
            valmax=wavelengths_values[-1],
            valinit=wavelengths_values[0],
            valstep = wavelengths_values,
            color='lightsteelblue',
            handle_style={'facecolor': 'white', 'edgecolor': '.05', 'size': 20}, # .75
        )
        wavelength_slider.label.set_size(20)

        
        def update(val):
            
            #Updates the graphs when the user interacts with the widget.

            raw_ydata = raw_data.get_trace(wavelength_slider.val)[0]
            line.set_ydata(raw_ydata/np.max(np.abs(raw_ydata)))
            
            fit_ydata = fit_data.abs.T[wavelengths_series[wavelength_slider.val]]
            line1.set_ydata(fit_ydata)

            residual_ydata = residual_data.abs.T[wavelengths_series[wavelength_slider.val]]
            line2.set_ydata(residual_ydata)            
            fig.canvas.draw_idle()
            
            try:
                ax.set_ylim(np.nanmin(raw_ydata/np.max(np.abs(raw_ydata)))-0.2, np.nanmax(raw_ydata/np.max(np.abs(raw_ydata)))+0.2) 
                ax1.set_ylim(np.nanmin(residual_ydata)-0.1, np.nanmax(residual_ydata)+0.1)
                
                #ax.set_title('Fit to the data at '+str(np.round(wavelengths_slider.val, 2))+'nm', fontsize = 25, y=1.02)
                #ax1.set_title('Residual at '+str(np.round(wavelengths_slider.val, 2))+'nm', fontsize = 25)
                # ^ can I get the title to update...
            except:
                pass
                
        wavelength_slider.on_changed(update)
        
        plt.show()

        return 

    
    def indep_roll(arr, shifts, axis=0):
        # is this step redundant because result = arr ???
        
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
        
        arr = np.swapaxes(arr, axis, -1)
        """
        JDP swaps the axis you want to shift with the last axis (so makes the shifted axis -1)
        so in this instance extracts each unique value from the elements in 'data_interp' (where each element in 'data_interp' contained 
        the same set of interpolated difference absorption values and recasts the array so that there are n * len(time_shifts) elements 
        and each element contains one of the interpolated difference absorption values from the elements in the original array repeated
        n * len(self.interpolated_times)
        """

        #JDP get all the indices for any dimensionality 
        all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]
        """
        creates a list of elements between element 0 and element n in array.shape where each element 
        in the list contains a sequence of indices between 0 and element n from the original array.shape
        in this case, all_idcs = ([0, ... arr.shape[0] - 1], [0, ... arr.shape[1] - 1])
        equivalent to: np.ogrid[:arr.shape[0], :arr.shape[1]])
        """
    
        # Convert to a positive shift 
        shifts[shifts < 0] += arr.shape[-1]
        """
        shifts = time_shifts
        len(time_shifts) = 256
        arr.shape[-1] = len(self.interpolated_time)
        taken each negative value from time_shifts (all negative in this case) and added arr.shape[-1]
        """
    
        #JDP shift the last index by the shift amount (semi witchcraft)
        all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]
        """
        shifts[:, np.newaxis] takes the array of positive shift values with shape (256, ) and converts this to an array with shape (256, 1)
        i.e converts the 'shapes' array from [a, b, c, ... len(time_shifts) - 1] to ([a], [b], .... [[time_shifts] - 1])
        Subtracts each element from the new 'shifts' array from each element in all_idcs[-1]
        How does this work because len(all_idcs[-1]) = 1 and len(shifts[:, np.newaxis]) = 256 ?
        Somehow converts all_idcs[-1] to an array (256, 20234) where each element in that array contains a list of each value from the original 
        indices in all_idcs[-1] (0, 1 ... len(self.interpolated_time) - 1) subtract the value within each element of shifts[:, np.newaxis]
        
        so the new array of all_idcs[-1] contains len(time_shifts) elements and takes the form:
        
        [[0 - 20232, 1 - 20232, ...., 20233 - 20232]                    [[-20232, -20231, ...., 1]
                           :                                 =                       :              
         [0 - 20198, 1 - 20198, .... 20233 - 20198]]                     [-20198, -20197, ...., 35]]  

        And now the list all_idcs contains two elements:
        The first element is an array of indices between 0 and len(time_shifts) - 1, and the second element is the array described above.
        """
        
        #JDP index the array to the new indices (fully witchcraft) 
        result = arr[tuple(all_idcs)]     
        # convert the list 'all_idcs' to a tuple  so can use it to index the array 'arr' which presumably has some excess and only want the values
        # indexed by the indices in 'all_indcs' (in this case all of the value are kept)
        
        #JDP swap the axes back to how they originally were
        arr = np.swapaxes(result, -1, axis)   
        # in this case axis = time_shifts
        # overwrites the array 'arr' with the elements from arr indexed by the indices stored in 'all_idcs' and swaps the axis
        return arr
        
    def get_t0_position(self, degree):

        vmin = np.nanmin(self.abs) # returns the minimum and maximum values of the array of difference absorption values ignoring any NaN elements
        vmax = np.nanmax(self.abs)
        
        fig, ax = plt.subplots()
        # create a figure so I can adjust the position of the labels using coordinates on the figure axis
        
        im = ax.pcolormesh(self.wavelengths, self.delays, self.abs, cmap='PuOr', 
                      norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        # # use a 'diverging' cmap (e.i. RdBu) so that the cmap can be set using colors.TwoSlopNorm so that zero array values are blank
        
        ax.tick_params(axis="both", labelsize = 15)
        ax.xaxis.set_label_coords(0, -0.05)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        #ax.title.set_label_coords(0.5, 0.5)
        
        ax.set_xlabel('Wavelength (nm)', fontsize = 20)
        ax.set_ylabel('Delay (ps)', fontsize = 20, rotation = 360)
        ax.set_yscale('symlog')
        ax.set_title(f"Click t0 at {degree+1} or more points, then press enter", fontsize = 25, y=1.02) # fontweight='bold'
        
        tick_range = np.linspace(vmin, vmax, 10)
        cbar = fig.colorbar(im, ticks=tick_range) # change the fontsize ? [vmin, 0, vmax]
        cbar.set_label(label = 'Δ O.D', fontsize = 20, y = 0.52, labelpad = 30, rotation=360)
        
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()             # make figure automatically full screen (note hides the taskbar)
        
        plt.draw()
        self.calib_points = plt.ginput(n=-1, timeout=-1, show_clicks=True) 
        plt.close(fig) # closes the current figure 'fig'
        """
        generates an interactive plot, an input, and stores the value at each selection as an attribute of self
        n = -1 allows the user to click on the plot indefinitely until the input is terminated
        timeout = 0 or -, the input will never timeout
        the 'enter' key terminates the input and stores the selected values as list of x, y coordinates ([x, y], [x, y], ....)
        """
        return 
        
    def fit_dispersion(self, degree):
        
        wavelengths = [point[0] for point in self.calib_points] # takes the first value from each element of self.calib_points (x data)
        times = [point[1] for point in self.calib_points]       # takes the second value from each element of self.calib_points (y data)
        fit = pn.Polynomial.fit(wavelengths, times, deg=degree) 
        # returns a series instance that is a least squares fit to the data y (times) sampled at x (wavelengths)
        
        # why doesn't 'fit' produce a polynomial that fits the data, why do you have to convert?
        
        coefs = fit.convert().coef  
        # converts the series 'fit' then extracts the coefficients from this series. 
        # A plot of fit.convert().coefs against wavelengths fits the change in 'times' with 'wavelengths
    
        self.corrected_time = -pn.polyval(self.wavelengths, coefs)  
        # the amount to subtract off each value in self.delays to correct for the spatial dispersion
        # pn.polyval(x, c)
        # evaluate a polynomial at points, x 
        
        self.dispersion_coefs = coefs  # stores the coefficients of the fit.convert() as an attribute of self
        self.disp_wavelengths = wavelengths
        self.disp_times = times


        return
        
    def apply_dispersion_correction(self, coefs=[None]):
        """
        'apply dispersion correction' using the values saved in 'self.corrected_time' from fit_dispersion
        interpolates between the data points in the original difference absorption data matrix and evaluates the difference absorption
        at the new delay time points on the CHIRP corrected axis and saves this array as an attribute of 'self' (self.abs)
        
        # RugPeek edit
        """
        #JDP if user inputs coefficients, prioritise using these
        if all(coefs):
            #JDP note that you need to use polyval from the Polynomial class (pn), not base numpy...
            #JDP need to edit so that it will enter this if block if there is an array coefs.
            
            coefs = np.asarray(coefs) # converts the list of coefficients (if coefficients supplied as an input) to an array
            self.corrected_time = -pn.polyval(self.wavelengths, coefs)
            # the amount to subtract off each value in self.delays to correct for the spatial dispersion
            self.dispersion_coefs = coefs #stores the supplied coefficients as an attribute of self
 
        
        #JDP otherwise, if the coefs have not been given, check if they exist from a "by eye" fit and use that
        elif hasattr(self, "dispersion_coefs"): # checks if the Rug object 'self' has any values saved to the attribute 'dispersion_coefs'
            self.corrected_time = -pn.polyval(self.wavelengths, self.dispersion_coefs)
        else:
            raise Exception("need to find the dispersion correction before applying it, obvs")
        
        interp_min = np.min(self.delays)+np.min(self.corrected_time) 
        # add the minimum (negative) value in self.corrected_time to the minimum value in self.delays to obtain the corrected minimum value
        interp_max = np.max(self.delays)+np.max(self.corrected_time)
        # add the miximum (negative) value in self.corrected_time to the maximum value in self.delays to obtain the corrected maximum value
        
        #JDP interpolate the time axis so we can apply the chirp correction
        
        dt = np.min(np.ediff1d(self.delays))
        # finds the minimum difference between any two consecutive elements in self.delays, this will set the delay 'resolution'
        
        samples = int((interp_max-interp_min)/dt)
        # number of interpolation points is latest time - earliest time / smallest step
        
        x = np.linspace(interp_min, interp_max, samples)
        #JDP create a new axis to get the new time points on
        
        t_interp = np.interp(x, self.delays, self.delays)
        #.np.interp(x, x_data, y_data) evaluate the interpolated values at x
        #JDP interpolate from the old time axis (self.delays) to the new one (x)
        # len(x) = 20234
        # len(t_intep) = 20234
        # len(self.delays) = 239
        
        self.interpolated_time = t_interp   #save the interpolated delay values as an attribute of self
        
        #JDP get a meshgrid with the new interpolated time axis so we can evaluate interpolated TA data
        wavelength_2D, times_2D = np.meshgrid(self.wavelengths, self.interpolated_time, indexing="ij")
        # np.meshgrid(xi, indexing, ..) xi are the 1D arrays that represent the coordinates of the grid
        # indexing takes cartesion 'xy' or matrix 'ij' form
        """
        generates a list containing two elements:
        the first element in the list is a 1D array of length len(self.wavelengths) where each element in that array is a list of the same value
        from self.wavelengths repeated len(self.interpolated_time) times. The array is arranged so that the first element contains repeated values 
        of the first element from self.wavelengths and the last element contains repeat values of the last element in self.wavelengths.

        the second element in the list is also a 1D array of length len(self.wavelengths) where each element in that array contains a list of the values
        in self.interpolated time starting from self.interpolated_time[0] and ending at self.interpolated_time[-1]
        Thus the meshgrid contains ([a], [b]):
    
        where a takes the form: 
        ([len(self.interpolated_time)*self.wavelengths[0]],
                              :
        [len(self.interpolated_time)*self_wavelengths[-1]])
        so there are len(self_wavelengths) different elements in the array
        
        and b takes the form:
        ([self.interpolated_time[0] ...... self.interpolated_time[-1]] * len(self.wavelengths)
        so again there are len(self_wavelengths) different elements in the array

        in this case wavelength_2D is equivalent to a 
        and times_2D is equivalent to b
        """
                
        #JDP interpolate the actual TA data from the matrix
        interp2D = sp.interpolate.RegularGridInterpolator((self.delays, self.wavelengths), self.abs,
                bounds_error=False, fill_value=-9999, method='nearest')
        # sp.interpolate.RegularGridInterpolator(points, values,...) # this is just multi-demensional interpolation 
        # interpolates between the values (self.abs) where the points (x/y data = self.delays, self, wavelengths) are in the form of a grid
        
        #JDP evaluate the TA interpolation function over the whole grid
        data_interp = interp2D((times_2D, wavelength_2D)).T 
        
        #JDP no understanding of why this needs transposing, some subtlety with the interpolator?
          
        #JDP note NB pcolormesh with nearest shading centers each time point on a grid cell, and extends the cell
        # to edges defined by the distance to the next time point. this can make it look like the plot is 
        # making fake data at different interpolation levels, but it isn't. explicitly limit the plot 
        # to the limits of the time axis to avoid confusion
        
        #JDP  turn this into a number of elements to shift given the interpolation scale
        time_shifts = np.rint(self.corrected_time/dt).astype(np.int64) 
        #np.rint() rounds each element to the nearest integer
        # take the nearest integer value of each element in self.correct_time / dt
        # where dt is the smallest difference between any consecutive values from self.delays i.e. the 'resolution'
        # then convert the values in the array from floats to integers


        """
        I don't think this code is used for anything...
        
        #JDP get the maximum shift, as we'll pad the interpolated matrix by this to ensure we don't roll over edges
        maxshift = np.max(np.abs(time_shifts)) #int(....)
        
        #JDP pad the matrix with an unphysical value ### why? ###
        
        matrix_padded = np.pad(data_interp, ((maxshift, maxshift), (0,0)), mode='constant',
            constant_values=-99)
        # generates a matrix of the same dimensions as the matrix filled with interpolated difference absorption data (data_interp)
        # where each value = -99
        """
        
        #JDP save the raw (uncorrected) matrix
        self.raw_abs = self.abs.copy()
        
        #JDP apply the chirp correction by rolling each wavelength's time axis by the necessary amount
        shifted_matrix = Rug.indep_roll(data_interp, time_shifts, axis=0) 
        # figure out how 'indep_roll' works at some point

        
        #JDP now need to "undo" the interpolation. should just be:
        time_indices = []
        
        for time in self.delays:
            time_idx, _ = Rug.find_nearest(self.interpolated_time, time)
            time_indices.append(time_idx)
        """
        for each value in 'self.delays', finds the closest value in 'self.interpolated_time' and stores the index of the value
        from 'self.interpolated_time' in the list 'time_indices'
        """
        print(shifted_matrix.shape)
        #plt.plot(shifted_matrix)
        
        self.abs = shifted_matrix[time_indices, :] 
        # saves the interpolated difference absorbance data from the array 'shifted_matrix' in the region specified by the indices stored 
        # in 'time_indices' as an attribute of 'self'
        
        return
        

    def interpolate_time_axis(rug, mintime=None, maxtime=None,
                              timestep=None, method='nearest',
                              undo=True, verbose=False):       # called by combine_rugs_wavelengths

        if mintime:
            interp_min = mintime
        else:
            interp_min = np.min(rug.delays)     # finds the minimum value in self.delays to use as the minimum value to interpolate from
                                                # if this has not been supplied as in input
        if maxtime:
            interp_max = maxtime
        else:
            interp_max = np.max(rug.delays)     # finds the maximum value in self.delays to use as the maximum value to interpolate up to
                                                # if this has not been supplied as an input
        if timestep:
            dt = timestep
        else:
            dt = np.min(np.ediff1d(rug.delays))   # obtains the minimum difference between any two consecutive elements in self.delays
                                                  # to obtain the 'resolution' of the interpolated delay axis
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
        t_interp = np.interp(x, rug.delays, rug.delays)

        #JDP get a meshgrid with the new interpolated time axis so we can evaluate interpolated TA data
        wavelength_2D, times_2D = np.meshgrid(rug.wavelengths, t_interp, indexing="ij")
        
        #JDP interpolate the actual TA data from the matrix (or, create a function to do it)
        interp2D = sp.interpolate.RegularGridInterpolator((rug.delays, rug.wavelengths), rug.abs,
                bounds_error=False, fill_value=-9999, method=method)

        data_interp = interp2D((times_2D, wavelength_2D)).T 
        
        if undo:
            time_indices = []
            for time in rug.delays:
                time_idx, _ = RugTools.find_nearest(t_interp, time)
                time_indices.append(time_idx)

            rug.abs = data_interp[time_indices, :]
            rug.delays = t_interp[time_indices]
        else:
            rug.abs = data_interp
            rug.raw_times = rug.delays
            rug.delays = t_interp
        return t_interp

    def undo_time_interpolation(rug, new_axis):                # called by combine_rugs_wavelengths
        
        time_indices = []
        for time in new_axis:
            time_idx, _ = Rug.find_nearest(rug.delays, time)   # stores the indices of the values from 'self.delays' closest to the values supplied
            time_indices.append(time_idx)                      # in the input 'new_axis' in the list 'time_indices'

        rug.abs = rug.abs[time_indices, :]                     
        rug.delays = rug.delays[time_indices]
        """
        returns a matrix of interpolated difference absorption values at the values from the interpolated difference absorption data matrix 
        at the indices stored in 'time_indices'
        and a matrix of interpolated delay time points from the interpolated delay time points matrix at the indicies stored in 'time_indices'
        """
        return

    def correct_offset(self, coefs):
        """
        adjust the Δ O.D values associated with each element of 'delays_ps' to correct for time zero error 
        and save the corrected axis in the list corrected_delays
        # RugPeek edit
        """  
        times = []
        count = 0

        vmin = np.nanmin(self.abs)
        vmax = np.nanmax(self.abs)
        
        #if not hasattr(self, "axis"):
        #    self.peek()
        
        while count < 1:
            
            fig, ax = plt.subplots()
            # create a figure so I can adjust the position of the labels using coordinates on the figure axis
            
            im = ax.pcolormesh(self.wavelengths, self.delays, self.abs, cmap='PuOr', 
                          norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
            # # use a 'diverging' cmap (e.i. RdBu) so that the cmap can be set using colors.TwoSlopNorm so that zero array values are blank
            
            ax.tick_params(axis="both", labelsize = 15)
            ax.xaxis.set_label_coords(0, -0.05)
            ax.yaxis.set_label_coords(-0.08, 0.5)
            #ax.title.set_label_coords(0.5, 0.5)
            
            ax.set_xlabel('Wavelength (nm)', fontsize = 20)
            ax.set_ylabel('Delay (ps)', fontsize = 20, rotation = 360)
            ax.set_yscale('symlog')
            ax.set_title(f"Click on the graph to indicate where time zero should be", fontsize = 25, y=1.02) # fontweight='bold'
            
            tick_range = np.linspace(vmin, vmax, 10)
            cbar = fig.colorbar(im, ticks=tick_range) # change the fontsize ? [vmin, 0, vmax]
            cbar.set_label(label = 'Δ O.D', fontsize = 20, y = 0.52, labelpad = 30, rotation=360)

            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()             # make figure automatically full screen (note hides the taskbar) 
                                                     # hides tabs so add plt.close(fig) otherwise have to use 'esc' key to exit                                  
            plt.draw()                               
            pts=plt.ginput(1)  # stores the x and y data at the point of the graph that is 'clicked' in an array ([x, y], [], ....)
            time=pts[0][1]     # stores the second value from the first element in the variable 'time' (this is the delay time)
            times.append(time)
            
            plt.hlines(y=time, xmin = self.wavelengths[0], xmax = self.wavelengths[-1])
            count += 1
            plt.pause(0.05)
            plt.close(fig)   # closes the current figure 'fig'
        
        
        offset = coefs - times # why do the coefs need to be included, why not just subtrace 'times' directly from each value in 'self.delays' ?
        
        corrected_delays = self.delays + offset   # offset each value in self.delays and save as a new variable 'correct_delays'
        self.delays = corrected_delays            # overright self.delays with the offset corrected delay times

        fig, ax = plt.subplots()
        # create a figure so I can adjust the position of the labels using coordinates on the figure axis
        
        im = ax.pcolormesh(self.wavelengths, self.delays, self.abs, cmap='PuOr', 
                      norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        # # use a 'diverging' cmap (e.i. RdBu) so that the cmap can be set using colors.TwoSlopNorm so that zero array values are blank
        
        ax.tick_params(axis="both", labelsize = 15)
        ax.xaxis.set_label_coords(0, -0.05)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        #ax.title.set_label_coords(0.5, 0.5)
        
        ax.set_xlabel('Wavelength (nm)', fontsize = 20)
        ax.set_ylabel('Delay (ps)', fontsize = 20, rotation = 360)
        ax.set_yscale('symlog')
        ax.set_title(f'corrected offset', fontsize = 25, y=1.02) # fontweight='bold'
        
        tick_range = np.linspace(vmin, vmax, 10)
        cbar = fig.colorbar(im, ticks=tick_range) # change the fontsize ? [vmin, 0, vmax]
        cbar.set_label(label = 'Δ O.D', fontsize = 20, y = 0.52, labelpad = 30, rotation=360)
        
        return

        
    def combine_rugs_wavelengths(rugs, fname, show=False):
        # have to input rugs in the order, SHG, FUN because of how the 'start' and 'end' wavelengths are selected for the new wavelengths axis
        
        #JDP assume first file is 'origin'
        #JDP if you want to combine more than 2 needs editing.
        #JDP assumes that you want to trim overlapping wavelengths from specturm 1, also needs changing

        #JDP need to interpolate everything onto the same time grid for this to be reliable
        #JDP get the min and max times for the interpolation
        
        mintime = np.min(np.array([i.delays for i in rugs]))
        # finds the minimum value from all of the values in the 'self.delays' attributes of the inputted 'rugs'
        maxtime = np.max([i.delays for i in rugs])
        # finds the maximum value from all of the values in the 'self.delays' attributes of the inputted 'rugs'
        
        original_time = rugs[0].delays
        #JDP interpolate the data onto a common time axis
        for rug in rugs:
            Rug.interpolate_time_axis(rug, mintime=mintime, maxtime=maxtime,  # generates an interpolated time axis between 'mintime' and 'maxtime'
                                           undo=False, verbose=False)
        
        start_wl = rugs[-1].wavelengths[0] #first wl of the most red spectrum
        end_wl = rugs[0].wavelengths[-1] #last wl of the most blue spectrum
        # ^ this indexing imposes strict rules on the order of the inputted rugs i.e must be: SHG, FUN and NOT: FUN, SHG
            
        overlap_point_blue, _ = Rug.find_nearest(rugs[0].wavelengths, start_wl)
        overlap_point_red, _ = Rug.find_nearest(rugs[-1].wavelengths, end_wl)
        # ^ finds the index of the closest value in self.wavelengths from the SHG rug to the value in 'start_wl'
        # and the index of the closest value in self.wavelengths from the FUN rug to the value in 'end_wl'
        
        overlap_points = [[0, overlap_point_blue], [0, -1]] # indices for the overlap region
        
        combined_rug = Rug(type=False) 
        # type = False: so the new Rug instance has attributes generated when a 'matrix.dat' as opposed to a '.dat' file is loaded as a Rug instance.
        combined_rug.abs = np.concatenate([i.abs[:, overlap_points[idx][0]:overlap_points[idx][1]] for idx, i in enumerate(rugs)], axis=1)
        """
        np.concatenate() joints array elements along the same axis. These arrays must have the same dimension except for along the axis
        on which they are being joined
        In this instance, takes self.abs at the indicies specified by first generating a an index value by iterating over the elements in 'rugs'
        then using this to index the elements in 'overlap_points' 
        
        So for i = SHG rug: takes difference absorption values from the array of difference absorption values from the SHG rug 
        indexed using [:, overlap_points[idx][0]:overlap_points[idx][1]] = [:, 0 : overlap_point_blue] for idx = 0 and concatenates these values 
        with the difference absorption values for i = FUN rug, idx = 1, [:, overlap_points[idx][0]:overlap_points[idx][1]] = [:, 0:-1]
        """
        combined_rug.wavelengths = np.concatenate([i.wavelengths[overlap_points[idx][0]:overlap_points[idx][1]] for idx, i in enumerate(rugs)], axis=0)
        # similarly concatenates the values from self.wavelengths from both rugs to 'join' the wavelength values between the SHG and FUN regions
        combined_rug.delays = rugs[0].delays
        """
        ^ temporarily assign the combined rug the same delay points as the SHG rug, the delay axis is subseqeuntly replaced by an axis 
        of interpolated delay time points at the values closest to the values in rugs[0].delays 
        """
        combined_rug.filename = fname   
        
        Rug.undo_time_interpolation(combined_rug, original_time)
        """
        stick it back onto the original axis for speed 
        i.e just take the difference absorption values at the time points from the original delay 
        axis rather than the interpolated time axis which sampled the difference absorption values at each point in the list 'samples'
        where len(samples) >>> len(self.delays)
        """
        if show:
            combined_rug.peek()

        return combined_rug
        
    
    #def cut_wavelengths(self, wlranges, fill=0):    # wlranges are the regions to remove [(a, b), (c, d)]
        """
        
        """
        """
        if hasattr(self, 'uncut_matrix'):
            pass
        else:
            self.uncut_matrix = np.copy(self.abs)   # saves an 'uncut' array as an attribute of self (in case want to retrieve 'uncut' data)

        for pair in wlranges:
            wlmin_idx, _ = Rug.find_nearest(self.wavelengths, pair[0])   
            wlmax_idx, _ = Rug.find_nearest(self.wavelengths, pair[1])
            # finds the nearest values in self.wavelengths to the values input in 'pair'
            self.abs[:, wlmin_idx:wlmax_idx] = fill  
            # the fill argument is set to 0:
            # assigns zero values between the limits wlmin_idx':wlmin_idx in the difference absorption array
        return
        """

    def cut_wavelengths(self, wl_range_keep):    # wlranges are the regions to remove [(a, b), (c, d)]
        """
        Store regions of the array containing difference absorption data you want to keep with step-like boundaries 
        """

        restricted_abs_shape = [0]
        restricted_wavelengths = []
        
        for range in wl_range_keep:
            wlmin_idx, _ = Rug.find_nearest(self.wavelengths, range[0])   
            wlmax_idx, _ = Rug.find_nearest(self.wavelengths, range[1])
            #restricted_abs.append(self.abs[wlmin_idx : wlmax_idx])
            #print(self.abs[:, wlmin_idx:wlmax_idx].shape)
            restricted_abs_shape.append(self.abs[:, wlmin_idx:wlmax_idx].shape[1])
            
            for i in self.wavelengths[wlmin_idx : wlmax_idx]:
                restricted_wavelengths.append(i)
                
        #print(np.sum(restricted_abs_shape))
        restricted_abs = np.zeros(shape=(len(self.delays), np.sum(restricted_abs_shape)))
        
        #print(restricted_abs_shape)
        #print(restricted_abs.shape)

        for idx, range in enumerate(wl_range_keep):
            wlmin_idx, _ = Rug.find_nearest(self.wavelengths, range[0])   
            wlmax_idx, _ = Rug.find_nearest(self.wavelengths, range[1])
            #print(restricted_abs_shape[idx], "\n")
            #print(restricted_abs_shape[idx+1], "\n")
            
            #print(restricted_abs[:, restricted_abs_shape[idx]:(restricted_abs_shape[idx]+restricted_abs_shape[idx+1])].shape)
            #print(self.abs[:, wlmin_idx:wlmax_idx].shape)
            
            restricted_abs[:, restricted_abs_shape[idx]:(restricted_abs_shape[idx]+restricted_abs_shape[idx+1])] = self.abs[:, wlmin_idx:wlmax_idx]
        
        self.abs = restricted_abs
        self.wavelengths = np.array(restricted_wavelengths) ########################################################
        
        return
        
    #def cut_wavelengths(self, wl_ranges):
        """
        if hasattr(self, 'uncut_matrix'):
            pass
        else:
            self.uncut_matrix = np.copy(self.abs)

        for pair in wl_ranges:
            wlmin_idx, _ = Rug.find_nearest(self.wavelengths, pair[0])   
            wlmax_idx, _ = Rug.find_nearest(self.wavelengths, pair[1])
            # finds the nearest values in self.wavelengths to the values input in 'pair'
            del self.abs[:, wlmin_idx:wlmax_idx] 
            del self.wavelengths[wlmin_idx:wlmax_idx]

        return
        """
        
    def limit_times(self, tmin=None, tmax=None):
        
        if not hasattr(self, 'uncut_matrix'):
            self.uncut_matrix = np.copy(self.abs)
        if not hasattr(self, 'uncut_times'):
            self.uncut_times = np.copy(self.delays)
            
        if tmax:
            tidx, _= Rug.find_nearest(self.delays, tmax)
            self.abs = self.abs[:tidx, :]
            self.delays = self.delays[:tidx]
        if tmin:
            tidx, _ = Rug.find_nearest(self.delays, tmin)
            self.abs = self.abs[tidx:, :]
            self.delays = self.delays[tidx:]
        return
        

    def background_subtract(self, verbose=False, show=False):
        """
        User defines an area for background removal by clicking two points on the plot. 
        An average value is calulated from the defined area and is then removed from the entire matrix. 
        It is important to remove the region of the spectrum contaminated by the pump/probe light prior to applying this function
        else the region subtracted form the spectrum will not reflect the true difference absorption spectrum due the background.
        """
        
        times = []
        
        vmin = np.nanmin(self.abs)  # obtains the minimum difference absorption value from 'self.abs'
        vmax = np.nanmax(self.abs)  # obtains the maximum difference absorption value from 'self.abs'

        fig, ax = plt.subplots()
        # create a figure so I can adjust the position of the labels using coordinates on the figure axis
        
        im = ax.pcolormesh(self.wavelengths, self.delays, self.abs, cmap='PuOr', 
                      norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        # # use a 'diverging' cmap (e.i. RdBu) so that the cmap can be set using colors.TwoSlopNorm so that zero array values are blank
        
        ax.tick_params(axis="both", labelsize = 15)
        ax.xaxis.set_label_coords(0, -0.05)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        #ax.title.set_label_coords(0.5, 0.5)
        
        ax.set_xlabel('Wavelength (nm)', fontsize = 20)
        ax.set_ylabel('Delay (ps)', fontsize = 20, rotation = 360)
        ax.set_yscale('symlog')
        ax.set_title(f"Click 2 lines to define area for background removal", fontsize = 25, y=1.02) # fontweight='bold'
        
        tick_range = np.linspace(vmin, vmax, 10)
        cbar = fig.colorbar(im, ticks=tick_range) # change the fontsize ? [vmin, 0, vmax]
        cbar.set_label(label = 'Δ O.D', fontsize = 20, y = 0.52, labelpad = 30, rotation=360)

        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()             # make figure automatically full screen (note hides the taskbar)
        
        plt.draw()
        self.background_points = plt.ginput(n = 2, timeout = -1, show_clicks=True)  
        time = [point[1] for point in self.background_points]
        print("The first delay point selected was:", np.round(time[0], 3), "ps", "\nThe second delay point selected was:", np.round(time[1], 3), "ps")
        times.append(time)
        #plt.hlines(y=time[0], xmin = self.wavelengths[0], xmax = self.wavelengths[-1])
        #plt.hlines(y=time[1], xmin = self.wavelengths[0], xmax = self.wavelengths[-1])
        plt.pause(0.05)
        plt.close(fig)

        maximum = np.max(times)
        minimum = np.min(times)
        
        max_index, actualmax = Rug.find_nearest(self.delays, maximum)
        min_index, actualmin = Rug.find_nearest(self.delays, minimum)
      
        area_indexes = []
        
        for i in range(min_index, max_index+1):
            area_indexes.append(i)
  
        background_region = self.abs[[area_indexes], :].T

        mean_background = np.mean(background_region, axis =1)
          
        background_matrix = np.zeros_like(self.abs)
        
        background_matrix = np.stack([mean_background for i in range(len(self.delays))], axis=0)[:,:,0]
        
        if not hasattr(self, 'unsubtracted_matrix'):
            self.unsubtracted_matrix = self.abs.copy()

        matrix = self.abs - background_matrix
        self.abs = matrix
        if show:
            self.peek(plotted_matrix='unsubtracted')
            self.peek()
        return

    def savefile(self, prefix=None): # coefc, dispersion, offset
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
        
        lent = len(self.delays)
        lenm = len(self.wavelengths)

        array = np.zeros((lent+1, lenm+1))

        array_wl = self.wavelengths
        array_times = self.delays[:,None].reshape((lent,))
        array_matrix = self.abs

        array[0,1:] = array_wl
        array[1:,0] = array_times
        array[1:, 1:] = array_matrix
        
        if hasattr(self, 'metadata'):
            metadata = str(self.metadata)
        else:
            metadata = ''

        today = date.today()

        current_time = time.localtime()
        timestring = time.strftime("%d/%m/%Y %H:%M:%S")
      
        file = np.savetxt(prefix+self.filename+'_processed'+'.dat', array,
                          header='XAxisTitleWavelength(nm)\nYAxisTitle Delay (ps)\n'
                          +metadata+'\n'+'Date&time: '+ timestring)
        
        # '\n'+coefc+'\n'+dispersion+'\n'+offset, + comments=''
        print("\033[1m" + 'Saved processed file of' + "\033[0;0m", self.filename)
       
        fig = self.peek() #colourbar=True
        plt.savefig(self.filename+"_processed.png")
        #plt.savefig('myfig')

        if hasattr(self, 'sig_x'):
            array = np.zeros((lent+1, lenm+1))

            array_wl = self.wavelengths
            array_times = self.delays[:,None].reshape((lent,))
            array_matrix = self.sig_x
    
            array[0,1:] = array_wl
            array[1:,0] = array_times
            array[1:, 1:] = array_matrix
            
            np.savetxt(prefix+self.filename+'_st_dev_arr'+'.dat', array,
                        header=metadata+'\n'+'Date&time: '+ timestring)

        else:
            return
        
        
        if hasattr(self, 'fit_data'):
            
            fit_data_arr = np.zeros((lent+1, len(self.fit_wavelengths)+1))
        
            fit_data_arr[0,1:] = self.fit_wavelengths  
            fit_data_arr[1:,0] = array_times
            fit_data_arr[1:, 1:] = np.array(self.fit_data).T # probably don't need to transpose here because transpose again in 'explore_fit'
            
            np.savetxt(prefix+self.filename+'_fit_data'+'.dat', fit_data_arr,
                          header='XAxisTitleWavelength(nm)\nYAxisTitle Delay (ps)\n'
                          +metadata+'\n'+'Date&time: '+ timestring)
        else:
            return

        if hasattr(self, 'residual_fit_data'):
            
            residual_data_arr = np.zeros((lent+1, len(self.fit_wavelengths)+1))

            residual_data_arr[0,1:] = self.fit_wavelengths  
            residual_data_arr[1:,0] = array_times
            residual_data_arr[1:, 1:] = np.array(self.residual_fit_data).T

            np.savetxt(self.filename+'_residual_fit_data'+'.dat', residual_data_arr,
                          header='XAxisTitleWavelength(nm)\nYAxisTitle Delay (ps)\n'
                          +metadata+'\n'+'Date&time: '+ timestring)
        else:
            return
            
         
        return


    
    def compute_SVD(self, threshold, tol=1.0E-10):

        try:
            U, S, V = np.linalg.svd(self.abs, full_matrices=True)
            """
            For the array m*n (239, 453) array 'self.abs' np.linal.svd() returns:
            V is an n*n (453, 453) array of 'principal spectra'
            S is an 1D array of length m (239), containing the singular values arranged in descending order so that S[0] >>> S[-1]
            U.T is an m*m (239, 239) array of 'principle kinetics'
            
            if set full_matrices = False, matrix V is n*m not m*m
            """
            
            S[np.abs(S) < tol] = 0   # makes any singular values in the array S < tol equal to 0
        
        except:

            raise Exception('SVD did not converge - correct data loaded?')

        self.singular_values = S
        self.principal_spectra = V       
        self.principal_kinetics = U.T    
        # is it U.T instead of V.T here because A (self.abs) is m*n (239, 453) and for V.T, A must be an n*m array i.e (453, 239)?
        
        # calculate the number of significant singular values (S) from the threshold input:
        singular_threshold = np.min(S[np.nonzero(S)]) * threshold
        print(f'Taking threshold for singular value significance as {threshold}x larger than smallest singular value ({np.round(singular_threshold, 9)}).')
        singular_differences = np.abs(np.ediff1d(self.singular_values))
        S_total = np.sum(self.singular_values**2)
        self.singular_fractions = self.singular_values**2 / S_total
        self.relevant_sing = np.array([ i for i in singular_differences if i > singular_threshold])
        print(f'Found {len(self.relevant_sing)} significant singular values')

        cpt_sum = 0
        for i in range(len(self.relevant_sing)):
            cpt_sum = cpt_sum + np.round(self.singular_fractions[i]*100, 2)
            print(f'{np.round(self.singular_fractions[i]*100,2)} % of data variance described by component {i+1}.')

        print(f'Overall, {np.round(cpt_sum,2)} % of data variance described by the first {len(self.relevant_sing)} components.')
        print(f'Leaving {np.round((100-cpt_sum),2)} % of data variance described by the the remaining {np.round(len(self.singular_values)-len(self.relevant_sing),2)} components which are below threshold (roughly {np.round((100-cpt_sum)/(len(self.singular_values) -len(self.relevant_sing)),2)} % per component).')

        self.relevant_spectra = self.principal_spectra[0:len(self.relevant_sing)]
        self.relevant_kinetics = self.principal_kinetics[0:len(self.relevant_sing)]
        self.relevant_matrices = np.array([self.relevant_sing[i]*(np.outer(self.relevant_kinetics[i],self.relevant_spectra[i]))
                                           for i in range(len(self.relevant_sing))])
        
        self.reconstructed_TA = np.sum(self.relevant_matrices, axis=0)
        
        return


    
    def SVD_explorer(self, x_axmin=-1, x_axmax=120):
        """
        generates a widget containing a plot of the SVD eigenvalues, traces, spectra for n components as well as colourmap of the original 
        array of difference absorption values 
        """
        
        fig = plt.figure(figsize=(10,10))
        grid = GridSpec(2, 2, width_ratios=[3, 3], 
                        height_ratios=[3, 3], wspace=0.3, hspace=0.3)
        
        ax1 = fig.add_subplot(grid[0])
        
        vmin = np.min(self.abs)
        vmax = np.max(self.abs)
    
        im = ax1.pcolormesh(self.wavelengths, self.delays, self.abs,
                cmap='PuOr', norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        
        ax1.set_ylabel('Time delay (ps)', fontsize = 15)
        ax1.set_xlabel('Wavelength (nm)', fontsize = 15)
        ax1.set_yscale('symlog')
        ax1.set_yscale('symlog')
        ax1.set_title(self.filename, fontsize = 20)
        
        _cmap = plt.get_cmap('viridis') # 'inferno'
        _colors = [_cmap(i) for i in np.linspace(0, 1, len(self.relevant_spectra)*100)]  
        # just number of components plus some extra to get better colors
        # be careful with indexing because currently will go out of range if more components are added
       
        ax2 = fig.add_subplot(grid[1])
        for i, trace in enumerate(self.relevant_kinetics):
            ax2.plot(self.delays, trace, color = _colors[i*100], label='Component '+str(i+1))
        ax2.set_ylabel('Amplitude (a.u)', fontsize = 15)
        ax2.set_xlabel('Time delay (ps)', fontsize = 15)
        #ax2.set_yscale('symlog')
        #ax2.set_xscale('symlog')
        ax2.set_xlim(x_axmin, self.delays[x_axmax]) # self.delays[0], self.delays[-1]
        #h, l = ax2.get_legend_handles_labels()
        #ax2.legend(h, l)
        ax2.legend()
        ax2.set_title('Principle Kinetics', fontsize = 20)


        ax3 = fig.add_subplot(grid[2])
        for i, spectrum in enumerate(self.relevant_spectra):
            ax3.plot(self.wavelengths, spectrum, color = _colors[i*100], label='Component '+str(i+1))
        ax3.set_ylabel('Amplitude (a.u)', fontsize = 15)
        ax3.set_xlabel('Wavelength (nm) ', fontsize = 15)
        ax3.set_xlim(left = self.wavelengths[0], right = self.wavelengths[-1])
        ax3.legend()
        ax3.set_title('Principle Spectra', fontsize = 20)

        ax4 = fig.add_subplot(grid[3])
        ax4.scatter(range(0, len(self.singular_values)), self.singular_values, s=10, c=_colors[1])
        ax4.set_ylabel('Amplitude (a.u)', fontsize = 15)
        ax4.set_xlabel('Component number', fontsize = 15)
        ax4.set_title('Singular Values', fontsize = 20)

        
        plt.show()
        
        
        for i, matrix in enumerate(self.relevant_matrices):
                    fig, ax = plt.subplots()
                    # create a figure so I can adjust the position of the labels using coordinates on the figure axis
                    
                    vmin = matrix.min()
                    vmax = matrix.max()/10
                    
                    im = ax.pcolormesh(self.wavelengths, self.delays, matrix, cmap='PuOr', 
                                  norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
                    # # use a 'diverging' cmap (e.i. RdBu) so that the cmap can be set using colors.TwoSlopNorm so that zero array values are blank
                    
                    ax.tick_params(axis="both", labelsize = 15)
                    ax.xaxis.set_label_coords(0, -0.05)
                    ax.yaxis.set_label_coords(-0.08, 0.5)
                    #ax.title.set_label_coords(0.5, 0.5)
            
                    tick_range = np.linspace(vmin, vmax, 10)
                    cbar = fig.colorbar(im, ticks=tick_range) # change the fontsize ? [vmin, 0, vmax]
                    cbar.set_label(label = 'Amplitude a.u', fontsize = 20, y = 0.52, labelpad = 50, rotation=360)
                    
                    ax.set_xlabel('Wavelength (nm)', fontsize = 20)
                    ax.set_ylabel('Delay (ps)', fontsize = 20, rotation = 360)
                    ax.set_yscale('symlog')
                    ax.set_title('Component '+str(i+1), fontsize = 20)
                    plt.show()
        
        return
        

    def find_component_spectra(self, train_size):
        """
        Function for predicting component species spectra from the raw difference absorption dataset
        Note that the current constraints passed to McrAR are that the data is both non - negative and normalized.
        """
        positive_abs = self.abs # - np.min(self.abs) 
        # comment out this step for datasets that already contain positive values only i.e pumped / unpumped absorption datasets
        
        est_pure_spectra, X_test = ks.train_test_split(positive_abs, train_size = train_size) 
        # train_size is the number of components contributing to the transient absorption difference spectrum
        
        mcrar = McrAR(max_iter=1000, st_regr='NNLS', c_regr='OLS', c_constraints=[]) #[ConstraintNonneg(), ConstraintNorm()]
        mcrar.fit(positive_abs, ST=est_pure_spectra)
        
        self.components = mcrar.ST_ + np.min(self.abs)
        
        
        with plt.style.context(('seaborn-whitegrid')):
            for i, val in enumerate(mcrar.ST_):
                
                fig, ax = plt.subplots(figsize=(8, 5))
                plt.plot(self.wavelengths, mcrar.ST_[i], 'black', label = 'Component '+str(i+1)) 
                
                plt.xlabel("Wavelength (nm)", fontsize=14)
                plt.ylabel("Difference Absorption (O.D)", fontsize=14)
                #plt.xlim(470, 700)
                #plt.ylim(0, 2.25)
                plt.legend()
                plt.show()

        return
        

    def global_fitter(self, fit_wavelengths, xdata, ydata, paramdata, system, extradata, ncpts=None, vars_per_cpt=None, prefixcpt='c', prefixdata='d',
                 fixed_vars=None, offset_vars=None, fixed_data_vars=None, 
                 model=lf.models.ExpressionModel, method=None, normalise=True, verbose=False, save_fit_report=False):
    
        # catch it in the case that you feed in 1D data
        if len(ydata.shape) == 1:
            ydata = np.expand_dims(ydata, axis=0)
    
        params = Rug.create_params_object(ydata, paramdata, prefixdata, prefixcpt, 
                                fixed_vars=fixed_vars, offset_vars=offset_vars, fixed_data_vars=fixed_data_vars,
                                 ncpts=ncpts, vars_per_cpt=vars_per_cpt)
        if verbose:
            params.pretty_print()
        
        out = lf.minimize(Rug.builtin_model, params, args = (xdata, ydata, ncpts, model, prefixdata, prefixcpt), method=method, 
                          nan_policy='omit', max_nfev=None)
        # nan_policy = 'omit' (handle NaN values) 
    

        # store information from the fit report as an array and save to a '.dat' file
        if save_fit_report:

            today = date.today()
            current_time = time.localtime()
            timestring = time.strftime("%d/%m/%Y %H:%M:%S")
            
            fit_report = []
            
            for i, val in enumerate(out.uvars):
                _fit_params = []
                _fit_params.append(val)
                _fit_params.append(out.uvars[val])
                j_params = ' '.join(map(str, _fit_params))
                fit_report.append(j_params)
                
            fit_report_arr = np.array(fit_report)
            
            np.savetxt('Fit_Report_Data/'+self.filename+'_fit_report'+'.dat', fit_report_arr,
                          header=self.filename+'_fit_report \n\nDate&time:\t'+timestring+
                      '\n\nFit Method\t'+str(out.method)+
                       '\nTotal Unique Traces in Dataset\t'+str(len(self.wavelengths))+
                       '\nNumber of Trace Datasets Included in Fit\t'+str(out.ndata/len(self.delays))+
                      '\nNumber of Function Evaluations\t'+str(out.nfev)+
                       '\n\nFit Statistics:'+
                       '\nChi-Squared\t'+str(out.chisqr)+
                      '\nReduced Chi-Sqaured\t'+str(out.redchi)+
                       '\nAkaikie Information Criterion\t'+str(out.aic)+
                       '\nBayesian Information Criterion\t'+str(out.bic)+
                       '\n\nFit Parameters:', fmt='%s'
                      )
            
        
        fit_data = []
        fit_data_array = np.zeros(shape=(len(fit_wavelengths), len(xdata)))
        
        residual_data = []
        residual_data_array = np.zeros(shape=(len(fit_wavelengths), len(xdata)))
        
        
        for i, _ in enumerate(ydata):
            resid = out.residual.reshape(len(ydata), len(ydata[0]))
            fit = ydata[i] - resid[i]

            fit_data.append(fit)
            fit_data_array[i] = np.array(fit)
            
            residual_data.append(resid[i])
            residual_data_array[i] = np.array(resid[i])
            
        
        self.fit_data = fit_data # list
        self.fit_data_arr = fit_data_array # stores the fit data in an array (not currently used anywhere)
        
        self.residual_fit_data = residual_data
        #self.residual_fit_data = residual_data_array
        
        self.output = out
        self.fit_wavelengths = fit_wavelengths # list
        
        return out
        

    def builtin_model(params, xdata, ydata, ncpts, model, prefixdata, prefixcpt, normalise=True): 
        # expect to get 3+2n params in dict for ncpts, so check this:
        # if len(params) != ncpts:
        #  raise Exception('you idiot')
        # print(ydata.shape)
        # is the problem that the whole thing keeps getting called over and over, 
        # so the model keeps getting redefined?
        
        residual = np.zeros_like(ydata)
        for idx, _ in enumerate(ydata):
            if normalise:
                ydata[idx] = ydata[idx]/np.max(np.abs(ydata[idx]))
        
            for jdx, n in enumerate(range(ncpts)): #loop over exponential components
                if (idx == 0 and jdx == 0):
                    mod = model(prefix=prefixdata+str(idx)+prefixcpt+str(jdx)+'_')
                else:
                    mod += model(prefix=prefixdata+str(idx)+prefixcpt+str(jdx)+'_')
        # here we have the model defined for that dataset

        #the problem here is that it is fitting each dataset to two components rather than fitting each to one component with the same t constant
         
            residual[idx,:] = ydata[idx,:] - mod.eval(params=params, x=xdata)
        #sys.exit()
    
        return residual.flatten()
        

        
    def create_params_object(ydata, paramdata, prefixdata, prefixcpt, fixed_vars=None, offset_vars=None, 
                                fixed_data_vars=None, ncpts=None, vars_per_cpt=None):
        ''' Creates a parameters object from given dictionary and ydata.

        Would you ever want each exponential component to have a different IRF?

        What i want to do is define the prefixes on the different datasets and then the stuff within each model can be initialised 
        as the model is created, otherwise you'd need two different sets of prefixes.

        prefix works as dNcM - N datasets, M components.

        need to re add the IRF stuff as a part of each exponential commponent (rather than defining it over and over)

        Assumes we are doing a global fit, so creates parameters for the number of datasets you've got,
        but fixes the values of the variables that shouldn't vary between fits.
        '''
        params = lf.Parameters()
        
        expected_length = (ncpts*vars_per_cpt)+len(offset_vars)
        
        if len(paramdata) > expected_length:
            print(f'Parameter dictionary passed in is longer than expected. Ignoring everything after the {expected_length}th element.')
            paramdata = dict(list(paramdata.items())[:expected_length])

        elif len(paramdata) < expected_length:
            raise Exception(f'Not enough parameters passed in. Expected {expected_length}, got {len(paramdata)}')
            
        # loop over all the y data being globally fit (datasets)
        for idx,  _ in enumerate(ydata):
            # loop over the param list, if first loop then want to vary the IRF params but fix for all others (components)
            cptcount = 0
            cptidx = 0
            for jdx, paramkey in enumerate(paramdata): 
                if paramkey not in offset_vars:
                    params.add(prefixdata+str(idx)+prefixcpt+str(cptidx)+'_'+paramkey.rstrip(string.digits), value=paramdata[paramkey][0], 
                                min=paramdata[paramkey][1], max=paramdata[paramkey][2], vary=paramdata[paramkey][3])


                    if ((cptidx != 0 or idx !=0) and paramkey.rstrip(string.digits) in fixed_vars):
                        params[prefixdata+str(idx)+prefixcpt+str(cptidx)+'_'+paramkey.rstrip(string.digits)].expr = prefixdata+'0'+prefixcpt+'0_'+paramkey.rstrip(string.digits)
                        params[prefixdata+str(idx)+prefixcpt+str(cptidx)+'_'+paramkey.rstrip(string.digits)].vary = False
                    
                # fix the parameters that are the same for each component, but not dataset    
                # elif (idx != 0 and paramkey in fixed_cpt_vars):
                # params[prefixdata+str(idx)+prefixcpt+str(cptidx)+'_'+paramkey.rstrip(string.digits)].vary = True
                    
                #otherwise if more than one dataset, fix the other variables to those from dataset 1
                    
                    elif (idx != 0 and any(var in paramkey for var in fixed_data_vars)):
                    #this needs fixing
                        params[prefixdata+str(idx)+prefixcpt+str(cptidx)+'_'+paramkey.rstrip(string.digits)].expr = prefixdata+'0'+prefixcpt+str(cptidx)+'_'+paramkey.rstrip(string.digits)
                        params[prefixdata+str(idx)+prefixcpt+str(cptidx)+'_'+paramkey.rstrip(string.digits)].vary = False

                    cptcount += 1    
                    
                    if cptcount%vars_per_cpt == 0:
                        cptidx += 1
                else:
                    params.add(prefixdata+str(idx)+prefixcpt+str(cptidx)+'_'+paramkey.rstrip(string.digits), value=paramdata[paramkey][0], 
                                min=paramdata[paramkey][1], max=paramdata[paramkey][2], vary=paramdata[paramkey][3])
                
                #fix the parameters that are the same for each component and dataset
        
        return params

    

    #def plot_fit_result(output, xaxis, traces, wavelengths, ncpts, print_report=False, xlabel='Delay [ps]', 
    #                        ylabel = r'$\Delta$OD [mOD]', xlim=(-5,50)):
    
        """
        JDP plotting function for the global fits, plots each fit on a separate graph
        """
        """
        for i, _ in enumerate(traces):
            resid = output.residual.reshape(len(traces), len(traces[0]))
            fitdata = traces[i] - resid[i]
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(xaxis, fitdata, color='blue', linewidth = 1.5, label='Fit')
            ax.scatter(xaxis, traces[i], color='lightsteelblue', linewidths = 1.5,  label='Raw')
            ax.tick_params(axis="both", labelsize = 15)
            ax.legend(fontsize = 15)
            ax.set_xlabel(xlabel, fontsize = 20)
            ax.set_ylabel(ylabel, fontsize = 20)
            ax.set_title(f'Fit to delay at {np.round(wavelengths[i], 2)} nm', fontsize = 25, y=1.02)     # remove/change if fitting to SVD trace
            ax.set_xlim(xlim[0], xlim[1])
            
        for n in range(ncpts):
            l1 = np.round(1/(output.params['d'+str(i)+'c'+str(n)+'_gamma'].value), 4)   
            l2 = output.params['d'+str(i)+'c'+str(n)+'_amplitude'].value
            
            #l1_err = np.round(RugFits.inverse_uncertainty(l1, output.params['d'+str(i)+'c'+str(n)+'_gamma'].stderr), 4)  
            print(f'Lifetime {n} is: {l1} ps.')
            
            print(f'amplitude {n} is: {l2}')
            #print(f'Lifetime {n} is: {l1} +/- {l1_err} ps.')
        
        if print_report:
            print(lf.fit_report(output))

        return
    """

    def plot_fit_result(output, xaxis, traces, wavelengths, ncpts, print_report=False, xlabel='Delay [ps]', 
                            ylabel = r'$\Delta$OD [mOD]', xlim=(-5,50)):

        """
        Plot the global fit data generated by the function 'global_fit' using a slider Widget to scan through the fit at the input wavelengths
        """
        fit_data = []
        
        for i, _ in enumerate(traces):
            resid = output.residual.reshape(len(traces), len(traces[0]))
            fit = (traces[i] - resid[i])
            fit_data.append(fit)

        trace = traces[0]
        fitdata = fit_data[0]
        
        wavelengths_indicies = []
        wavelengths_values = []

        for idx, val in enumerate(wavelengths):
            wavelengths_indicies.append(idx)
            wavelengths_values.append(val)

        wavelengths_series = pd.Series(wavelengths_indicies, index=wavelengths_values)
                
        #fig, ax = plt.subplots() 
        
        #fig = plt.figure()
        #ax = fig.gca()

        fig = plt.figure(figsize=(10,10))
        grid = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], 
                        wspace=0.3, hspace=0.3)
        
        ax = fig.add_subplot(grid[0:7])
        
        
        line, = ax.plot(xaxis, traces[0], color='lightsteelblue', linewidth = 3,  label='Raw') # linewidths = 1.5
        line1, = ax.plot(xaxis, fit_data[0], color='blue', linewidth = 1.5, label='Fit')
        
        ax.tick_params(axis="both", labelsize = 15)
        ax.legend(fontsize = 15)
        ax.set_xlabel(xlabel, fontsize = 20)
        ax.set_ylabel(ylabel, fontsize = 20)
        ax.set_xlim(xlim[0], xlim[1])
        

        # Adjustable slider to scan through the fits/traces at the input wavelengths
        global wavelength_slider
        ax_wave = fig.add_axes([0.15, 0.9, 0.80, 0.02])        # 0.1, 0.05, 0.80, 0.02 = bottom, 0.93
        wavelength_slider = Slider(
            ax=ax_wave,
            label='Wavelength (nm)',
            valmin=wavelengths[0],
            valmax=wavelengths[-1],   
            valinit=wavelengths[0],
            valstep=wavelengths,
            color = 'lightsteelblue',
            handle_style={'facecolor': 'white', 'edgecolor': '.75', 'size': 20}
        )
        wavelength_slider.label.set_size(20)

        def update(val):
            
            # Updates the graphs when the user interacts with the widget.
            trace_ydata = traces[wavelengths_series[wavelength_slider.val]]
            line.set_ydata(trace_ydata)

            fit_ydata = fit_data[wavelengths_series[wavelength_slider.val]]
            line1.set_ydata(fit_ydata)
            fig.canvas.draw_idle()
            
            try:
                ax.set_ylim(np.nanmin(trace_ydata)-0.2, np.nanmax(trace_ydata)+0.2)
            except:
                pass
            
        wavelength_slider.on_changed(update)
        plt.show()
        
        """
        for n in range(ncpts):
            l1 = np.round(1/(output.params['d'+str(i)+'c'+str(n)+'_gamma'].value), 4)   
            l2 = output.params['d'+str(i)+'c'+str(n)+'_amplitude'].value
            
            #l1_err = np.round(RugFits.inverse_uncertainty(l1, output.params['d'+str(i)+'c'+str(n)+'_gamma'].stderr), 4)  
            print(f'Lifetime {n} is: {l1} ps.')
            
            print(f'amplitude {n} is: {l2}')
            #print(f'Lifetime {n} is: {l1} +/- {l1_err} ps.')
        """
        
        if print_report:
            print(lf.fit_report(output))

        return

        

    def plot_exist_fit(fit_data, output, xaxis, traces, wavelengths, ncpts, print_report=False, xlabel='Delay [ps]',
                            ylabel = r'$\Delta$OD [mOD]', xlim=(-5,50)):
        """
        Plot the global fit data saved as an attribute of self using a slider Widget to scan through the fit at each of the input wavelengths
        """
        trace = traces[0]
        fitdata = fit_data[0]
        
        wavelengths_indicies = []
        wavelengths_values = []

        for idx, val in enumerate(wavelengths):
            wavelengths_indicies.append(idx)
            wavelengths_values.append(val)

        wavelengths_series = pd.Series(wavelengths_indicies, index=wavelengths_values)
                
        #fig, ax = plt.subplots() 
        
        #fig = plt.figure()
        #ax = fig.gca()

        fig = plt.figure(figsize=(10,10))
        grid = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], 
                        wspace=0.3, hspace=0.3)

        ax = fig.add_subplot(grid[0:7])    
        
        line, = ax.plot(xaxis, traces[0]/np.max(np.abs(traces[0])), color='lightsteelblue', linewidth = 3,  label='Raw') # linewidths = 1.5
        line1, = ax.plot(xaxis, fit_data[0], color='blue', linewidth = 1.5, label='Fit')
        
        ax.tick_params(axis="both", labelsize = 15)
        ax.legend(fontsize = 15)
        ax.set_xlabel(xlabel, fontsize = 20)
        ax.set_ylabel(ylabel, fontsize = 20)
        ax.set_xlim(xlim[0], xlim[1])
        ax.grid(visible=True)
        

        # Adjustable slider to scan through the fits/traces at the input wavelengths
        global wavelength_slider
        ax_wave = fig.add_axes([0.15, 0.9, 0.80, 0.02])        # 0.1, 0.05, 0.80, 0.02 = bottom, 0.93
        wavelength_slider = Slider(
            ax=ax_wave,
            label='Wavelength (nm)',
            valmin=wavelengths[0],
            valmax=wavelengths[-1],   
            valinit=wavelengths[0],
            valstep=wavelengths,
            color = 'lightsteelblue',
            handle_style={'facecolor': 'white', 'edgecolor': '.75', 'size': 20}
        )
        wavelength_slider.label.set_size(20)

        def update(val):
            
            # Updates the graphs when the user interacts with the widget.
            trace_ydata = traces[wavelengths_series[wavelength_slider.val]]
            line.set_ydata(trace_ydata/np.max(np.abs(trace_ydata)))

            fit_ydata = fit_data[wavelengths_series[wavelength_slider.val]]
            line1.set_ydata(fit_ydata)
            fig.canvas.draw_idle()
            
            try:
                ax.set_ylim(np.nanmin(trace_ydata/np.max(np.abs(trace_ydata)))-0.2, np.nanmax(trace_ydata/np.max(np.abs(trace_ydata)))+0.2)
            except:
                pass
            
        wavelength_slider.on_changed(update)
        plt.show()
        
        """
        for n in range(ncpts):
            l1 = np.round(1/(output.params['d'+str(i)+'c'+str(n)+'_gamma'].value), 4)   
            l2 = output.params['d'+str(i)+'c'+str(n)+'_amplitude'].value
            
            #l1_err = np.round(RugFits.inverse_uncertainty(l1, output.params['d'+str(i)+'c'+str(n)+'_gamma'].stderr), 4)  
            print(f'Lifetime {n} is: {l1} ps.')
            
            print(f'amplitude {n} is: {l2}')
            #print(f'Lifetime {n} is: {l1} +/- {l1_err} ps.')
        """
        
        #if print_report:
        #    print(lf.fit_report(output))

        return

    










    # the following is all an attempt to fit the data to a hard model using a 2D fit (sum of exp gaussian along y axis and sum of gaussians along x axis)

    def global_fit_2D_test(self):

        X, Y = np.meshgrid(self.wavelengths, self.delays, indexing="ij")
        Z = self.abs
        
        # define some functions called by the 2D function
        def gaussian(x, amplitude, center, sigma):
            return (amplitude/(np.sqrt(np.pi*2)*sigma)) * (np.exp(-0.5*(((x-center)/sigma)**2)))
    

        def expgaussian(x, amplitude, center, sigma, gamma):
            return amplitude * (gamma/2) * np.exp(center*gamma + (gamma*sigma)**2/2 - gamma*x) * sp.special.erfc((center + gamma*sigma**2 - x)/(np.sqrt(2)*sigma))


        p = 1 
        """
        ^ number of iterations to loop to generate random data with an exponentially modified 
          gaussian lineshape and the number of exponentially modified gaussians to fit
        
          called by 'objective' function
        """

        # function to generate two peaks with one exponential component each 
        def exp_gauss_2D(x, y, amp, amp1, amp2, cen_x, cen_x1, cen_x2,
                         cen_y, cen_y1, cen_y2, sig_x, sig_x1, sig_x2,
                         sig_y, sig_y1, sig_y2, gam_, gam_1, gam_2):

            z = (amp**2 * gaussian(x, amplitude=1, center=cen_x, sigma=sig_x) * expgaussian(y, amplitude=1, center=cen_y, sigma=sig_y, gamma=gam_)) + (-1 * (amp1**2) * gaussian(x, amplitude=1, center=cen_x1, sigma=sig_x1) * expgaussian(y, amplitude=1, center=cen_y1, sigma=sig_y1, gamma=gam_1)) + (-1*(amp2**2) * gaussian(x, amplitude=1, center=cen_x2, sigma=sig_x2) * expgaussian(y, amplitude=1, center=cen_y2, sigma=sig_y2, gamma=gam_2))

            return z
        
        
        
        def exp_gauss_2D_dataset(params, i, x, y):
            # peak 1
            amp = params[f'amp']
            cen_x = params[f'cen_x']
            cen_y = params[f'cen_y']
            sig_x = params[f'sig_x']
            sig_y = params[f'sig_y']
            gam_ = params[f'gam_']
            
            # peak 2
            amp1 = params[f'amp{i+1}']
            cen_x1 = params[f'cen_x{i+1}']
            cen_y1 = params[f'cen_y{i+1}']
            sig_x1 = params[f'sig_x{i+1}']
            sig_y1 = params[f'sig_y{i+1}']
            gam_1 = params[f'gam_{i+1}']

            #peak 3
            amp2 = params[f'amp{i+2}']
            cen_x2 = params[f'cen_x{i+2}']
            cen_y2 = params[f'cen_y{i+2}']
            sig_x2 = params[f'sig_x{i+2}']
            sig_y2 = params[f'sig_y{i+2}']
            gam_2 = params[f'gam_{i+2}']
            return exp_gauss_2D(x, y, amp, amp1, amp2, cen_x, cen_x1, cen_x2,
                                cen_y, cen_y1, cen_y2, sig_x, sig_x1, sig_x2,
                                sig_y, sig_y1, sig_y2, gam_, gam_1, gam_2)


        def objective(params, x, y, data):
    
            residual = np.zeros_like(data)
            
            for idx in np.arange(p): 
                for jdx, val in enumerate(y[0]):
                    for kdx, vals in enumerate(x.T[0]):
                        residual[jdx, kdx] = data[jdx, kdx] - exp_gauss_2D_dataset(params=params, i=idx, x=x[kdx, jdx], y=y[kdx, jdx])
                                          
            print(residual.flatten())
            """
            for jdx, val in enumerate(x.T[0]):
                for kdx, vals in enumerate(y[0]):
                    residual[jdx, kdx] = data[jdx, kdx] - exp_gauss_2D_dataset(params=params, i=idx, x=x[jdx, kdx], y=y[jdx, kdx])
            """
            """     
            for idx in np.arange(p):         
                for i, val in enumerate(data):
                    residual[i] = data[i] - exp_gauss_2D_dataset(params=params, i=idx, x=x, y=y)[i]
            """
            return residual.flatten()



        fit_params = lf.Parameters()

        # fit to two exponentially modified gaussians along y and two gaussians along x
        for i in np.arange(p):
            #peak 1
            fit_params.add(f'amp', value = 80.1314445, min=79.5, max=80.5) 
            fit_params.add(f'cen_x', value = 400, min = 395, max = 405)
            fit_params.add(f'cen_y', value = 0, min = -0.1, max = 0.1)
            fit_params.add(f'sig_x', value = 20, min = 19.5, max = 20.5)
            fit_params.add(f'sig_y', value = 0.07, min = 0.065, max = 0.075)
            fit_params.add(f'gam_', value = 0.15, min = 0.1, max = 0.2)
            
            # peak 2
            fit_params.add(f'amp{i+1}', value = 57.2321010, min = 56.5, max = 57.5) 
            fit_params.add(f'cen_x{i+1}', value = 406, min = 402, max = 410)
            fit_params.add(f'cen_y{i+1}', value = 0, min = -0.1, max = 0.1)
            fit_params.add(f'sig_x{i+1}', value = 8, min = 7.5, max = 8.5)
            fit_params.add(f'sig_y{i+1}', value = 0.07, min = 0.065, max = 0.075)
            fit_params.add(f'gam_{i+1}', value = 0.3, min = 0.15, max = 0.45)
            
            # peak 3
            fit_params.add(f'amp{i+2}', value = 46.4276229, min = 46, max = 47) 
            fit_params.add(f'cen_x{i+2}', value = 395, min = 390, max = 400)
            fit_params.add(f'cen_y{i+2}', value = 0, min = -0.1, max = 0.1)
            fit_params.add(f'sig_x{i+2}', value = 10, min = 9.5, max = 10.5)
            fit_params.add(f'sig_y{i+2}', value = 0.07, min = 0.065, max = 0.075)
            fit_params.add(f'gam_{i+2}', value = 0.3, min = 0.15, max = 0.45)

        fit_params[f'sig_y1'].expr = 'sig_y'
        fit_params[f'sig_y2'].expr = 'sig_y'
        
        fit_params[f'cen_y1'].expr = 'cen_y'
        fit_params[f'cen_y2'].expr = 'cen_y'
        fit_params[f'gam_2'].expr = 'gam_1'

        out = lf.minimize(objective, fit_params, method='leastsq', args=(X, Y, Z), calc_covar=True)
        lf.report_fit(out.params)

        out.params.pretty_print()
        fit = exp_gauss_2D_dataset(params=out.params, i=0, x=X, y=Y)

        self.z = Z
        self.fit = fit
        self.x = X
        self.y = Y

    def plot_fit_result_test(self):
        """
        Plot colourmaps of the raw data, the fit, and the residual
        """
        fig, axs = plt.subplots(3, 1) # , figsize=(10, 10))

        vmax = np.nanpercentile(self.z, 99.9)
        vmin = np.nanpercentile(self.z, 0.01)
        vcenter = 0
        
        
        ax = axs[1]
        #fit = exp_gauss_2D_dataset(params=out.params, i=0, x=X, y=Y)   
         
        im = ax.pcolormesh(self.x, self.y, self.fit, cmap='PuOr', norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax), shading='nearest')
        tick_range = np.linspace(vmin, vmax, 10)
        cbar = fig.colorbar(im, ticks=tick_range) 
        #ax.set_yscale('symlog')
        ax.set_ylim(0, 10)
        ax.set_xlabel('Wavelengths (nm)')
        ax.set_ylabel('Delays (ps)')
        ax.set_title('Fit')
        
        
        ax = axs[0]
        im = ax.pcolormesh(self.x, self.y, self.z.T, cmap='PuOr', norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax), shading='nearest')
        tick_range = np.linspace(vmin, vmax, 10)
        cbar = fig.colorbar(im, ticks=tick_range) 
        #ax.set_yscale('symlog')
        ax.set_ylim(0, 10)
        ax.set_xlabel('Wavelengths (nm)')
        ax.set_ylabel('Delays (ps)')
        ax.set_title('Raw')
        
        
        ax = axs[2]
        im = ax.pcolormesh(self.x, self.y, self.z.T - self.fit, cmap='PuOr', norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax), shading='nearest')
        tick_range = np.linspace(vmin, vmax, 10)
        cbar = fig.colorbar(im, ticks=tick_range) 
        #ax.set_yscale('symlog')
        ax.set_ylim(0, 10)
        ax.set_xlabel('Wavelengths (nm)')
        ax.set_ylabel('Delays (ps)')
        ax.set_title('Residual')
        
        plt.show()
        return

    def explore_trace_test(self):
        """
        Interactive slider to scan through the fit at each wavelength
        """
        X_index = []
        X_values = []
        
        for idx, val in enumerate (self.x.T[0]):
            X_index.append(idx)
            X_values.append(val)
        
        X_series = pd.Series(X_index, index=X_values)
        
        
        raw_trace = self.z.T[0]
        fit_trace = self.fit[0]
        
        fig = plt.figure(figsize=(10,10))
        grid = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], 
                        wspace=0.3, hspace=0.3)
        
        ax1 = fig.add_subplot(grid[0:7])
        
        line, = ax1.plot(self.y[0], raw_trace, '.C0', label = 'raw')
        #line, = ax1.plot(Y[0], raw_trace, color='lightsteelblue', linewidth=2, label = 'raw')
        
        line1, = ax1.plot(self.y[0], fit_trace, color='lightsteelblue', linewidth=2, label = 'fit')
        
        ax1.tick_params(axis="both", labelsize = 15)
        #ax1.set_ylim(-0.009, 0.009)
        ax1.set_xlim(-0.1, 5)
        ax1.set_xlabel('Delays (ps)', fontsize = 20)
        ax1.set_ylabel(r'$\Delta$OD [mOD]', fontsize = 20) 
        ax1.legend(fontsize=20)
        ax1.set_title(self.filename+' Trace', fontsize = 25, y=1.02)
        ax1.grid(visible=True)
        
        # Sliders
        global trace_slider
        
        axwave = plt.axes([0.1, 0.03, 0.8, 0.03])
        trace_slider = Slider(
            ax=axwave,
            label='Wavelength (nm)',
            valmin=self.x.T[0, 0],
            valmax=self.x.T[0, -1],
            valinit=self.x.T[0, 0],
            valstep=self.x.T[0, 0:],
            color='lightsteelblue',
            handle_style={'facecolor': 'white', 'edgecolor': '.05', 'size': 20} # .75
        )
        trace_slider.label.set_size(20)
        
        def update(val):
            
            #Updates the graphs when the user interacts with the widget.
        
            raw_ydata = self.z.T[X_series[trace_slider.val]]
            line.set_ydata(raw_ydata)
            fit_ydata = self.fit[X_series[trace_slider.val]]
            line1.set_ydata(fit_ydata)
            fig.canvas.draw_idle()
            
            try:
                ax1.set_ylim((np.nanmin(raw_ydata) -5), (np.nanmax(raw_ydata) + 5) )
                
            except:
                pass
            
        trace_slider.on_changed(update)
        
        plt.show()

    def explore_spectra_test(self):
        """
        Interactive widget with a slider to inspect slices Z data at slices through the X axis
        """

        Y_index = []
        Y_values = []
        
        for idx, val in enumerate (self.y[0]):
            Y_index.append(idx)
            Y_values.append(val)
        
        Y_series = pd.Series(Y_index, index=Y_values)
        
        
        raw_spectrum = self.z[0]
        fit_spectrum = self.fit.T[0]
        
        
        fig = plt.figure(figsize=(10,10))
        grid = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], 
                        wspace=0.3, hspace=0.3)
        
        ax1 = fig.add_subplot(grid[0:7])
        
        line, = ax1.plot(self.x.T[0], raw_spectrum, '.C0', label = 'raw')
        line1, = ax1.plot(self.x.T[0], fit_spectrum, color='lightsteelblue', linewidth=2, label = 'fit')
        
        #ax1.set_ylim(-0.001, 0.0004)
        ax1.tick_params(axis="both", labelsize = 15)
        ax1.set_xlabel('Wavelength (nm)', fontsize = 20)
        ax1.set_ylabel(r'$\Delta$OD [mOD]', fontsize = 20) 
        ax1.legend(fontsize=20)
        ax1.set_title(self.filename+' Spectrum', fontsize = 25, y=1.02)
        ax1.grid(visible=True)
        
        # Sliders
        global spectrum_slider
        
        axwave = plt.axes([0.1, 0.03, 0.8, 0.03])
        spectrum_slider = Slider(
            ax=axwave,
            label='Delay (ps)',
            valmin=self.y[0, 0],
            valmax=self.y[0, 150], 
            valinit=self.y[0, 0],
            valstep=self.y[0][0:150],
            color='lightsteelblue',
            handle_style={'facecolor': 'white', 'edgecolor': '.05', 'size': 20} # .75
        )
        spectrum_slider.label.set_size(20)
        
        def update(val):
            
            #Updates the graphs when the user interacts with the widget.
        
            raw_ydata = self.z[Y_series[spectrum_slider.val]]
            line.set_ydata(raw_ydata)
            fit_ydata = self.fit.T[Y_series[spectrum_slider.val]]
            line1.set_ydata(fit_ydata)
            
            fig.canvas.draw_idle()
            
            try:
                ax1.set_ylim(np.nanmin(raw_ydata) - 5, np.nanmax(raw_ydata) + 5)
                
            except:
                pass
            
               
        spectrum_slider.on_changed(update)
        
        plt.show()
        return
        
















    

    """
    Existing 2D global fitting functions are in trial phase and currently don't work, delete these after adding a working 2D global fitting function
    """

    def global_fit_2D(self):
        """
        Fit the entire array of difference absorption data to n components where each component is an exponentially modified gaussian along the 
        self.delays axis and a gaussian distribution along the self.wavelengths axis

        NOT GENERAL
        """
        
        x, y = self.wavelengths, self.delays
        X, Y = np.meshgrid(x, y)
        Z = self.abs
        

        def expgaussian(x, amplitude, center, sigma, gamma):
            return amplitude * (gamma/2) * np.exp(center*gamma + (gamma*sigma)**2/2 - gamma*x) * sp.special.erfc((center + gamma*sigma**2 - x)/(np.sqrt(2)*sigma))

        

        def gaussian(x, amplitude, center, sigma):
            return (amplitude/(np.pi*sigma)) * ((np.exp(-(x-center)**2) / (2*sigma**2)))
            
            

        def exp_gauss_2D(x, y, amplitude, centre_x, centre_y, sigma_x, sigma_y, gamma):
            """
            Function to generate a exponentially modified gaussian distribution along one axis and a normal gaussian distribution along the orthogonal axis
        
            Want to fit to an exponentially modified gaussian along the self.delays axis and a normal gaussian distribution along the self.wavelengths axis
            """
            z = amplitude * (gaussian(x, amplitude=1, center=centre_x, sigma=sigma_x) * expgaussian(y, amplitude=1, center=centre_y, sigma=sigma_y, gamma=gamma))
            return z
            
        n_params = 6
        
        def _exp_gauss_2D(M, *args):
            """
            NOT GENERAL currently set to fit processed Met-Mb_409_SHG/FUN dataset
            """
            x, y = M
            arr = np.zeros(x.shape)
            for i in range(len(args)//n_params):
                arr += exp_gauss_2D(x, y, *args[i*n_params:i*n_params+n_params])
            return arr

        # params = amplitude, centre_x, centre_y, sigma_x, sigma_y, gamma
        
        init_params = [(0.1,  4.27494339e+02, 1.4, 0.0688, 0.0688,  0.2500),          # positive
                       (0.1,  4.30791558e+02, 1.4, 0.0688, 0.0688,  1.959),
                       
                       (-0.1, 3.82407945e+02, -1.4, 0.0688,  0.0688,  0.2500),          # negative
                       (-0.1, 3.82597215e+02, -1.4, 0.0688,  0.0688,  1.959)]

        p0 = [p for prms in init_params for p in prms] 
        
        x_data = np.vstack((X.ravel(), Y.ravel()))

        popt, pcov = curve_fit(_exp_gauss_2D, x_data, Z.ravel(), p0, bounds=((-0.2, 425, 0, 0.01, 1E-6, 0.1,
                                                                              -0.2, 425, 0, 0.01, 1E-6, 1.5,
                                                                              -0.2, 380, -2, 0.01, 1E-7, 0.1,
                                                                              -0.2, 380, -2, 0.01, 1E-7, 1.5),
                                                                              (0.2, 435, 2, 0.1, 0.1, 0.5,  
                                                                               0.2, 435, 2, 0.1, 0.1, 5, 
                                                                               0.2, 385, 0, 0.1, 0.1, 0.5, 
                                                                               0.2, 385, 0, 0.1, 0.1, 5),
                                                                             ))
        
        
        fit = np.zeros(Z.shape)
        for i in range(len(popt)//n_params):
            fit += exp_gauss_2D(X, Y, *popt[i*n_params:i*n_params+n_params])

        print(popt)
        print('Fitted parameters:')
        print(f'Amplitudes =', popt[0:24:6], "\n")
        print(f'Centre_x =', popt[1:24:6], "\n")
        print(f'Centre_y =', popt[2:24:6], "\n")
        print(f'Sigma_x =', popt[3:24:6], "\n")
        print(f'Sigma_y =', popt[4:24:6], "\n")
        print(f'Gamma =', popt[5:24:6], "\n")
        for i in popt[5:24:6]:
            print(f'Decay constant =', 1/i)
        

        #vmin = np.nanmin(self.abs)
        #vmax = np.nanmax(self.abs)
        
        vmax = np.nanpercentile(Z, 99.9)
        vmin = np.nanpercentile(Z, 0.1)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        ax = axs[0, 0]
    
        # art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
        # plt.colorbar(im_1, ax=ax, label='z')
        im_1 = ax.pcolormesh(X, Y, Z, cmap='PuOr', norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(im_1)
        ax.set_yscale('symlog')
        ax.set_title('Raw data')
        
        ax = axs[0, 1]
        #art = ax.pcolor(X, Y, Z-fit, vmin=0, vmax=vmax, shading='auto')
        #plt.colorbar(im_2, ax=ax, label='z')
        im_2 = ax.pcolormesh(X, Y, Z-fit, cmap='PuOr', norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(im_2)
        ax.set_yscale('symlog')
        ax.set_title('Residual')
        
        ax = axs[1, 1]
        #art = ax.pcolor(X, Y, fit, vmin=0, vmax=10, shading='auto')
        #plt.colorbar(im_3, ax=ax, label='z')
        im_3 = ax.pcolormesh(X, Y, fit, cmap='PuOr', norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(im_3)
        ax.set_yscale('symlog')
        ax.set_title('Fit')
        
        plt.show()
        
        print(X.T[33]) # [33] = 376nm, [35] = #380nm, [55] = 425nm
        
        plt.figure()
        plt.scatter(Y.T[0], Z.T[55], color='black', s=10, label='raw')
        plt.plot(Y.T[0], fit.T[55], color='magenta', label='fit')
        plt.plot(Y.T[0], Z.T[55] - fit.T[55], label='residual')
        plt.xscale('log')
        plt.title('Trace at 425nm')
        plt.legend()

        plt.figure()
        plt.scatter(Y.T[0], Z.T[56], color='black', s=10, label='raw')
        plt.plot(Y.T[0], fit.T[56], color='magenta', label='fit')
        plt.plot(Y.T[0], Z.T[56] - fit.T[56], label='residual')
        plt.xscale('log')
        #plt.title('Trace at 425nm')
        plt.legend()

        plt.figure()
        plt.scatter(Y.T[0], Z.T[57], color='black', s=10, label='raw')
        plt.plot(Y.T[0], fit.T[57], color='magenta', label='fit')
        plt.plot(Y.T[0], Z.T[57] - fit.T[57], label='residual')
        plt.xscale('log')
        #plt.title('Trace at 425nm')
        plt.legend()

        plt.figure()
        plt.scatter(Y.T[0], Z.T[58], color='black', s=10, label='raw')
        plt.plot(Y.T[0], fit.T[58], color='magenta', label='fit')
        plt.plot(Y.T[0], Z.T[58] - fit.T[58], label='residual')
        plt.xscale('log')
        #plt.title('Trace at 425nm')
        plt.legend()

        

        plt.figure()
        plt.scatter(Y.T[0], Z.T[35], color='black', s=10, label='raw')
        plt.plot(Y.T[0], fit.T[35], color='magenta', label='fit')
        plt.plot(Y.T[0], Z.T[35] - fit.T[35], label='residual')
        plt.xscale('log')
        plt.title('Trace at 380nm')
        plt.legend()

        plt.figure()
        plt.scatter(Y.T[0], Z.T[34], color='black', s=10, label='raw')
        plt.plot(Y.T[0], fit.T[34], color='magenta', label='fit')
        plt.plot(Y.T[0], Z.T[34] - fit.T[34], label='residual')
        plt.xscale('log')
        #plt.title('Trace at 380nm')
        plt.legend()

        plt.figure()
        plt.scatter(Y.T[0], Z.T[33], color='black', s=10, label='raw')
        plt.plot(Y.T[0], fit.T[33], color='magenta', label='fit')
        plt.plot(Y.T[0], Z.T[33] - fit.T[33], label='residual')
        plt.xscale('log')
        #plt.title('Trace at 380nm')
        plt.legend()



        
        print(Y[35]) # [10] = -0.018ps [11] = 0.048ps [15] = 0.24ps, [20] = 0.498, [30] = 0.9985, [31] = 1.048, [35] = 1.248ps
        #print(X[0])
        
        plt.figure()
        plt.scatter(X[0], Z[35], color='black', s=10, label='raw')
        plt.plot(X[0], fit[35], color='magenta', label='fit')
        plt.plot(X[0], Z[35] - fit[35], label='residual')
        plt.title('Spectrum at 1.048ps')
        plt.legend()
        
        return

    def global_fit_2D_630(self):
        """
        Fit the entire array of difference absorption data to n components where each component is an exponentially modified gaussian along the 
        self.delays axis and a gaussian distribution along the self.wavelengths axis

        NOT GENERAL
        """
        
        x, y = self.wavelengths, self.delays
        X, Y = np.meshgrid(x, y)
        Z = self.abs
        

        def expgaussian(x, amplitude, center, sigma, gamma):
            return amplitude * (gamma/2) * np.exp(center*gamma + (gamma*sigma)**2/2 - gamma*x) * sp.special.erfc((center + gamma*sigma**2 - x)/(np.sqrt(2)*sigma))

        

        def gaussian(x, amplitude, center, sigma):
            return (amplitude/(np.pi*sigma)) * ((np.exp(-(x-center)**2) / (2*sigma**2)))
            
            

        def exp_gauss_2D(x, y, amplitude, centre_x, centre_y, sigma_x, sigma_y, gamma):
            """
            Function to generate a exponentially modified gaussian distribution along one axis and a normal gaussian distribution along the orthogonal axis
        
            Want to fit to an exponentially modified gaussian along the self.delays axis and a normal gaussian distribution along the self.wavelengths axis
            """
            z = amplitude * (gaussian(x, amplitude=1, center=centre_x, sigma=sigma_x) * expgaussian(y, amplitude=1, center=centre_y, sigma=sigma_y, gamma=gamma))
            return z

        
            
        n_params = 6

        def _exp_gauss_2D(M, *args):
            """
            NOT GENERAL currently set to fit processed Met-Mb_630_SHG/FUN dataset
            """
            x, y = M
            arr = np.zeros(x.shape)
            for i in range(len(args)//n_params):
                arr += exp_gauss_2D(x, y, *args[i*n_params:i*n_params+n_params])
            return arr
            

        # params = amplitude, centre_x, centre_y, sigma_x, sigma_y, gamma
        
        init_params = [(0.5,  433, 1, 0.07, 0.0688,  0.01),          # positive
                       (-1,  406, 0.3, 0.07, 0.0688,  0.01)]
                       
                       #(-0.1, 3.82407945e+02, -1.4, 0.0688,  0.0688,  0.2500),          # negative
                       #(-0.1, 3.82597215e+02, -1.4, 0.0688,  0.0688,  1.959)]

        p0 = [p for prms in init_params for p in prms] 
        
        x_data = np.vstack((X.ravel(), Y.ravel()))

        # method='trf' for bounded optimization
        popt, pcov = curve_fit(_exp_gauss_2D, x_data, Z.ravel(), p0, maxfev = 10000)#, bounds=((-0.2, 400, 0, 0.01, 1E-6, 0.1,
                                                                    #          -0.2, 400, 0, 0.01, 1E-6, 1.5,
                                                                    #          -0.2, 380, -2, 0.01, 1E-7, 0.1,
                                                                    #          -0.2, 380, -2, 0.01, 1E-7, 1.5),
                                                                    #          (0.2, 435, 2, 0.1, 0.1, 0.5,  
                                                                    #           0.2, 435, 2, 0.1, 0.1, 5, 
                                                                    #           0.2, 385, 0, 0.1, 0.1, 0.5, 
                                                                    #           0.2, 385, 0, 0.1, 0.1, 5),
                                                                    #         ))
        
        
        fit = np.zeros(Z.shape)
        for i in range(len(popt)//n_params):
            fit += exp_gauss_2D(X, Y, *popt[i*n_params:i*n_params+n_params])

        print(popt)
        """
        print('Fitted parameters:')
        print(f'Amplitudes =', popt[0:24:6], "\n")
        print(f'Centre_x =', popt[1:24:6], "\n")
        print(f'Centre_y =', popt[2:24:6], "\n")
        print(f'Sigma_x =', popt[3:24:6], "\n")
        print(f'Sigma_y =', popt[4:24:6], "\n")
        print(f'Gamma =', popt[5:24:6], "\n")
        for i in popt[5:24:6]:
            print(f'Decay constant =', 1/i)
        """
        

        #vmin = np.nanmin(self.abs)
        #vmax = np.nanmax(self.abs)
        
        vmax = np.nanpercentile(Z, 99.9)
        vmin = np.nanpercentile(Z, 0.1)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        ax = axs[0, 0]
    
        # art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
        # plt.colorbar(im_1, ax=ax, label='z')
        im_1 = ax.pcolormesh(X, Y, Z, cmap='PuOr', norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(im_1)
        ax.set_yscale('symlog')
        ax.set_title('Raw data')
        
        ax = axs[0, 1]
        #art = ax.pcolor(X, Y, Z-fit, vmin=0, vmax=vmax, shading='auto')
        #plt.colorbar(im_2, ax=ax, label='z')
        im_2 = ax.pcolormesh(X, Y, Z-fit, cmap='PuOr', norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(im_2)
        ax.set_yscale('symlog')
        ax.set_title('Residual')
        
        ax = axs[1, 1]
        #art = ax.pcolor(X, Y, fit, vmin=0, vmax=10, shading='auto')
        #plt.colorbar(im_3, ax=ax, label='z')
        im_3 = ax.pcolormesh(X, Y, fit, cmap='PuOr', norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(im_3)
        ax.set_yscale('symlog')
        ax.set_title('Fit')
        
        plt.show()
        
        # traces
        #print(X.T[70, 0])
        
        plt.figure()
        plt.scatter(Y.T[0], Z.T[95], color='black', s=10, label='raw')
        plt.plot(Y.T[0], fit.T[95], color='magenta', label='fit')
        plt.plot(Y.T[0], Z.T[95] - fit.T[95], label='residual')
        plt.xscale('log')
        plt.title('Trace at 427nm')
        plt.legend()

        plt.figure()
        plt.scatter(Y.T[0], Z.T[70], color='black', s=10, label='raw')
        plt.plot(Y.T[0], fit.T[70], color='magenta', label='fit')
        plt.plot(Y.T[0], Z.T[70] - fit.T[70], label='residual')
        plt.xscale('log')
        plt.title('Trace at 406nm')
        plt.legend()

        #spectra
        #print(Y[15, 0])
        
        plt.figure()
        plt.scatter(X[0], Z[0], color='black', s=10, label='raw')
        plt.plot(X[0], fit[0], color='magenta', label='fit')
        plt.plot(X[0], Z[0] - fit[0], label='residual')
        plt.title('Spectrum at 0.318s')
        plt.legend()

        plt.figure()
        plt.scatter(X[0], Z[15], color='black', s=10, label='raw')
        plt.plot(X[0], fit[15], color='magenta', label='fit')
        plt.plot(X[0], Z[15] - fit[15], label='residual')
        plt.title('Spectrum at 1.068s')
        plt.legend()
        
        return
                    
            
        
    

