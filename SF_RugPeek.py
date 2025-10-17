#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import re
from os import listdir

from datetime import date
import time

from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button
import matplotlib.colors as colors

## NESTED SAMPLING ##
import dynesty
from dynesty import plotting as dyplot

from dynesty import utils as dyfunc
from dynesty.pool import Pool

## allows for saving and loading nested sampling data ##
import dill
import dynesty.utils
dynesty.utils.pickle_module = dill


def load_csv_file(directory, Sample_concentration, Sample, H2O2_conc, HS=False, print_information=False):

    Sample_concentration = np.round(Sample_concentration / 1e-6, 3)
    
    filenames = listdir(directory)
    
    file_data = []
    file_metadata = []
    
    for file in filenames:    
        
        # read and store experiment data
        data = pd.read_csv(directory+f'/{file}', skiprows=26, on_bad_lines='skip')
        if data.keys()[0] == "Wavelength":
            
            # read and store experiment metadata
            meta_data = pd.read_csv(directory+f'/{file}', sep='\t')
        
            metadata = []
        
            metadata.append(meta_data.values[3, 0][1:])
            metadata.append(meta_data.values[4, 0][1:])
            metadata.append(meta_data.values[5, 0][1:])
            metadata.append(meta_data.values[10, 0][5:30])
            metadata.append(meta_data.values[10, 0][-85:-79])
    
            
            file_metadata.append(metadata)
    
            
            # read and store experiment data
            data = pd.read_csv(directory+f'/{file}', skiprows=27, on_bad_lines='skip')
            
            # store the time points in an array
            time_points = np.array(list(map(float, data.keys()[1:-1])))
            
            # store the wavelength values in an array
            wavelength_data = []
            
            for i, v in enumerate(data.values):
                try:
                    wavelength_data.append(float(v[0]))
                except ValueError:
                    pass
                
            wavelengths = []
            
            for x in wavelength_data:
                if x not in wavelengths and math.isnan(x) == False:
                    wavelengths.append(x)
    
            wavelengths = np.array(wavelengths)
            
    
            abs = data.values[0:len(wavelengths), 1:-1].T
    
            for idx, val in enumerate(abs):
                for jdx, wal in enumerate(val):
                    abs[idx, jdx] = float(wal)
    
    
            abs_array_df = pd.DataFrame(i for i in abs)
            abs_array_df.columns = wavelengths
            abs_array_df.index = time_points
    
            file_data.append(abs_array_df)
            
    
        elif data.keys()[0] == "Time":
            # read and store experiment metadata
            meta_data = pd.read_csv(directory+f'/{file}', sep='\t')
        
            metadata = []
        
            metadata.append(meta_data.values[3, 0][1:])
            metadata.append(meta_data.values[4, 0][1:])
            metadata.append(meta_data.values[5, 0][1:])
            metadata.append(meta_data.iloc()[9, 0][meta_data.iloc()[9, 0].rindex('e,')+2:meta_data.iloc()[9, 0].rindex('s,')+1])
            metadata.append(meta_data.iloc()[9, 0][meta_data.iloc()[9, 0].rindex('s,')+2:meta_data.iloc()[9, 0].rindex('over')-1]) 
            
            file_metadata.append(metadata)
            
            
            # read and store experiment data
            data = pd.read_csv(directory+f'/{file}', skiprows=27, on_bad_lines='skip')
            
            # store the wavelength data in an array
            wavelengths = np.array(list(map(float, data.keys()[1:-1])))
    
    
            # store the time data in an array
            time_data = []
    
            for i, v in enumerate(data.values.T[0]):
                try:
                    time_data.append(float(v))
                except ValueError:
                    pass
                
            time_points = []
    
            for x in time_data:
                if x not in time_points and math.isnan(x) == False:
                    time_points.append(x)
    
            time_points = np.array(time_points)
    
    
            
            abs = data.values[0:time_points.shape[0], 1:-1]
    
            for idx, val in enumerate(abs):
                for jdx, wal in enumerate(val):
                    abs[idx, jdx] = float(wal)
    
    
    
            abs_array_df = pd.DataFrame(i for i in abs)
            abs_array_df.columns = wavelengths
            abs_array_df.index = time_points
    
            file_data.append(abs_array_df)
            
        else:
            data = pd.read_csv(directory+f'/{file}', skiprows=25, on_bad_lines='skip')
            if data.keys()[0] == "Wavelength":
                
                # read and store experiment metadata
                meta_data = pd.read_csv(directory+f'/{file}', sep='\t')
            
                metadata = []
            
                metadata.append(meta_data.values[3, 0][1:])
                metadata.append(meta_data.values[4, 0][1:])
                metadata.append(meta_data.values[8, 0][11:45])
                metadata.append(meta_data.iloc()[9, 0][meta_data.iloc()[9, 0].rindex('e,')+2:meta_data.iloc()[9, 0].rindex('s,')+1])
                metadata.append(meta_data.iloc()[9, 0][meta_data.iloc()[9, 0].rindex('s,')+2:meta_data.iloc()[9, 0].rindex('over')-1]) 
    
                file_metadata.append(metadata)
    
                # read and store experiment data
                data = pd.read_csv(directory+f'/{file}', skiprows=26, on_bad_lines='skip')
                
                # store the time points in an array
                time_points = np.array(list(map(float, data.keys()[1:-1])))
                
                # store the wavelength values in an array
                wavelength_data = []
                
                for i, v in enumerate(data.values):
                    try:
                        wavelength_data.append(float(v[0]))
                    except ValueError:
                        pass
                    
                wavelengths = []
                
                for x in wavelength_data:
                    if x not in wavelengths and math.isnan(x) == False:
                        wavelengths.append(x)
        
                wavelengths = np.array(wavelengths)
                
        
                abs = data.values[0:len(wavelengths), 1:-1].T
        
                for idx, val in enumerate(abs):
                    for jdx, wal in enumerate(val):
                        abs[idx, jdx] = float(wal)
    
                abs_array_df = pd.DataFrame(i for i in abs)
                abs_array_df.columns = wavelengths
                abs_array_df.index = time_points
        
                file_data.append(abs_array_df)
        
        
            elif data.keys()[0] == "Time":
                
                # read and store experiment metadata
                meta_data = pd.read_csv(directory+f'/{file}', sep='\t')
            
                metadata = []
            
                metadata.append(meta_data.values[3, 0][1:])
                metadata.append(meta_data.values[4, 0][1:])
                
                metadata.append(meta_data.values[9, 0][meta_data.iloc()[9, 0].rindex('th,')+3:meta_data.iloc()[9, 0].rindex('nm,')+2])
                metadata.append(meta_data.iloc()[8, 0][meta_data.iloc()[8, 0].rindex('e,')+2:meta_data.iloc()[8, 0].rindex('s,')+1])
                metadata.append(meta_data.iloc()[8, 0][meta_data.iloc()[8, 0].rindex('s,')+2:meta_data.iloc()[8, 0].rindex(',o')])
    
                file_metadata.append(metadata)
    
                # read and store experiment data
                data = pd.read_csv(directory+f'/{file}', skiprows=26, on_bad_lines='skip')
                
                # store the wavelength data in an array
                wavelengths = np.array(list(map(float, data.keys()[1:-1])))
        
        
                # store the time data in an array
                time_data = []
        
                for i, v in enumerate(data.values.T[0]):
                    try:
                        time_data.append(float(v))
                    except ValueError:
                        pass
                    
                time_points = []
        
                for x in time_data:
                    if x not in time_points and math.isnan(x) == False:
                        time_points.append(x)
        
                time_points = np.array(time_points)
        
        
                abs = data.values[0:time_points.shape[0], 1:-1]
        
                for idx, val in enumerate(abs):
                    for jdx, wal in enumerate(val):
                        abs[idx, jdx] = float(wal)
    
                abs_array_df = pd.DataFrame(i for i in abs)
                abs_array_df.columns = wavelengths
                abs_array_df.index = time_points
        
                file_data.append(abs_array_df)
    
    
    
    
    data_series = pd.Series(file_data, index=filenames)
    metadata_series = pd.Series(file_metadata, index=filenames)
    
    peroxide_series = pd.Series(H2O2_conc, index=filenames)
    
    if HS==True:
        ## assumes [HS] = [H2O2] ##
        HS_series = pd.Series(H2O2_conc, index=filenames)
        HS_conc = H2O2_conc


    if print_information==True:
        for i, v in enumerate(filenames):
            
            if HS == True:
                print(f'filename = {v}\nfile index = {i}\n[H202] = {H2O2_conc[i]} equ.\n[HS] = {H2O2_conc[i]} equ.\n{len(data_series[v].index)} {metadata_series[i][-1]}ly-spaced time points over {data_series[v].index[-1]}s\n\n')
                print(f'Convert a dataframe into a set of arrays by inputting a file index value from the printed list into', 'sf.SF_Rug.return_arr(sample_1, file_index)')
                
            else:
                print(f'filename = {v}\nfile index = {i}\n[H202] = {H2O2_conc[i]} equ.\n{len(data_series[v].index)} {metadata_series[i][-1]}ly-spaced time points over {data_series[v].index[-1]}s\n\n')
                print(f'Convert a dataframe into a set of arrays by inputting a file index value from the printed list into', 'sf.SF_Rug.return_arr(sample_1, file_index)')

    if HS == True:
        return data_series, Sample_concentration, Sample, H2O2_conc, HS_conc, filenames, metadata_series

    return data_series, Sample_concentration, Sample, H2O2_conc, filenames, metadata_series



def load_dat_file(directory, file):
    count = 0
    metadata = []
    
    with open(directory+'/'+file, encoding="utf-8") as f:
        for line in f:
            
            line = line.strip()
            
            if line.startswith('# ['):
                m_idx = []
                for idx, val in enumerate(line):
                    if val == "'":
                        m_idx.append(idx)

                metadata = []
                
                for i in range((len(m_idx) // 2)):
                    metadata.append(line[m_idx[i*2]+1:m_idx[(i*2)+1]])
            
            if np.char.isnumeric(line[0]):
                break
    
            count = count + 1
            # opens the file as f then for each line in f removes whitespace characters and increases the value of the variable count by 
            # +1 until it reaches a line that begins with a numeric character
    
    data = np.genfromtxt(directory+'/'+file, skip_header=count)

    filename = file[:file.rindex('equ')+8]
    
    wavelengths_nm = data[0, 1:]
    delays_ps = data[1:, 0]
    delta_OD = data[1:, 1:]

    return wavelengths_nm, delays_ps, delta_OD, filename, metadata


## initialize one of the datasets as a class ##
class SF_Rug:
    def __init__(self,
                 directory=None,
                 filename=None,
                 Sample_concentration=None,
                 Sample=None,
                 H2O2_conc=None,
                 HS=False,
                 print_information=False): 
        
        # print(f'{self}\n{directory}\n{filename}\n{Sample_concentration}\n{Sample}\n{H2O2_conc}\n{HS}\n{print_information}')
        
        if filename:
            self.wavelengths, self.delays, self.abs, self.fig_title, self.metadata = load_dat_file(directory, filename)
            ## assuming no bounds have been applied, will neeed to change this bit if applying bounds ##
            self.bounds = np.array([[self.wavelengths[0], self.wavelengths[-1]]])
            self.wl_range_keep = [self.wavelengths[0], self.wavelengths[-1]]
            
        elif filename == None and HS == True:
            self.data_series, self.Sample_concentration, self.Sample, self.H2O2_conc, self.HS_conc, self.filenames, self.metadata = load_csv_file(directory, Sample_concentration, Sample, H2O2_conc, HS, print_information)
            
        elif filename == None and HS == False:
            self.data_series, self.Sample_concentration, self.Sample, self.H2O2_conc, self.filenames, self.metadata = load_csv_file(directory, Sample_concentration, Sample, H2O2_conc, HS, print_information)
            
        else:
            return
            
        return


    
    def return_arr(self, file_index=0):
            
        self.wavelengths = np.array(self.data_series[self.filenames[file_index]].T.index)
        
        self.delays = np.array(self.data_series[self.filenames[file_index]].index)
        self.delays = np.array(self.data_series[self.filenames[file_index]].index)
        
        self.abs = np.array(self.data_series[self.filenames[file_index]].values)
        
        if hasattr(self, 'HS_conc'):
            self.fig_title = f'{self.Sample_concentration}'+r'$\mu$'+f'M_{self.Sample[:len(self.Sample)]}_+{self.H2O2_conc[file_index]}_equ-H2O2_+{self.HS_conc[file_index]}_equ_HS'.replace('_', ' ')
            self.filename = f'{self.Sample_concentration}uM_{self.Sample[:len(self.Sample)]}_+{self.H2O2_conc[file_index]}_equ-H2O2_+{self.HS_conc[file_index]}_equ_HS'.replace('_', ' ')
            
        else:
            self.fig_title = f'{self.Sample_concentration}'+r'$\mu$'+f'M_{self.Sample[:len(self.Sample)]}_+{self.H2O2_conc[file_index]}_equ-H2O2'.replace('_', ' ')
            self.filename = f'{self.Sample_concentration}uM_{self.Sample[:len(self.Sample)]}_+{self.H2O2_conc[file_index]}_equ-H2O2'.replace('_', ' ')

        ## assuming no bounds have been applied, will neeed to change this bit if applying bounds ##
        self.bounds = np.array([[self.wavelengths[0], self.wavelengths[-1]]])
        self.wl_range_keep = [self.wavelengths[0], self.wavelengths[-1]]
        
        print(f'Loaded dataframe:\n{self.filename}\n{self.delays.shape[0]} {self.metadata[file_index][-1]}ly-spaced time points over {self.delays[-1]}s\n')
        
        return

    

    def find_nearest(array, value):
    
        array = np.asarray(array)           
        idx = (np.abs(array - value)).argmin() 

        return idx, array[idx] 
        
        

    def savefile(self, directory=None):
        
        lent = len(self.delays)
        lenm = len(self.wavelengths)

        array = np.zeros((lent+1, lenm+1))

        array_wl = self.wavelengths
        array_times = self.delays[:, None].reshape((lent,))
        
        if hasattr(self, 'unbound_abs'):
            array_matrix = self.unbound_abs
        else:
            array_matrix = self.abs

        array[0, 1:] = array_wl
        array[1:, 0] = array_times
        array[1:, 1:] = array_matrix
        
        if hasattr(self, 'meta_data'):
            metadata = str(self.meta_data)
        else:
            metadata = ''

        today = date.today()

        current_time = time.localtime()
        timestring = time.strftime("%d/%m/%Y %H:%M:%S")
      
        file = np.savetxt(directory+'/'+self.filename.replace(' ', '_')+'_processed'+'.dat', array,
                          header='XAxisTitleWavelength(nm)\nYAxisTitle Delay (ps)\n'
                          +metadata+'\n'+'Date&time: '+ timestring)
        
        print("\033[1m" + 'Saved processed file of' + "\033[0;0m", self.filename)
       

        if hasattr(self, 'sig_x'):
            array = np.zeros((lent+1, lenm+1))

            array_wl = self.wavelengths
            array_times = self.delays[:,None].reshape((lent,))
            
            if hasattr(self, 'unbound_sig_x'):
                array_matrix = self.unbound_sig_x
            else:
                array_matrix = self.sig_x
    
            array[0, 1:] = array_wl
            array[1:, 0] = array_times
            array[1:, 1:] = array_matrix
            
            np.savetxt(directory+'/'+self.filename.replace(' ', '_')+'_st_dev_arr'+'.dat', array,
                        header=metadata+'\n'+'Date&time: '+ timestring)

        else:
            return

        return

    
        
 
    def explore_spectra(self,
                        df_idx=None,
                        ground_state=None,
                        normal=False,
                        sl_min=0,
                        sl_max=-1,
                        tick_size=25,
                        axis_fontsize=30,
                        title_fontsize=25):
        """
        Interactive widget with a slider to inspect absorption spectra at a given delay
        """
        
        if hasattr(self, 'HS_conc'):
            HS_conc = self.HS_conc[df_idx]
        else:
            HS_conc = 0

        
        if hasattr(self, 'dict_keys'):
            title = f'{self.Sample_concentration}'+r'$\mu$'+f'M_{self.Sample[:len(self.Sample)]}_+{self.dict_keys[df_idx]}_equ'.replace('_', ' ')+r' H$_{2}$O$_{2}$'+f' +{HS_conc} equ. HS'
        else:
            title = self.fig_title.replace('_', ' ')[:-5]+r' H$_{2}$O$_{2}$'+f' +{HS_conc} equ. HS'
        
            
        if ground_state:
            if ground_state.abs.shape[0] != ground_state.wavelengths.shape[0]:
                gs_avg = np.mean(ground_state.abs, axis=0)

            else:
                gs_avg = ground_state.abs
                
            
        X, Y = np.meshgrid(self.wavelengths, self.delays, indexing="ij")
        Z = self.abs
        
        Y_index = []
        Y_values = []
        
        for idx, val in enumerate (Y[0]):
            Y_index.append(idx)
            Y_values.append(val)
        
        Y_series = pd.Series(Y_index, index=Y_values)

        
        fig = plt.figure(figsize=(10,10))
        grid = GridSpec(2, 2,
                        width_ratios=[3, 3],
                        height_ratios=[3, 3], 
                        wspace=0.3,
                        hspace=0.3)
        
        tick_size = tick_size
        axis_fontsize = axis_fontsize
        title_fontsize = title_fontsize
        sl_min = sl_min
        sl_max = sl_max
        
        ax1 = fig.add_subplot(grid[0:7]) 
        
        if normal == True:
            ax1.plot(self.wavelengths, Z[sl_min]/np.max(Z[sl_min]), color='darkslateblue', linewidth=2, label=str('%#.3g' % self.delays[sl_min])+'s')
        else:
            ax1.plot(self.wavelengths, Z[sl_min], color='darkslateblue', linewidth=2, label=str('%#.3g' % self.delays[sl_min])+'s')
        
        ax1.tick_params(axis="both", labelsize = tick_size)
        ax1.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize)
        ax1.set_ylabel('Absorption [OD]', fontsize = axis_fontsize)
        ax1.set_xlim(self.wavelengths[0], self.wavelengths[-1])
        #ax1.set_ylim(-55, 15)
        ax1.legend(fontsize=tick_size, loc='upper right')
        
        
        ax1.set_title(f'{title}\nStopped-Flow Absorption Spectrum at {np.round(self.delays[sl_min], 2)}s', fontsize=title_fontsize, y=1.02)
        ax1.grid(visible=True)

        
        if normal == False:
            try:
                ax1.set_ylim(np.nanmin(Z[sl_min]) - (np.abs(np.nanmin(Z[sl_min]))/4), np.nanmax(Z[sl_min]) + (np.abs(np.nanmax(Z[sl_min]))/4))     
            except:
                pass
        
        # Sliders
        global delay_slider
        
        axwave = plt.axes([0.1, 0.01, 0.8, 0.02])
        delay_slider = Slider(
            ax=axwave,
            label='Delay',
            valmin=self.delays[sl_min],
            valmax=self.delays[sl_max], 
            valinit=self.delays[sl_min],
            valstep=self.delays,
            color='lightsteelblue',
            handle_style={'facecolor': 'white', 'edgecolor': '.05', 'size': 20} # .75
        )
        delay_slider.label.set_size(20)

        def update(val):
            
            # Updates the graphs when the user interacts with the widget.
            
            spectrum_ydata = Z[Y_series[delay_slider.val]]
            
            ax1.cla()

            if ground_state:
                if normal == True:
                    ax1.plot(self.wavelengths, gs_avg/np.max(gs_avg), color='red', linewidth=2, label = f'{self.Sample} ground state')
                else:
                    ax1.plot(self.wavelengths, gs_avg, color='red', linewidth=2, label = f'{self.Sample} ground state')
            else:
                pass
                
            if len(mem_values) > 0:
                
                mem_values.sort() 
                
                cmap_ = plt.get_cmap('viridis')
                colors_ = [cmap_(i) for i in np.linspace(0, 1, len(mem_values))]

                for i, color in enumerate(colors_, start = 0):
                    if normal == True:
                        ax1.plot(self.wavelengths, Z[Y_series[mem_values[i]]] / np.max(Z[Y_series[mem_values[i]]]), color=color, linewidth = 2, label = str('%#.3g' % mem_values[i])+'s')
                    else:
                        ax1.plot(self.wavelengths, Z[Y_series[mem_values[i]]], color=color, linewidth = 2, label = str('%#.3g' % mem_values[i])+'s')

            if normal == True:
                ax1.plot(self.wavelengths, Z[Y_series[delay_slider.val]]/np.max(Z[Y_series[delay_slider.val]]), color='darkslateblue', linewidth=2, label = str('%#.3g' % delay_slider.val)+'s')
            else:
                ax1.plot(self.wavelengths, Z[Y_series[delay_slider.val]], color='darkslateblue', linewidth=2, label = str('%#.3g' % delay_slider.val)+'s')

            ax1.tick_params(axis="both", labelsize = tick_size)
            ax1.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize)
            ax1.set_ylabel('Absorption [OD]', fontsize = axis_fontsize)
            ax1.set_xlim(self.wavelengths[0], self.wavelengths[-1])
            #ax1.set_ylim(-60, 15)
            ax1.legend(fontsize=tick_size, loc='upper right')
            ax1.set_title(f'{title}\nStopped-Flow Absorption Spectrum at {np.round(delay_slider.val, 2)}s', fontsize=title_fontsize, y=1.02)
            ax1.grid(visible=True)
            
            fig.canvas.draw_idle()
            
            if normal == False:
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

            ## script for plotting the selected spectra on a separate figure ##
            """
            cmap_ = plt.get_cmap('viridis')
            colors_ = [cmap_(i) for i in np.linspace(0, 1, len(mem_values))]

            
            fig_1 = plt.figure(figsize=(10,10))
            grid_1 = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], 
                        wspace=0.3, hspace=0.3)
        
            ax2 = fig_1.add_subplot(grid_1[0:7])
            ax2.cla()

            
            for i, color in enumerate(colors_, start = 0):

                ax2.plot(self.wavelengths, Z[Y_series[mem_values[i]]], color=color, linewidth = 2, label = f'{np.round(mem_values[i], 2)}s')
                
                ax2.tick_params(axis="both", labelsize = tick_size)
                
                #ax2.tick_params(axis="y",
                #                which='both',
                #                left=False,
                #                labelleft=False)
                
                ax2.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize) # , labelpad=20)
                ax2.set_ylabel(r'$\Delta$OD [mOD]', fontsize = axis_fontsize, labelpad=20)
                ax2.set_xlim(self.wavelengths[0], self.wavelengths[-1])
                #ax2.set_ylim(-55, 15)
                ax2.legend(fontsize=tick_size, loc='upper right')
                ax2.set_title(f'{self.filename} SF Absorption Spectra at {", ".join(str(time) for time in mem_values[:-1])} and {mem_values[-1]}s', fontsize = title_fontsize, y=1.02)
                ax2.grid(visible=True)
                
                fig_1.canvas.draw_idle()
            """
            return
        
        
        def reset(event):
            
            mem_values.clear()
            delay_slider.reset()
            
            ax1.cla()

            if normal == True:
                ax1.plot(self.wavelengths, Z[sl_min]/np.max(Z[sl_min]), color='darkslateblue', linewidth = 2, label=str('%#.3g' % self.delays[sl_min])+'s')
            else:
                ax1.plot(self.wavelengths, Z[sl_min], color='darkslateblue', linewidth = 2, label=str('%#.3g' % self.delays[sl_min])+'s')
                
            ax1.tick_params(axis="both", labelsize = tick_size)
            ax1.set_xlabel('Wavelength (nm)', fontsize=axis_fontsize)
            ax1.set_ylabel('Absorption [OD]', fontsize=axis_fontsize) 
            ax1.set_xlim(self.wavelengths[0], self.wavelengths[-1])
            #ax1.set_ylim(-60, 15)
            ax1.legend(fontsize=tick_size, loc='upper right')
            ax1.set_title(f'{title}\nStopped-Flow Spectrum at '+str(np.round(self.delays[0], 2))+' s', fontsize = title_fontsize, y=1.02)
            ax1.grid(visible=True)
            
            if normal == False:
                try:
                    ax1.set_ylim(np.nanmin(Z[sl_min]) - (np.abs(np.nanmin(Z[sl_min]))/4), np.nanmax(Z[sl_min]) + (np.abs(np.nanmax(Z[sl_min]))/4))
                except:
                    pass
            
            fig.canvas.draw_idle()


        reset_button.on_clicked(reset)
        resetax._button = reset_button
        
        add_click_button.on_clicked(add_click)
        add_clickax._button = add_click_button


        plt.show()

        self.delays_list = mem_values
        
        return


    
    def explore_traces(self,
                       df_idx=None,
                       normal=False,
                       sl_min=0,
                       sl_max=-1,
                       ax_min=-1,
                       ax_max=-1,
                       tick_size=25,
                       axis_fontsize=30,
                       title_fontsize=25):
        
        """
        Interactive widget with a slider to inspect traces at a given wavelength
        """        

        X, Y = np.meshgrid(self.wavelengths, self.delays, indexing="ij")
        Z = self.abs
        
        X_index = []
        X_values = []
        
        for idx, val in enumerate (X.T[0]):
            X_index.append(idx)
            X_values.append(val)
        
        X_series = pd.Series(X_index, index=X_values)
        
        tick_size = tick_size
        axis_fontsize = axis_fontsize
        title_fontsize = title_fontsize
        sl_min = sl_min
        sl_max = sl_max

        if hasattr(self, 'HS_conc'):
            HS_conc = self.HS_conc[df_idx]
        else:
            HS_conc = '0'

        
        if hasattr(self, 'dict_keys'):
            title = f'{self.Sample_concentration}'+r'$\mu$'+f'M_{self.Sample[:len(self.Sample)]}_+{self.dict_keys[df_idx]}_equ'.replace('_', ' ')+r' H$_{2}$O$_{2}$'+f' +{HS_conc} equ. HS'
        else:
            title = self.fig_title.replace('_', ' ')[:-5]+r' H$_{2}$O$_{2}$'+f' +{HS_conc} equ. HS'

        
        fig = plt.figure(figsize=(10,10))
        grid = GridSpec(2, 2,
                        width_ratios=[3, 3],
                        height_ratios=[3, 3], 
                        wspace=0.3,
                        hspace=0.3)
        
        
        ax1 = fig.add_subplot(grid[0:7])

        if normal == True:
            ax1.plot(self.delays, Z.T[sl_min]/np.max(np.abs(Z.T[sl_min])), color='darkslateblue', linewidth=2, label=str('%#.3g' % self.wavelengths[sl_min])+'nm')
        else:
            ax1.plot(self.delays, Z.T[sl_min], color='darkslateblue', linewidth=2, label=str('%#.3g' % self.wavelengths[sl_min])+'nm')
        
        ax1.tick_params(axis="both", labelsize = tick_size)
        ax1.set_xlabel('Time (s)', fontsize = axis_fontsize)
        ax1.set_ylabel(r'Absorption (O.D)', fontsize = axis_fontsize)
        ax1.set_xlim(ax_min, self.delays[ax_max])
        
        ax1.set_title(f'{title} trace at '+str('%#.3g' % self.wavelengths[sl_min])+' nm', fontsize = title_fontsize, y=1.02)
        ax1.grid(visible=True)
        
        # Sliders
        global spectrum_slider
        
        if hasattr(self, 'wlstep'):
                valstep=self.wlstep
        else:
            valstep=self.wavelengths[0:]
            
        axwave = plt.axes([0.1, 0.01, 0.8, 0.02])
        spectrum_slider = Slider(
            ax=axwave,
            label='Wavelength',
            valmin=self.wavelengths[0],
            valmax=self.wavelengths[-1],
            valinit=self.wavelengths[0],
            valstep=valstep,
            color='lightsteelblue',
            handle_style={'facecolor': 'white', 'edgecolor': '.05', 'size': 20} # .75
        )
        spectrum_slider.label.set_size(20)

        def update(val):
            
            ## Updates the graphs when the user interacts with the widget ##  

            ax1.cla()
            
            if len(mem_values) > 0:
                mem_values.sort() 
    
                cmap_ = plt.get_cmap('viridis')
                colors_ = [cmap_(i) for i in np.linspace(0, 1, len(mem_values))]

                for i, color in enumerate(colors_, start = 0):
                    
                    if normal == True:
                        ax1.plot(self.delays, Z.T[X_series[mem_values[i]]]/np.max(np.abs(Z.T[X_series[mem_values[i]]])), color=color, linewidth=2, label = str('%#.3g' % mem_values[i])+' nm')
                    else:
                        ax1.plot(self.delays, Z.T[X_series[mem_values[i]]], color=color, linewidth=2, label=str('%#.3g' % mem_values[i])+' nm')

                
            if normal == True:
                ax1.plot(self.delays, Z.T[X_series[spectrum_slider.val]]/np.max(np.abs(Z.T[X_series[spectrum_slider.val]])), color='darkslateblue', linewidth=2, label = str('%#.3g' % spectrum_slider.val)+' nm')
            else:
                ax1.plot(self.delays, Z.T[X_series[spectrum_slider.val]], color='darkslateblue', linewidth=2, label = str('%#.3g' % spectrum_slider.val)+' nm')
            
            ax1.tick_params(axis="both", labelsize = tick_size)
            ax1.set_xlabel('Time (s)', fontsize = axis_fontsize)
            ax1.set_ylabel('Absorption (O.D)', fontsize = axis_fontsize) 
            ax1.set_xlim(ax_min, self.delays[ax_max])
            ax1.legend(fontsize=tick_size, loc='upper right')
            ax1.set_title(f'{title} trace at '+str('%#.6g' % spectrum_slider.val)+' nm', fontsize = title_fontsize, y=1.02)
            ax1.grid(visible=True)
            
            fig.canvas.draw_idle()

            if normal == False:
                try:
                    ax1.set_ylim(np.nanmin(Z.T[X_series[spectrum_slider.val]]) - (np.abs(np.nanmin(Z.T[X_series[spectrum_slider.val]]))/4), np.nanmax(Z.T[X_series[spectrum_slider.val]]) + (np.abs(np.nanmax(Z.T[X_series[spectrum_slider.val]]))/4))
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
            """
            cmap_ = plt.get_cmap('viridis')
            colors_ = [cmap_(i) for i in np.linspace(0, 0.5, len(mem_values))]
            
            fig_1 = plt.figure(figsize=(10,10))
            grid_1 = GridSpec(2, 2, width_ratios=[3, 3], height_ratios=[3, 3], 
                        wspace=0.3, hspace=0.3)
        
            ax2 = fig_1.add_subplot(grid_1[0:7])
            ax2.cla()
            
            for i, color in enumerate(colors_, start = 0):
                #line1, = ax1.plot(self.wavelengths, self.get_spectrum(i)[0], linewidth = 2, label = str(np.round(i, 2))+' s')
                
                ax2.plot(self.delays, self.get_trace(mem_values[i])[0], color=color, linewidth = 2, label = str(np.round(mem_values[i], 2))+' nm')
                
                ax2.tick_params(axis="both", labelsize = tick_size)
                ax2.set_xlabel('Delay (s)', fontsize = axis_fontsize)
                ax2.set_ylabel(r'$\Delta$OD [mOD]', fontsize = axis_fontsize) # 'Difference Absorption (mOD)'
                ax2.set_xlim(ax_min, self.delays[ax_max])
                ax2.legend(fontsize=tick_size, loc='upper right')
                ax2.set_title(self.filename+' Traces', fontsize = title_fontsize, y=1.02)
                ax2.grid(visible=True)
                
                fig_1.canvas.draw_idle()
            """
            
            self.wavelengths_list = mem_values
            
            return
            
            
        
        def reset(event):
            
            mem_values.clear()
            spectrum_slider.reset()
            
            ax1.cla()
            
            if normal == True:
                ax1.plot(self.delays, Z.T[sl_min]/np.max(np.abs(Z.T[sl_min])), color='darkslateblue', linewidth=2, label=str('%#.3g' % self.wavelengths[sl_min])+'nm')
            else:
                ax1.plot(self.delays, Z.T[sl_min], color='darkslateblue', linewidth=2, label=str('%#.3g' % self.wavelengths[sl_min])+'nm')
                
            ax1.tick_params(axis="both", labelsize = tick_size)
            ax1.set_xlabel('Time (s)', fontsize = axis_fontsize)
            ax1.set_ylabel('Absorption (O.D)', fontsize = axis_fontsize) # 'Difference Absorption (mOD)'
            ax1.set_xlim(ax_min, self.delays[ax_max])
            ax1.legend(fontsize=tick_size, loc='upper right')
            ax1.set_title('Trace at '+str('%#.3g' % self.wavelengths[sl_min])+' nm', fontsize = title_fontsize, y=1.02)
            ax1.grid(visible=True)
            
            fig.canvas.draw_idle()


        reset_button.on_clicked(reset)
        resetax._button = reset_button
        
        add_click_button.on_clicked(add_click)
        add_clickax._button = add_click_button

        plt.show()
        
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
        
        self.reconstructed = np.sum(self.relevant_matrices, axis=0)
        
        return


    
    def explore_SVD(self,
                    df_idx=None,
                    x_axmin=0,
                    x_axmax=-50,
                    tick_size=15,
                    label_fontsize=15,
                    title_fontsize=20,
                    cmap='Reds',
                    save_plot=False,
                    output_directory=None,
                    save_dpi=200):
        """
        generates a widget containing a plot of the SVD eigenvalues, traces, spectra for n components as well as colourmap of the original 
        array of absorption values 
        """
        if hasattr(self, 'HS_conc'):
            HS_conc = self.HS_conc[df_idx]
        else:
            HS_conc = 0

        if hasattr(self, 'data_series'):
            metadata = self.metadata[df_idx, -1]
        else:
            metadata = self.metadata[-1]
        
        fig = plt.figure(figsize=(10, 10)) # 10, 10
        grid = GridSpec(2, 10, # 2,2 
                        width_ratios=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 
                        height_ratios=[3, 3],
                        wspace=0.3,
                        hspace=0.3,)
        
        ax1 = fig.add_subplot(grid[0, :6]) # grid[0]
        
        vmin = np.min(self.abs)
        vmax = np.max(self.abs)
        
        im = ax1.pcolormesh(self.wavelengths,
                            self.delays,
                            self.abs,
                            cmap=cmap) # 'hot_r')

        ax1.tick_params(axis="both", labelsize=tick_size)
        ax1.set_ylabel('Time (s)', fontsize=label_fontsize, labelpad=20)
        ax1.set_title(self.fig_title.replace('_', ' ')[:-5]+r' H$_{2}$O$_{2}$'+f' +{HS_conc} equ. HS\n{self.delays.shape[0]} {metadata}ly-spaced time points over {int(self.delays[-1])}s', x=0.55, fontsize=title_fontsize, pad=20)
        
        tick_range = np.linspace(vmin, vmax, 10)
        
        cbar = fig.colorbar(im,
                            ticks=tick_range,
                            fraction=0.121,
                            pad=0.05) # 0.01)
                            # location='bottom')
        
        cbar.ax.tick_params(labelsize=tick_size)
        cbar.set_label(label=r'O.D', fontsize=label_fontsize, labelpad=10)


        _cmap = plt.get_cmap('inferno') # 'inferno'
        _colors = [_cmap(i) for i in np.linspace(0, 1, len(self.relevant_spectra)*100)]  
        # just the number of components plus some extra to get better colors
        # be careful with indexing because currently will go out of range if more components are added

        
        ax2 = fig.add_subplot(grid[1, 5:])
        for i, trace in enumerate(self.relevant_kinetics):
            ax2.plot(self.delays, trace, color = _colors[i*100], label='Component '+str(i+1))

        ax2.tick_params(axis="x", labelsize=tick_size)
        ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax2.set_xlabel('Time (s)', fontsize=label_fontsize, labelpad=20)
        ax2.set_xlim(self.delays[x_axmin], self.delays[x_axmax]) # self.delays[0], self.delays[-1]
        ax2.set_title('Principle Kinetics', fontsize=title_fontsize, pad=20)
        ax2.legend()
        

        ax3 = fig.add_subplot(grid[1, :5]) 
        
        for i, spectrum in enumerate(self.relevant_spectra):
            ax3.plot(self.wavelengths, spectrum, color=_colors[i*100], label='Component '+str(i+1))

        ax3.tick_params(axis="both", labelsize=tick_size)
        ax3.set_ylabel('Amplitude (a.u)', fontsize=label_fontsize, labelpad=20)
        ax3.set_xlabel('Wavelength (nm) ', fontsize=label_fontsize, labelpad=20)
        ax3.set_xlim(left=self.wavelengths[0], right=self.wavelengths[-1])
        ax3.legend()
        ax3.set_title('Principle Spectra', fontsize=title_fontsize, pad=20)

        
        ax4 = fig.add_subplot(grid[0, 6:])
        ax4.scatter(range(0, len(self.singular_values)), self.singular_values, s=10, c=_colors[1])
        
        ax4.tick_params(axis="both", labelsize=tick_size)
        ax4.set_ylabel('Amplitude (a.u)', fontsize=label_fontsize, labelpad=20)
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position("right")
        
        ax4.set_xlabel('Component number', fontsize=label_fontsize, labelpad=10)
        ax4.set_title('Singular Values', fontsize=title_fontsize, pad=20)
        
        
        plt.show()
        
        if save_plot:
            ## make the figure full screen before saving to retain formatting ##
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle() 
            
            fig.savefig(f'{output_directory}\{self.filename}_SVD.png', dpi=save_dpi)
            plt.close('all')
        
        
        for i, matrix in enumerate(self.relevant_matrices):
                    fig, ax = plt.subplots()
                    # create a figure so I can adjust the position of the labels using coordinates on the figure axis
                    
                    vmin = np.min(matrix)
                    vmax = np.max(matrix)
            
            
                    im = ax.pcolormesh(self.wavelengths, self.delays, matrix, cmap='Reds') # 'hot'
                                  
                    ax.tick_params(axis="both", labelsize=tick_size)
                    ax.xaxis.set_label_coords(0, -0.05)
                    ax.yaxis.set_label_coords(-0.08, 0.5)
            
                    # tick_range = np.linspace(vmin, vmax, 10)
                    # cbar = fig.colorbar(im, ticks=tick_range) 
                    # cbar.ax.tick_params(labelsize=tick_size)
                    # cbar.set_label(label='Amplitude [a.u]', fontsize=20, y=0.52, labelpad=100, rotation=360)
                    
                    ax.set_xlabel('Wavelength (nm)', fontsize=20, labelpad=20)
                    ax.set_ylabel('Time (s)', fontsize=20, rotation=360, labelpad=20)
                    ax.set_title('SVD Component '+str(i+1), x=0.55, fontsize=40, pad=40)
                    plt.show()
            
                    if save_plot:
                        ## make the figure full screen before saving to retain formatting ##
                        manager = plt.get_current_fig_manager()
                        manager.full_screen_toggle() 
                        
                        fig.savefig(f'{output_directory}\{self.filename}_SVD_rugplot_{i+1}.png', dpi=save_dpi)
                        plt.close('all')
        
        return


    ## NESTED SAMPLING ##
    ## MODIFY TO FIT SF DATA TO PSEUDO FIRST-ORDER KINETICS ##
    ########################################################################################################################################################
    def exp(t, k, c):
        return np.exp(-k*t) + c

    def normal_exp(t, k):
        return np.exp(-k*t)

    
    def prior_transform(utheta, tau_lowlim, tau_uplim, c_lowlim, c_uplim, nparams, Ep):
        
        # for use with nested sampling
        # transfrom from unit cube (0, 1) to parameter of interest
        # scale and shift
        
        # if Ep:
        #    nparams = nparams - len(Ep)
            
        nc = len(c_lowlim)
        
        taulimdiff = tau_uplim[0:nparams-nc] - tau_lowlim[0:nparams-nc]
        
        utaus = utheta[0:nparams-nc]
        taus = utaus*taulimdiff + tau_lowlim[0:nparams-nc]
        
        if nc > 0:
            c_limdiff = c_uplim[-nc:] - c_lowlim[-nc:] 
            uc = utheta[-nc:]
            c_s = uc*c_limdiff + c_lowlim[-nc:]
            theta = list(taus) + list(c_s)
            
        else:
            theta = list(taus)
            
        """
        if Ep:
            Ep_tau = []
            
            if len(Ep) > 1:
                for i in Ep:
                    Ep_tau.append(theta[i])
                    
            elif len(Ep) == 1:
                theta.append(theta[Ep[0]])
                             
            theta = theta + Ep_tau
        """
        return np.asarray(theta)



    def theta_to_E_matrix(theta, E):
        
        nrows, ncols = E.shape
        Etheta = np.zeros_like(E).astype(float) 
        krows, kcols = np.nonzero(E)
        
        for idx, row in enumerate(krows):
            Etheta[row, kcols[idx]] = theta[idx]

        return Etheta

    
    def log_likelihood(theta, E, C0, t, data, nparams, sigma, Ep, var_c, T=1):
        # for use with nested sampling
        # print(theta, E, C0, t.shape, data.shape, nparams, Ep)
        
    
        Etheta = SF_Rug.theta_to_E_matrix(theta, E)
        
        if theta.shape[0] > np.count_nonzero(E):
            c_theta = theta[np.count_nonzero(E):]
            fitted_data, _ = SF_Rug.calculate_SF_dynamics(Etheta, C0, c_theta, t, data, Ep, var_c, log=False, verbose=False)
        else:
            c_theta=None
            fitted_data, _ = SF_Rug.calculate_SF_dynamics(Etheta, C0, c_theta, t, data, Ep, var_c, log=False, verbose=False)
        
        diff = fitted_data - data
        logp = -np.sum(( (diff)**2) / (sigma**2)) / T
        
        return logp
        

    def E_matrix_to_K_matrix(E):
        
        nrows, ncols = E.shape
        K = np.zeros_like(E)
        
        for idx in range(nrows):
            for jdx in range(ncols):
                K[idx, idx] += - E[idx, jdx]
                if idx != jdx:
                    K[jdx, idx] += E[idx, jdx]
                    
        return K

    

    def calculate_SF_dynamics(E, C0, c_theta, t, data, Ep, var_c=False, log=False, verbose=False):
        # put guesses into matrix E, and then have this transformed to the actual K matrix first
        # E already contains the guesses for the rate constants, and we know the indexing matches
        
        E = np.asarray(E)
        C0 = np.asarray(C0)

        K = SF_Rug.E_matrix_to_K_matrix(E)

        if log:
            # kest = [1/(10**tau) for tau in tauest] # convert log space proposals back to linear space
            K = np.divide(1, 10**K, out=np.zeros_like(K), where=K!=0)
        else:
            K = np.divide(1, K, out=np.zeros_like(K), where=K!=0)

        
        # eigenvals and vects of k matrix to diagonalise
        evals, evects = np.linalg.eig(K) 
     #  if len(evals) < len(kvec):
     #     raise ValueError('Defined K matrix has not got enough independent eigenvalues to diagonalise, reconsider model.')

        evects_inv = np.linalg.inv(evects) # inverse of U for similarity transform of K matrix

        # vector  of the initial population fraction in each component 
        source = C0.T

        # define the A matrix that transforms the conc profiles from the diagonal case to the case defined by the K matrix
        A = evects @ np.diagflat(evects_inv @ source)

        # construct dynamics in the eigenbasis (we will transform this back into the original basis through A)
        # use -k here because the eigenvalues will be negative rate constants as per our definition of K
        
        if var_c == True:
            kc_params = []
            
            kc_params.append(np.array(evals))
            kc_params.append(np.array(c_theta))
            
            kcp = np.array(kc_params).T
 
            cD_arr = np.array([SF_Rug.exp(t, -k[0], k[1]) for k in kcp]).T 
            
            """
            if Ep:
                
                
                if len(Ep) > 1:
                    Ep_arr = np.array([1 - SF_Rug.exp(t, -kcp[i, 0], kcp[i, 1]) for i in Ep]).T 
                    cD = np.concatenate((cD_arr, Ep_arr), axis=1)
                
                elif len(Ep) == 1:
                    Ep_arr = np.array([1 - SF_Rug.exp(t, -kcp[Ep[0], 0], kcp[Ep[0], 1])]).T 
                    cD = np.concatenate((cD_arr, Ep_arr), axis=1)
                
            else:
                cD = cD_arr
            """
            cD = cD_arr
            
        elif var_c==True and c_theta != None:                
            cD_arr = np.array([SF_Rug.exp(t, -k, c_theta) for k in evals]).T 
            """
            if Ep:
                pass
                
                if len(Ep) > 1:
                    Ep_arr = np.array([1 - SF_Rug.exp(t, -evals[i], c_theta) for i in Ep]).T 
                    cD = np.concatenate((cD_arr, Ep_arr), axis=1)

                elif len(Ep) == 1:
                    Ep_arr = np.array([1 - SF_Rug.exp(t, -evals[Ep[0]], c_theta)]).T 
                    cD = np.concatenate((cD_arr, Ep_arr), axis=1)
                
            else:
                cD = cD_arr
            """
            cD = cD_arr
        

        else:
            cD_arr = np.array([SF_Rug.normal_exp(t, -k) for k in evals]).T 
            
            cD = cD_arr
            """
            if Ep:
                if len(Ep) > 1:
                    pass
                    # Ep_arr = np.array([1 - SF_Rug.normal_exp(t, -evals[i]) for i in Ep]).T 
                    # cD = np.concatenate((cD_arr, Ep_arr), axis=1)
                    
                    # cD = np.array([cD_arr[i] = 1 - SF_Rug.normal_exp(t, -evals[i]) for i in Ep])
                
                elif len(Ep) == 1:
                    print(evals[:-1])
                    cD_arr = np.array([SF_Rug.normal_exp(t, -k) for k in evals[:Ep[0]]])
                    Ep_arr = np.array([1 - SF_Rug.normal_exp(t, -evals[Ep[0]])])
    
                    cD = np.concatenate((cD_arr, Ep_arr), axis=0).T
                
            else:
                cD = cD_arr
            """


        # multiply by A transpose to get back the conc profiles from the case defined by the K matrix
        # c = cD @ A.T 

        if Ep: # assumes that, for a compartment that decays to a stable photoproduct, all of the decay pathways from that compartment lead to the stable product
            if len(Ep) == 1:
                c_arr = cD @ A.T

                Ep_vals = evals.tolist()
                Ep_vals.reverse()                
                
                Ep_arr = np.array([1 - SF_Rug.normal_exp(t, -Ep_vals[Ep[0]])])

                c = np.concatenate((c_arr.T, Ep_arr), axis=0).T
                
        else:
            c = cD @ A.T 



        s, resid, rank, sing = np.linalg.lstsq(c, data, rcond=None)

        #compute a fitted data matrix from the modelled kinetics and extracted spectra
        SF_matrix = c @ s
        
        return SF_matrix, [c, s]


        

    def run_nested_sampling(wt_kwargs,
                            stop_kwargs,
                            E, C0, self, sigma, taulims, c_lims, Ep=None, var_c=False,
                            sample_method='auto',
                            bound_method='multi',
                            log_likelihood=log_likelihood, 
                            prior_transform=prior_transform,
                            checkpoint_file=None):


        self.unbound_abs = self.abs # carried over from workaround in RugPeek for plotting data with modified axis ranges.

        
        tau_lowlim, tau_uplim = taulims
        c_lowlim, c_uplim = c_lims


        nparams = np.count_nonzero(E) + len(c_lims[0])

        dsampler = dynesty.DynamicNestedSampler(log_likelihood,
                                                prior_transform,
                                                ndim=nparams,
                                                bound=bound_method,
                                                sample=sample_method,
                                                ptform_args=(tau_lowlim, tau_uplim, c_lowlim, c_uplim, nparams, Ep),
                                                logl_args=(E, C0, self.delays, self.unbound_abs, nparams, sigma, Ep, var_c))
        
        
        dsampler.run_nested(wt_kwargs=wt_kwargs,
                            checkpoint_file=checkpoint_file)
        
        self.dres = dsampler.results

        self.dres.summary()

        samples, weights = self.dres.samples, self.dres.importance_weights()
        mean, cov = dyfunc.mean_and_cov(samples, weights)

        self.DNS_fit_params = mean        

        ## compute average spectra and concentration profiles ##
        Etheta = SF_Rug.theta_to_E_matrix(self.DNS_fit_params, E)
        tempdata, temp = SF_Rug.calculate_SF_dynamics(Etheta, C0, mean[-len(c_lims[0]):], self.delays, self.unbound_abs, Ep, var_c, log=False, verbose=False)
        
        self.DNS_matrix = tempdata
        self.DNS_CP = temp[0]
        self.DNS_SP = temp[1]
        
        # Resample weighted samples.
        samples_equal = self.dres.samples_equal()

        # Generate a new set of results with sampling uncertainties.
        self.results_sim = dyfunc.resample_run(self.dres)
        
        return 
        

    def load_DNS(self,
                 E=None,
                 C0=None,
                 Ep=None,
                 c_lims=None,
                 var_c=False,
                 fname=None):
        
        dsampler = dynesty.DynamicNestedSampler.restore(fname=fname)
        self.dres = dsampler.results

        samples, weights = self.dres.samples, self.dres.importance_weights()
        mean, cov = dyfunc.mean_and_cov(samples, weights)

        self.DNS_fit_params = mean
        """
        if Ep:
            if len(Ep) == 1:
                E_p = self.DNS_fit_params.tolist()
                E_p.append(E_p[Ep[0]])
                self.DNS_fit_params = np.array(E_p)
            else:
                pass
        """
        
        

        ## compute average spectra and concentration profiles ##
        Etheta = SF_Rug.theta_to_E_matrix(self.DNS_fit_params, E)
        tempdata, temp = SF_Rug.calculate_SF_dynamics(Etheta, C0, mean[-len(c_lims[0]):], self.delays, self.abs, Ep, var_c, log=False, verbose=False)
        
        self.DNS_matrix = tempdata
        self.DNS_CP = temp[0]
        self.DNS_SP = temp[1]

        """
        # Resample weighted samples.
        samples_equal = self.dres.samples_equal()

        # Generate a new set of results with sampling uncertainties.
        self.results_sim = dyfunc.resample_run(self.dres)
        """
        
        return


    def plot_nested_sampling(self,
                             E=None,
                             C0=None,
                             Ep=None,
                             c_lims=None,
                             var_c=False,
                             df_idx=None,
                             nsamples=20,
                             stdfactor=1,
                             modelname='',
                             fname=None,
                             labelstau=None,
                             e_labels=None,
                             
                             span=None,
                             
                             alignment='horizontal',
                             
                             plot_traceplot=True,
                             plot_runplot=True,
                             plot_cornerplot=True,
                             plot_cornerpoints=True,
                             plot_SF=True,
                             plot_SF_tileplot=False,
                             # figsize=8,
                             
                             save_dynesty_plots=False,
                             save_SF_plots=False,
                             save_corner_plots=False,
                             save_SF_tileplot=False,
                             output_directory='./',
                             save_dpi=200,
                             title_fontsize=25,
                             label_fontsize=25,
                             tick_size=10,
                             text_size=15,
                             ud_idx=-1,
                             SF_cmap='Reds'):
        

        samples, weights = self.dres.samples, self.dres.importance_weights() 
        mean, cov = dyfunc.mean_and_cov(samples, weights)
        flat_samples = self.dres.samples_equal() 
        nparams = flat_samples.shape[1]
        
        figsize=nparams*2
        
        if hasattr(self, 'HS_conc'):
            HS_conc = self.HS_conc[df_idx]
        else:
            HS_conc = 0

        """
        if hasattr(self, 'dict_keys'):
            title = f'{self.Sample_concentration}'+r'$\mu$'+f'M_{self.Sample[:len(self.Sample)]}_+{self.dict_keys[df_idx]}_equ'.replace('_', ' ')+r' H$_{2}$O$_{2}$'+f' +{HS_conc} equ. HS'
        else:
            title = self.fig_title.replace('_', ' ')[:-5]+r' H$_{2}$O$_{2}$'+f' +{HS_conc} equ. HS'
        """
        if hasattr(self, 'data_series'):
            metadata = self.metadata[df_idx, -1]
        else:
            metadata = self.metadata[-1]

        

        if labelstau == None:
            labelstau = [r'$\tau_{{'+str(i+1)+str(i)+'}}$' for i in np.flip(range(nparams-len(c_lims[0])))]
            
        labels_offset = [r'C$_{{'+str(i+1)+'}}$' for i in range(len(c_lims[0]))]

        labels = labelstau + labels_offset

        if e_labels == None:
            if Ep:
                e_labels = [r'e$_{{'+str(i+1)+'}}$' for i in range(E.shape[0] + len(Ep))]
            else:
                e_labels = [r'e$_{{'+str(i+1)+'}}$' for i in range(E.shape[0])]
        
        if modelname:
            fig_title = self.fig_title.replace('_', ' ')[:-5]
            title = f'{fig_title}'+r' H$_{2}$O$_{2}$'+f'\n\n{modelname}'
            
            save_fname = f'{self.filename} {fname}'
            savestring = save_fname.replace(' ', '_')
        
        else:
            title = self.filename
            savestring = title.replace(' ', '_')
        


        
        if plot_traceplot:
            if span:
                fig, axes = dyplot.traceplot(self.dres,
                                             span=span,
                                             labels=labels, 
                                             fig=plt.subplots(nparams, 2, figsize=(figsize, figsize), layout='tight'))
            else:
                fig, axes = dyplot.traceplot(self.dres,
                                             labels=labels, 
                                             fig=plt.subplots(nparams, 2, figsize=(figsize, figsize), layout='tight'))

            for i in range(nparams):
                axes[i, 0].set_ylabel(labels[i], fontsize=20, ha='right', labelpad=20, rotation=360)
                axes[i, 1].set_xlabel(labels[i], fontsize=10, labelpad=-1)


            fig.suptitle(title, fontsize=title_fontsize)  

            if save_dynesty_plots:
                ## make the figure full screen before saving to retain formatting ##
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle() 
                
                fig.savefig(f'{output_directory}\{savestring}_traceplot.png', dpi=save_dpi)
                plt.close('all')

        
        if plot_runplot:
            fig, axes = dyplot.runplot(self.dres,
                                       logplot=True) # ,
                                       # fig=plt.subplots(4, layout='constrained'))
            
            ns_labels = ['Live Points',
                         'Likelihood\n(normalized)',
                         'Importance\nWeight PDF',
                         'Evidence']

            for i in range(4):
                axes[i].set_ylabel(ns_labels[i], fontsize=15, ha='right', x=0.15, rotation=360)

            
            fig.suptitle(title, fontsize=title_fontsize)  
            fig.tight_layout()
            
            if save_dynesty_plots:
                ## make the figure full screen before saving to retain formatting ##
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle() 
                
                fig.savefig(f'{output_directory}\{savestring}_runplot.png', dpi=save_dpi)
                plt.close('all')
                


        
        if plot_cornerplot:
            fig, axes = dyplot.cornerplot(self.dres,
                                          show_titles=True, 
                                          labels=labels,
                                          title_kwargs={'fontsize': title_fontsize,
                                                        'x' : 0.6},
                                          title_fmt='.3g',
                                          fig=plt.subplots(nparams, nparams, figsize=(figsize, figsize)))
            """
            for i, ax in enumerate(fig.get_axes()[len(fig.get_axes())-len(labels):len(fig.get_axes())-1]):
                ax.set_xlabel(labels[i], fontsize=label_fontsize, labelpad=20)
                fig.get_axes()[-1].set_xlabel('')
        
        
            for i, ax in enumerate(fig.get_axes()[0+len(labels):len(fig.get_axes()):len(labels)]):
                ax.set_ylabel(labels[i+1], fontsize=label_fontsize, ha='right', x=-0.2, rotation=360)
        
            for ax in fig.get_axes():
                ax.tick_params(axis='both', labelsize=tick_size)
            """ 
            fig.suptitle(title, x=0.6, fontsize=title_fontsize)  
            
            if save_dynesty_plots or save_corner_plots:
                ## make the figure full screen before saving to retain formatting ##
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle() 
                
                fig.savefig(f'{output_directory}\{savestring}_cornerplot.png', dpi=save_dpi)
                plt.close('all')

        
        if plot_cornerpoints:
            fig, ax = dyplot.cornerpoints(self.dres, 
                                          cmap='viridis',
                                          kde=False,
                                          labels=labels,
                                          label_kwargs={'fontsize' : label_fontsize,
                                                        'rotation' : 360})

                
            fig.suptitle(title, x=0.6, fontsize=title_fontsize)  
            
            if save_dynesty_plots:
                ## make the figure full screen before saving to retain formatting ##
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle() 
                
                fig.savefig(f'{output_directory}\{savestring}_cornerpoints.png', dpi=save_dpi)
                plt.close('all')
                


        

        if plot_SF and alignment=='vertical':
            #if data is None:
            #    raise Error('Need to pass in a data Rug to use plot the actual TA fit.')
            if E is None:
                raise Error('need to pass in an encoding (E) matrix to define the kinetic model.')
            if C0 is None:
                raise Error('need to pass in initial concentration vector (C0) to fit the kinetic model.')


            grid = GridSpec(3, 2, hspace=0.3, wspace=0.5)
            fig = plt.figure(figsize=(12, 15))
            
            means = np.mean(flat_samples, axis=0)
            sdevs = np.std(flat_samples, axis=0)
            
            rng = np.random.default_rng()
            sampled_params = rng.choice(flat_samples, nsamples, axis=0)
            
            if Ep:
                ncpts = E.shape[0] + len(Ep)
            else:
                ncpts = E.shape[0]

            sampled_matrix = []
            sampled_kinetics = []
            sampled_spectra = []

            title_fontsize=18
            #text_size=10
            
            fig1 = fig.add_subplot(grid[2, 1])
            fig2 = fig.add_subplot(grid[1, 1])
            fig3 = fig.add_subplot(grid[0, 1])

            fig1.set_title('Concentration Profiles', fontsize=title_fontsize, pad=10)
            fig2.set_title('Spectra', fontsize=title_fontsize, pad=10)
            fig3.set_title('Information', fontsize=title_fontsize, pad=20)

            # takes these from the original rug object (data)
            t = self.delays
            w = self.wavelengths

            #cycler for the component colours
            cptcolours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

            # loop over the randomly drawn samples and calculate kinetics and spectra and matrices for each one
            for theta in sampled_params:
                
                # needs to be log=false if you already undid the log space sample
                Etheta = SF_Rug.theta_to_E_matrix(theta, E)

                
                tempdata, temp = SF_Rug.calculate_SF_dynamics(Etheta, C0, theta[-len(c_lims[0]):], t, self.abs, Ep, var_c, log=False, verbose=False)
                
                sampled_matrix.append(tempdata)
                sampled_kinetics.append(temp[0])
                sampled_spectra.append(temp[1])

                # find mean and sdev
                mean_kinetics = np.mean(np.asarray(sampled_kinetics), axis=0)
                mean_spectra = np.mean(np.asarray(sampled_spectra), axis=0).T
                std_kinetics = np.std(np.asarray(sampled_kinetics), axis=0)
                std_spectra = np.std(np.asarray(sampled_spectra), axis=0).T
                fitted_SF_map = np.mean(sampled_matrix, axis=0)

                mean_lifetimes = ['%#.3g' % mean for mean in means[:len(means)-len(c_lims[0])]]
                sdev_lifetimes = ['%#.3g' % sdev for sdev in sdevs[:len(means)-len(c_lims[0])]]


            ## plot it all ##
            for j in range(ncpts):
                fig1.plot(t, mean_kinetics[:,j], color=cptcolours[j], label=f'{e_labels[j]}', lw=1)
                fig1.fill_between(t, mean_kinetics[:, j]-(stdfactor*std_kinetics[:, j]), mean_kinetics[:,j]+(stdfactor*std_kinetics[:, j]), color=cptcolours[j], alpha=0.5)

                for idx in range(self.bounds.shape[0]):
                    if idx == 0:
                        
                        fig2.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color=cptcolours[j], label=f'{e_labels[j]}', lw=1)
                        
                        fig2.fill_between(w[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] - (stdfactor*std_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), mean_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]+(stdfactor*std_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), color=cptcolours[j], alpha=0.5)

                    else:
                        
                        fig2.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color=cptcolours[j], label=f'{e_labels[j]}', lw=1)

                        fig2.fill_between(w[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] - (stdfactor*std_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), mean_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] + (stdfactor*std_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), color=cptcolours[j], alpha=0.5)



                
                h, l = fig2.get_legend_handles_labels() 
                ha = 'left'
                x_c = 0
                
                fig3.axis('off')
                
                fig3.text(x_c, 0.95, fr'File: {fig_title}'+r' H$_{2}$O$_{2}$'+f' +{HS_conc} equ. HS\n', ha=ha, va='center', fontsize=text_size)
                fig3.text(x_c, 0.76, fr'Fit to {modelname}'+'\n'+f'{self.delays.shape[0]} {metadata}ly-spaced time points over {int(self.delays[-1])}s', ha=ha, va='center', fontsize=text_size)

                fig3.text(x_c, 0.48, fr'Uncertainties shown are $\pm$ {stdfactor}$\sigma$'+'\n', ha=ha, va='center', fontsize=text_size)
                fig3.text(x_c, 0.35, 'Component lifetime means: \n'+fr' {", ".join(str(mean) for mean in mean_lifetimes)} s', ha=ha, va='center', fontsize=text_size)
                fig3.text(x_c, 0.18, r'Component lifetime $\sigma$:'+f'\n{", ".join(str(sdev) for sdev in sdev_lifetimes)} s', ha=ha, va='center', fontsize=text_size)
                
                niter = self.dres['niter']
                ncall = np.sum(self.dres['ncall'])
                
                summary = []
                summary.append(self.dres['eff']) 
                summary.append(self.dres['logz'][-1]) 
                summary.append(self.dres['logzerr'][-1]) 
                
                summary = ['%#.3g' % x for x in summary]
                
                fig3.text(x_c, -0.03, f'niter = {niter} eff(%) = {summary[0]}', ha=ha, va='center', fontsize=text_size)
                fig3.text(x_c, -0.13, f'ncall = {ncall} log(z) = {summary[1]}', ha=ha, va='center', fontsize=text_size)
                # fig3.text(x_c, -0.09, f'eff(%) = {summary[0]}', ha=ha, va='center', fontsize=text_size)
                # fig3.text(x_c, -0.19, f'log(z) = {summary[1]} '+r'$\pm$'+f' {summary[2]}', ha=ha, va='center', fontsize=text_size)


                fig1.set_xlim(self.delays[0], self.delays[ud_idx])
                fig1.set_xlabel('Time [s]', fontsize=text_size, labelpad=5)
                fig2.set_xlim(self.wavelengths[0], self.wavelengths[-1])
                fig2.set_xlabel('Wavelength [nm]', fontsize=10, labelpad=5)
                fig1.set_ylabel('Concentration [au]', fontsize=text_size, labelpad=20)
                fig2.set_ylabel('Absorption [OD]', fontsize=text_size, labelpad=20)
                fig2.grid(visible=True)



                vmin = np.min(self.abs)
                vmax = np.max(self.abs)
    
                trunc_fitted_SF = np.zeros_like(fitted_SF_map)
                trunc_raw_SF = np.zeros_like(self.abs)
    
                resid_map = fitted_SF_map - self.abs
                trunc_resid_map = np.zeros_like(resid_map)
    
                for idx in range(self.bounds.shape[0]):
                    
                    if idx == 0:
                        trunc_fitted_SF[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] = fitted_SF_map[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]
    
                        trunc_raw_SF[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]
                        
                        trunc_resid_map[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] = resid_map[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]

                    elif 0 < idx < self.bounds.shape[0]-1:
                            trunc_fitted_SF[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]
                
                            trunc_raw_SF[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]
                
                            trunc_resid_map[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]
                        
                    else:
                        trunc_fitted_SF[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]+1] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]+1]
                        
                        trunc_raw_SF[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]+1] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]+1]

                        trunc_resid_map[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]+1] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]+1]
    
    
                ax0 = fig.add_subplot(grid[0, 0])
                ax1 = fig.add_subplot(grid[1, 0])
                ax2 = fig.add_subplot(grid[2, 0])
    
                if hasattr(self, 'bounds'):
                    im0 = ax0.pcolormesh(self.wavelengths, t, trunc_fitted_SF, cmap=SF_cmap)
                    im1 = ax1.pcolormesh(self.wavelengths, t, trunc_raw_SF, cmap=SF_cmap)
                    
                    vmin_r = np.min(resid_map)/10
                    vmax_r = np.max(resid_map)/10
    
                    im2 = ax2.pcolormesh(self.wavelengths, t, trunc_resid_map, cmap=SF_cmap)
                    
                else:
                    im0 = ax0.pcolormesh(self.wavelengths, t, fitted_SF_map, cmap=SF_cmap)
                    im1 = ax1.pcolormesh(self.wavelengths, t, self.abs, cmap=SF_cmap)
                
                    vmin_r = np.min(resid_map)/10
                    vmax_r = np.max(resid_map)/10
                    
                    im2 = ax2.pcolormesh(self.wavelengths, t, resid_map, cmap=SF_cmap)
                
                ax0.set_title('Fitted Data', fontsize=20, pad=10)
                fig.colorbar(im0, ax=ax0, ticks=[np.ceil(vmin), 0, np.floor(vmax)])
                ax0.set_ylabel('Time [s]', fontsize=text_size, labelpad=20)
                
                
                ax1.set_title('Real Data', fontsize=20, pad=10)
                fig.colorbar(im1, ax=ax1, ticks=[np.ceil(vmin), 0, np.floor(vmax)])
                ax1.set_ylabel('Time [s]', fontsize=text_size, labelpad=10)
                
                ax2.set_title('Residual (fit-real)', fontsize=20, pad=10)
                fig.colorbar(im2, ax=ax2, ticks=[vmin_r, 0, vmax_r])
                ax2.set_ylabel('Time [s]', fontsize=text_size, labelpad=10)
                ax2.set_xlabel('Wavelength [nm]', fontsize=text_size, labelpad=10)

            if save_SF_plots:
                fig.savefig(f'{output_directory}\{savestring}_v_fitted_TA_plot.png', dpi=save_dpi)
                plt.close('all')


        


        if plot_SF and alignment=='horizontal':
            #if data is None:
            #    raise Error('Need to pass in a data Rug to use plot the actual TA fit.')
            if E is None:
                raise Error('need to pass in an encoding (E) matrix to define the kinetic model.')
            if C0 is None:
                raise Error('need to pass in initial concentration vector (C0) to fit the kinetic model.')

            
            grid = GridSpec(2, 3, hspace=0.5, wspace=0.3)
            fig = plt.figure(figsize=(figsize, figsize/2))
            
            means = np.mean(flat_samples, axis=0)
            sdevs = np.std(flat_samples, axis=0)
            
            rng = np.random.default_rng()
            sampled_params = rng.choice(flat_samples, nsamples, axis=0)
            
            if Ep:
                ncpts = E.shape[0] + len(Ep)
            else:
                ncpts = E.shape[0]

            sampled_matrix = []
            sampled_kinetics = []
            sampled_spectra = []

            fig1 = fig.add_subplot(grid[0, 0])
            fig2 = fig.add_subplot(grid[0, 1])
            fig3 = fig.add_subplot(grid[0, 2])

            fig1.set_title('Concentration Profiles', fontsize=title_fontsize, pad=20)
            fig2.set_title('Spectra', fontsize=title_fontsize, pad=20)
            fig3.set_title('Information', fontsize=title_fontsize, pad=20)

            #takes these from the original rug object (data)
            t = self.delays
            w = self.wavelengths

            #cycler for the component colours
            cptcolours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

            # loop over the randomly drawn samples and calculate kinetics and spectra and matrices for each one
            for theta in sampled_params:
                
                # needs to be log=false if you already undid the log space sample
                Etheta = SF_Rug.theta_to_E_matrix(theta, E)

                tempdata, temp = SF_Rug.calculate_SF_dynamics(Etheta, C0, theta[-len(c_lims[0]):], t, self.abs, Ep, var_c, log=False, verbose=False)
                
                sampled_matrix.append(tempdata)
                sampled_kinetics.append(temp[0])
                sampled_spectra.append(temp[1])

                # find mean and sdev
                mean_kinetics = np.mean(np.asarray(sampled_kinetics), axis=0)
                mean_spectra = np.mean(np.asarray(sampled_spectra), axis=0).T
                std_kinetics = np.std(np.asarray(sampled_kinetics), axis=0)
                std_spectra = np.std(np.asarray(sampled_spectra), axis=0).T
                fitted_SF_map = np.mean(sampled_matrix, axis=0)

                mean_lifetimes = ['%#.3g' % mean for mean in means[:len(means)-len(c_lims[0])]]
                sdev_lifetimes = ['%#.3g' % sdev for sdev in sdevs[:len(means)-len(c_lims[0])]]

            ## plot it all ##
            for j in range(ncpts):
                fig1.plot(t, mean_kinetics[:,j], color=cptcolours[j], label=f'{e_labels[j]}', lw=1)
                fig1.fill_between(t, mean_kinetics[:, j]-(stdfactor*std_kinetics[:, j]), mean_kinetics[:,j]+(stdfactor*std_kinetics[:, j]), color=cptcolours[j], alpha=0.5)

                for idx in range(self.bounds.shape[0]):
                    if idx == 0:
                        
                        fig2.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color=cptcolours[j], label=f'{e_labels[j]}', lw=1)
                        
                        fig2.fill_between(w[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] - (stdfactor*std_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), mean_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]+(stdfactor*std_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), color=cptcolours[j], alpha=0.5)

                    else:
                        
                        fig2.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color=cptcolours[j], label=f'{e_labels[j]}', lw=1)

                        fig2.fill_between(w[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] - (stdfactor*std_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), mean_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] + (stdfactor*std_spectra[:, j][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), color=cptcolours[j], alpha=0.5)
                        

            h, l = fig2.get_legend_handles_labels() 
            ha = 'left'
            x_c = 0
            
            fig3.axis('off')
            
            fig3.text(x_c, 0.95, fr'File: {self.fig_title}'+f' +{HS_conc} equ. HS\n', ha=ha, va='center', fontsize=text_size)
            fig3.text(x_c, 0.75, fr'Fit to {modelname}'+'\n'+f'{self.delays.shape[0]} {metadata}ly-spaced time points over {int(self.delays[-1])}s', ha=ha, va='center', fontsize=text_size)

            fig3.text(x_c, 0.55, fr'Uncertainties shown are $\pm$ {stdfactor}$\sigma$'+'\n', ha=ha, va='center', fontsize=text_size)
            fig3.text(x_c, 0.45, 'Component lifetime means: \n'+fr' {", ".join(str(mean) for mean in mean_lifetimes)} s', ha=ha, va='center', fontsize=text_size)
            fig3.text(x_c, 0.3, r'Component lifetime $\sigma$:'+f'\n{", ".join(str(sdev) for sdev in sdev_lifetimes)} s', ha=ha, va='center', fontsize=text_size)
            
            niter = self.dres['niter'] 
            ncall = np.sum(self.dres['ncall']) 
            
            summary = []
            summary.append(self.dres['eff']) 
            summary.append(self.dres['logz'][-1])
            summary.append(self.dres['logzerr'][-1])
            
            summary = ['%#.3g' % x for x in summary]
            
            fig3.text(x_c, 0.15, f'niter = {niter}', ha=ha, va='center', fontsize=text_size)
            fig3.text(x_c, 0.05, f'ncall = {ncall}', ha=ha, va='center', fontsize=text_size)
            fig3.text(x_c, -0.05, f'eff(%) = {summary[0]}', ha=ha, va='center', fontsize=text_size)
            fig3.text(x_c, -0.15, f'log(z) = {summary[1]} '+r'$\pm$'+f' {summary[2]}', ha=ha, va='center', fontsize=text_size)



            
            fig1.set_xlim(self.delays[0], self.delays[ud_idx])
            fig1.set_xlabel('Time (s)', fontsize=text_size, labelpad=20)
            fig2.set_xlim(self.wavelengths[0], self.wavelengths[-1])
            fig2.set_xlabel('Wavelength (nm)', fontsize=text_size, labelpad=20)
            fig1.set_ylabel('Concentration (a.u)', fontsize=text_size, labelpad=20)
            fig2.set_ylabel('Absorption (O.D)', fontsize=text_size, labelpad=20)
            fig2.grid(visible=True)


            vmin = np.min(self.abs)
            vmax = np.max(self.abs)

            trunc_fitted_SF = np.zeros_like(fitted_SF_map)
            trunc_raw_SF = np.zeros_like(self.abs)

            resid_map = fitted_SF_map - self.abs
            trunc_resid_map = np.zeros_like(resid_map)

            for idx in range(self.bounds.shape[0]):
                
                if idx == 0:
                    trunc_fitted_SF[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] = fitted_SF_map[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]

                    trunc_raw_SF[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]
                    
                    trunc_resid_map[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] = resid_map[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]

                elif 0 < idx < self.bounds.shape[0]-1:
                    trunc_fitted_SF[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]
                
                    trunc_raw_SF[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]
                
                    trunc_resid_map[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]
                        
                else:
                    trunc_fitted_SF[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]+1] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]+1]
                        
                    trunc_raw_SF[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]+1] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]+1]

                    trunc_resid_map[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]+1] = self.abs[:, SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]+1]


            ax0 = fig.add_subplot(grid[1, 0])
            ax1 = fig.add_subplot(grid[1, 1])
            ax2 = fig.add_subplot(grid[1, 2])

            if hasattr(self, 'bounds'):
                im0 = ax0.pcolormesh(self.wavelengths, t, trunc_fitted_SF, cmap=SF_cmap)
                im1 = ax1.pcolormesh(self.wavelengths, t, trunc_raw_SF, cmap=SF_cmap)
                
                vmin_r = np.min(resid_map)/10
                vmax_r = np.max(resid_map)/10

                im2 = ax2.pcolormesh(self.wavelengths, t, trunc_resid_map, cmap=SF_cmap)
                
            else:
                im0 = ax0.pcolormesh(self.wavelengths, t, fitted_SF_map, cmap=SF_cmap)
                im1 = ax1.pcolormesh(self.wavelengths, t, self.abs, cmap=SF_cmap)
                
                vmin_r = np.min(resid_map)/10
                vmax_r = np.max(resid_map)/10
                
                im2 = ax2.pcolormesh(self.wavelengths, t, resid_map, cmap=SF_cmap)
           

            
            ax0.set_title('Fitted Data', fontsize=label_fontsize, pad=20)
            fig.colorbar(im0, ax=ax0, ticks=[np.ceil(vmin), 0, np.floor(vmax)])
            ax0.set_ylabel('Time [s]', fontsize=text_size, labelpad=20)
            ax0.set_xlabel('Wavelength [nm]', fontsize=text_size, labelpad=20)
            
            ax1.set_title('Real Data', fontsize=label_fontsize, pad=20)
            fig.colorbar(im1, ax=ax1, ticks=[np.ceil(vmin), 0, np.floor(vmax)])
            ax1.set_ylabel('Time [s]', fontsize=text_size, labelpad=20)
            ax1.set_xlabel('Wavelength [nm]', fontsize=text_size, labelpad=20)

            ax2.set_title('Residual (fit-real)', fontsize=label_fontsize, pad=20)
            fig.colorbar(im2, ax=ax2, ticks=[vmin_r, 0, vmax_r])
            ax2.set_ylabel('Time [s]', fontsize=text_size, labelpad=20)
            ax2.set_xlabel('Wavelength [nm]', fontsize=text_size, labelpad=20)

        if save_SF_plots:
            ## make the figure full screen before saving to retain formatting ##
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle() 
            
            fig.savefig(f'{output_directory}\{savestring}_h_fitted_TA_plot.png', dpi=save_dpi)
            plt.close('all')



        
        if plot_SF_tileplot and alignment=='horizontal':
            
            rng = np.random.default_rng()
            sampled_params = rng.choice(flat_samples, nsamples, axis=0)
            
            if Ep:
                ncpts = E.shape[0] + len(Ep)
            else:
                ncpts = E.shape[0]

            sampled_matrix = []
            sampled_kinetics = []
            sampled_spectra = []


            t = self.delays
            w = self.wavelengths

            # cycler for the component colours
            cptcolours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

            # loop over the randomly drawn samples and calculate kinetics and spectra and matrices for each one
            for theta in sampled_params:
                
                # needs to be log=false if you already undid the log space sample
                Etheta = SF_Rug.theta_to_E_matrix(theta, E)

                tempdata, temp = SF_Rug.calculate_SF_dynamics(Etheta, C0, theta[-len(c_lims[0]):], t, self.abs, Ep, var_c, log=False, verbose=False)
                
                sampled_matrix.append(tempdata)
                sampled_kinetics.append(temp[0])
                sampled_spectra.append(temp[1])

                # find mean and sdev
                mean_kinetics = np.mean(np.asarray(sampled_kinetics), axis=0)
                mean_spectra = np.mean(np.asarray(sampled_spectra), axis=0).T
                std_kinetics = np.std(np.asarray(sampled_kinetics), axis=0)
                std_spectra = np.std(np.asarray(sampled_spectra), axis=0).T

            
            fig, axs = plt.subplots(2, ncpts, layout='tight')
            label_fontsize=20
            title_fontsize=20
        
            for i in range(ncpts):

                ax = axs[0, i]
                ax.plot(t, mean_kinetics[:,i], color=cptcolours[i])
                ax.fill_between(t, mean_kinetics[:, i]-(stdfactor*std_kinetics[:, i]), mean_kinetics[:,i]+(stdfactor*std_kinetics[:, i]), color=cptcolours[i], alpha=0.5)
                ax.set_xlim(self.delays[0], self.delays[ud_idx])
                ax.set_xlabel('Time (s)', fontsize=text_size)
                ax.set_title(f'{np.flip(e_labels)[i]} Concentration Profile', fontsize=title_fontsize, pad=10)
                
                if i == 0:
                    ax.set_ylabel('Amplitude (a.u)', fontsize=label_fontsize, labelpad=20)
                    
                                   
                ax = axs[1, i]
                
                for idx in range(self.bounds.shape[0]):
                    if idx == 0:
                        
                        ax.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color=cptcolours[i])
                        
                        ax.fill_between(w[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] - (stdfactor*std_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), mean_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]+(stdfactor*std_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), color=cptcolours[i], alpha=0.5)

                    else:
                        
                        ax.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color=cptcolours[i])

                        ax.fill_between(w[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] - (stdfactor*std_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), mean_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] + (stdfactor*std_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), color=cptcolours[i], alpha=0.5)
                        
                
                ax.set_xlim(w[0], w[-1])
                ax.set_xlabel('Wavelength (nm)', fontsize=text_size, labelpad=10)
                ax.set_title(f'{np.flip(e_labels)[i]} Spectrum', fontsize=title_fontsize, pad=10)
                ax.grid(visible=True)
                
                if i == 0:
                    ax.set_ylabel('Amplitude (a.u)', fontsize=label_fontsize, labelpad=20)

            if save_SF_tileplot:
                ## make the figure full screen before saving to retain formatting ##
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle() 
                
                fig.savefig(f'{output_directory}\{savestring}_h_tile_plot.png', dpi=save_dpi)
                plt.close('all')
                    

                
        if plot_SF_tileplot and alignment=='vertical':
            
            rng = np.random.default_rng()
            sampled_params = rng.choice(flat_samples, nsamples, axis=0)

            if Ep:
                ncpts = E.shape[0] + len(Ep)
            else:
                ncpts = E.shape[0]
            
            sampled_matrix = []
            sampled_kinetics = []
            sampled_spectra = []


            t = self.delays
            w = self.wavelengths

            # cycler for the component colours
            cptcolours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

            # loop over the randomly drawn samples and calculate kinetics and spectra and matrices for each one
            for theta in sampled_params:
                
                # needs to be log=false if you already undid the log space sample
                Etheta = SF_Rug.theta_to_E_matrix(theta, E)

                tempdata, temp = SF_Rug.calculate_SF_dynamics(Etheta, C0, theta[-len(c_lims[0]):], t, self.abs, Ep, var_c, log=False, verbose=False)
                
                sampled_matrix.append(tempdata)
                sampled_kinetics.append(temp[0])
                sampled_spectra.append(temp[1])

                # find mean and sdev
                mean_kinetics = np.mean(np.asarray(sampled_kinetics), axis=0)
                mean_spectra = np.mean(np.asarray(sampled_spectra), axis=0).T
                std_kinetics = np.std(np.asarray(sampled_kinetics), axis=0)
                std_spectra = np.std(np.asarray(sampled_spectra), axis=0).T

            
            fig, axs = plt.subplots(ncpts, 2, figsize=(10, 15), layout='tight')

            label_fontsize=20
            title_fontsize=20

            
            for i in range(ncpts):
                
                ax = axs[i, 0]                
                ax.plot(t, mean_kinetics[:,i], color=cptcolours[i])
                ax.fill_between(t, mean_kinetics[:, i]-(stdfactor*std_kinetics[:, i]), mean_kinetics[:,i]+(stdfactor*std_kinetics[:, i]), color=cptcolours[i], alpha=0.5)
                ax.tick_params(axis='both', which='both', labelsize=text_size)
                ax.set_xlim(self.delays[0], self.delays[ud_idx])
                ax.set_ylabel('Amplitude (a.u)', fontsize=label_fontsize, labelpad=20)
                ax.set_title(f'{np.flip(e_labels)[i]} Concentration Profile', fontsize=title_fontsize, pad=10)
                ax.grid(visible=True)
                
                if i < ncpts-1:
                    ax.tick_params(axis='x',
                                   which='both',
                                   bottom=False,
                                   labelbottom=False)
                                    
                if i == ncpts-1:
                    ax.set_xlabel('Time (s)', fontsize=label_fontsize, labelpad=20)
                    
                                   
                
                ax = axs[i, 1]    
                for idx in range(self.bounds.shape[0]):
                    if idx == 0:
                        
                        ax.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color=cptcolours[i])
                        
                        ax.fill_between(w[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] - (stdfactor*std_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), mean_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]+(stdfactor*std_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), color=cptcolours[i], alpha=0.5)

                    else:
                        
                        ax.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color=cptcolours[i])

                        ax.fill_between(w[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], mean_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] - (stdfactor*std_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), mean_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] + (stdfactor*std_spectra[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]]), color=cptcolours[i], alpha=0.5)
                        

                ax.tick_params(axis='both', which='both', labelsize=text_size)
                ax.set_xlim(w[0], w[-1])
                ax.set_title(f'{np.flip(e_labels)[i]} Spectrum', fontsize=title_fontsize, pad=10)
                ax.grid(visible=True)
                
                if i < ncpts-1:
                    ax.tick_params(axis='x',
                                   which='both',
                                   bottom=False,
                                   labelbottom=False)
                    
                if i == ncpts-1:
                    ax.set_xlabel('Wavelength (nm)', fontsize=label_fontsize, labelpad=20)

            if save_SF_tileplot:
                fig.savefig(f'{output_directory}\{savestring}_v_tile_plot.png', dpi=save_dpi)
                plt.close('all')

        return



    def explore_DNS_spectra(self,
                            e_labels=None,
                            Sample_concentration=None,
                            sl_min=0,
                            sl_max=-1,
                            tick_size=25,
                            axis_fontsize=30,
                            title_fontsize=25,
                            loc='upper left'):
        
        """
        Interactive widget to inspect the fit to the series of spectra
        """
        
        wavelengths = self.wavelengths
        times = self.delays
        
        X, Y = np.meshgrid(wavelengths, times, indexing="ij")
        Z = self.abs
        
        Y_index = []
        Y_values = []
        
        for idx, val in enumerate (Y[0]):
            Y_index.append(idx)
            Y_values.append(val)
        
        Y_series = pd.Series(Y_index, index=Y_values)
        
        
        sl_min = sl_min 
        sl_max = sl_max
        
        tick_size = tick_size
        axis_fontsize = axis_fontsize
        title_fontsize = title_fontsize
        
        
        Cpt_arr = np.zeros(shape=(self.DNS_SP.shape[0], len(self.DNS_SP[0])))
        
        cp_max = []
        
        for i in range(Cpt_arr.shape[0]):
            Cpt_arr[i] = self.DNS_SP[i] * self.DNS_CP.T[i, sl_min]
            
            cp_max.append(np.max(self.DNS_CP.T[i]))
            
        cp_max = np.cumsum(cp_max, axis=0)[-1]

        
        ## integrate over the area of each component and store these values in an array ##
        int_max = np.zeros(shape=(Cpt_arr.shape[0], self.delays.shape[0]))
                                  
        for idx in range(Cpt_arr.shape[0]):
            for jdx, wal in enumerate(self.delays):
                
                comp = self.DNS_SP.T[:, idx] * self.DNS_CP.T[idx, jdx]
                int_max[idx, jdx] = np.trapz(comp, self.wavelengths)
            

        
        if e_labels == None:
            e_labels = [r'e$_{{'+str(i+1)+'}}$' for i in range(Cpt_arr.shape[0])]
        
        _cmap = plt.get_cmap('inferno')
        _colors = [_cmap(i) for i in np.linspace(0, 1, Cpt_arr.shape[0]*100)]  
        
        spectrum_sum = np.cumsum(Cpt_arr, axis=0)
        
        reconstructed_spectra_init = spectrum_sum[-1] 
        
        
        fig = plt.figure(layout="constrained")
        grid = GridSpec(3, 1, figure=fig, wspace=0.2, hspace=0.15)
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)
        
        ax1 = fig.add_subplot(grid[1:, :])
         
        # ax1.plot(wavelengths, Z[sl_min], '.C0', label='Raw Spectrum at '+str('%#.3g' % times[sl_min])+' s')
        # ax1.plot(wavelengths, reconstructed_spectra_init, color='darkslateblue', linewidth=2, label='reconstructed spectrum')

        for idx in range(self.bounds.shape[0]):
                
            if idx == 0:
                ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], Z[sl_min][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], '.C0', label='Raw Spectrum at '+str('%#.3g' % times[sl_min])+' s')
                
                ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], reconstructed_spectra_init[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color='darkslateblue', linewidth=2, label='reconstructed spectrum\n')

            else:
                ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], Z[sl_min][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], '.C0')
                
                ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], reconstructed_spectra_init[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color='darkslateblue', linewidth=2)
        
        
        for i in range(Cpt_arr.shape[0]):
            # proportion of each compartment relative to the cumulative sum of the maximum amplitude of each compartment (over all values of t) at time, t:
            # ratio_1 = self.DNS_CP[sl_min, i] / cp_max 
            
            # ratio of the amplitude of compartment, i, at time t, to its maximum amplitude:
            # ratio_1 = self.DNS_CP[sl_min, i] / np.max(self.DNS_CP.T[i])
            
            # ratio of the integral of compartment i over self.wavelengths.shape[0], at time, t, to the maximum integral of compartment i:
            comp = self.DNS_SP.T[:, i] * self.DNS_CP.T[i, sl_min]
            ratio_1 = np.trapz(comp, self.wavelengths) / np.max(int_max[i])

             # proportion of the amplitude of each compartment, relative to the cumulative sum of the amplitudes of each compartment, at time, t:
            # ratio_2 = self.DNS_CP[sl_min, i] / np.cumsum(self.DNS_CP[sl_min], axis=0)[-1]

            # ratio of the integral of the contribution of compartment i at time, t over self.wavelengths.shape[0] to the spectrum at time, t
            ratio_2 = np.trapz(comp, self.wavelengths) / np.trapz(reconstructed_spectra_init, self.wavelengths)
            
            
            # ax1.plot(wavelengths, self.DNS_SP.T[:, i] * self.DNS_CP.T[i, sl_min], color=_colors[i*100], ls='--', label=r'['+str(np.flip(e_labels)[i])+']$_{t}$ / $\sum$[e$_{i}$, e$_{j}$]$_{0}}$'+f': {np.round(ratio_1*100, 2)}%\n'+r'['+str(np.flip(e_labels)[i])+']$_{t}$ / $\sum$[e$_{i}$, e$_{j}$]$_{t}}$'+f': {np.round(ratio_2*100, 2)}%\n')

            
            for idx in range(self.bounds.shape[0]):
                
                if idx == 0:
                    ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], self.DNS_SP.T[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] * self.DNS_CP.T[i, sl_min], color=_colors[i*100], ls='--', 
                             label=r'['+str(np.flip(e_labels)[i])+']$_{t}$ / max('+str(np.flip(e_labels)[i])+')'+f': {np.round(ratio_1*100, 2)}%\n'+r'['+str(np.flip(e_labels)[i])+']$_{t}$ / $\sum$[e$_{i}$, e$_{j}$]$_{t}}$'+f': {np.round(ratio_2*100, 2)}%\n'+r'['+str(np.flip(e_labels)[i])+']'+f'={np.round(ratio_2*Sample_concentration/1e-6, 2)}'+r'$\mu$M'+'\n')
                    

                else:
                    ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], self.DNS_SP.T[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] * self.DNS_CP.T[i, sl_min], color=_colors[i*100], ls='--')

        
        ax1.tick_params(axis="both", labelsize = tick_size)
        ax1.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize)
        ax1.set_ylabel('Absorption (OD)', fontsize = axis_fontsize) 
        ax1.set_xlim(wavelengths[0], wavelengths[-1])
        ax1.legend(fontsize=15, loc=loc)
        ax1.grid(visible=True)
        
        ax2 = fig.add_subplot(grid[0, :])
        
        ax2.plot(wavelengths, Z[sl_min] - reconstructed_spectra_init, '.C3')
        
        ax2.tick_params(axis="both", labelsize=tick_size)
        ax2.set_xlim(wavelengths[0], wavelengths[-1])
        ax2.set_title('Residual at '+str('%#.3g' % times[sl_min])+'s', fontsize = title_fontsize, y=1.02)
        ax2.grid(visible=True)
        
        # Sliders
        global time_slider
        
        axwave = plt.axes([0.15, 0.635, 0.79, 0.015]) # ([0.1, 0.03, 0.8, 0.03])
        time_slider = Slider(
            ax=axwave,
            label='Time (s)',
            valmin=times[sl_min],
            valmax=times[sl_max], 
            valinit=times[sl_min],
            valstep=times,
            color='lightsteelblue',
            handle_style={'facecolor': 'white', 'edgecolor': '.05', 'size': 20}
        )
        time_slider.label.set_size(20)
        
        
        
        def update(val):
            raw_ydata = Z[Y_series[time_slider.val]]
            Cpt_arr = np.zeros(shape=(self.DNS_SP.shape[0], len(self.DNS_SP[0])))
        
            for i in range(Cpt_arr.shape[0]):
                Cpt_arr[i] = self.DNS_SP[i] * self.DNS_CP.T[i, Y_series[time_slider.val]]
            
            spectrum_sum = np.cumsum(Cpt_arr, axis=0)
            
            rec_ydata = spectrum_sum[-1] 
            
            ax1.cla()
            #ax1.plot(wavelengths, raw_ydata, '.C0', label='Raw Spectrum at '+str('%#.3g' % time_slider.val)+' s')
            #ax1.plot(wavelengths, rec_ydata, color='darkslateblue', linewidth=2, label='reconstructed spectrum')

            for idx in range(self.bounds.shape[0]):
                
                if idx == 0:
                    ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], raw_ydata[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], '.C0', label='Raw Spectrum at '+str('%#.3g' % time_slider.val)+' s')
                    
                    ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], rec_ydata[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color='darkslateblue', linewidth=2, label='reconstructed spectrum\n')
    
                else:
                    ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], raw_ydata[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], '.C0')
                    
                    ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], rec_ydata[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color='darkslateblue', linewidth=2)
                
            
            for i in range(Cpt_arr.shape[0]):
                # proportion of each compartment relative to the cumulative sum of the maximum amplitude of each compartment (over all values of t) at time, t:
                # ratio_1 = self.DNS_CP[Y_series[time_slider.val], i] / cp_max # np.max(self.DNS_CP.T[i])
                # ratio_1 = self.DNS_CP[Y_series[time_slider.val], i] / np.max(self.DNS_CP.T[i])

                # ratio of the integral of compartment i over self.wavelengths.shape[0], at time, t, to the maximum integral of compartment i:
                comp = self.DNS_SP.T[:, i] * self.DNS_CP.T[i, Y_series[time_slider.val]]
                ratio_1 = np.trapz(comp, self.wavelengths) / np.max(int_max[i])
                
                # proportion of each compartment in the spectrum, relative to the cumulative sum of the amplitudes of each compartment at time, t:
                # ratio_2 = self.DNS_CP[Y_series[time_slider.val], i] / np.cumsum(self.DNS_CP[Y_series[time_slider.val]], axis=0)[-1]

                # ratio of the integral of the contribution of compartment i at time, t over self.wavelengths.shape[0] to the spectrum at time, t
                ratio_2 = np.trapz(comp, self.wavelengths) / np.trapz(rec_ydata, self.wavelengths)
                
                # ax1.plot(wavelengths, self.DNS_SP.T[:, i] * self.DNS_CP.T[i, Y_series[time_slider.val]], color=_colors[i*100], ls='--', label=r'['+str(np.flip(e_labels)[i])+']$_{t}$ / $\sum$[e$_{i}$, e$_{j}$]$_{0}}$'+f': {np.round(ratio_1*100, 2)}%\n'+r'['+str(np.flip(e_labels)[i])+']$_{t}$ / $\sum$[e$_{i}$, e$_{j}$]$_{t}}$'+f': {np.round(ratio_2*100, 2)}%\n')
                
                for idx in range(self.bounds.shape[0]):
                
                    if idx == 0:
                        ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], self.DNS_SP.T[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] * self.DNS_CP.T[i, Y_series[time_slider.val]], color=_colors[i*100], ls='--', label=r'['+str(np.flip(e_labels)[i])+']$_{t}$ / max('+str(np.flip(e_labels)[i])+')'+f': {np.round(ratio_1*100, 2)}%\n'+r'['+str(np.flip(e_labels)[i])+']$_{t}$ / $\sum$[e$_{i}$, e$_{j}$]$_{t}}$'+f': {np.round(ratio_2*100, 2)}%\n'+r'['+str(np.flip(e_labels)[i])+']'+f'={np.round(ratio_2*Sample_concentration/1e-6, 2)}'+r'$\mu$M'+'\n')
        
                    else:
                        ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], self.DNS_SP.T[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] * self.DNS_CP.T[i, Y_series[time_slider.val]], color=_colors[i*100], ls='--')
                
        
            ax1.tick_params(axis="both", labelsize = tick_size)
            ax1.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize)
            ax1.set_ylabel('Absorption (OD)', fontsize = axis_fontsize) 
            ax1.set_xlim(wavelengths[0], wavelengths[-1])
            ax1.legend(fontsize=15, loc=loc)
            ax1.grid(visible=True)
        
            ax2.cla()
            ax2.plot(wavelengths, raw_ydata - rec_ydata, '.C3')
        
            ax2.tick_params(axis="both", labelsize = tick_size)
            ax2.set_xlim(wavelengths[0], wavelengths[-1])
            ax2.set_title('Residual at '+str('%#.3g' % time_slider.val)+'s', fontsize = title_fontsize, y=1.02)
            ax2.grid(visible=True)
            
            
            fig.canvas.draw_idle()
            """
            try:
                ax1.set_ylim((np.nanmin(raw_ydata) - (np.abs(np.nanmin(raw_ydata)))), (np.nanmax(raw_ydata) + (np.abs(np.nanmax(raw_ydata)))))
                ax_1.set_ylim((np.nanmin(res_ydata) - (np.abs(np.nanmin(res_ydata)))), (np.nanmax(res_ydata) + (np.abs(np.nanmax(res_ydata)))))
        
            except:
                pass
            """
        time_slider.on_changed(update)
        
        
        # Buttons 
        resetax = fig.add_axes([0.01, 0.66, 0.05, 0.03])  # 0.15
        add_clickax = fig.add_axes([0.01, 0.62, 0.05, 0.03])
        
        reset_button = Button(resetax, 'Reset', color='lightsteelblue', hovercolor='gainsboro')
        reset_button.label.set_fontsize(20)
        
        add_click_button = Button(add_clickax, 'Add', color='lightsteelblue', hovercolor='gainsboro')
        add_click_button.label.set_fontsize(20)
        
        mem_values = []
        
        
        
        def add_click(event):
            # stores the current time value selected with the slider in a list
            mem_values.append(time_slider.val)
            """
            cmap_ = plt.get_cmap('viridis')
            colors_ = [cmap_(i) for i in np.linspace(0, 1, len(mem_values))]
            
            fig_Cpt_arr = np.zeros(shape=(self.DNS_SP.shape[0], len(self.DNS_SP[0])))
        
            fig_1 = plt.figure(layout="constrained")
            grid_1 = GridSpec(3, 1, figure=fig_1, wspace=0.2, hspace=0.15)
        
            ax3 = fig_1.add_subplot(grid_1[1:, :])
            ax4 = fig_1.add_subplot(grid_1[0, :])
            
            for i, color in enumerate(colors_, start=0):
                
                for idx in range(fig_Cpt_arr.shape[0]):
                    fig_Cpt_arr[idx] = self.DNS_SP[idx] * self.DNS_CP.T[idx, Y_series[mem_values[i]]]
                
                fig_spectrum_sum = np.cumsum(fig_Cpt_arr, axis=0)
                fig_rec_ydata = fig_spectrum_sum[-1] 
                
        
                #ax3.plot(wavelengths, Z[Y_series[mem_values[i]]], f'.C{i}', label='Raw Difference Spectrum at '+str('%#.3g' % mem_values[i])+'s')
                #ax3.plot(wavelengths, fig_rec_ydata, color=color)

                for idx in range(self.bounds.shape[0]):
                
                    if idx == 0:
                        ax3.scatter(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], Z[Y_series[mem_values[i]]][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color=color, label=str('%#.3g' % mem_values[i])+'s')
                        
                        ax3.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], fig_rec_ydata[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color=color)
        
                    else:
                        ax3.scatter(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], Z[Y_series[mem_values[i]]][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color=color)
                        
                        ax3.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], fig_rec_ydata[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color=color)
                
                ax3.tick_params(axis="both", labelsize=tick_size)
                ax3.set_xlabel('Wavelength (nm)', fontsize=axis_fontsize)
                ax3.set_ylabel('Absorption (O.D)', fontsize=axis_fontsize)
                ax3.set_xlim(wavelengths[0], wavelengths[-1])
                title = self.filename.replace('_', ' ')
                ax3.set_title(f'{title} Difference Spectra', fontsize=title_fontsize, pad=20)
                ax3.legend(fontsize=tick_size)       
                ax3.grid(visible=True)
        
                    
                ax4.plot(wavelengths, Z[Y_series[mem_values[i]]] - fig_rec_ydata, f'.C{i}')
                
                ax4.tick_params(axis="both", labelsize = tick_size)
                #ax2.set_ylabel('Absorption (O.D)', fontsize = axis_fontsize)
                ax4.set_xlim(wavelengths[0], wavelengths[-1])
                ax4.set_title('Residual', fontsize = title_fontsize, y=1.02)
                ax4.grid(visible=True)
                
                fig_1.canvas.draw_idle()
            """
            return
            
        
        def reset(event):
            
            mem_values.clear()
            time_slider.reset()
            
            ax1.cla()
            ax2.cla()
            
            for idx in range(self.bounds.shape[0]):
                
                if idx == 0:
                    ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], Z[sl_min][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], '.C0', label='Raw Spectrum at '+str('%#.3g' % times[sl_min])+' s')
                    
                    ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], reconstructed_spectra_init[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color='darkslateblue', linewidth=2, label='reconstructed spectrum\n')
    
                else:
                    ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], Z[sl_min][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], '.C0')
                    
                    ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], reconstructed_spectra_init[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], color='darkslateblue', linewidth=2)
        
        
            for i in range(Cpt_arr.shape[0]):
                # proportion of each compartment relative to the cumulative sum of the maximum amplitude of each compartment (over all values of t) at time, t:
                # ratio_1 = self.DNS_CP[sl_min, i] / cp_max 
                
                # ratio of the amplitude of compartment, i, at time t, to its maximum amplitude:
                # ratio_1 = self.DNS_CP[sl_min, i] / np.max(self.DNS_CP.T[i])
                
                # ratio of the integral of compartment i over self.wavelengths.shape[0], at time, t, to the maximum integral of compartment i:
                comp = self.DNS_SP.T[:, i] * self.DNS_CP.T[i, sl_min]
                ratio_1 = np.trapz(comp, self.wavelengths) / np.max(int_max[i])
    
                 # proportion of the amplitude of each compartment, relative to the cumulative sum of the amplitudes of each compartment, at time, t:
                # ratio_2 = self.DNS_CP[sl_min, i] / np.cumsum(self.DNS_CP[sl_min], axis=0)[-1]
    
                # ratio of the integral of the contribution of compartment i at time, t over self.wavelengths.shape[0] to the spectrum at time, t
                ratio_2 = np.trapz(comp, self.wavelengths) / np.trapz(reconstructed_spectra_init, self.wavelengths)
                
                
                # ax1.plot(wavelengths, self.DNS_SP.T[:, i] * self.DNS_CP.T[i, sl_min], color=_colors[i*100], ls='--', label=r'['+str(np.flip(e_labels)[i])+']$_{t}$ / $\sum$[e$_{i}$, e$_{j}$]$_{0}}$'+f': {np.round(ratio_1*100, 2)}%\n'+r'['+str(np.flip(e_labels)[i])+']$_{t}$ / $\sum$[e$_{i}$, e$_{j}$]$_{t}}$'+f': {np.round(ratio_2*100, 2)}%\n')
    
                
                for idx in range(self.bounds.shape[0]):
                    
                    if idx == 0:
                        ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], self.DNS_SP.T[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] * self.DNS_CP.T[i, sl_min], color=_colors[i*100], ls='--', 
                                 label=r'['+str(np.flip(e_labels)[i])+']$_{t}$ / max('+str(np.flip(e_labels)[i])+')'+f': {np.round(ratio_1*100, 2)}%\n'+r'['+str(np.flip(e_labels)[i])+']$_{t}$ / $\sum$[e$_{i}$, e$_{j}$]$_{t}}$'+f': {np.round(ratio_2*100, 2)}%\n'+r'['+str(np.flip(e_labels)[i])+']'+f'={np.round(ratio_2*Sample_concentration/1e-6, 2)}'+r'$\mu$M'+'\n')
                        
    
                    else:
                        ax1.plot(self.wavelengths[SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]], self.DNS_SP.T[:, i][SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 0])[0]+1:SF_Rug.find_nearest(self.wavelengths, self.bounds[idx, 1])[0]] * self.DNS_CP.T[i, sl_min], color=_colors[i*100], ls='--')
                    
                        
                        
            ax1.tick_params(axis="both", labelsize=tick_size)
            ax1.set_xlabel('Wavelength (nm)', fontsize=axis_fontsize)
            ax1.set_ylabel('Absorption (OD)', fontsize=axis_fontsize) 
            ax1.set_xlim(wavelengths[0], wavelengths[-1])
            ax1.legend(fontsize=15, loc=loc)
            ax1.grid(visible=True)
            
            # ax2 = fig.add_subplot(grid[0, :])
            
            ax2.plot(wavelengths, Z[sl_min] - reconstructed_spectra_init, '.C3')
            
            ax2.tick_params(axis="both", labelsize=tick_size)
            ax2.set_xlim(wavelengths[0], wavelengths[-1])
            ax2.set_title('Residual at '+str('%#.3g' % times[sl_min])+'s', fontsize=title_fontsize, y=1.02)
            ax2.grid(visible=True)
            
            """
            ax1.tick_params(axis="both", labelsize = tick_size)
            ax1.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize)
            ax1.set_ylabel('Absorption (OD)', fontsize = axis_fontsize) 
            ax1.set_xlim(wavelengths[0], wavelengths[-1])
            ax1.set_ylim(-0.01, 1.1)
            ax1.legend(fontsize=20)
            ax1.grid(visible=True)

            
            ax2.plot(wavelengths, Z[sl_min] - reconstructed_spectra_init, '.C3')
            
            ax2.tick_params(axis="both", labelsize = tick_size)
            ax2.set_xlim(wavelengths[0], wavelengths[-1])
            ax2.set_title('Residual at '+str('%#.3g' % total_time[sl_min])+'s', fontsize = title_fontsize, y=1.02)
            ax2.grid(visible=True)
            """
            
            fig.canvas.draw_idle()
        
        
        reset_button.on_clicked(reset)
        resetax._button = reset_button
        
        add_click_button.on_clicked(add_click)
        add_clickax._button = add_click_button
        
        plt.show()
        
        return


    
    def explore_DNS_traces(self,
                           e_labels=None,
                           sl_min=0,
                           sl_max=-1,
                           lx_lim=-1, 
                           ux_lim=120,
                           tick_size=25,
                           axis_fontsize=30,
                           title_fontsize=25):
        """
        Interactive widget to inspect the fit of a given kinetic model evaluated using DNS to the wavelength dependent traces in the original data
        """
        
        wavelengths = self.wavelengths
        total_time = self.delays
        
        X, Y = np.meshgrid(wavelengths, total_time, indexing="ij")
        Z = self.abs
        
        X_index = []
        X_values = []
        
        for idx, val in enumerate(X.T[0]):
            X_index.append(idx)
            X_values.append(val)
        
        X_series = pd.Series(X_index, index=X_values)
        
        
        sl_min = sl_min
        sl_max = sl_max
        
        lx_lim = lx_lim
        ux_lim = ux_lim
        
        tick_size = tick_size
        axis_fontsize = axis_fontsize
        title_fontsize = title_fontsize
        
        
        cpt_arr = np.zeros(shape=(self.DNS_CP.T.shape[0], self.DNS_CP.T.shape[1]))
        
        for i, v in enumerate(cpt_arr):
            cpt_arr[i] = self.DNS_SP[i, sl_min] * self.DNS_CP.T[i]
        
        reconstructed_trace_init = np.cumsum(cpt_arr, axis=0)[-1] 
        
        ## compute the sum of the absolute magnitude of each component to determine the contribution from each component from the ratio of the integral ##
        int_ydata = np.zeros_like(cpt_arr)
        
        for i, v in enumerate(int_ydata):
            int_ydata[i] = np.abs(self.DNS_SP[i, sl_min] * self.DNS_CP.T[i])
            
        int_ydata = np.cumsum(int_ydata, axis=0)[-1]

        if e_labels == None:
            e_labels = [r'e$_{{'+str(i+1)+'}}$' for i in range(cpt_arr.shape[0])]
        
        _cmap = plt.get_cmap('inferno')
        _colors = [_cmap(i) for i in np.linspace(0, 1, cpt_arr.shape[0]*100)] 
        
        fig = plt.figure(layout="constrained")
        grid = GridSpec(3, 1, figure=fig, wspace=0.2, hspace=0.15)
        
        ax1 = fig.add_subplot(grid[1:, :])
         
        ax1.plot(total_time, Z.T[sl_min], '.C0', label='Raw Trace at '+str('%#.3g' % wavelengths[sl_min])+' nm')
        ax1.plot(total_time, reconstructed_trace_init, color='darkslateblue', linewidth=2, label='Fitted trace')
        
        for i in range(cpt_arr.shape[0]):
            ratio = np.trapz(np.abs(cpt_arr[i]), total_time) / np.trapz(int_ydata, total_time)
            
            ax1.plot(total_time, cpt_arr[i], ls='--', color=_colors[i*100], label=r'$\int_{t_{min}}^{t_{max}}$['+str(np.flip(e_labels)[i])+']$_{\lambda}$ / $\int_{t_{min}}^{t_{max}}$[e$_{i}$, e$_{j}$]$_{\lambda}$'+f': {np.round(ratio*100, 2)}%')
            
                     # label = r'$\dfrac{\int_{t_{min}}^{t_{max}}['+str(np.flip(e_labels)[i])+']_{\lambda}}{\int_{t_{min}}^{t_{max}}[e_{i}, e_{j}]_{\lambda}}$'+f': {np.round(ratio*100, 2)}%')
        
# ^ should be able to use something more like this with a proper numerator and denominator 
        
        ax1.tick_params(axis="both", labelsize = tick_size)
        ax1.set_xlabel('Delay (ps)', fontsize = axis_fontsize)
        ax1.set_ylabel(r'$\Delta$O.D', fontsize = axis_fontsize) 
        ax1.set_xlim(lx_lim, total_time[ux_lim])
        # ax1.set_ylim(-0.2, 1.1) # (-0.01, 0.2)
        
        ax1.legend(fontsize=20)
        ax1.grid(visible=True)
        
        ax2 = fig.add_subplot(grid[0, :])
        
        ax2.plot(total_time, Z.T[sl_min] - reconstructed_trace_init, '.C3')
        
        ax2.tick_params(axis="both", labelsize=tick_size)
        ax2.set_xlim(lx_lim, total_time[ux_lim])
        ax2.set_title('Residual at '+str('%#.3g' % wavelengths[sl_min])+' nm', fontsize = title_fontsize, y=1.02)
        ax2.grid(visible=True)
        
        # Sliders
        global wavelength_slider

                
        if hasattr(self, 'wlstep'):
                valstep=self.wlstep
        else:
            valstep=self.wavelengths
        
        axwave = plt.axes([0.2, 0.635, 0.69, 0.015]) # ([0.1, 0.03, 0.8, 0.03])
        wavelength_slider = Slider(
            ax=axwave,
            label='Wavelength (nm)',
            valmin=wavelengths[sl_min],
            valmax=wavelengths[sl_max], 
            valinit=wavelengths[sl_min],
            valstep=valstep,
            color='lightsteelblue',
            handle_style={'facecolor': 'white', 'edgecolor': '.05', 'size': 20}
        )
        wavelength_slider.label.set_size(20)
        
        
        def update(val):
            
            raw_ydata = Z.T[X_series[wavelength_slider.val]]
            
            cpt_arr = np.zeros(shape=(self.DNS_CP.T.shape[0], self.DNS_CP.T.shape[1]))
            
            for i, v in enumerate(cpt_arr):
                cpt_arr[i] = self.DNS_SP[i, X_series[wavelength_slider.val]] * self.DNS_CP.T[i]
        
            rec_ydata = np.cumsum(cpt_arr, axis=0)[-1]
        
            ## compute the sum of the absolute magnitude of each component to determine the contribution from each component from the ratio of the integral ##
            int_ydata = np.zeros_like(cpt_arr)
            
            for i, v in enumerate(int_ydata):
                int_ydata[i] = np.abs(self.DNS_SP[i, X_series[wavelength_slider.val]] * self.DNS_CP.T[i])
                
            int_ydata = np.cumsum(int_ydata, axis=0)[-1]
        
            ax1.cla()
            ax1.plot(total_time, raw_ydata, '.C0', label='Raw Trace at '+str('%#.3g' % wavelength_slider.val)+' nm')
            ax1.plot(total_time, rec_ydata, color='darkslateblue', linewidth=2, label='fitted trace')
        
            for i in range(cpt_arr.shape[0]):
                ratio = np.trapz(np.abs(cpt_arr[i]), total_time) / np.trapz(int_ydata, total_time)
                ax1.plot(total_time, cpt_arr[i], ls='--', color=_colors[i*100], label=r'$\int_{t_{min}}^{t_{max}}$['+str(np.flip(e_labels)[i])+']$_{\lambda}$ / $\int_{t_{min}}^{t_{max}}$[e$_{i}$, e$_{j}$]$_{\lambda}$'+f': {np.round(ratio*100, 2)}%')
        
            
            ax1.tick_params(axis="both", labelsize = tick_size)
            ax1.set_xlabel('Time (s)', fontsize = axis_fontsize)
            ax1.set_ylabel('Absorption (OD)', fontsize = axis_fontsize) 
            ax1.set_xlim(lx_lim, total_time[ux_lim])
            # ax1.set_ylim(-0.2, 1.1) # (-0.01, 0.2)
        
            ax1.legend(fontsize=20)
            ax1.grid(visible=True)
            
            ax2.cla()
            ax2.plot(total_time, raw_ydata - rec_ydata, '.C3')
        
            ax2.tick_params(axis="both", labelsize = tick_size)
            ax2.set_xlim(lx_lim, total_time[ux_lim])
            ax2.set_title('Residual at '+str('%#.3g' % wavelength_slider.val)+' nm', fontsize = title_fontsize, y=1.02)
            ax2.grid(visible=True)
            
            
            fig.canvas.draw_idle()
            """
            try:
                ax1.set_ylim((np.nanmin(raw_ydata) - (np.abs(np.nanmin(raw_ydata)))), (np.nanmax(raw_ydata) + (np.abs(np.nanmax(raw_ydata)))))
                ax_1.set_ylim((np.nanmin(res_ydata) - (np.abs(np.nanmin(res_ydata)))), (np.nanmax(res_ydata) + (np.abs(np.nanmax(res_ydata)))))
        
            except:
                pass
            """
        wavelength_slider.on_changed(update)
        
        
        # Buttons 
        
        resetax = fig.add_axes([0.01, 0.656, 0.05, 0.03])  # 0.15
        add_clickax = fig.add_axes([0.01, 0.62, 0.05, 0.03])
        
        reset_button = Button(resetax, 'Reset', color = 'lightsteelblue', hovercolor='gainsboro')
        reset_button.label.set_fontsize(20)
        
        add_click_button = Button(add_clickax, 'Add', color='lightsteelblue', hovercolor='gainsboro')
        add_click_button.label.set_fontsize(20)
        
        
        mem_values = []
        
        def add_click(event):
            # stores the current time value selected with the slider in a list
            mem_values.append(wavelength_slider.val)
            """
            cmap_ = plt.get_cmap('viridis')
            colors_ = [cmap_(i) for i in np.linspace(0, 1, len(mem_values))]
            
            fig_Cpt_arr = np.zeros(shape=(self.DNS_CP.T.shape[0], self.DNS_CP.T.shape[1]))
                                          
            fig_1 = plt.figure(layout="constrained")
            grid_1 = GridSpec(3, 1, figure=fig_1, wspace=0.2, hspace=0.15)
        
            ax3 = fig_1.add_subplot(grid_1[1:, :])
            ax4 = fig_1.add_subplot(grid_1[0, :])
        
            
            for i, color in enumerate(colors_, start=0):
                
                for idx in range(fig_Cpt_arr.shape[0]):
                    fig_Cpt_arr[idx] = self.DNS_SP[idx, X_series[mem_values[i]]] * self.DNS_CP.T[idx]
                    
                fig_spectrum_sum = np.cumsum(fig_Cpt_arr, axis=0)
                
                fig_rec_ydata = fig_spectrum_sum[-1] 
                
                
                ax3.scatter(total_time, Z.T[X_series[mem_values[i]]], color=color, label = 'Raw Trace at '+str('%#.3g' % mem_values[i])+' nm')
                ax3.plot(total_time, fig_rec_ydata, color=color)
        
                
                ax3.tick_params(axis="both", labelsize=tick_size)
                ax3.set_xlabel('Time (ps)', fontsize=axis_fontsize)
                ax3.set_ylabel('Absorption (O.D)', fontsize=axis_fontsize)
                ax3.set_xlim(lx_lim, total_time[ux_lim])
                ax3.legend(fontsize=tick_size, loc='best')
                ax3.grid(visible=True)
        
                    
                ax4.plot(total_time, Z.T[X_series[mem_values[i]]] - fig_rec_ydata, f'.C{i}')
                
                ax4.tick_params(axis="both", labelsize=tick_size)
                ax4.set_xlim(lx_lim, total_time[ux_lim])
                ax4.set_title('Residual at '+str('%#.3g' % mem_values[i])+' nm', fontsize=title_fontsize, y=1.02)
                ax4.grid(visible=True)
                
                fig_1.canvas.draw_idle()
            """
            return
            
        
        def reset(event):
            
            mem_values.clear()
            wavelength_slider.reset()
            
            ax1.cla()
            ax2.cla()
            
            ax1.plot(total_time, Z.T[sl_min], '.C0', label='Raw Trace at '+str('%#.3g' % wavelengths[sl_min])+' nm')
            ax1.plot(total_time, reconstructed_trace_init, color='darkslateblue', linewidth=2, label='Fitted trace')
            
            for i in range(cpt_arr.shape[0]):
                ratio = np.trapz(np.abs(cpt_arr[i]), total_time) / np.trapz(int_ydata, total_time)
                ax1.plot(total_time, cpt_arr[i], ls='--', color=_colors[i*100], label=r'$\int_{t_{min}}^{t_{max}}$['+str(np.flip(e_labels)[i])+']$_{\lambda}$ / $\int_{t_{min}}^{t_{max}}$[e$_{i}$, e$_{j}$]$_{\lambda}$'+f': {np.round(ratio*100, 2)}%')
            
            ax1.tick_params(axis="both", labelsize = tick_size)
            ax1.set_xlabel('Wavelength (nm)', fontsize = axis_fontsize)
            ax1.set_ylabel('Absorption (OD)', fontsize = axis_fontsize) 
            ax1.set_xlim(lx_lim, total_time[ux_lim])
            # ax1.set_ylim(-0.01, 1.1)
        
            ax1.legend(fontsize=20)
            ax1.grid(visible=True)
            
            ax2.plot(total_time, Z.T[sl_min] - reconstructed_trace_init, '.C3')
            
            ax2.tick_params(axis="both", labelsize = tick_size)
            ax2.set_xlim(lx_lim, total_time[ux_lim])
            ax2.set_title('Residual at '+str('%#.3g' % wavelengths[sl_min])+' nm', fontsize = title_fontsize, y=1.02)
            ax2.grid(visible=True)
            
            fig.canvas.draw_idle()
        
        
        reset_button.on_clicked(reset)
        resetax._button = reset_button
        
        add_click_button.on_clicked(add_click)
        add_clickax._button = add_click_button
        
        plt.show()

        return


