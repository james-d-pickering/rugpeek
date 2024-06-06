import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pn
import lmfit as lf
import scipy as sp
import matplotlib.colors as colors

class RugTools:
   # class for generic tools to help data processing

    def read_harpia_matrix(file, extension):
    # Harpia matrix is the overall TA matrix from a measurement
    # not divided into individual scans
        data = np.genfromtxt(file+extension, skip_header=2)
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
        self.filename = fname

        self.chirp_corrected = False

        #consistent filename convention: split with underscores. define a function to split it up
        #and get the relevant bits, so the metadata is stored so it can be correlated with the dispersion fit


    def open_clickable_rug(self, scale='log'):
        fig = plt.figure()
        ax = fig.gca()
        
        ax.pcolormesh(self.wavelengths, self.times, self.matrix)
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Delay [ps]')
        
        if scale == 'log':
            ax.set_yscale('symlog')
        elif scale == 'linear':
            ax.set_yscale('linear')
        else:
            raise NameError
        
        return ax
 
    def get_t0_positions(self, ax, degree):
        ax.set_title(f"Click t0 at {degree} or more points, then press enter")
        plt.draw()
        self.calib_points = plt.ginput(n=-1, timeout=-1, show_clicks=True)
        return

    # def interpolate_time_axis(self, dt=0.05):
        # interpolate the time axis so we can apply the chirp correction
    #    samples = np.size(self.times)/dt
     #   x = np.arange(np.min(self.times), np.max(self.times), samples)
      #  t_index = np.arange(np.size(self.times))
      #  t_interp = np.interp(x, t_index, self.times)
      #  plt.figure()
      #  plt.plot(t_index, self.times)
      #  plt.plot(x, t_interp)
      #  return

    def generate_time_axis(self, degree):
        wavelengths = [point[0] for point in self.calib_points]
        times = [point[1] for point in self.calib_points]
        fit = pn.Polynomial.fit(wavelengths, times, deg=degree)
        coefs = fit.convert().coef
        time_fit = pn.polyval(self.wavelengths, coefs)
        # corrected_time are the time corrections we need to use
        self.corrected_time = -time_fit
        self.dispersion_coefs = coefs
        return 

    def plot_chirp_correction(self, ax):
        ax.plot(self.wavelengths, -self.corrected_time, color='k')
        return
       

    def peek(self, ax=None, cmap='inferno', min_max=None, aspect='equal', interpolation='none',
                norm=None,scale='log', title=None, colourbar=False, xlabel=True, ylabel=True,
                yticks=True, xticks=True):
        if not ax:
            fig = plt.figure(figsize=(6,6))
            ax0 = fig.gca()
        else:
            ax0 = ax
        
        if min_max:
            vmin = min_max[0]
            vmax = min_max[1]
        else:
            min_max = (None, None)
            
        #print(vmin, vmax)
        im = ax0.pcolormesh(self.wavelengths, self.times, self.matrix,
        cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))
        #ax0.set_xticks(self.wavelengths)
        #ax0.set_yticks(self.times)
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

        return im

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
       
 
    def apply_dispersion_correction(self):
        # if chirp has been corrected, extend the interpolation range to the largest possible
        if hasattr(self, "corrected_time"):
            interp_min = np.min(self.times)+np.min(self.corrected_time)
            interp_max = np.max(self.times)+np.max(self.corrected_time)
        else:
            raise Exception("need to find the dispersion correction before applying it, obvs")

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
        self.raw_matrix = self.matrix

        #JDP apply the chirp correction by rolling each wavelength's time axis by the necessary amount
        shifted_matrix = RugTools.indep_roll(data_interp, time_shifts, axis=0)
        fig = plt.figure(); ax0 = fig.gca()
       # ax0.set_title('chirp corrected, interpolated')
       # ax0.pcolormesh(self.wavelengths, self.interpolated_time, shifted_matrix, shading='nearest') 
       # ax0.set_yscale('symlog')


        #JDP now need to "undo" the interpolation. should just be:
        time_indices = []
        for time in self.times:
            time_idx, _ = RugTools.find_nearest(self.interpolated_time, time)
            time_indices.append(time_idx)

        shifted_coarse = shifted_matrix[time_indices, :]

        fig = plt.figure(); ax0 = fig.gca()
        ax0.set_title('Dispersion Corrected')
        ax0.pcolormesh(self.wavelengths, self.times, shifted_coarse, shading='nearest')
        ax0.set_yscale('symlog')
        plt.show()

        return

   

    def get_dispersion_correction(self, degree=3):
        ax = self.open_clickable_rug(scale='log')
        self.get_t0_positions(ax, degree)
        self.generate_time_axis(degree=degree)
        self.plot_chirp_correction(ax)
        return
        
    def get_time_trace(self, wavelength, plot=False):
        wavelength_idx, wavelength_real = RugTools.find_nearest(self.wavelengths, wavelength)
        time_trace = self.matrix[:, wavelength_idx]
        if plot:
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(self.times, time_trace)
            ax.set_xscale("symlog")
        return time_trace
    
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


