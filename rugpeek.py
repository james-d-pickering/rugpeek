import rugpeek_funcs as rp
import matplotlib.pyplot as plt
import numpy as np
import lmfit as lf
import matplotlib as mpl
import sys

mpl.use('tkagg')

plt.close('all')

#JDP data files - edit so they are wherever they need to be for your local directory
matrix_file = './example_data/Ferric-Mb_80uM_409nm_SHG_TA_2_matrix'
solvent_file = "./example_data/milliQ_409nm_SHG_chirp_1_matrix"


#JDP create a Rug object from one of the data files
data = rp.Rug(solvent_file, '.dat')

#sys.exit() #JDP use to exit the code after a certain point - useful for testing

#JDP get the dispersion coefficients "by eye"
data.get_dispersion_correction_eye()

#JDP alternatively state the coefficients (lowest degree to highest - i.e. coefficient of degree n is element n)
#coefs = [-3.99E01, 2.649E-01, -5.84E-04, 4.40E-07]

#JDP apply these dispersion corrections 
data.apply_dispersion_correction()

#JDP look at the raw data, with the dispersion correction overlaid
data.peek(raw=True, plot_dispersion=True)

#JDP now we have applied a correction, the default data matrix is the corrected one - look at it:
data.peek()


plt.show() #because otherwise stuff gets destroyed when the program ends, as the data class is destroyed.
