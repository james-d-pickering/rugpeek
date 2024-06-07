import rugpeek_funcs as rp
import matplotlib.pyplot as plt
import numpy as np
import lmfit as lf
import matplotlib as mpl

mpl.use('tkagg')

plt.close('all')
matrix_file = '../2024-01-09/Ferric-Mb_80uM_409nm_SHG_TA_2_matrix'
solvent_file = "../2024-01-16/milliQ_409nm_SHG_chirp_1_matrix"

#matrix_file = './fitting_test'

data = rp.Rug(solvent_file, '.dat')

data.get_dispersion_correction_eye()

coefs = [-3.99E01, 2.649E-01, -5.84E-04, 4.40E-07]

data.apply_dispersion_correction(coefs = coefs)

data.peek(raw=True, plot_dispersion=True)

data.peek()


plt.show() #because otherwise stuff gets destroyed when the program ends, as the data class is destroyed.
