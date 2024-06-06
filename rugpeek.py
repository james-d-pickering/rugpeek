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


data = rp.Rug(matrix_file, '.dat')

data.get_dispersion_correction()
data.apply_dispersion_correction()

