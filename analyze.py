
from friedrich.analysis import MCMCResults
from friedrich.lightcurve import hat11_params_morris
import matplotlib.pyplot as plt

archive_path = '/Users/bmmorris/data/chains024.hdf5'

m = MCMCResults(archive_path)
m.plot_lnprob()
m.plot_corner()
m.plot_max_lnp(hat11_params_morris())
plt.show()
