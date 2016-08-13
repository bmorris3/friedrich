
from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import sys
sys.path.insert(0, '../')

from friedrich.stsp import friedrich_results_to_stsp_inputs
from friedrich.lightcurve import hat11_params_morris

results_dir = '/local/tmp/friedrich/hat11'
transit_params = hat11_params_morris()

friedrich_results_to_stsp_inputs(results_dir, transit_params)
