# cat {command_list_path} | parallel -j {n_jobs}

import os


n_jobs = 4
n_solar_two_year_bins = 35
fit_sun_path = 'fit_sun.py'
fit_hat11_path = 'fit_hat11.py'
command_list_path = 'parallel.csh'
python_path = 'python'

with open(command_list_path, 'w') as command_list:
    command_list.write(" ".join([python_path, fit_hat11_path, "\n"]))
    
    for i in range(n_solar_two_year_bins):
        command_list.write(" ".join([python_path, fit_sun_path, str(i), "\n"]))

os.system("cat {command_list_path} | parallel -j {n_jobs}"
          .format(n_jobs=n_jobs, command_list_path=command_list_path))