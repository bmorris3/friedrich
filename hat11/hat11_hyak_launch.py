# Import dev version of friedrich:
import sys
sys.path.insert(0, '../')

from friedrich.hyak import launch_hyak_run

launch_hyak_run(n_transits=205, 
                job_name='friedrich_hat11',
                run_dir='/usr/lusers/bmmorris/git/friedrich/hat11',
                run_script='/usr/lusers/bmmorris/git/friedrich/hat11/hat11_hyak.py')
