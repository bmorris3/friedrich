# Import dev version of friedrich:
import sys
sys.path.insert(0, '../')

from friedrich.hyak2 import launch_hyak_run

launch_hyak_run(n_transits=583,
                run_script='/usr/lusers/bmmorris/git/friedrich/k17/k17_hyak.py',
                run_dir='/usr/lusers/bmmorris/git/friedrich/k17',
                job_name='friedrich_k17')
