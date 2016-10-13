# Import dev version of friedrich:
import sys
sys.path.insert(0, '../')

from friedrich.hyak import launch_hyak_run

launch_hyak_run(n_transits=205,
                output_dir='/gscratch/stf/bmmorris/friedrich/sun_inactive/',
                job_name='friedrich_sun',
                run_dir='/usr/lusers/bmmorris/git/friedrich/sun_inactive',
                run_script='/usr/lusers/bmmorris/git/friedrich/sun_inactive/sun_hyak.py')
