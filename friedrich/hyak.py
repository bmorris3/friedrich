# Licensed under the MIT License - see LICENSE.rst
"""
Launch a batch of hyak runs
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

submit_template = """#!/bin/bash
## --------------------------------------------------------
## NOTE: to submit jobs to Hyak use
##       qsub <script.sh>
##
## #PBS is a directive requesting job scheduling resources
## and ALL PBS directives must be at the top of the script,
## standard bash commands can follow afterwards.
## NOTE: Lines that begin with #PBS are commands to PBS,
##       and they are not comment lines.  To comment out
##       use "#  PBS".
## --------------------------------------------------------

## Job name
#PBS -N {job_name}

## DIRECTORY where this job is run
#PBS -d {run_dir}

## GROUP to run under
## PBS -W group_list=hyak-stf
#PBS -q bf

## NUMBER nodes, CPUs per node, and MEMORY
#PBS -l nodes=1:ppn=8,mem=12gb,feature=intel

## WALLTIME (defaults to 1 hour as the minimum, specify > 1 hour longer jobs)
#PBS -l walltime={walltime}

## LOG the (stderr and stdout) job output in the directory
#PBS -j oe -o {log_dir}

## EMAIL to send when job is aborted, begins, and terminates
#PBS -m abe -M {email}

## --------------------------------------------------------
## END of PBS commands ... only BASH from here and below
## --------------------------------------------------------

## LOAD any appropriate environment modules and variables
## module load git_2.4.4
module load gcc_4.4.7-impi_5.1.2

## --------------------------------------------------------
## DEBUGGING information (include jobs logs in any help requests)
## --------------------------------------------------------
## Total Number of nodes and processors (cores) to be used by the job
echo "== JOB DEBUGGING INFORMATION=========================="
HYAK_NNODES=$(uniq $PBS_NODEFILE | wc -l )
HYAK_NPE=$(wc -l < $PBS_NODEFILE)
echo "This job will run on $HYAK_NNODES nodes with $HYAK_NPE total CPU-cores"
echo ""
echo "Node:CPUs Used"
uniq -c $PBS_NODEFILE | awk '{{print $2 ":" $1}}'
echo ""
echo "ENVIRONMENT VARIABLES"
set
echo ""
echo "== END DEBUGGING INFORMATION  ========================"


## --------------------------------------------------------
## RUN your specific applications/scripts/code here
## --------------------------------------------------------

## CHANGE directory to where job was submitted
## (careful, PBS defaults to user home directory)
cd $PBS_O_WORKDIR

mpirun -np 8 {run_script} {transit_number}

## python {run_script} {transit_number}
"""

def launch_hyak_run(n_transits, run_script, run_dir, job_name='friedrich',
                    log_dir='/gscratch/stf/bmmorris/friedrich/logs',
                    submit_script_dir='/gscratch/stf/bmmorris/friedrich/submit_scripts'):

    for i in range(n_transits):
        walltime = '01:00:00'
        email = 'bmmorris@uw.edu'

        transit_number = str(i)
        submit_script_name = 'submit_script_{0}.sh'.format(transit_number)

        submit_script = submit_template.format(job_name=job_name,
                                               run_dir=run_dir,
                                               log_dir=log_dir,
                                               walltime=walltime,
                                               email=email,
                                               run_script=run_script,
                                               transit_number=transit_number)

        submit_script_path = os.path.join(submit_script_dir, submit_script_name)
        with open(submit_script_path, 'w') as f:
            f.write(submit_script)

        os.system('qsub {0}'.format(submit_script_path))
