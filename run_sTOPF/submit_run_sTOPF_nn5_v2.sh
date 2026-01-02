#!/bin/bash

# HTCondor/Juseless variables 
init_dir='/data/project/brainvar_sexdiff_movies/sTOPF' # main directory where the project is stored
code_dir='code/run_sTOPF' # subdirectory containing the code
name_py='run_sTOPF.py'
name_wrapper='wrap_run_sTOPF.sh'  # name of the bash-wrapper
proj='v2'

# print the .submit header
printf "# The environment

universe              = vanilla
getenv                = True
request_cpus          = 1
request_memory        = 1G

# Execution
initialdir            = ${init_dir}/${code_dir}
executable            = ${init_dir}/${code_dir}/${name_wrapper}
transfer_executable   = False
transfer_input_files = ${name_py}
\n"

# set up arguments for PCA
# project dir 
wkdir=${init_dir}
project=${proj}
nn=5

# set up log folder
logs_dir=${init_dir}/${code_dir}/logs/ #${dataset} #location where log files will be saved
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir" # create the logs dir if it doesn't exist

printf "arguments   = %s %s %s\n" "$wkdir" "$project" "$nn"
printf "log         = ${logs_dir}/\$(Cluster).\$(Process)_movie.log\n"
printf "output      = ${logs_dir}/\$(Cluster).\$(Process)_movie.out\n"
printf "error       = ${logs_dir}/\$(Cluster).\$(Process)_movie.err\n"
printf "Queue\n\n"

