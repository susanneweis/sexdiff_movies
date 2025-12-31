#!/bin/bash

# set environment name
env_name="hormone_env"

# activate virtual environment
source /home/sweis/.venvs/$env_name/bin/activate

# run the Python script
python run_sTOPF.py "$1" "$2" "$3"

# deactivate environment
deactivate

