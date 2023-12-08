#!/bin/bash

#OAR -n python

#OAR --project faultscan

#OAR -l /nodes=1/core=2,walltime=2:30:00 

source /applis/environments/conda.sh
conda activate py3

# path to use the python
export PATH="/applis/environments/conda.sh:$PATH"
# export PYTHONPATH="pycorr/v1.0:$PYTHONPATH"

#python 3_compute_Us.py
python 2_perform_inversion.py
#python craft_Gaussian.py
