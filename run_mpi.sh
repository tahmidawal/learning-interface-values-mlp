#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fenics-env
export PYTHONPATH=/Users/tahmidawal/Desktop/M-ORAS/Learning\ Boudary\ Conditions\ Subdomation/.venv/lib/python3.11/site-packages:$PYTHONPATH
python pde_solver_ml_mpi.py 