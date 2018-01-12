#!/bin/bash
#PBS -q pdlab
#PBS -N mpi-mm
#PBS -j oe
#PBS -l nodes=4:ppn=8

module load mpi
cd $PBS_O_WORKDIR

mpiexec -n 32 -ppn 8  ../Bin/NBL 60000 30 8 > ../Results/Results_60k_NBL/NBL_4_8_8.txt
