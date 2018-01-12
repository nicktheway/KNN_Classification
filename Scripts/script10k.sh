#!/bin/bash
#PBS -q pdlab
#PBS -N mpi-mm
#PBS -j oe
#PBS -l nodes=2:ppn=8

module load mpi
cd $PBS_O_WORKDIR

mpiexec -n 16 -ppn 8  ../Bin/BL 10000 784 8 > ../Results/Results_10k_BL/BL_2_8_8.txt

