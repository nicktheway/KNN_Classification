# KNN_Classification
### KNN Classification with MPI and openMP support

The source code of the program is in the **Source** folder. There are 3 versions:
- The MPI Blocking version
- The MPI Non-Blocking version
- A serial version

The data used to produce the resulting graphs found on the **images** folder 
can be found in the **Files** directory here:
 * [Dropbox link](https://www.dropbox.com/sh/p0qn8vo9xkse578/AADnDMDxAFhfhxU7DRRpn4YYa/Source?dl=0)
 
 To compile and run the programs:
 1. download the **Source** and the **Files** 
 directories from the [Dropbox link](https://www.dropbox.com/sh/p0qn8vo9xkse578/AADnDMDxAFhfhxU7DRRpn4YYa/Source?dl=0) 
 and put them on the same directory.
 2. Go to the **Source** directory (`cd path/Source`)
 3. Compile the programs:
    * Blocking: `mpicc -O3 -fopenmp functions.c MPI_B_KNN.c -o BL`
    * Non Blocking: `mpicc -O3 -fopenmp functions.c MPI_NB_KNN.c -o NBL`
    * Serial: `gcc -O3 Serial_KNN.c -o SE -lrt
 4. Run the programs on a single computer, example with the data provided in the **Files** directory and 4 procs:
    * Blocking: `mpiexec -np 4 ./BL 10000 784`
    * Non Blocking: `mpiexec -np 4 ./NBL 60000 30`
 
 For running the scripts on the hellasgrid cluster you can use scripts like the ones provided in the **Scripts** folder.
 The matlab script was used to extract the plots shown at the **images** folder.
