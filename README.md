# KNN_Classification
### KNN Classification with MPI and openMP support

The source code of the program is in the **Source** folder. There are 3 versions:
- The MPI Blocking version _MPI_B_KNN.c_
- The MPI Non-Blocking version _MPI_NB_KNN.c_
- A serial version _Serial_KNN.c_

The data used to produce the resulting graphs found on the **images** folder 
can be found here:
 * [Dropbox link](https://www.dropbox.com/s/y53wvtfk2lhsx46/KNNTrainingFiles.zip?dl=0)
 
 To compile and run the programs:
 1. Clone this repository.
 1. Download and extract the training data from the [Dropbox link](https://www.dropbox.com/s/y53wvtfk2lhsx46/KNNTrainingFiles.zip?dl=0) 
 into a new "Files" directory **inside** the cloned repo's directory. In other words, the extracted files should be in the `repo_path/Files` directory.
 1. Go to the **Source** directory (`cd repo_path/Source`)
 1. Compile the programs:
    * Blocking: `mpicc -O3 -fopenmp functions.c MPI_B_KNN.c -o BL`
    * Non Blocking: `mpicc -O3 -fopenmp functions.c MPI_NB_KNN.c -o NBL`
    * Serial: `gcc -O3 Serial_KNN.c -o SE -lrt
 1. Run the programs on a single computer (example with the data provided in the **Files** directory and **4** procs):
    * Blocking: `mpiexec -np 4 ./BL 10000 784`
    * Non Blocking: `mpiexec -np 4 ./NBL 60000 30`
 
 For running the scripts on the hellasgrid cluster you can use scripts like the ones provided in the **Scripts** folder.
 The matlab script was used to extract the plots shown at the **images** folder.
