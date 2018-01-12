/**
 * Filename: MPI_NB_KNN.c 
 * Description:
 *  MPI non-blocking KNN classification algorithm with multithreading support
 *	Written for "Parallel and Distributed Systems" class
 *		Faculty of Electrical and Computer Engineering AUTH.
 * Authors:
 *  Katomeris Nikolaos, AEM: 8551, ngkatomer@auth.gr  
 *  Kyriazis Leandros Giorgos, AEM: 7711, gkyriazt@auth.gr
 * Date:
 *  December 2017
 * 
 * Compile command (example):
 *  mpicc -O3 -fopenmp filename.c functions.c -o exec_name
 *  (mpicc -O3 -fopenmp mergF.c functions.c)
 * Run command (example):
 *  mpiexec -np <number_of_processes> ./exec_name <number_of_elements> <number_of_attributes> <number_of_threads_per_process> (optional)
 *  (mpiexec -np 8 ./a.out 60000 30 4)
 */ 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "functions.h"

#define K 30

/**
 * ---------------------------------Main Description---------------------------------
 * line 52      : User arguments are passed into Initialize and the right bin file 
 * 			      is chosen according to them(10k or 60k) MPI is initialized there too.
 * lines 60-61  : OpenMP is set to not dynamic and number of threads is set
 * line 66      : Buffer (other_elements) is prepared to receive the data from other processes
 * lines 90-92  : Clock starts at process rank == 0. 
 * lines 97-136 : All processes transfer data in ring pattern using the non - blocking Isend , Irecv
 *				  while updating their KNN_tables. Also they use mpi_Waits for synchronization. 
 * lines 143-153: clock stops and duration is printed
 * lines 158-184: For cross validation with matlab the percentage of the correct matches is calculated
 */
 
int main(int argc, char** argv)
{
    /**
     * M : the number of elements,
     * N : the number of attributes,
     * P : the number of processes,
     * TN: the number of threads to run each process
     */
    int M, N, P, TN;
    FILE *table_data_fp, *table_labels_fp;
    if(Initialize(&M, &N, &P, &TN, &table_data_fp, &table_labels_fp, argc, argv)){
        printf("Unsuccessful initialization. Exiting...\n");
        return 0;
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request request[3];
    MPI_Status status;
    omp_set_dynamic(0);
    omp_set_num_threads(TN);
    /**
     * Initialize the respecting tables of each process.
     */
    element* my_elements = loadMyElements(table_data_fp, table_labels_fp, M/P, N, rank);
    element* other_elements = prepareBuffer(M/P, N);
    element* buffer = prepareBuffer(M/P, N);

    double** my_knn = initializeKNNTable(M/P, N);
    int** my_label_knn = initializeLabelKNNTable(M/P, N);

    if (my_elements == NULL || other_elements == NULL || buffer == NULL || my_knn == NULL || my_label_knn == NULL){
        exit(1);
    }

    int size_of_elements = M/P*sizeof(int)+M/P*N*sizeof(double);

    /**
     * Initialize counter for counting how many data
     * have been calculated in each process in order
     * to know when to finish the KNN algorithm.
     */
    int i = 2;

    /**
     * Wait everyone and start timer on rank = 0.
     */
    struct timespec start, finish;
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank){
        clock_gettime(CLOCK_REALTIME, &start);
    }

    /**
     * Main algorithm (NON BLOCKING).
     */
    if (rank % 2){
        MPI_Isend(my_elements[0].label, size_of_elements, MPI_BYTE, (rank+1) % P, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(other_elements[0].label, size_of_elements, MPI_BYTE, rank -1, 0, MPI_COMM_WORLD, &request[1]);
        findKNN(my_knn, my_label_knn, M/P, K, N, my_elements, my_elements);
        
        MPI_Wait(&request[1], &status);
        while(i < P){
            MPI_Isend(other_elements[0].label, size_of_elements, MPI_BYTE, (rank+1) % P, 0, MPI_COMM_WORLD, &request[2]);
            MPI_Irecv(buffer[0].label, size_of_elements, MPI_BYTE, rank -1, 0, MPI_COMM_WORLD, &request[1]);

            findKNN(my_knn, my_label_knn, M/P, K, N, my_elements, other_elements);

            MPI_Wait(&request[1], &status);
            MPI_Wait(&request[2], &status);
            swapElementP(&other_elements, &buffer);
            i++;
        }
        //Last one
        findKNN(my_knn, my_label_knn, M/P, K, N, my_elements, other_elements);
    }
    else{
        MPI_Isend(my_elements[0].label, size_of_elements, MPI_BYTE, (rank+1) % P, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(other_elements[0].label, size_of_elements, MPI_BYTE, (P+rank-1) % P, 0, MPI_COMM_WORLD, &request[1]);
        
        findKNN(my_knn, my_label_knn, M/P, K, N, my_elements, my_elements);

        MPI_Wait(&request[1], &status);
        while(i < P){
            MPI_Isend(other_elements[0].label, size_of_elements, MPI_BYTE, (rank+1) % P, 0, MPI_COMM_WORLD, &request[2]);
            MPI_Irecv(buffer[0].label, size_of_elements, MPI_BYTE, (P+rank-1) % P, 0, MPI_COMM_WORLD, &request[1]);
            
            findKNN(my_knn, my_label_knn, M/P, K, N, my_elements, other_elements);

            MPI_Wait(&request[1], &status);
            MPI_Wait(&request[2], &status);
            swapElementP(&other_elements, &buffer);
            i++;
        }
        findKNN(my_knn, my_label_knn, M/P, K, N, my_elements, other_elements);
    }
    //Wait everyone.
    MPI_Barrier(MPI_COMM_WORLD);

    /**
     * Get time on rank = 0.
     */
    if (!rank){
        clock_gettime(CLOCK_REALTIME, &finish);
        long seconds = finish.tv_sec - start.tv_sec; 
        long nano_seconds = finish.tv_nsec - start.tv_nsec; 
        
        if (start.tv_nsec > finish.tv_nsec) {
        seconds--; 
        nano_seconds += 1e9; 
        } 
        printf("_time: %lf\n", seconds + (double)nano_seconds/1e9);
    }

    /**
     * Calculate and print result percentage.
     */
    int j, counter = 0;
    for (i = 0; i < M/P; i++){
        int count[10] = {0};
        for (j = 0; j < K; j++){
            count[my_label_knn[i][j]-1]++;
        }
        if(iMax(count, 10)+1 == *my_elements[i].label){
            counter++;
        }
    }
    double result = (double) counter / (M/P);
    double *result_array;
    if (!rank){
        result_array = (double*) malloc(P*sizeof(double));
        if (result_array == NULL){
            printf("Algorithm finished but there was no memory for calculating the complete result.");
        }
    }

    MPI_Gather(&result, 1, MPI_DOUBLE, result_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(!rank){
        for (i = 1; i < P; i++){
            result += result_array[i];
        }
        printf("Final %% is: %lf\n",  result / P);
    }

    /**
     * Clean up.
     */
    for (i = 0; i < M/P; i++){
        free(my_knn[i]);
        free(my_label_knn[i]);
    }
    free(my_knn);
    free(my_label_knn);
    free(my_elements[0].label);
    free(my_elements);
    free(other_elements[0].label);
    free(other_elements);
    free(buffer[0].label);
    free(buffer);
    fclose(table_data_fp);
    fclose(table_labels_fp);
    MPI_Finalize();

    return 0;
}
