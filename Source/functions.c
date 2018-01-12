/**
 * File: functions.c
 * Description:
 *  Helper file for the 2nd exercise (KNN classification using MPI),
 *  of the parallel and distributed systems class, THMMY AUTH.
 *  
 *  The declarations of the functions defined here are in:
 *  File: functions.h
 * 	That file also contains COMMENTS about these functions.
 * 
 * Authors:
 *  Katomeris Nikolaos, AEM: 8551, ngkatomer@auth.gr
 *  Kyriazis Leandros Giorgos, AEM: 7711, gkyriazt@auth.gr
 * 
 * Date:
 *  December 2017
 */
#include "functions.h"


int Initialize(int* number_of_elements, int* number_of_attributes, int* number_of_processes, int* number_of_threads,FILE** data_fp, FILE** labels_fp, int argc, char** argv){
    if (argc != 3 && argc != 4){
        printf("Usage:./%s <number_of_elements> <number_of_attributes> <number_of_threads_per_process> (optional)\n", argv[0]);
		return 1;
    }
    // MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, number_of_processes);
    if (*number_of_processes % 2){
        printf("Odd number of processes <%d> is not supported.\n", *number_of_processes);
		return 1;
    }
    *number_of_elements = atoi(argv[1]);
	*number_of_attributes = atoi(argv[2]);
	if (argc == 4)	*number_of_threads = atoi(argv[3]);
	else *number_of_threads = 1;

	if(*number_of_threads == 0) *number_of_threads = 1;

	if (*number_of_elements % *number_of_processes){
		printf("number_of_elements %% number_of_attributes must be zero\n");
		return 1;
	}
    if((*number_of_attributes == 784 && *number_of_elements > 10000) || (*number_of_attributes == 30 && *number_of_elements > 60000) || *number_of_attributes <= 0 || number_of_elements <= 0){
        printf("These arguments are not supported. Exiting...\n");
        return 1;
    }

    if(*number_of_attributes == 784){
        *data_fp = fopen("../Files/train_X_10k_x_784.bin","rb");
        *labels_fp = fopen("../Files/Labels_10k_x_1.bin","rb");
        if (*data_fp == NULL){
		    printf("Couldn't open data file. Exiting...\n");
		    return 1;
	    }
        if (*labels_fp == NULL){
		    printf("Couldn't open label file. Exiting...\n");
		    return 1;
	    }
    }
    else if(*number_of_attributes == 30){
        *data_fp = fopen("../Files/train_X_SVD_60k_x_30.bin","rb");
        *labels_fp = fopen("../Files/Labels_60k_x_1.bin","rb");
        if (*data_fp == NULL){
		    printf("Couldn't open data file. Exiting...\n");
		    return 1;
	    }
        if (*labels_fp == NULL){
		    printf("Couldn't open label file. Exiting...\n");
		    return 1;
	    }
    }
    else{
        printf("No file with %d attributes found\n", *number_of_attributes);
        return 1;
    }

    return 0;
}

element* prepareBuffer(int number_of_elements, int number_of_attributes){
    element* my_elements = (element*)malloc(number_of_elements*sizeof(element));
    void* packet = malloc(number_of_elements*sizeof(int) +number_of_elements*number_of_attributes*sizeof(double));
    if (packet == NULL || my_elements == NULL){
		fprintf(stderr, "Couldn't allocate memory at func: prepareBuffer\n");
		return NULL;
	}
    int* label_packet = &((int*) packet)[0];
    double* data_packet = (double*) &((int*) packet)[number_of_elements];
    int i;
    for (i = 0; i < number_of_elements; i++){
        my_elements[i].label = &label_packet[i];
        my_elements[i].attributes = &data_packet[number_of_attributes*i];
    }
    return my_elements;
}

void swapElementP(element **one, element **two){
	element *temp = *one;
	*one = *two;
	*two = temp;
}

int iMax(int* array, int size){
    int i;
    int max_index = 0;
    for (i = 1; i < size; i++){
        if(array[i] > array[max_index]){
            max_index = i;
        }
    }
    return max_index;
}

void findKNN(double** distKNN, int** labelKNN,int lines, int k, int attr_number,element* my_elements, element* other_elements)
{
	int i;
	
	#pragma omp parallel for
	for (i = 0; i < lines; i++)
	{
		double distance;
		int j, attr, position, counter;
		for (j = 0; j < lines; j++)
		{
			distance = 0;
			
			for (attr = 0; attr < attr_number; attr++){
				distance += (my_elements[i].attributes[attr]-other_elements[j].attributes[attr])*(my_elements[i].attributes[attr]-other_elements[j].attributes[attr]);
			}
			if (distance == 0) continue;
			
			for (position = k - 1; position >= 0; position--){
				if (distance < distKNN[i][position]) continue;
				else break;
			}
			position++;
			
			if (position >= k) continue;
			
			for (counter = k - 2; counter >= position; counter--){
				distKNN[i][counter+1] = distKNN[i][counter];
                labelKNN[i][counter+1] = labelKNN[i][counter];
			}
			distKNN[i][position] = distance;
            labelKNN[i][position] = *other_elements[j].label;
		}
	}
}

double** initializeKNNTable(int rows, int col)
{
	double** my_knn = (double**) malloc(rows*sizeof(double*));
	if (my_knn == NULL) {
		fprintf(stderr, "Couldn't allocate memory for the knn table");
		return NULL;
	}
	int i, j;
	for (i = 0; i < rows; i++){
		my_knn[i] = (double*) malloc(col*sizeof(double));
		if (my_knn[i] == NULL) {
			fprintf(stderr, "Couldn't allocate memory for the knn table row %d", i);
			return NULL;
		}
	}
	
	for (i = 0; i < rows; i++){
		for (j = 0; j < col; j++){
			my_knn[i][j] = DBL_MAX;
		}
	}
	return my_knn;
}

int** initializeLabelKNNTable(int rows, int col)
{
	int** my_knn = (int**) malloc(rows*sizeof(int*));
	if (my_knn == NULL) {
		fprintf(stderr, "Couldn't allocate memory for the knn table");
		return NULL;
	}
	int i, j;
	for (i = 0; i < rows; i++){
		my_knn[i] = (int*) malloc(col*sizeof(int));
		if (my_knn[i] == NULL) {
			fprintf(stderr, "Couldn't allocate memory for the knn table row %d", i);
			return NULL;
		}
	}
	
	for (i = 0; i < rows; i++){
		for (j = 0; j < col; j++){
			my_knn[i][j] = -1;
		}
	}
	return my_knn;
}

element* loadMyElements(FILE* data_fp, FILE* label_fp, int number_of_elements, int number_of_attributes, int offset)
{
    element* my_elements = (element*)malloc(number_of_elements*sizeof(element));
    if (my_elements == NULL){
		fprintf(stderr, "Couldn't allocate memory at func: loadMyElements\n");
		return NULL;
	}
    void* packet = loadElements(label_fp, data_fp, number_of_elements, number_of_attributes, offset);
	if (packet == NULL){
		return NULL;
	}
    int* label_packet = &((int*) packet)[0];
    
    double* data_packet = (double*) &((int*) packet)[number_of_elements];
    int i;
    for (i = 0; i < number_of_elements; i++){
		my_elements[i].label = &label_packet[i];
        my_elements[i].attributes = &data_packet[number_of_attributes*i];
	}

    return my_elements;
}

void* loadElements(FILE* label_fp, FILE* data_fp, int number_of_elements, int number_of_attributes, int offset){
    void* elements = malloc(number_of_elements*sizeof(int)+number_of_attributes*number_of_elements*sizeof(double));
    if (elements == NULL){
		fprintf(stderr ,"Couldn't allocate memory for the table segment"
				" at offset %d\n", offset);
		return NULL;	
	}
    int n;

    //Get Labels
    int* labels_start = (int *)elements;
	if (fseek(label_fp, offset*number_of_elements*sizeof(int), SEEK_SET))
	{
		fprintf(stderr, "Error at setting the label file's offset, offset: %d\n", offset);
		return NULL;
	}
	if((n = fread(labels_start, sizeof(int), number_of_elements, label_fp)) != number_of_elements){
		fprintf(stderr, "Error at reading the label data, offset: %d, size = %d\n", offset, number_of_elements);
		return NULL;	
	}

    //Get attributes
    if (fseek(data_fp, offset*number_of_elements*number_of_attributes*sizeof(double), SEEK_SET)){
		fprintf(stderr, "Error at setting the attr file's offset, offset: %d\n", offset);
		return NULL;
	}

    double* data_start = (double*)&((int*) elements)[number_of_elements];
    if((n = fread(data_start, sizeof(double), number_of_elements*number_of_attributes, data_fp)) != number_of_elements*number_of_attributes){
		fprintf(stderr, "Error at reading the attr data, offset: %d, size = %d\n", offset, number_of_elements);
		return NULL;	
	}

    return elements;
}