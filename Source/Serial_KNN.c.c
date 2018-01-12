#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define N 30
#define M 60000 
#define K 30


int main()
{
	double **table;
	int *labels = (int*) malloc(M*sizeof(int));
	FILE *table_data_fp , *table_labels_fp;
	
	table = (double**) malloc(M*sizeof(double *));
	
	if (table == NULL){
		printf("Couldn't allocate memory for the table. Exiting...");
		exit(1);
	}
	int table_line;
	for (table_line = 0; table_line < M; table_line++){
		table[table_line] = (double *)malloc(N*sizeof(double));
		if (table[table_line] == NULL){
			printf("Couldn't allocate memory for the table line %d." 
					"Exiting...", table_line);
			exit(1);
		}
	}

	//table_data_fp = fopen("train_X_10k_x_784.bin","rb");
	table_data_fp = fopen("train_X_SVD_60k_x_30.bin","rb");


	table_labels_fp = fopen("Labels_60k_x_1.bin","rb");

	int i,j,c1,c2,k,count;
	int l; 

	//***LOAD DATA FROM THE FILES***
	//Load every line of the table_data file
	for (i = 0 ; i < M   ; i++) {
		fread(table[i] ,sizeof(double),N,  table_data_fp);
		//fread(table[1] ,sizeof(double),N,  table_data_fp);
	}
	fclose(table_data_fp);

	fread (labels , sizeof(int) , M , table_labels_fp);
	fclose(table_labels_fp);
	//-----------------------------

	//Declare and initialize final tables
	double dist = 0; 
	double **dist_nn = (double**) malloc(M*sizeof(double *));
	int **labels_nn = (int**) malloc(M*sizeof(int *));
	for (i = 0; i < M; i++)
	{
		dist_nn[i] = (double *)malloc(K*sizeof(double));
		labels_nn[i] = (int*) malloc(K*sizeof(int));
	}
	//double dist_nn[M][K];
	//int labels_nn[M][K];
	
	for (i = 0 ; i < M ; i ++ ){
		for (j = 0 ; j < K ; j ++ ){
		
		dist_nn[i][j] = 100000 ;
		labels_nn[i][j] = -1;
		}
	}
	//----------------------------
	double temp ; 
	struct timespec start, finish;
	clock_gettime(CLOCK_REALTIME, &start);
	// Distance of spot k from all the M points
	for (l = 0; l < M ; l++){
		for (c1 = 0; c1 < M; c1++){
			dist = 0 ;
			if (l == c1)
				continue;
			
			for ( c2 = 0 ; c2 < N ; c2 ++) {
				dist+=(table[l][c2]-table[c1][c2])*(table[l][c2]-table[c1][c2]);
			}
		
			for (count = K-1; count >= 0; count--) {
				if (dist < dist_nn[l][count]) { 
					continue;
				}      
				else{  
					break;
				}
			}
			count++;

			if ( count >= K )
				continue;     
		     
			for (i = K-2; i >= count; i--){
				dist_nn[l][i+1] = dist_nn[l][i];
				labels_nn[l][i+1] = labels_nn[l][i];
			}
			dist_nn[l][count] = dist;
			labels_nn[l][count] = labels[c1];
		}
		//if(i ==10000)break ;
	}

	//#######################################################//
	clock_gettime(CLOCK_REALTIME, &finish);
	long seconds = finish.tv_sec - start.tv_sec; 
	long nano_seconds = finish.tv_nsec - start.tv_nsec; 
	
	if (start.tv_nsec > finish.tv_nsec) {
	seconds--; 
	nano_seconds += 1e9; 
	} 
	printf("_time: %lf\n", seconds + (double)nano_seconds/1e9);
        
	
	//Clean up
	for ( table_line = 0 ; table_line < M ; table_line ++ ) {
		free(table[table_line]);
	} 
	for ( table_line = 0 ; table_line < K ; table_line ++ ) {
		free(labels_nn[table_line]);
		free(dist_nn[table_line]);
	} 
	free(dist_nn);
	free(labels_nn);
	free(table);
	free(labels);
	//Finish
	return 0;
}
