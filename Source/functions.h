/**
 * File: functions.h
 * Description:
 *  Helper file for the 2nd exercise (KNN classification using MPI),
 *  of the parallel and distributed systems class, THMMY AUTH.
 *  
 *  The definitions of the functions declared here are in:
 *  File: functions.c
 * 
 * Authors:
 *  Katomeris Nikolaos, AEM: 8551, ngkatomer@auth.gr
 *  Kyriazis Leandros Giorgos, AEM: 7711, gkyriazt@auth.gr
 * 
 * Date:
 *  December 2017
 */

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mpi.h>
#include <omp.h>

/**
 * Definition of the type: <element>
 * This type will be used for storing an element(point) of 
 * data. Each element has a label and a number of attributes.
 * This type contains two pointers:
 * -label: int*, should point to an int that will be the label
 * -attributes: double*, should point to a double/double_table
 */
typedef struct{
    int* label;
    double* attributes;
} element;

/**
 * Method: Initialize()
 * Description:
 * -Reads the input arguments
 * -Initializes the appropriate files (the path of the files is hardcoded)
 * -Initializes MPI and the number_of_processes pointed variable.
 * Return value:
 * -0, if no error occured
 * -1, if something went wrong.
 */ 
int Initialize(int* number_of_elements, int* number_of_attributes, int* number_of_processes, int* number_of_threads, FILE** data_fp, FILE** labels_fp, int argc, char** argv);

/**
 * Method: loadElements()
 * Description:
 *  Allocates a continuous space in memory (heap) where it loads the labels
 *  and the (#<number_of_attributes>) attributes of #<number_of_elements> 
 *  elements from the files pointed by the two FILE*. The starting position
 *  of the elements read is specified by the offset.
 * Return value:
 *  -Returns a pointer pointing to the start of the above mentioned space
 *  in the heap, that containes (first) all the labels and (then) all the
 *  attributes of the asked elements.
 *  -NULL, if something goes wrong this function will return NULL and print
 *  an appropriate error message on the error stream (stderr).
 * Memory Map (continuous memory):
 *  Returned value -> |1st label|2nd label|...|Nth label|1st array of attr|
 *      |2nd array of attr|...|Nth array of attr|
 *  where N is the number_of_elements and the size of the attribute arrays 
 *  is number_of_attributes. Every label is an integer and the attribute
 *  arrays are double arrays. Therefore, the total size allocated in the heap
 *  and pointed by the returned value will be: 
 *  N*sizeof(int)+N*number_of_attributes*sizeof(double).
 */
void* loadElements(FILE* label_fp, FILE* data_fp, int number_of_elements, int number_of_attributes, int offset);

/**
 * Method: loadMyElements()
 * Description:
 *  This function allocates and initializes an array of type <element>.
 *  It uses the method <loadElements> to load the data from the files pointed
 *  by the FILE* and specified by the other 3 <int> arguments on the heap. Then,
 *  it initializes every <element> of the array: The *label and the *attributes
 *  of each element will now point at the right place on the heap.
 * Return value:
 *  -Returns a pointer to the allocated and initialized element array.
 *  -NULL if an error occurs. (check stderr for error messages)
 */ 
element* loadMyElements(FILE* data_fp, FILE* label_fp, int number_of_elements, int number_of_attributes, int offset);

/**
 * Method: prepareBuffer()
 * Description:
 *  This function allocates and initializes an array of type <element> as
 *  <loadMyElements()> method does with the difference that the memory 
 *  allocated does not contain any useful data. The return value should be
 *  used as a buffer for saving element arrays' data.
 * Return value:
 *  -Return a pointer to the allocated and initialized element array.
 *  -NULL if an error occurs. (check stderr for error messages)
 */
element* prepareBuffer(int number_of_elements, int number_of_attributes);

/**
 * Method: swapElementP()
 * Description:
 *  Swaps the element arrays at position one and two.
 */
void swapElementP(element **one, element **two);

/**
 * Method: iMax()
 * Description:
 *  Finds the position of the max value of the int* array of size <size>.
 * Return Value:
 *  Returns the index of the max value of the array.
 */
int iMax(int* array, int size);

/**
 * Method: initializeKNNTable()
 * Description:
 *  This method allocates memory for a 2D double array and initializes all
 *  of its values to DBL_MAX.
 * Return Value:
 *  -Returns a pointer to the 2D allocated array.
 *  -NULL, if it couldn't allocate the asked memory (rows*col*sizeof(double)).
 *      (check stderr for error messages)
 */
double** initializeKNNTable(int rows, int col);

/**
 * Method: initializeLabelKNNTable()
 * Description:
 *  This method allocates memory for a 2D int array and initializes all
 *  of its values to -1.
 * Return Value:
 *  -Returns a pointer to the 2D allocated array.
 *  -NULL, if it couldn't allocate the asked memory (rows*col*sizeof(double)).
 *      (check stderr for error messages)
 */
int** initializeLabelKNNTable(int rows, int col);

/**
 * Method: findKNN()
 * Description:
 *  Updates the KNN tables of the distances and labels (firts 2 arguments).
 *  Checks the distances between the attributes of the "other_elements" and
 *  and each of the "my_elements". If it finds an element that is close enough
 *  to be in the distance table, it updates that table with the new distance and
 *  the label table with the label of that element.
 *  The sizes of the argument' tables are specified by the three <int> arguments
 */
void findKNN(double** distKNN, int** labelKNN, int lines, int k, int attr_number, element* my_elements, element* other_elements);

#endif