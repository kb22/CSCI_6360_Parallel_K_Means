#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<time.h>
#include<mpi.h>
#include<string.h>
#include<math.h>

#define X 1049088				// Total Number of pixels in the image, to be given as an input. 
#define Y 3						// No change required basically the RBG components of an image
#define K 4						// NUMBER OF CLUSTERS TO DIVIDE THE DATA INTO
#define MAX_ITERS 10  			// NUMBER OF ITERATIONS

double *num=NULL;				 // pointer to all the data to be stored
double *centroids_c=NULL;		 // pointer to the data of centroids
double *centroids_cresult=NULL;	
int *idx=NULL;					 // pointer to data  which stores the index of the centroid nearest to each pixel


typedef unsigned long long ticks;


// Function to calculate time as given ny Professor
static __inline__ ticks getticks(void)
{
  unsigned int tbl, tbu0, tbu1;

  do {
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
    __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
  } while (tbu0 != tbu1);

  return (((unsigned long long)tbu0) << 32) | tbl;
}


// This function is used to find the index of the centroid that is nearest to each pixel.

void findclosestcentroids(double *num, double *centroids_c, int* idx, int each_chunk){

	int i, j, l, min_ind; 
	double sum, dist[K],min_dist;
	
	for (i=0;i<each_chunk;i++){
		for (j=0;j<K;j++){
			sum=0;
			for (l=0;l<Y;l++){

				sum=sum+(*(num+i*Y+l)-*(centroids_c+j*Y+l))*(*(num+i*Y+l)-*(centroids_c+j*Y+l));
			}
			//printf("Distance %e\n",sum);
			dist[j]=sqrt(sum);

		}
		min_dist=dist[0];
		min_ind=0;
		for (j=0; j<K; j++){
			if (dist[j]<min_dist) {
				min_dist=dist[j];
				min_ind=j;
			}
		}
	*(idx+i)=min_ind;
	//printf("Index is %d \n",*(idx+i));
	}

}


// This function is used to update the centroids in each iteration. this is basically a reduction function where the
// mean of all the data points belonging to one cluster is calculated.


void computeCentroids(double* num, int* idx, double* centroids_c, int each_chunk){

	int i,j, l, m, count;
	double sum[Y];
	for(i=0;i<Y;i++) //sum[i]=0.0;

	for (i=0;i<K;i++){

		count=0;
		for(m=0;m<Y;m++) sum[m]=0.0;

		for(j =0; j<each_chunk; j++){

			if(idx[j]==i){

					count++;
					for (l=0;l<Y;l++){

						sum[l]=sum[l]+ *(num+j*Y+l);
						//printf("Sum is %e \n",sum[l]);
					
					}
			
			}

		}
		if (count==0) continue; 
		for (l=0;l<Y;l++){

			*(centroids_c+i*Y+l)=sum[l]/count;
			//printf ("Centroids counts  %e %d ",*(centroids_c+i*Y+l), count );							
		}

	//printf("\n");
	} 

}

int main(int argc, char *argv[]){

        // Define variables to keep track of time
                 unsigned long long start = 0;
   unsigned long long finish = 0;

	int myrank, numranks, result;
	int i,each_chunk,j,k;
    double starttime, endtime;
   
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Request request1, request2, request3, request4; 
    MPI_Status status;
    MPI_File fh;


    //Determination of each chunk of the file each process is going to read

    each_chunk=X/numranks;
	if(myrank==numranks-1)
		each_chunk=each_chunk+X%numranks;
	//printf("Each chunk %d \n", each_chunk);

    num=(double*)calloc(each_chunk*Y, sizeof(double));
	centroids_c=(double*)calloc(K*Y,sizeof(double));
	centroids_cresult=(double*)calloc(K*Y,sizeof(double));
	idx=(int*)calloc(each_chunk,sizeof(int));
	


	result=MPI_File_open(MPI_COMM_WORLD, "input.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
		if(result != MPI_SUCCESS) {printf("Error in opening the file\n"); exit(-1);}

	result=MPI_File_read_at(fh, myrank*each_chunk*Y*sizeof(double), num, each_chunk*Y, MPI_DOUBLE, &status);
		if(result != MPI_SUCCESS) {printf("Error in reading the file\n"); exit(-1);}
	
		/*
		if (myrank==1)	
			for (i=each_chunk;i<each_chunk*Y;i++){
				printf("The read numbers are %e \n", *(num+i));
			}
		*/

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_File_close(&fh);

	if (myrank==0)
	{	
    // Let us start calculation of time
                start = getticks();
		int lower =0;
		int upper =each_chunk-1;
		srand(time(0));


		//Random initialization of centroids 
		for (int i = 0; i < K; i++) {

			int rnd_num = (rand()%(upper-lower + 1)) + lower;
			//printf("%d ", rnd_num);  
			
			for (int j=0;j<Y;j++){ 
        		*(centroids_c+i*Y+j) = *(num+rnd_num*Y+j);
        		//printf("Centroids are %e",*(centroids_c+i*Y+j)); 
        	} 
        //printf("\n");
    	}

	}

	// Broadcast of the initialized centroids to all ranks from rank 0

	MPI_Bcast(centroids_c, K*Y, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 
    for (i=0;i<MAX_ITERS;i++){

	    findclosestcentroids((double *)num, (double *)centroids_c, &idx[0],each_chunk);
	    computeCentroids((double *)num, &idx[0], (double *)centroids_c, each_chunk);
	    MPI_Barrier(MPI_COMM_WORLD);
	    MPI_Allreduce(centroids_c, centroids_cresult, K*Y, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	    	for (int a=0; a<K;a++){
			
				for (int b=0;b<Y;b++){

					*(centroids_c+a*Y+b)=*(centroids_cresult+a*Y+b)/numranks;
				}
			}

    }
    
	if(!myrank)
    for (i=0; i<K;i++){
		
		for (k=0;k<Y;k++){

			printf ("Centroids %d new %e", myrank, *(centroids_c+i*Y+k)/numranks);
		}
		printf("\n");
	}
    	
    
    //Repalcement of each pixel in the image by the centroid it is closest to. This is the step that quantizes the image.	

    for (i=0; i<each_chunk;i++){
		//printf("%d==%d\n",i+1, idx[i]+1);

		for (k=0;k<K;k++){

			if (idx[i]==k){

					for (j=0;j<Y;j++){			
						*(num+i*Y+j)=*(centroids_cresult+k*Y+j)/numranks;
					}
			}
				
		}

	}


        if (myrank == 0) {
                // Close timer and print total time taken
           finish = getticks();
           printf("Total time taken to run K-Means on %d pixel image with %d clusters is %lf seconds.\n", X, K, (finish-start)/512000000.0f);
           }
     
    //Each process writes its own chunk of data to a particular file.

   	result=MPI_File_open(MPI_COMM_WORLD, "output.bin",  MPI_MODE_CREATE|MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
		if(result != MPI_SUCCESS) {printf("Error in opening the file\n"); exit(-1);}

	result=MPI_File_write_at(fh, myrank*each_chunk*Y*sizeof(double), num, each_chunk*Y, MPI_DOUBLE, &status);
		if(result != MPI_SUCCESS) {printf("Error in writing the file\n"); exit(-1);}


	MPI_Barrier(MPI_COMM_WORLD);
	MPI_File_close(&fh);

    MPI_Finalize();
    free(num);
    free(centroids_c);
    free(centroids_cresult);
    free(idx);
	return 0;
}
