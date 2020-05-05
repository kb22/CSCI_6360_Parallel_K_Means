#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#define X 1049088     // X Total Number of pixels in the image, to be given as an input. 
#define Y 3			  // No change required basically the RBG components of an image
#define K 4			  // NUMBER OF CLUSTERS TO DIVIDE THE DATA INTO
#define MAX_ITERS 10  // NUMBER OF ITERATIONS in k K-Means
double *num;		  // pointer to all the data to be stored
double* centroids;	  // pointer to the data of centroids
int* idx;			  // pointer to data  which stores the index of the centroid nearest to each pixel

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

void findclosestcentroids(double *num, double *centroids, int* idx){

	int i, j, l, min_ind; 
	double sum, dist[K],min_dist;
	for (i=0;i<X;i++){
		for (j=0;j<K;j++){
			sum=0;
			for (l=0;l<Y;l++){

				sum=sum+(*(num+i*Y+l)-*(centroids+j*Y+l))*(*(num+i*Y+l)-*(centroids+j*Y+l));
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
	}
}



// This function is used to update the centroids in each iteration. this is basically a reduction function where the
// mean of all the data points belonging to one cluster is calculated.

void computeCentroids(double* num, int* idx, double* centroids){

	int i,j, l, m, count;
	double sum[Y];
	for(i=0;i<Y;i++) sum[i]=0.0;

	for (i=0;i<K;i++){

		count=0;
		for(m=0;m<Y;m++) sum[m]=0.0;

		for(j =0; j<X; j++){

			if(idx[j]==i){

					count++;
					for (l=0;l<Y;l++){

						sum[l]=sum[l]+ *(num+j*Y+l);
					
					}
			
			}

		}
		//printf("COunts is %d \n", count);
		for (l=0;l<Y;l++){

			*(centroids+i*Y+l)=sum[l]/count;					
		}
	} 

}
void main(){

	// Define variables to keep track of time
	unsigned long long start = 0;
	unsigned long long finish = 0;

	FILE *fp, *fw;
	double num1;
	int i, j,k,rnd_num;
	num=(double*)calloc(X*Y, sizeof(double));
	centroids=(double*)calloc(K*Y,sizeof(double));
	idx=(int*)calloc(X,sizeof(int));
	//int    idx[X];

	//Upper and lower bounds for the random numbers 
	int lower=0;
	int upper =X-1;
	
	srand(time(0));
	
	//File open for reading

	fp=fopen("input.txt","r");
	if(fp==NULL) {
		printf("Exiting no file with such name \n");
		exit(-1);
	}
	//printf("Hello World \n");
	for (i=0;i<X;i++){
		for (j=0;j<Y;j++){
			fscanf(fp,"%lf", &num1);
			*(num+i*Y+j)=num1;
		}
	}
	fclose(fp);

	// Starting K-Means clustering

	// Let us start calculation of time
	start = getticks();


	//Creation of random number
	for (i = 0; i < K; i++) {

			rnd_num = (rand()%(upper-lower + 1)) + lower;
			//printf("%d ", rnd_num);  
			for (j=0;j<Y;j++){ 
        		*(centroids+i*Y+j) = *(num+rnd_num*Y+j); 
        	} 
        //printf("\n");
    }

	//printf("Double deref %e Single deref %p NO deref %p\n",**centroids, *centroids, centroids);

    //run the loop for a fixed number of iterations

	for (i=0; i<MAX_ITERS; i++){

		findclosestcentroids((double *)num, (double *)centroids, &idx[0]);
		computeCentroids((double *)num, &idx[0], (double *)centroids);

	}

	
	for(i=0; i<K;i++){
			for(j=0; j<Y;j++){

				printf("Centroids are %lf  ",*(centroids+i*Y+j));

			}
		printf("\n");
	}
	

	//Repalcement of each pixel in the image by the centroid it is closest to. This is the step that quantizes the image.

	for (i=0; i<X;i++){
		//printf("%d==%d\n",i+1, idx[i]+1);

		for (k=0;k<K;k++){

			if (idx[i]==k){

					for (j=0;j<Y;j++){			
						*(num+i*Y+j)=*(centroids+k*Y+j);
					}
			}
				
		}

	}

	// KMeans algorithm completed

	// Close timer and print total time taken
	finish = getticks();
	printf("Total time taken to run K-Means on %d pixel image with %d clusters is %e seconds.\n", X, K, (finish-start)/512000000.0f);


	//Writing the data to the output file
	fw=fopen("output.txt","w");
	
	for(i=0; i<X;i++){
	
		for(j=0; j<Y;j++){

				fprintf(fw,"%lf  ",*(num+i*Y+j));
				//	printf("%lf  ",num[i][j]);
			}
		fprintf(fw, "\n");
		//printf("\n");
	}
	fclose(fw);
	free (num);
	free (centroids);
	free(idx);
	
}
