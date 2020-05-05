#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<cuda.h>
#include<cuda_runtime.h>

#define Y 3					
#define K 4				//NUMBERR OF CLUSTERS


// This kermel is used to find the index of the centroid that is nearest to each pixel.
//Ech thread is responsible for a particular pixel

__global__
void findclosestcentroids(double* num, double* centroids_c, int* idx, int each_chunk){

	int index=blockIdx.x*blockDim.x+threadIdx.x;
	int stride=blockDim.x*gridDim.x;
	int offset=0; //offset keeps track if the same thread (number enters the loop the next time, as the thread id will be same)
	for(int i=index; i<each_chunk; i+=stride){
		
		int x=index+offset*stride;
		int j, l, min_ind; 
		double sum, dist[K],min_dist;
		
		for (j=0;j<K;j++){
			
			sum=0;
			for (l=0;l<Y;l++){

					sum=sum+(*(num+x*Y+l)-*(centroids_c+j*Y+l))*(*(num+x*Y+l)-*(centroids_c+j*Y+l));

			}
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
		*(idx+x)=min_ind;
		offset++;
	}
	
}


// This kernel is launched  to update the centroids in each iteration. this is basically a reduction function where the
// mean of all the data points belonging to one cluster is calculated.

__global__
void computeCentroids(double* num, int* idx, double* centroids_c, int each_chunk){

	
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	int stride=blockDim.x*gridDim.x;
	int offset=0; //offset keeps track if the same thread (number enters the loop the next time, as the thread id will be same)
	
	int  m, j, l, count;
	double sum[Y]; 					//for(i=0;i<Y;i++) sum[i]=0.0;//is it reqd ?
	for(int i=index; i<K; i+=stride){

		int x=index+offset*stride;
		count=0;
		for(m=0;m<Y;m++) sum[m]=0.0;

		for(j =0; j<each_chunk; j++){

			if(idx[j]==x){

					count++;
					for (l=0;l<Y;l++){

						sum[l]=sum[l]+ *(num+j*Y+l);
					
					}
			
			}

		}
		if (count==0) continue;
		//printf("Counts is %d \n", count);
		for (l=0;l<Y;l++){

			*(centroids_c+x*Y+l)=sum[l]/count;					
		}
	}

}


//Kernel that performs the repalcement of each pixel in the image by the centroid it is closest to. 
//This is basically the step that quantizes the image.	


__global__
void assign_thru_gpu(double* num, double* centroids_c, int* idx, int each_chunk){
	
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	int stride=blockDim.x*gridDim.x;
	int offset=0; //offset keeps track if the same thread number enters the loop the next time, as the thread id will be same
	for(int i=index; i<each_chunk; i+=stride){
		
		int x=index+offset*stride;
		int i, j, k ; 

		for (k=0;k<K;k++){

			if (idx[x]==k){

					for (j=0;j<Y;j++){			
						*(num+x*Y+j)=*(centroids_c+k*Y+j);
					}
			}
				
		}
		offset++;
	}

}



//Assignment of each CUDA device to a particular rank

extern "C" void cuda_init(int each_chunk, int myrank, int numranks){

	int cudaDeviceCount=-1;
	cudaError_t cE;
	//Check if cuda device exists and get the number of  working cuda devices
    if ((cE=cudaGetDeviceCount( &cudaDeviceCount))!=cudaSuccess){
	    printf("Unable to determine cuda Device count, error is %d count is %d \n", cE, cudaDeviceCount);
	    exit (-1);
	}
	
	//Set cuda device for each MPI rank 	
	if ((cE=cudaSetDevice(myrank%cudaDeviceCount))!=cudaSuccess){
	    printf("Unable to have rank %d set to cuda device %d, error is %d \n", myrank, (myrank % cudaDeviceCount), cE);
	    exit (-1);
	}

	extern double* num;
	extern double* centroids_c;
	extern double* centroids_cresult;
	extern int* idx;

	num=NULL;
	centroids_c=NULL;
	centroids_cresult=NULL;
	idx=NULL; 

	cudaMallocManaged( &num, sizeof(double)*each_chunk*Y);
    cudaMallocManaged( &centroids_c, sizeof(double)*K*Y);
    cudaMallocManaged( &centroids_cresult, sizeof(double)*K*Y);
	cudaMallocManaged(&idx, sizeof(int)*each_chunk);


}
    
//function that initiates kernel launches from the main function.

extern "C" void k_means_kernel_launch(double* num, double* centroids_c, int* idx, int each_chunk, int n_blocks, int no_of_threads){
	
	int cudaDeviceCount;
	cudaError_t cE1,cE2, cE3;

	findclosestcentroids<<< n_blocks, no_of_threads>>>(num, centroids_c, idx, each_chunk);
		
	cE1=cudaGetDeviceCount( &cudaDeviceCount);
	cE2=cudaDeviceSynchronize();
	//printf("The two errors are %d %d \n",cE1,cE2);
	//const char* x_err=cudaGetErrorString (cE2);
	//printf("%s \n",x_err); 

	computeCentroids<<<1, 32>>>(num, &idx[0], centroids_c,each_chunk);

	cE3=cudaDeviceSynchronize();
	//printf("The error is %d\n",cE3);
	//x_err=cudaGetErrorString (cE3);
	//printf("%s \n",x_err); 

}

//Repalcement of each pixel in the image by the centroid it is closest to. This is the step that quantizes the image.	

extern "C" void assign(double* num, double* centroids_c, int* idx, int each_chunk, int n_blocks, int no_of_threads){
	
	assign_thru_gpu<<<n_blocks, no_of_threads>>>(num, centroids_c, idx, each_chunk);
	cudaDeviceSynchronize();

}


//Freeing the dynamic memory 

extern "C" void cuda_free(double* num, double* centroids_c, double* centroids_cresult, int* idx){
	
	cudaFree(num);
	cudaFree(centroids_c);
	cudaFree(centroids_cresult);
	cudaFree(idx);

}
