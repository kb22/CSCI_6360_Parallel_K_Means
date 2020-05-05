all: k_cpart.c k_cudapart.cu
	mpicc -g k_cpart.c -c -o k_cpart.o
	nvcc -O3 -gencode arch=compute_70,code=sm_70 k_cudapart.cu -c -o k_cudapart.o
	mpicc -g k_cpart.o k_cudapart.o -o k_all_exe -L /usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++