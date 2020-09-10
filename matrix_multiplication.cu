#include <iostream>
#include <vector>
#include <string>
#include <taskflow/taskflow.hpp>


int dimension;


/***
   first argument is the approach
       0: serial, 1: gpu without shared memory, 2: gpu with shared memory
       3: taskflow with parallel_for, 4: taskflow with cuda 
   second argument is the dimension
***/


void matrix_multiplication_sequential(){
    std::vector<std::vector<int>> h_A(dimension, std::vector<int>(dimension, 1));
    std::vector<std::vector<int>> h_B(dimension, std::vector<int>(dimension, 1));
    std::vector<std::vector<int>> h_C(dimension, std::vector<int>(dimension, 0));

    for(int i = 0; i < dimension; ++i){
        for(int j = 0; j < dimension; ++j){
            for(int k = 0; k < dimension; ++k)
                h_C[i][j] += h_A[i][k] * h_B[k][j];
	}
    }
}



void matrix_multiplication_parallel_for(){
    std::vector<std::vector<int>> h_A(dimension, std::vector<int>(dimension, 1));
    std::vector<std::vector<int>> h_B(dimension, std::vector<int>(dimension, 1));
    std::vector<std::vector<int>> h_C(dimension, std::vector<int>(dimension, 0));

    tf::Taskflow taskflow;
    tf::Executor executor;

    tf::Task task = taskflow.for_each_index(0, dimension, 1, [&] (int i){
        for(int j = 0; j < dimension; ++j){
            for(int k = 0; k < dimension; ++k){
                h_C[i][j] += h_A[i][k] * h_B[k][j];
            }
        }
    });
    executor.run(taskflow).wait();
}


__global__ void kernel_global_memory(int* d_A, int* d_B, int* d_C, const int dimension){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int result = 0;

    for(int i = 0; i < dimension; ++i)
        result += d_A[y * dimension + i] * d_B[i * dimension + x];

    d_C[y * dimension + x] = result;
}


__global__ void kernel_shared_memory(int* d_A, int* d_B, int* d_C, const int dimension){
    __shared__ int s_d_A[32][32];
    __shared__ int s_d_B[32][32];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int result = 0;

    for(int i = 0; i < dimension/32; ++i){
        s_d_A[threadIdx.y][threadIdx.x] = d_A[y * dimension + (i * 32 + threadIdx.x)];
	s_d_B[threadIdx.y][threadIdx.x] = d_B[(i * 32 + threadIdx.y) * dimension + x];

	__syncthreads();

	for(int i = 0; i < 32; ++i)
	    result += s_d_A[threadIdx.y][i] * s_d_B[i][threadIdx.x];
	__syncthreads();
    }

    d_C[y + dimension + x] = result;
}



void matrix_multiplication_gpu(const int approach){
    int size = dimension * dimension * sizeof(int);
    std::vector<int> h_A(dimension*dimension, 1);
    std::vector<int> h_B(dimension*dimension, 1);
    std::vector<int> h_C(dimension*dimension, 0);

    int* d_A;
    int* d_B;
    int* d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(32, 32);
    dim3 dimGrid(dimension/32, dimension/32);

    if(approach == 1)    kernel_global_memory<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, dimension);
    if(approach == 2)    kernel_shared_memory<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, dimension);

    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    /***
    for(int i = 0; i < dimension*dimension; ++i){
	if(i%dimension == 0)    std::cout << '\n';
        std::cout << h_C[i] << ' ';
    }
    ***/
}



void matrix_multiplication_taskflow_gpu(){
    tf::Taskflow taskflow;
    tf::Executor executor;
    std::vector<int> h_A, h_B, h_C;
    int* d_A;
    int* d_B;
    int* d_C;

    auto allocate_a = taskflow.emplace([&](){
	h_A.resize(dimension*dimension, dimension+dimension);
        TF_CHECK_CUDA(cudaMalloc(&d_A, dimension*dimension*sizeof(int)), "failed to allocate a");	
    }).name("allocate_a");

    auto allocate_b = taskflow.emplace([&](){
	h_B.resize(dimension*dimension, dimension+dimension);
        TF_CHECK_CUDA(cudaMalloc(&d_B, dimension*dimension*sizeof(int)), "failed to allocate b");	
    }).name("allocate_b");

    auto allocate_c = taskflow.emplace([&](){
	h_C.resize(dimension*dimension, dimension+dimension);
        TF_CHECK_CUDA(cudaMalloc(&d_C, dimension*dimension*sizeof(int)), "failed to allocate c");	
    }).name("allocate_c");

    auto cudaFlow = taskflow.emplace([&](tf::cudaFlow& cf){
        auto copy_da = cf.copy(d_A, h_A.data(), dimension*dimension).name("HostToDevice_a");
        auto copy_db = cf.copy(d_B, h_B.data(), dimension*dimension).name("HostToDevice_b");
        auto copy_hc = cf.copy(h_C.data(), d_C, dimension*dimension).name("DeviceToHost_c");
    
	dim3 dimGrid(dimension/32, dimension/32);
	dim3 dimBlock(32, 32);

	auto kmatmul = cf.kernel(dimGrid, dimBlock, 0, kernel_global_memory, d_A, d_B, d_C, dimension).name("matmul");

	kmatmul.succeed(copy_da, copy_db).precede(copy_hc);
    }).name("cudaFlow");

    auto free = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(d_A), "failed to free d_A");
        TF_CHECK_CUDA(cudaFree(d_B), "failed to free d_B");
        TF_CHECK_CUDA(cudaFree(d_C), "failed to free d_C");	
    }).name("free");

    cudaFlow.succeed(allocate_a, allocate_b, allocate_c).precede(free);
    executor.run(taskflow).wait();
}



int main(int argc, char* argv[]){ 
    int approach = std::stoi(argv[1]);
    dimension = std::stoi(argv[2]);

    if(approach == 0)    matrix_multiplication_sequential();
    if(approach == 1)    matrix_multiplication_gpu(approach);
    if(approach == 2)    matrix_multiplication_gpu(approach);
    if(approach == 3)    matrix_multiplication_parallel_for();
    if(approach == 4)    matrix_multiplication_taskflow_gpu();


    return 0;
}

