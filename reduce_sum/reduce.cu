#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>

int blockThreads{256};


/***
   first argument is the approach
   0: cpu_outplace, 1: cpu_inplace, 2: gpu_outplace, 3: gpu_inplace
   second argument is row
   third argument is column
***/

__device__ void warpReduce(volatile unsigned int* sdata, const unsigned int tid, const unsigned int elements){
    if(elements > 32)    sdata[tid] += sdata[tid + 32];
    if(elements > 16)    sdata[tid] += sdata[tid + 16];
    if(elements > 8)     sdata[tid] += sdata[tid + 8];
    if(elements > 4)     sdata[tid] += sdata[tid + 4];
    if(elements > 2)     sdata[tid] += sdata[tid + 2];
    if(elements > 1)     sdata[tid] += sdata[tid + 1];
}


__global__ void kernel5(unsigned int* d_in, unsigned int* d_out, const unsigned int elements){
    extern __shared__ unsigned int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * (blockDim.x * 2);

    sdata[tid] = 0;

    if(i < elements){
	 if(gridDim.x > 1)    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
         else    sdata[tid] = d_in[i];
    }
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 32; s >>= 1){
        if(tid < s)    sdata[tid] += sdata[tid + s];
	__syncthreads();
    }

    if(tid < 32)    warpReduce(sdata, tid, elements);

    if(tid == 0)    d_out[blockIdx.x] = sdata[0];
}


__global__ void kernel4(unsigned int* d_in, unsigned int* d_out, const unsigned int elements){
    extern __shared__ unsigned int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * (blockDim.x * 2);

    sdata[tid] = 0;
    if(i < elements){
        if(gridDim.x > 1)    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
        else    sdata[tid] = d_in[i];
    }
    __syncthreads();


    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s)    sdata[tid] += sdata[tid + s];
	__syncthreads();
    }

    if(tid == 0)    d_out[blockIdx.x] = sdata[0];
}



__global__ void kernel3(unsigned int* d_in, unsigned int* d_out, const unsigned int elements){
    extern __shared__ unsigned int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    sdata[tid] = (i < elements) ? d_in[i] : 0;
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s)    sdata[tid] += sdata[tid + s];
	__syncthreads();

    }
    if(tid == 0)    d_out[blockIdx.x] = sdata[0];
}



__global__ void kernel2(unsigned int* d_in, unsigned int* d_out, const unsigned int elements){
    extern __shared__ unsigned int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;


    sdata[tid] = (i < elements) ? d_in[i] : 0;
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s*=2){
	unsigned int index = 2 * s * tid;
        if(index < blockDim.x)    sdata[index] += sdata[index + s];
	__syncthreads();
    }

    if(tid == 0)    d_out[blockIdx.x] = sdata[0];
}


__global__ void kernel1(unsigned int* d_in, unsigned int* d_out, const unsigned int elements){
    extern __shared__ unsigned int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    sdata[tid] = (i < elements) ? d_in[i] : 0;
    __syncthreads();

    for(unsigned int s = 1; (s < blockDim.x) && (tid+s) < elements; s*=2){
        if(tid % (2*s) == 0)    sdata[tid] += sdata[tid + s];
	__syncthreads();

    }

    if(tid == 0)    d_out[blockIdx.x] = sdata[0];
}



void reduce_sum(const int approach, unsigned int& elements){
    unsigned int numBlocks = 0;
    int numThreads = 256;

    unsigned int originalElements = elements;

    unsigned int size = elements * sizeof(unsigned int);
    
    std::vector<unsigned int> h_in;
    std::vector<unsigned int> h_out(elements, 0);

    for(unsigned int i = 0; i < elements; ++i)    h_in.push_back(i%2);
    unsigned int gold_sum = elements/2;

    unsigned int* d_in;
    unsigned int* d_out;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);
    
    auto start = std::chrono::steady_clock::now();
    while(elements > 1){
	if(elements < numThreads){
            numBlocks = 1;
	    numThreads = elements;
	}
	else    numBlocks = (elements - 1) / numThreads + 1;


        if(approach == 1)    
	    kernel1<<<numBlocks, numThreads, numThreads * sizeof(unsigned int)>>>(d_in, d_out, elements);
        else if(approach == 2)    
	    kernel2<<<numBlocks, numThreads, numThreads * sizeof(unsigned int)>>>(d_in, d_out, elements);
        else if(approach == 3)
            kernel3<<<numBlocks, numThreads, numThreads * sizeof(unsigned int)>>>(d_in, d_out, elements);
        else if(approach == 4)
	    kernel4<<<numBlocks, numThreads, numThreads * sizeof(unsigned int)>>>(d_in, d_out, elements);
	else
	    kernel5<<<numBlocks, numThreads, numThreads * sizeof(unsigned int)>>>(d_in, d_out, elements);
	
        elements = numBlocks;

        if(approach >= 4)    elements = elements / 2;
	
	d_in = d_out;
    }	
    auto end = std::chrono::steady_clock::now();
    
    cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    
    assert(h_out[0] == gold_sum);

    std::cout << "reduce " << approach << " took " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << " ms to sum " << originalElements << " elements\n"; 
}



void reduce_sum_sequential(const unsigned int elements){
    std::vector<unsigned int> h_in;
    
    unsigned int gold_sum, sequential_sum;

    for(unsigned int i = 0; i < elements; ++i)    h_in.push_back(i%2);
    gold_sum = elements / 2;
    
    auto start = std::chrono::steady_clock::now();
    for(auto i : h_in)    sequential_sum += i;
    auto end = std::chrono::steady_clock::now();
    
    assert(gold_sum == sequential_sum);

    std::cout << "sequential reduce took " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << " ms to sum " << elements << " elements\n"; 
}



int main(int argc, char* argv[]){ 
    int approach = std::stoi(argv[1]);
    unsigned int elements = std::stoi(argv[2]);
    
    if(approach == 0)    reduce_sum_sequential(elements);
    else    reduce_sum(approach, elements);



    return 0;
}

