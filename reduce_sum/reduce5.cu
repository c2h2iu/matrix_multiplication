#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>


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
	if(gridDim.x > 1)    
	    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
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




void reduce_sum(unsigned int& elements){
    unsigned int numBlocks = 0;
    int numThreads = 256;

    unsigned int size = elements * sizeof(unsigned int);
    
    std::vector<unsigned int> h_in;
    std::vector<unsigned int> h_out(elements, 0);

    for(unsigned int i = 0; i < elements; ++i)    h_in.push_back(i%2);
    unsigned int gold_sum = elements/2;

    //auto start = std::chrono::steady_clock::now();

    unsigned int* d_in;
    unsigned int* d_out;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);
    
    //auto start = std::chrono::steady_clock::now();
    while(elements > 1){
	if(elements < numThreads){
            numBlocks = 1;
	    numThreads = elements;
	}
	else    numBlocks = (elements - 1) / numThreads  + 1;

	std::cout << "elemnts = " << elements << '\n';
	kernel5<<<numBlocks, numThreads, numThreads * sizeof(unsigned int)>>>(d_in, d_out, elements);

        elements = numBlocks/2;
	
	d_in = d_out;
    }	

    //auto end = std::chrono::steady_clock::now();
    cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    
    assert(h_out[0] == gold_sum);

    //std::cout << "Elapsed time in milliseconds : " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << '\n';
}





int main(int argc, char* argv[]){ 
    unsigned int elements = std::stoi(argv[1]);
    
    reduce_sum(elements);

    return 0;
}

