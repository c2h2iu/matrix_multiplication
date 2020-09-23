#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>




__global__ void kernel2(unsigned int* d_in, unsigned int* d_out, const unsigned int elements){
    extern __shared__ unsigned int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    sdata[tid] = 0;

    if(i < elements)    sdata[tid] = d_in[i];
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s*=2){
	unsigned int index = 2 * s * tid;
        if(index < blockDim.x)    sdata[index] += sdata[index + s];
	__syncthreads();

    }

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
	else    numBlocks = (elements - 1) / numThreads + 1;

	//std::cout << "elemnts = " << elements << '\n';
        auto start = std::chrono::steady_clock::now();
	kernel2<<<numBlocks, numThreads, numThreads * sizeof(unsigned int)>>>(d_in, d_out, elements);
        auto end = std::chrono::steady_clock::now();

	std::cout << elements << " elements ";
        std::cout << "took " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " ms\n";
        elements = numBlocks;
	d_in = d_out;
    }	
    //auto end = std::chrono::steady_clock::now();

    cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    //for(int i = 0; i < h_out.size(); ++i)    std::cout << h_out[i] << ' ';
    assert(h_out[0] == gold_sum);

    //std::cout << "Elapsed time in microseconds : " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << '\n';
}




int main(int argc, char* argv[]){ 
    unsigned int elements = std::stoi(argv[1]);

    reduce_sum(elements);

    return 0;
}

