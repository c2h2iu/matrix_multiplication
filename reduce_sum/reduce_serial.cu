#include <iostream>
#include <vector>
#include <chrono>




void reduce_sum(const unsigned int elements){
    std::vector<unsigned int> h_in;
    unsigned int gold_sum = 0;

    for(unsigned int i = 0; i < elements; ++i)    h_in.push_back(i%2);
    
    auto start = std::chrono::steady_clock::now();
    for(auto i : h_in)    gold_sum += i;
    auto end = std::chrono::steady_clock::now();
   
    std::cout << elements << " elements "; 
    std::cout << "took " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " ms\n";
    //std::cout << gold_sum;
}



int main(int argc, char* argv[]){ 
    unsigned int elements = std::stoi(argv[1]);
    
    reduce_sum(elements);

    return 0;
}

