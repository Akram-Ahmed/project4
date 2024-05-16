#include <iostream>
#include <math.h>


// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float *x, float *y) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n; i+=stride)
    {
        y[i] = x[i] + y[i];
    } 
}

int main(void)
{
    int N = 1<<29; // 1M elements
    int blockSize = 256;
    int numBlocks = (N+blockSize-1) / blockSize;

    std::cout << "Calculating vector addition of a " << N << " elements using "<< blockSize << " threads for each block, and " << numBlocks << " blocks."<< std::endl; 

    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
   
 // initialize x and y arrays on the host
    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // run kernal on 1M elements on the CPU   
    add<<<numBlocks,blockSize>>>(N, x, y); //256 gpu threads

    // Wait for GPU to finish.
    cudaDeviceSynchronize();

    // Check for errors
    float maxError = 0.0f;
    for(int i=0; i<N; i++) {
        maxError=fmax(maxError, fabs(y[i]-3.0f));
    }

    std::cout << "Max Error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}
