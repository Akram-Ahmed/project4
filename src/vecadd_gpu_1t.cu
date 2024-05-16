#include <iostream>
#include <math.h>


// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++){
        y[i] = x[i] + y[i];
    }
    
}

int main(void)
{
    int N = 1<<29; // 1M elements
    std::cout << "Doing vector addition of " << N << " elements." << std::endl; 
    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
   
 // initialize x and y arrays on the host
    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // run kernal on 1M elements on the CPU   
    add<<<1,1>>>(N, x, y); // launch 1 gpu per thread

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
