#include <iostream>
#include <math.h>
#include <unistd.h>
#include <chrono>

// function to add the elements of two arrays
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++){
        y[i] = x[i] + y[i];
    }
    
}

int main(int argc, char const *argv[]){
    int N = 1<<29; // 1M elements
    std::cout << "Doing vector addition of " << N << " elements." << std::endl;
    float *x = new float[N];
    float *y = new float[N];

    // initialize x and y arrays on host
    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // run kernal on 1M elements on the CPU
    // start timer
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
    add(N, x, y);
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for(int i=0; i<N; i++) {
        maxError=fmax(maxError, fabs(y[i]-3.0f));
    }
    std::cout << "Max Error: " << maxError << std::endl;
    std::cout << "Elapsed time : " << elapsed.count()*1000 << " ms" << std::endl;

    delete []x;
    delete []y;

    return 0;
}
