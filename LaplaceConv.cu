#include <iostream>
#include <cuda_runtime.h>

/**
 * Plan:
 * 
 * Kernel That implements laplace convolution:
 * 
 * 
 * Helper function for kernel launch:
 *   - Pass initialized host memory as parameters
 *   - Device memory allocation
 *   - Copying from host to device
 * 
 * Main function:
 *  - Will contain opencv image loading
 *  - Host memory allocation and helper function
 *    calling
 * 
 * 
 * 
*/

/**
 * Define compile time constants, will need for shared memory,
 * halo regions, and shared memory tile size
*/
#define TILE_SIZE 16
#define HALO_SIZE 1
#define SHARED_SIZE (TILE_SIZE + 2 * HALO_SIZE)

// Error checking constant Macros



// Laplacian Kernel
// width and height are the dimensons of the image
__global__ LaplacianKernel(float* input, float* output, int width, int height) {
    // Load constants and Halo info into shared memory
    __shared__ float shared_data[SHARED_SIZE][SHARED_SIZE];

    // initialize row and col using thread and block
    // (Reminder it is 2 dimensions)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute shared memory coordinates (where to store the shared memory)
    int ty = threadIdx.y + HALO_SIZE
    int tx = threadIdx.x + HALO_SIZE:

    // Apply the 3x3 laplacian kernel, (consists of calculating halo regions)
    // Implement like: left, right, top, bottom

    // then compute laplacian convolution using HALO's
}

/**
 * Cuda kernel launch helper
 * represent input and output messages as mat arrays. Pass them by address
 * Will mainly contain device memory implementation
 * inputImg and outputImg are the host memory
*/
void runLaplacian(cv::Mat& inputImg, cv::Mat& outputImg) {
    // Calculate Memory Size

    // Allocate Device memory using cudaMalloc

    // Copy from host parameters to device using cudaMemcpy

    // allocate block and grid dimensions

    // Launch kernel using gridim and blockdim and parameters

    // Error checking

    // Copy from device to host output

    // free device memory
}

/**
 * Main function:
 * Will be used for cv image processing and cv intertwined host memory 
 * Will utilize command line arguments
*/
int main(int argc, char** argv) {
    // Check command line arguments

    // Load greayscale images using OpenCv, perhaps have images
    // in same directory

    // check if image loaded successully

    // convert to flag32 and normalize

    // create output matrix with same dimensions

    // call helper function

    // convert output to uint8 and denormalize

    // save the result image

    // perhaps display images for documentation

    return 0;
}