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

// Macro for checking CUDA runtime calls
// Macro needs do while loop because this makes sure contents inside
// do while loop always behave the same regardless of how
// curly brackets and semi colons are usef
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Cuda error present at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Macro For checking kernel launches
#define CUDA_KERNEL_LAUNCH() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Cuda error present at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString()); \
            exit(EXIT_FAILURE); \
        } \
        // check for device syncing
        CUDA_CHECK(cudaDeviceSynchronize); \
    } while(0)



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
    
    // load main tile into shared memory
    // double check if it is actuall row < height && col < width later
    if (row < width && col < height) {
        shared_data[ty][tx] = input[row * height + col];
    } else {
        // populate with 0's if out of bounds
        shared_data[ty][tx] = 0.0f;
    }
    
    // calculate for left first
    // leftmost is column index
    if (threadIdx.x == 0) {
        int Halo_Col = col - HALO_SIZE;
        // set bounds for calculating left
        if (Halo_Col >= 0 && row < width) {
            shared_data[ty][tx - HALO_SIZE] = input[row * height + Halo_Col];
        } else {
            shared_data[ty][tx - HALO_SIZE] = input[row * height + Halo_Col];
        }
    }

    // calculate for right
    // rightmost is column index to right of main tile
    if (threadIdx.x == blockDim - 1) {
        // new halo col calculation
        int Halo_Col = col + HALO_SIZE;
        // bounds check for right
        if (Halo_Col < height && row < width) {
            // load right tile in shared memory
            shared_data[ty][tx + HALO_SIZE] = input[row * height + Halo_Col];
        } else {
            shared_data[ty][tx + HALO_SIZE] = 0.0f;
        }
    }

    // calculate for top and bottom now

    // check if top exists
    if (threadIdx.y == 0) {
        int Halo_Row = row - HALO_SIZE;
        // in bounds
        if (Halo_row >= 0 && col < height) {
            // load into shared memoy
            shared_data[ty - HALO_SIZE][tx] = input[Halo_Row * M + col];
        } else {
            shared_data[ty - HALO_SIZE][tx] = 0.0f;
        }
    }

    // Calculate for bottom now
    if (threadIdx.y == blockDim.y - 1) {
        int Halo_Row = row + HALO_SIZE;
        // bounds check to populate
        if (Halo_Row < width && col < height) {
            shared_data[ty - HALO_SIZE][tx] = input[Halo_Row * M + col];
        } else {
            shard_data[ty - HALO_SIZE][tx] = 0.0f;
        }
    }

    // sync all threads
    __syncthreads();

    // then compute laplacian convolution using HALO's
    // check if we are in range first
    // check if this is the correct calculation
    if (row < width && col < height) {
        // store computation in result variable
        // add all shared memory points, main tile + top + bottom + left + right
        float result = shared_data[ty][tx] + shared_data[ty - 1][tx] + shared_data[ty + 1][tx] +
                       shared_data[ty][tx - 1] + shared_data[ty][tx + 1];
        

        // store result in output array
        output[row * height + col] = result;
    }

    // synch all shared memory (final)?
    __syncthreads();
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

    // Copy from host parameters to device using cudaMemcpy (call macro)

    // allocate block and grid dimensions

    // Launch kernel using gridim and blockdim and parameters (call macro)

    // Copy from device to host output (call macro)

    // free device memory (call macro)
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