#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

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
//#define CUDA_CHECK(call) \
   // do { \
       // cudaError_t err = call; \
       // if (err != cudaSuccess) { \
            //fprintf(stderr, "Cuda error present at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
           // exit(EXIT_FAILURE); \
       // } \
   // } while(0)

// Macro For checking kernel launches
//#define CUDA_KERNEL_LAUNCH() \
   // do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Cuda error present at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Cuda error present at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)



// Laplacian Kernel
// width and height are the dimensons of the image
__global__ void LaplacianKernel(float* input, float* output, int width, int height) {
    // Load constants and Halo info into shared memory
    __shared__ float shared_data[SHARED_SIZE][SHARED_SIZE];

    // initialize row and col using thread and block
    // (Reminder it is 2 dimensions)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute shared memory coordinates (where to store the shared memory)
    int ty = threadIdx.y + HALO_SIZE;
    int tx = threadIdx.x + HALO_SIZE;

    // Apply the 3x3 laplacian kernel, (consists of calculating halo regions)
    // Implement like: left, right, top, bottom
    
    // load main tile into shared memory
    // double check if it is actuall row < height && col < width later
    if (row < height && col < width) {
        shared_data[ty][tx] = input[row * width + col];
    } else {
        // populate with 0's if out of bounds
        shared_data[ty][tx] = 0.0f;
    }
    
    // calculate for left first
    // leftmost is column index
    if (threadIdx.x == 0) {
        int Halo_Col = col - HALO_SIZE;
        // set bounds for calculating left
        if (Halo_Col >= 0 && row < height) {
            shared_data[ty][tx - HALO_SIZE] = input[row * width + Halo_Col];
        } else {
            shared_data[ty][tx - HALO_SIZE] = 0.0f;
        }
    }

    // calculate for right
    // rightmost is column index to right of main tile
    if (threadIdx.x == blockDim.x - 1) {
        // new halo col calculation
        int Halo_Col = col + HALO_SIZE;
        // bounds check for right
        if (Halo_Col < width && row < height) {
            // load right tile in shared memory
            shared_data[ty][tx + HALO_SIZE] = input[row * width + Halo_Col];
        } else {
            shared_data[ty][tx + HALO_SIZE] = 0.0f;
        }
    }

    // calculate for top and bottom now

    // check if top exists
    if (threadIdx.y == 0) {
        int Halo_Row = row - HALO_SIZE;
        // in bounds
        if (Halo_Row >= 0 && col < width) {
            // load into shared memoy
            shared_data[ty - HALO_SIZE][tx] = input[Halo_Row * width + col];
        } else {
            shared_data[ty - HALO_SIZE][tx] = 0.0f;
        }
    }

    // Calculate for bottom now
    if (threadIdx.y == blockDim.y - 1) {
        int Halo_Row = row + HALO_SIZE;
        // bounds check to populate
        if (Halo_Row < height && col < width) {
            shared_data[ty + HALO_SIZE][tx] = input[Halo_Row * width + col];
        } else {
            shared_data[ty + HALO_SIZE][tx] = 0.0f;
        }
    }

    // sync all threads
    __syncthreads();

    // then compute laplacian convolution using HALO's
    // check if we are in range first
    // check if this is the correct calculation
    if (row < height && col < width) {
        // store computation in result variable
        // add all shared memory points, main tile + top + bottom + left + right
        // error1: fix calc
        float result = 4.0f * shared_data[ty][tx] - shared_data[ty - 1][tx] - shared_data[ty + 1][tx] -
                       shared_data[ty][tx - 1] - shared_data[ty][tx + 1];
        

        // store result in output array
        output[row * width + col] = result;
    }

    // synch all shared memory (final)?
    // Not sure if i need this, probably not as syncthreads
    // likely only needed after halo's loaded into shared mem
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
    size_t size = inputImg.rows * inputImg.cols * sizeof(float);

    // Allocate Device memory using cudaMalloc
    float *d_input;
    cudaError_t err = cudaMalloc(&d_input, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_input failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    float *d_output;
    err = cudaMalloc(&d_output, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_output failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        return;
    }

    // Copy from host to device
    err = cudaMemcpy(d_input, inputImg.ptr<float>(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // allocate block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((inputImg.cols + TILE_SIZE - 1) / TILE_SIZE, 
                (inputImg.rows + TILE_SIZE - 1) / TILE_SIZE);

    printf("Image size: %dx%d\n", inputImg.cols, inputImg.rows);
    printf("Grid size: %dx%d, Block size: %dx%d\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Launch kernel
    LaplacianKernel<<<gridDim, blockDim>>>(d_input, d_output, inputImg.cols, inputImg.rows);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }
    
    // Wait for kernel to finish and check for execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    printf("Kernel executed successfully!\n");

    // Copy from device to host
    err = cudaMemcpy(outputImg.ptr<float>(), d_output, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("Memory operations completed successfully!\n");
}

/**
 * Work on main before the helper function to better understand how helper works
 * Main function:
 * Will be used for cv image processing and cv intertwined host memory 
 * Will utilize command line arguments
*/
int main(int argc, char** argv) {
    std::string imagePath;
    
    // Use bundled lena.jpg if no argument provided
    if (argc == 2) {
        imagePath = argv[1];
    } else {
        imagePath = "Lena.jpg";
        std::cout << "Using bundled test image: " << imagePath << std::endl;
    }

    // Read and open image as grayscale
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    // Check if image loaded successfully
    if (img.empty()) {
        std::cerr << "Error: Could not load image: " << imagePath << std::endl;
        return -1;
    }

    // Convert to float32 and normalize to [0, 1]
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32F, 1.0/255.0);

    // Create output matrix
    cv::Mat output(img.rows, img.cols, CV_32F);

    // Run CUDA Laplacian kernel
    runLaplacian(imgFloat, output);

    // Convert output back to 8-bit for saving
    cv::Mat outputUint8;
    cv::Mat absOutput;
    cv::absdiff(output, cv::Scalar::all(0), absOutput);
    
    double minVal, maxVal;
    cv::minMaxLoc(absOutput, &minVal, &maxVal);
    
    if (maxVal > 0) {
        absOutput.convertTo(outputUint8, CV_8U, 255.0/maxVal);
    } else {
        outputUint8 = cv::Mat::zeros(img.rows, img.cols, CV_8U);
    }

    // Save the result image
    if (!cv::imwrite("laplacian_result.jpg", outputUint8)) {
        std::cerr << "Error saving image to file" << std::endl;
        return -1;
    }

    std::cout << "Laplacian edge detection completed! Output saved as laplacian_result.jpg" << std::endl;
    return 0;
}

