#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <chrono>

#define BLOCK_SIZE 16

using namespace cv;
using namespace std;

__global__
void downscaleImageKernel(float* input, float* output, int width, int height, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height / 4 && col < width / 4) {
        for (int channel = 0; channel < channels; ++channel) {
            int inputStartRow = row * 4;
            int inputStartCol = col * 4;

            float sum = 0.0f;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    sum += input[(inputStartRow + i) * width * channels + (inputStartCol + j) * channels + channel];
                }
            }
            output[(row * width / 4 + col) * channels + channel] = sum / 16.0f;
        }
    }
}
__device__ 
void UpscaleBodyKernel(float* downscaled, float* output, int width, int height, int channels){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float parameterMatrix[4*2] = {7.0f/8.0f, 1.0f/8.0f, 5.0f/8.0f, 3.0f/8.0f, 3.0f/8.0f, 5.0f/8.0f, 1.0f/8.0f, 7.0f/8.0f};

    float C[2 * 4];  // Transpose of A

    // Transposing matrix A to get matrix C
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {

            C[(i * 4 + j)] = parameterMatrix[(j * 2 + i)];

        }
    }

    if (row < height -1 && col < width -1) {
        for (int channel = 0; channel < channels; ++channel) {
            int calRow = row * 4;
            int calCol = col *4;

            float submatrix[2*2] = {downscaled[(row * width + col) * channels + channel],downscaled[(row * width + (col + 1)) * channels + channel],
                                    downscaled[((row + 1) * width + col) * channels + channel],downscaled[((row + 1) * width + (col + 1)) * channels + channel]};

            float tempMatrix[4 * 2];

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 2; ++j) {
                    tempMatrix[(i * 2 + j)] = 0.0f;
                    for (int k = 0; k < 2; ++k) {
                        tempMatrix[(i * 2 + j)] += parameterMatrix[(i * 2 + k)] * submatrix[(k * 2 + j)];
                    }
                }
            }

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    output[((calRow + 2 + i) * width*4 + (calCol + 2 + j)) * channels + channel] = 0.0f;
                    for (int k = 0; k < 2; ++k) {
                        output[((calRow + i + 2) * width*4 + (calCol + j + 2)) * channels + channel] +=tempMatrix[i * 2 + k] * C[k * 4 + j];
                    }
                }
            }
        }
        
    }    
}

__global__
void UpscaleOperationKernel(float* downscaled, float* upscaled, int width, int height, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

        
    // Upscale the first row
    if (row == 0 && col % 4 == 0) {
        for (int channel = 0; channel < channels; ++channel) {
            // Calculate values except the last three elements
            if (col < width - 4) {
                upscaled[(row * width + col) * channels + channel] = downscaled[(col / 4) * channels + channel];
                upscaled[(row * width + col + 1) * channels + channel] = (3.0f / 4.0f) * downscaled[(col / 4) * channels + channel] +
                                                                           (1.0f / 4.0f) * downscaled[((col / 4) + 1) * channels + channel];
                upscaled[(row * width + col + 2) * channels + channel] = (2.0f / 4.0f) * downscaled[(col / 4) * channels + channel] +
                                                                           (2.0f / 4.0f) * downscaled[((col / 4) + 1) * channels + channel];
                upscaled[(row * width + col + 3) * channels + channel] = (1.0f / 4.0f) * downscaled[(col / 4) * channels + channel] +
                                                                           (3.0f / 4.0f) * downscaled[((col / 4) + 1) * channels + channel];
                upscaled[(row * width + col + 4) * channels + channel] = downscaled[((col / 4) + 1) * channels + channel];
            }
            // Copy the third last element's value from the fourth last value
            else if (col == width - 4) {
                upscaled[(row * width + col + 1) * channels + channel] = upscaled[(row * width + col - 3) * channels + channel];
                upscaled[(row * width + col + 2) * channels + channel] = upscaled[(row * width + col - 2) * channels + channel];
                upscaled[(row * width + col + 3) * channels + channel] = upscaled[(row * width + col - 1) * channels + channel];
            }
        }
    }

    // Upscale the first column
    else if (col == 0 && row % 4 == 0) {
        for (int channel = 0; channel < channels; ++channel) {
            // Calculate values except the last three elements
            if (row < height - 4) {
                upscaled[(row * width + col) * channels + channel] = downscaled[(row / 4) * (width / 4) * channels+ channel];
                upscaled[((row + 1) * width + col) * channels + channel] = 3.0f / 4.0f * downscaled[(row / 4) * (width / 4) * channels+ channel] +
                                                                           1.0f / 4.0f * downscaled[((row + 4) / 4) * (width / 4) * channels+ channel];
                upscaled[((row + 2) * width + col) * channels + channel] = 2.0f / 4.0f * downscaled[(row / 4) * (width / 4) * channels+ channel] +
                                                                           2.0f / 4.0f * downscaled[((row + 4) / 4) * (width / 4) * channels+ channel];
                upscaled[((row + 3) * width + col) * channels + channel] = 1.0f / 4.0f * downscaled[(row / 4) * (width / 4) * channels+ channel] +
                                                                           3.0f / 4.0f * downscaled[((row + 4) / 4) * (width / 4) * channels+ channel];
                upscaled[((row + 4) * width + col) * channels + channel] = downscaled[((row + 4) / 4) * (width / 4) * channels];
            }
            // Copy the third last element's value from the fourth last value
            else if (row == height - 4) {
                upscaled[((row + 1) * width + col) * channels + channel] = upscaled[((row - 3) * width) * channels + channel];
                upscaled[((row + 2) * width + col) * channels + channel] = upscaled[((row - 2) * width) * channels + channel];
                upscaled[((row + 3) * width + col) * channels + channel] = upscaled[((row - 1) * width) * channels + channel];
            }
        }
    }


    // Upscale the penultimate row
    else if (row == height - 2 && col % 4 == 0) {
        for (int channel = 0; channel < channels; ++channel) {
            if (col < width - 4) {
                upscaled[(row * width + col)* channels + channel] = downscaled[((row-2)/4 * (width/4) + col/4)* channels + channel ];  
                upscaled[(row * width + col +1)* channels + channel] = 3.0f / 4.0f * downscaled[((row-2)/4 * (width/4) + col/4)* channels + channel] + 1.0f / 4.0f * downscaled[((row-2)/4 * (width/4) + col/4 + 1)* channels + channel];
                upscaled[(row * width + col + 2)* channels + channel] = 2.0f / 4.0f * downscaled[((row-2)/4 * (width/4) + col/4)* channels + channel] + 2.0f / 4.0f * downscaled[((row-2)/4 * (width/4) + col/4 + 1)* channels + channel];
                upscaled[(row * width + col + 3)* channels + channel] = 1.0f / 4.0f * downscaled[((row-2)/4 * (width/4) + col/4)* channels + channel] + 3.0f / 4.0f * downscaled[((row-2)/4 * (width/4) + col/4 + 1)* channels + channel];
                upscaled[(row * width + col + 4)* channels + channel] = downscaled[((row-2)/4 * (width/4) + col/4 +1)* channels + channel];
                // upscaled[row * width + col + 4] = downscaled[(col/4)+ (width /4) * 3 +1];
            }
            // Copy the third last element's value from the fourth last value
            else if (col == width - 4) {
                upscaled[(row * width + col +1)* channels + channel] = upscaled[(row * width + col - 3)* channels + channel];
                upscaled[(row * width + col + 2)* channels + channel] = upscaled[(row * width + col -2)* channels + channel];
                upscaled[(row * width + col + 3)* channels + channel] = upscaled[(row * width + col - 1)* channels + channel];
            }
        }
    }

    // Upscale the penultimate column
    else if (col == width - 2 && row % 4 == 0) {
        for (int channel = 0; channel < channels; ++channel) {        
            if (row < height - 4){
                // upscaled[row * width+col] = downscaled[(row/ 4) *(width / 4) +(width / 4) -1];
                upscaled[(row * width+col)* channels + channel] = downscaled[((row/ 4) *(width / 4) +(col -2)/4)* channels + channel];
                upscaled[((row + 1) * width+col)* channels + channel] = 3.0f / 4.0f * downscaled[((row/ 4) *(width / 4) +(col -2)/4)* channels + channel]+ 1.0f / 4.0f * downscaled[(((row+4)/ 4) *(width / 4) +(col -2)/4)* channels + channel];
                upscaled[((row + 2) * width+col)* channels + channel] = 2.0f / 4.0f * downscaled[((row/ 4) *(width / 4) +(col -2)/4)* channels + channel] + 2.0f / 4.0f * downscaled[(((row+4)/ 4) *(width / 4) +(col -2)/4)* channels + channel];
                upscaled[((row + 3) * width+col)* channels + channel] = 1.0f / 4.0f * downscaled[((row/ 4) *(width / 4) +(col -2)/4)* channels + channel] + 3.0f / 4.0f * downscaled[(((row+4)/ 4) *(width / 4) +(col -2)/4)* channels + channel];
                upscaled[((row + 4) * width+col)* channels + channel] = downscaled[(((row+4)/ 4) *(width / 4) +(col -2)/4)* channels + channel];

            }
            else if (row == height - 4){
                upscaled[((row +1)* width+col)* channels + channel] =upscaled[(row -3)*width* channels + channel];
                upscaled[((row +2)* width+col)* channels + channel] =upscaled[(row -2)*width* channels + channel];
                upscaled[((row +3)* width+col)* channels + channel] =upscaled[(row -1)*width* channels + channel];
            }
        }
    
    }


            
    else if (col == 1 && row % 4 == 0) {
        // Calculate values except the last three elements
        for (int channel = 0; channel < channels; ++channel) {        
            if (row < height - 4){
                // upscaled[row * width+col] = downscaled[(row/ 4) *(width / 4) +(width / 4) -1];
                upscaled[(row * width+col)* channels + channel] = downscaled[((row/ 4) *(width / 4))* channels + channel];
                upscaled[((row + 1) * width+col)* channels + channel] = 3.0f / 4.0f * downscaled[((row/ 4) *(width / 4))* channels + channel]+ 1.0f / 4.0f * downscaled[((row/ 4) *(width / 4) + (width /4))* channels + channel];
                upscaled[((row + 2) * width+col)* channels + channel] = 2.0f / 4.0f * downscaled[((row/ 4) *(width / 4))* channels + channel] + 2.0f / 4.0f * downscaled[((row/ 4) *(width / 4) + (width /4))* channels + channel];
                upscaled[((row + 3) * width+col)* channels + channel] = 1.0f / 4.0f * downscaled[((row/ 4) *(width / 4))* channels + channel] + 3.0f / 4.0f * downscaled[((row/ 4) *(width / 4) + (width /4))* channels + channel];
                upscaled[((row + 4) * width+col)* channels + channel] = downscaled[((row/ 4) *(width / 4) + (width /4))* channels + channel];

            }
            else if (row == height - 4){
                upscaled[((row +1)* width+col)* channels + channel] =upscaled[(row -3)*width* channels + channel];
                upscaled[((row +2)* width+col)* channels + channel] =upscaled[(row -2)*width* channels + channel];
                upscaled[((row +3)* width+col)* channels + channel] =upscaled[(row -1)*width* channels + channel];
            }
        }        
    
    }
    
    else if (row == 1 && col % 4 == 0) {
        // Calculate values except the last three elements
        for (int channel = 0; channel < channels; ++channel) {
            if (col < width - 4) {
                upscaled[(row * width + col)* channels + channel] = downscaled[(col/4)* channels + channel ];  
                upscaled[(row * width + col +1)* channels + channel] = 3.0f / 4.0f * downscaled[((col ) / 4)* channels + channel] + 1.0f / 4.0f * downscaled[((col / 4) +1)* channels + channel];
                upscaled[(row * width + col + 2)* channels + channel] = 2.0f / 4.0f * downscaled[((col ) / 4)* channels + channel] + 2.0f / 4.0f * downscaled[((col / 4) +1)* channels + channel];
                upscaled[(row * width + col + 3)* channels + channel] = 1.0f / 4.0f * downscaled[((col ) / 4)* channels + channel] + 3.0f / 4.0f * downscaled[((col / 4) +1)* channels + channel];
                upscaled[(row * width + col + 4)* channels + channel] = downscaled[((col / 4) +1)* channels + channel];
                // upscaled[row * width + col + 4] = downscaled[(col/4)+ (width /4) * 3 +1];
            }
            // Copy the third last element's value from the fourth last value
            else if (col == width - 4) {
                upscaled[(row * width + col +1)* channels + channel] = upscaled[(row * width + col - 3)* channels + channel];
                upscaled[(row * width + col + 2)* channels + channel] = upscaled[(row * width + col -2)* channels + channel];
                upscaled[(row * width + col + 3)* channels + channel] = upscaled[(row * width + col - 1)* channels + channel];
            }
        }        
    }

    // Copy to the last column
    else if (col == width - 1) {
        for (int channel = 0; channel < channels; ++channel) {        
            if (row < height - 4){
                // upscaled[row * width+col] = downscaled[(row/ 4) *(width / 4) +(width / 4) -1];
                upscaled[(row * width+col)* channels + channel] = downscaled[((row/ 4) *(width / 4) +(col -3)/4)* channels + channel];
                upscaled[((row + 1) * width+col)* channels + channel] = 3.0f / 4.0f * downscaled[((row/ 4) *(width / 4) +(col -3)/4)* channels + channel]+ 1.0f / 4.0f * downscaled[(((row+4)/ 4) *(width / 4) +(col -3)/4)* channels + channel];
                upscaled[((row + 2) * width+col)* channels + channel] = 2.0f / 4.0f * downscaled[((row/ 4) *(width / 4) +(col -3)/4)* channels + channel] + 2.0f / 4.0f * downscaled[(((row+4)/ 4) *(width / 4) +(col -3)/4)* channels + channel];
                upscaled[((row + 3) * width+col)* channels + channel] = 1.0f / 4.0f * downscaled[((row/ 4) *(width / 4) +(col -3)/4)* channels + channel] + 3.0f / 4.0f * downscaled[(((row+4)/ 4) *(width / 4) +(col -3)/4)* channels + channel];
                upscaled[((row + 4) * width+col)* channels + channel] = downscaled[(((row+4)/ 4) *(width / 4) +(col -3)/4)* channels + channel];

            }
            else if (row == height - 4){
                upscaled[((row +1)* width+col)* channels + channel] =upscaled[(row -3)*width* channels + channel];
                upscaled[((row +2)* width+col)* channels + channel] =upscaled[(row -2)*width* channels + channel];
                upscaled[((row +3)* width+col)* channels + channel] =upscaled[(row -1)*width* channels + channel];
            }
        }     
    }


    // Copy to the last row
    else if (row == height - 1) {      
        for (int channel = 0; channel < channels; ++channel) {
            if (col < width - 4) {
                upscaled[(row * width + col)* channels + channel] = downscaled[((row-3)/4 * (width/4) + col/4 )* channels + channel ];  
                upscaled[(row * width + col +1)* channels + channel] = 3.0f / 4.0f * downscaled[((row-3)/4 * (width/4) + col/4 )* channels + channel] + 1.0f / 4.0f * downscaled[((row-3)/4 * (width/4) + col/4 +1)* channels + channel];
                upscaled[(row * width + col + 2)* channels + channel] = 2.0f / 4.0f * downscaled[((row-3)/4 * (width/4) + col/4 )* channels + channel] + 2.0f / 4.0f * downscaled[((row-3)/4 * (width/4) + col/4 +1)* channels + channel];
                upscaled[(row * width + col + 3)* channels + channel] = 1.0f / 4.0f * downscaled[((row-3)/4 * (width/4) + col/4 )* channels + channel] + 3.0f / 4.0f * downscaled[((row-3)/4 * (width/4) + col/4 +1)* channels + channel];
                upscaled[(row * width + col + 4)* channels + channel] = downscaled[((row-3)/4 * (width/4) + col/4 +1)* channels + channel];
                // upscaled[row * width + col + 4] = downscaled[(col/4)+ (width /4) * 3 +1];
            }
            // Copy the third last element's value from the fourth last value
            else if (col == width - 4) {
                upscaled[(row * width + col +1)* channels + channel] = upscaled[(row * width + col - 3)* channels + channel];
                upscaled[(row * width + col + 2)* channels + channel] = upscaled[(row * width + col -2)* channels + channel];
                upscaled[(row * width + col + 3)* channels + channel] = upscaled[(row * width + col - 1)* channels + channel];
            }
        } 
    }

    // Upscale the main body

    UpscaleBodyKernel(downscaled, upscaled, static_cast<int>(width / 4), static_cast<int>(height /4), channels);

    
}

__global__
void CalculatePError(float* original, float* upscaled, float* pError, int width, int height, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        for (int channel =0; channel < channels; ++channel){
            pError[(row * width + col) * channels + channel] = original[(row * width + col)* channels + channel] - upscaled[(row * width + col)* channels + channel];
        }
    }
}


__global__
void SobelOperatorKernel(float* input, float* pEdge, int width, int height, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        for (int channel =0; channel< channels; ++channel){
            // Filling the border
            if (row == 0 || col == 0 || row == height - 1 || col == width - 1) {
                pEdge[(row * width + col)* channels + channel] = 0.0;
            } else {
                // Filling the body using Sobel operator
                float sobelX = -1.0 * input[((row - 1) * width + (col - 1))* channels + channel] + 0.0 * input[((row - 1) * width + col)* channels + channel] + 1.0 * input[((row - 1) * width + (col + 1))* channels + channel]
                            -2.0 * input[(row * width + (col - 1))* channels + channel] + 0.0 * input[(row * width + col)* channels + channel] + 2.0 * input[(row * width + (col + 1))* channels + channel]
                            -1.0 * input[((row + 1) * width + (col - 1))* channels + channel] + 0.0 * input[((row + 1) * width + col)* channels + channel] + 1.0 * input[((row + 1) * width + (col + 1))* channels + channel];

                float sobelY = 1.0 * input[((row - 1) * width + (col + 1))* channels + channel] + 2.0 * input[(row * width + (col + 1))* channels + channel] + 1.0 * input[((row + 1) * width + (col + 1))* channels + channel]
                            -1.0 * input[((row - 1) * width + (col - 1))* channels + channel] - 2.0 * input[(row * width + (col - 1))* channels + channel] - 1.0 * input[((row + 1) * width + (col - 1))* channels + channel];

                // pEdge[row * width + col] = fabs(sobelX) + fabs(sobelY); // Absolute sum of horizontal and vertical derivatives
                pEdge[(row * width + col)* channels + channel] = sqrt(pow(sobelX, 2) + pow(sobelY, 2));
            }
        }
    }
}

float CalculateMean(float* pEdge, int width, int height, int channels) {
    float mean = 0.0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int channel = 0; channel < channels; ++channel) {
                mean += pEdge[(i * width + j) * channels + channel];
            }
        }
    }
    return mean / static_cast<float>(width * height * channels);
}

__global__ 
void preliminarySharpenedKernel(float* result, float* pEdge, float* pError, float* upscaleMatrix, int width, int height, float mean, float lightStrength, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;    
    if (row < height && col < width) {
        for (int channel=0; channel< channels; ++channel){
        // Apply brightness adjustment to pEdge array 
            pEdge[(row * width + col)* channels + channel] = pEdge[(row * width + col)* channels + channel] * lightStrength - mean;

            result[(row * width + col)* channels + channel] = (pError[(row * width + col)* channels + channel] + pEdge[(row * width + col)* channels + channel])* (2.0f+ lightStrength) + upscaleMatrix[(row * width + col)* channels + channel];   
        }   
    }
}
// __global__ 
// void preliminarySharpenedKernel(float* result, float* pEdge, float* pError, float* upscaleMatrix, int width, int height, float mean, float lightStrength, int channels) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     // Define the 1D sharpening filter
//     const float sharpeningFilter[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

//     float sum = 0;    
//     if (row < height && col < width) {
//         for (int channel=0; channel< channels; ++channel){
//             // Apply brightness adjustment to pEdge array 
//             pEdge[(row * width + col)* channels + channel] = pEdge[(row * width + col)* channels + channel] * lightStrength - mean;

//             result[(row * width + col)* channels + channel] = (pError[(row * width + col)* channels + channel] + pEdge[(row * width + col)* channels + channel])* (2.0f+ lightStrength) + upscaleMatrix[(row * width + col)* channels + channel];   
//             // Apply the sharpening filter to the result array
//             for (int i = -1; i <= 1; ++i) {
//                 for (int j = -1; j <= 1; ++j) {
//                     int neighborRow = row + i;
//                     int neighborCol = col + j;

//                     // Check if the neighbor is within the image bounds
//                     if (neighborRow >= 0 && neighborRow < height && neighborCol >= 0 && neighborCol < width) {
//                         int filterIndex = (i + 1) * 3 + (j + 1);
//                         result[(neighborRow * width + neighborCol)* channels + channel] += result[(neighborRow * width + neighborCol)* channels + channel] * sharpeningFilter[filterIndex];
//                     }
//                 }
//             }
//         }
    
//     }
// }



// Overshoot control kernel using the max values array
__global__ 
void OvershootControlKernel(float* finalSharpened, float* preliminarySharpened, float* original, int width, int height, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        for (int channel =0; channel<channels ; ++channel){
            // Matrix Border
            if (row == 0 || col == 0 || row == height - 1 || col == width - 1) {
                finalSharpened[(row * width + col)* channels + channel] = preliminarySharpened[(row * width + col)* channels + channel];
            } else {
                // Matrix Body
                // int submatrixSize = 3;  // Size of the submatrix (3x3)
                float maxVal = -1.0f;   // Initialize max value to a small value
                float minVal = 256.0f;  // Initialize min value to a large value 

                // Find the max and min values in the 3x3 submatrix
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        float value = original[((row + i) * width + (col + j))* channels + channel];
                        maxVal = fmaxf(maxVal, value);
                        minVal = fminf(minVal, value);
                    }
                }

                // float oscMax = abs(maxVal - preliminarySharpened[row * width + col]);
                // float oscMin = abs(preliminarySharpened[row * width + col] - minVal);
                float oscMax = (preliminarySharpened[(row * width + col)* channels + channel] - maxVal) + preliminarySharpened[(row * width + col)* channels + channel];
                float oscMin = (minVal - preliminarySharpened[(row * width + col)* channels + channel]) + preliminarySharpened[(row * width + col)* channels + channel];
                

                // Adjust each element of the preliminarySharpened matrix
                if (preliminarySharpened[(row * width + col)* channels + channel] > maxVal) {
                    finalSharpened[(row * width + col)* channels + channel] = fminf(oscMax, 255.0f);
                } else if (preliminarySharpened[(row * width + col)* channels + channel] < minVal) {
                    finalSharpened[(row * width + col)* channels + channel] = fminf(fmaxf(oscMin, 0.0f), 255.0f);
                } else {
                    finalSharpened[(row * width + col)* channels + channel] = fminf(fmaxf(preliminarySharpened[row * width + col], 0.0f), 255.0f);
                }

                // finalSharpened[row * width + col] += original[row * width + col];
                // Store the final value in the finalSharpened matrix
                // finalSharpened[row * width + col] = fminf(fmaxf(preliminarySharpened[row * width + col], 0.0f), 255.0f);
            }
        }
    }
}

void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

void sharpenAndUpscaleImage(const cv::Mat& input, cv::Mat& output) {
    // Allocate device memory
    float *d_input, *d_downscaled, *d_upscaled, *d_pError, *d_pEdge, *d_preliminary, *d_finalSharpened;
    size_t inputSize = input.rows * input.cols * input.channels() * sizeof(float);
    size_t downscaledSize = (input.rows / 4) * (input.cols / 4) * input.channels() * sizeof(float);
    size_t upscaledSize = inputSize;    
    size_t pErrorSize = inputSize;
    size_t pEdgeSize = inputSize; 
    size_t preliminarySize = inputSize;
    size_t finalSharpenedSize = inputSize;   

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_downscaled, downscaledSize);
    cudaMalloc(&d_upscaled, upscaledSize);
    cudaMalloc(&d_pError, pErrorSize);
    cudaMalloc(&d_pEdge, pEdgeSize);
    cudaMalloc(&d_preliminary, preliminarySize);
    cudaMalloc(&d_finalSharpened, finalSharpenedSize);

    // Copy input data to device
    cudaMemcpy(d_input, input.ptr<float>(), inputSize, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float elapsedTime;

    // Create events for timing
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Record the start time
    checkCudaErrors(cudaEventRecord(start, 0));

    // Set grid and block dimensions for downscale
    dim3 blockDimDownscale(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDimDownscale((input.cols + blockDimDownscale.x - 1) / blockDimDownscale.x, (input.rows + blockDimDownscale.y - 1) / blockDimDownscale.y);

    // Launch the downscale kernel
    downscaleImageKernel<<<gridDimDownscale, blockDimDownscale>>>(d_input, d_downscaled, input.cols, input.rows, input.channels());

    // Record the stop time
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    // Calculate and print the elapsed time for each function
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Downscale Time: " << elapsedTime << " ms" << std::endl;

    // Set grid and block dimensions for upscale
    dim3 blockDimUpscale(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDimUpscale((input.cols  + blockDimUpscale.x - 1) / blockDimUpscale.x, (input.rows  + blockDimUpscale.y - 1) / blockDimUpscale.y);

    // Launch the upscale kernel
    checkCudaErrors(cudaEventRecord(start, 0));
    UpscaleOperationKernel<<<gridDimUpscale, blockDimUpscale>>>(d_downscaled, d_upscaled, input.cols, input.rows, input.channels());
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Upscale Time: " << elapsedTime << " ms" << std::endl;

    // Set grid and block dimensions for CalculatePError
    dim3 blockDimCalculatePError(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDimCalculatePError((input.cols + blockDimCalculatePError.x - 1) / blockDimCalculatePError.x, (input.rows + blockDimCalculatePError.y - 1) / blockDimCalculatePError.y);

    // Launch the CalculatePError kernel
    checkCudaErrors(cudaEventRecord(start, 0));
    CalculatePError<<<gridDimCalculatePError, blockDimCalculatePError>>>(d_input, d_upscaled, d_pError, input.cols, input.rows, input.channels());
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "d_pError Time: " << elapsedTime << " ms" << std::endl; 

    // Set grid and block dimensions for SobelOperatorKernel
    dim3 blockDimSobel(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDimSobel((input.cols + blockDimSobel.x - 1) / blockDimSobel.x, (input.rows + blockDimSobel.y - 1) / blockDimSobel.y);

    // Launch the SobelOperatorKernel kernel
    checkCudaErrors(cudaEventRecord(start, 0));
    SobelOperatorKernel<<<gridDimSobel, blockDimSobel>>>(d_input, d_pEdge, input.cols, input.rows, input.channels());
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "sobel Time: " << elapsedTime << " ms" << std::endl; 

    // Allocate host memory for the mean calculation
    float* h_pEdge = new float[input.cols * input.rows* input.channels()];

    // Copy the result from d_pEdge to the host
    cudaMemcpy(h_pEdge, d_pEdge, pEdgeSize, cudaMemcpyDeviceToHost);

    // Calculate the mean using the host array
    float mean = CalculateMean(h_pEdge, input.cols, input.rows, input.channels());

    // Free the host array
    delete[] h_pEdge;

    // Set grid and block dimensions for preliminarySharpened
    dim3 blockDimPreliminary(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDimPreliminary((input.cols + blockDimPreliminary.x - 1) / blockDimPreliminary.x, (input.rows + blockDimPreliminary.y - 1) / blockDimPreliminary.y);


    float lightStrength = 0.205f;
    // Launch the preliminarySharpenedKernel kernel
    checkCudaErrors(cudaEventRecord(start, 0));
    preliminarySharpenedKernel<<<gridDimPreliminary, blockDimPreliminary>>>(d_preliminary, d_pEdge, d_pError, d_upscaled, input.cols, input.rows, mean, lightStrength, input.channels());
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Preliminary Time: " << elapsedTime << " ms" << std::endl;

    // Set grid and block dimensions for OvershootControl
    dim3 blockDimOvershootControl(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDimOvershootControl((input.cols + blockDimOvershootControl.x - 1) / blockDimOvershootControl.x, (input.rows + blockDimOvershootControl.y - 1) / blockDimOvershootControl.y);

    // Launch the OvershootControlKernel kernel
    checkCudaErrors(cudaEventRecord(start, 0));
    OvershootControlKernel<<<gridDimOvershootControl, blockDimOvershootControl>>>(d_finalSharpened, d_preliminary, d_input, input.cols, input.rows, input.channels());
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "OvershootControl Time: " << elapsedTime << " ms" << std::endl;  
  

    output.create(input.rows, input.cols, CV_32FC(input.channels()));
    cudaMemcpy(output.ptr<float>(), d_preliminary, preliminarySize, cudaMemcpyDeviceToHost);

    // Destroy events
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_downscaled);
    cudaFree(d_upscaled);
    cudaFree(d_pError);
    cudaFree(d_pEdge);
    cudaFree(d_preliminary);
    cudaFree(d_finalSharpened);

}

int main() {
    // Read the input image
    cv::Mat inputImage = cv::imread("C:/Users/Admin/Desktop/parralelComputing/color1.png");

    if (inputImage.empty()) {
        std::cerr << "Error: Could not read the input image." << std::endl;
        return -1;
    }
    // Get user input for kRescaleFactor
    double kRescaleFactor;
    cout << "Enter the rescale factor (VD: 0.75): ";
    cin >> kRescaleFactor;

    // Check if the input is valid
    // if (fmod(kRescaleFactor, 4.0) != 0) {
    //     cout << "Invalid rescale factor. It must be divided by 4." << endl;
    //     return -1;
    // }

    Mat rescaledMat;
    resize(inputImage, rescaledMat, Size(0, 0), kRescaleFactor, kRescaleFactor);

    // Convert input image to float type
    cv::Mat inputFloat;
    rescaledMat.convertTo(inputFloat, CV_32F);



    // Apply the sharpening and upscaling filter
    cv::Mat sharpenedUpscaledOutput;
    sharpenAndUpscaleImage(inputFloat, sharpenedUpscaledOutput);

    // Convert the result back to uint8 type
    cv::Mat sharpenedUpscaledOutputUint8;
    sharpenedUpscaledOutput.convertTo(sharpenedUpscaledOutputUint8, CV_8U);

    // Save the result
    cv::imwrite("C:/Users/Admin/Desktop/imageSharpening/finalSharpened.png", sharpenedUpscaledOutputUint8);

    std::cout << "Sharpened and upscaled image saved as finalSharpened.jpg" << std::endl;

    return 0;
}