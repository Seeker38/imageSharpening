
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;
using namespace std;
float* downscaleImageCPU(float* input, int width, int height, int channels) {
    int newWidth = width / 4;
    int newHeight = height / 4;

    // Allocate memory for the output matrix
    float* output = new float[newWidth * newHeight * channels];

    for (int row = 0; row < newHeight; ++row) {
        for (int col = 0; col < newWidth; ++col) {
            for (int channel = 0; channel < channels; ++channel) {
                int inputStartRow = row * 4;
                int inputStartCol = col * 4;

                float sum = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        sum += input[(inputStartRow + i) * width * channels + (inputStartCol + j) * channels + channel];
                    }
                }
                output[(row * newWidth + col)* channels + channel] = sum / 16.0f;
            }
        }
    }

    return output;
}


void UpscaleBody(float* downscaled, float* output, int width, int height, int channels){

    float parameterMatrix[8] = {7.0f/8.0f, 1.0f/8.0f, 5.0f/8.0f, 3.0f/8.0f, 3.0f/8.0f, 5.0f/8.0f, 1.0f/8.0f, 7.0f/8.0f};

    float C[2 * 4];  // Transpose of A

    // Transposing matrix A to get matrix C
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            C[i * 4 + j] = parameterMatrix[j * 2 + i];
        }
    }

    for (int row = 0; row < height -1; ++row) {
        for (int col = 0; col < width -1; ++col) {
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
    
}
float* upscaleOperationCPU(float* downscaled, int width, int height, int channels) {

    // Allocate memory for the output matrix
    float* upscaled = new float[width * height* channels];
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {    
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

                // UpscaleBodyKernel(downscaled, upscaled, width, height);
                // upscaled[row * width + col] = 0;

        }
    }
    UpscaleBody(downscaled, upscaled, static_cast<int>(width / 4), static_cast<int>(height /4), channels);

    return upscaled;   
}

float* calculatePError(float* original, float* upscaled, int width, int height, int channels) {
    float* pError = new float[width * height* channels];

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            for (int channel =0; channel < channels; ++channel){
                pError[(row * width + col) * channels + channel] = original[(row * width + col)* channels + channel] - upscaled[(row * width + col)* channels + channel];
            }
        }
    }
    return pError;
}

float* SobelOperator(float* input, int width, int height, int channels) {
    float* pEdge = new float[width * height* channels];
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            for (int channel =0; channel< channels; ++channel){
                // Filling the border
                if (row == 0 || col == 0 || row == height - 1 || col == width - 1) {
                    pEdge[(row * width + col)* channels + channel] = 0.0;
                } else {
                    // Filling the body using Sobel operator
                    float sobelX = -1.0f * input[((row - 1) * width + (col - 1))* channels + channel] + 0.0f * input[((row - 1) * width + col)* channels + channel] + 1.0f * input[((row - 1) * width + (col + 1))* channels + channel]
                                -2.0f * input[(row * width + (col - 1))* channels + channel] + 0.0f * input[(row * width + col)* channels + channel] + 2.0f * input[(row * width + (col + 1))* channels + channel]
                                -1.0f * input[((row + 1) * width + (col - 1))* channels + channel] + 0.0f * input[((row + 1) * width + col)* channels + channel] + 1.0f * input[((row + 1) * width + (col + 1))* channels + channel];

                    float sobelY = 1.0f * input[((row - 1) * width + (col + 1))* channels + channel] + 2.0f * input[(row * width + (col + 1))* channels + channel] + 1.0f * input[((row + 1) * width + (col + 1))* channels + channel]
                                -1.0f * input[((row - 1) * width + (col - 1))* channels + channel] - 2.0f * input[(row * width + (col - 1))* channels + channel] - 1.0f * input[((row + 1) * width + (col - 1))* channels + channel];

                    // pEdge[row * width + col] = fabs(sobelX) + fabs(sobelY); // Absolute sum of horizontal and vertical derivatives
                    pEdge[(row * width + col)* channels + channel] = static_cast<float>(sqrt(pow(sobelX, 2) + pow(sobelY, 2)));
                }
            }
        }
    }
    return pEdge;
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
float* preliminarySharpened(float* pEdge, float* pError, float* upscaleMatrix, int width, int height, float mean, float lightStrength, int channels) {
    float* result = new float[width * height* channels];
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            for (int channel=0; channel< channels; ++channel){
            // Apply brightness adjustment to pEdge array 
                pEdge[(row * width + col)* channels + channel] = pEdge[(row * width + col)* channels + channel] * lightStrength - mean;

                result[(row * width + col)* channels + channel] = (pError[(row * width + col)* channels + channel] + pEdge[(row * width + col)* channels + channel])* (2.0f+ lightStrength) + upscaleMatrix[(row * width + col)* channels + channel];   
            }
        }
    }
    return result;
}

float* OvershootControl(float* preliminarySharpened, float* original, int width, int height, int channels) {
    float* finalSharpened = new float[width * height* channels];
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
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
                    float oscMax = ((preliminarySharpened[(row * width + col)* channels + channel] - maxVal) + preliminarySharpened[(row * width + col)* channels + channel])* ( 0.775f);
                    float oscMin = ((minVal - preliminarySharpened[(row * width + col)* channels + channel]) + preliminarySharpened[(row * width + col)* channels + channel])* ( 0.775f);
                    

                    // Adjust each element of the preliminarySharpened matrix
                    if (preliminarySharpened[(row * width + col)* channels + channel] > maxVal) {
                        finalSharpened[(row * width + col)* channels + channel] = std::min(oscMax, 255.0f);
                    } else if (preliminarySharpened[(row * width + col)* channels + channel] < minVal) {
                        finalSharpened[(row * width + col)* channels + channel] = std::max(oscMin, 0.0f);
                    } else {
                        finalSharpened[(row * width + col)* channels + channel] = std::min(std::max(preliminarySharpened[(row * width + col)* channels + channel], 0.0f), 255.0f);
                    }
                }
            }
        }
    }
    return finalSharpened;
}



// Function to measure the runtime of another function
template<typename Func, typename... Args>
long long measureRuntime(Func func, Args&&... args) {
    auto start_time = std::chrono::high_resolution_clock::now();

    func(std::forward<Args>(args)...);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    return duration.count();
}


int main() {
    // Read the color image using OpenCV
    cv::Mat inputImage = cv::imread("C:/Users/Admin/Desktop/imageSharpening/color1.png");

    if (inputImage.empty()) {
        std::cerr << "Error: Unable to read the image." << std::endl;
        return -1;
    }

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
    
    // Convert the image to a flattened float array
    int width = rescaledMat.cols;
    int height = rescaledMat.rows;
    int channels = rescaledMat.channels();
    float* input = new float[width * height * channels];

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            for (int channel = 0; channel < channels; ++channel) {
                input[(row * width + col) * channels + channel] = static_cast<float>(rescaledMat.at<cv::Vec3b>(row, col)[channel]);
            }
        }
    }

    // Apply the downscaleImageCPU function
    float* h_downscaled = downscaleImageCPU(input, width, height, channels);

    // Upscale the downscaled image
    float* h_upscaled = upscaleOperationCPU(h_downscaled, width, height,channels);

    float* h_pError = calculatePError(input, h_upscaled, width, height,channels);

    float* h_pEdge =SobelOperator(input, width, height,channels);

    float mean = CalculateMean(h_pEdge, width, height,channels);

    float lightStrength = 0.205f;
    float* h_preliminary = preliminarySharpened( h_pEdge, h_pError, h_upscaled, width, height, mean, lightStrength,channels);

    float* h_finalSharpened = OvershootControl( h_preliminary, input, width, height,channels);


    // Display the original and downscaled images
    cv::Mat sharpenedImage(height, width, CV_8UC3);

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            for (int channel = 0; channel < channels; ++channel) {
                sharpenedImage.at<cv::Vec3b>(row, col)[channel] = static_cast<uchar>(h_finalSharpened[(row * width + col) * channels + channel]);
            }
        }
    }

    std::string outputPath = "C:/Users/Admin/Desktop/imageSharpening/finalSharpened.png";
    cv::imwrite(outputPath, sharpenedImage);

    // Clean up memory
    delete[] input;
    delete[] h_downscaled;
    delete[] h_upscaled;
    delete[] h_pError;
    delete[] h_pEdge;
    delete[] h_preliminary;
    delete[] h_finalSharpened;

    return 0;
}