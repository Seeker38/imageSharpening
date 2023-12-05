
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
float* downscaleImageCPU(float* input, int width, int height) {
    int newWidth = width / 4;
    int newHeight = height / 4;

    // Allocate memory for the output matrix
    float* output = new float[newWidth * newHeight];

    for (int row = 0; row < newHeight; ++row) {
        for (int col = 0; col < newWidth; ++col) {
            int inputStartRow = row * 4;
            int inputStartCol = col * 4;

            float sum = 0.0f;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    sum += input[(inputStartRow + i) * width + inputStartCol + j];
                }
            }
            output[row * newWidth + col] = sum / 16.0f;
        }
    }

    return output;
}


void UpscaleBody(float* downscaled, float* output, int width, int height) {
    float parameterMatrix[8] = {7.0f/8.0f, 1.0f/8.0f, 5.0f/8.0f, 3.0f/8.0f, 3.0f/8.0f, 5.0f/8.0f, 1.0f/8.0f, 7.0f/8.0f};
    float C[2 * 4];

    // Transpose of matrix A to get matrix C
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            C[i * 4 + j] = parameterMatrix[j * 2 + i];
        }
    }

    for (int row = 0; row < height - 1; ++row) {
        for (int col = 0; col < width - 1; ++col) {
            int calRow = row * 4;
            int calCol = col * 4;

            float submatrix[2][2] = {{downscaled[row * width + col], downscaled[row * width + (col + 1)]},
                                     {downscaled[(row + 1) * width + col], downscaled[(row + 1) * width + (col + 1)]}};

            float tempMatrix[4][2];

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 2; ++j) {
                    tempMatrix[i][j] = 0.0f;
                    for (int k = 0; k < 2; ++k) {
                        tempMatrix[i][j] += parameterMatrix[i * 2 + k] * submatrix[k][j];
                    }
                }
            }

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    output[(calRow + 2 + i) * width * 4 + (calCol + 2 + j)] = 0.0f;
                    for (int k = 0; k < 2; ++k) {
                        output[(calRow + i + 2) * width * 4 + (calCol + j + 2)] += tempMatrix[i][k] * C[k * 4 + j];
                    }
                }
            }
        }
    }
}
float* upscaleOperationCPU(float* downscaled, int width, int height) {

    // Allocate memory for the output matrix
    float* upscaled = new float[width * height];
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {    
            // Upscale the first row
            if (row == 0 && col % 4 == 0) {
                // Calculate values except the last three elements
                if (col< width - 4) {
                    upscaled[row * width + col] = downscaled[(col/4)];  
                    upscaled[row * width + col +1] = (3.0f / 4.0f) * downscaled[(col ) / 4] + (1.0f / 4.0f) * downscaled[(col / 4) +1];
                    upscaled[row * width + col + 2] = (2.0f / 4.0f) * downscaled[(col) / 4] + (2.0f / 4.0f) * downscaled[(col / 4) +1];
                    upscaled[row * width + col + 3] = (1.0f / 4.0f) * downscaled[(col) / 4] + (3.0f / 4.0f) * downscaled[(col / 4) +1];
                    upscaled[row * width + col + 4] = downscaled[(col/4)+1];
                }
                // Copy the third last element's value from the fourth last value
                else if (col == width - 4) {
                    upscaled[row * width + col+1] = upscaled[row * width + col - 3];
                    upscaled[row * width + col+2] = upscaled[row * width + col -2];
                    upscaled[row * width + col+3] = upscaled[row * width + col - 1];
                }
            }

            // Upscale the first column 
            else if (col == 0 && row % 4 == 0) {
                // Calculate values except the last three elements
                if (row < height - 4){
                    upscaled[row * width+col] = downscaled[(row/ 4) *(width / 4)];
                    upscaled[(row + 1) * width+col] = 3.0f / 4.0f * downscaled[(row/ 4) *(width / 4)]+ 1.0f / 4.0f * downscaled[((row+4)/ 4) *(width / 4)];
                    upscaled[(row + 2) * width+col] = 2.0f / 4.0f * downscaled[(row/ 4) *(width / 4)] + 2.0f / 4.0f * downscaled[((row+4)/ 4) *(width / 4)];
                    upscaled[(row + 3) * width+col] = 1.0f / 4.0f * downscaled[(row/ 4) *(width / 4)] + 3.0f / 4.0f * downscaled[((row+4)/ 4) *(width / 4)];
                    upscaled[(row + 4) * width+col] = downscaled[((row+4)/ 4) *(width / 4)];

                }
                else if (row == height - 4){
                    upscaled[(row +1)* width+col] =upscaled[(row -3)*width];
                    upscaled[(row +2)* width+col] =upscaled[(row -2)*width];
                    upscaled[(row +3)* width+col] =upscaled[(row -1)*width];
                }
            }


            // Upscale the penultimate row
            else if (row == height - 2 && col % 4 == 0) {
                if (col < width - 4) {
                    upscaled[row * width + col] = downscaled[(row-2)/4 * (width/4) + col/4 ];  
                    upscaled[row * width + col +1] = 3.0f / 4.0f * downscaled[(row-2)/4 * (width/4) + col/4] + 1.0f / 4.0f * downscaled[(row-2)/4 * (width/4) + col/4 +1];
                    upscaled[row * width + col + 2] = 2.0f / 4.0f * downscaled[(row-2)/4 * (width/4) + col/4] + 2.0f / 4.0f * downscaled[(row-2)/4 * (width/4) + col/4 + 1];
                    upscaled[row * width + col + 3] = 1.0f / 4.0f * downscaled[(row-2)/4 * (width/4) + col/4] + 3.0f / 4.0f * downscaled[(row-2)/4 * (width/4) + col/4 +1];
                    upscaled[row * width + col + 4] = downscaled[(row-2)/4 * (width/4) + col/4 +1];
                    // upscaled[row * width + col + 4] = downscaled[(col/4)+ (width /4) * 3 +1];
                }
                // Copy the third last element's value from the fourth last value
                else if (col == width - 4) {
                    upscaled[row * width + col+1] = upscaled[row * width + col - 3];
                    upscaled[row * width + col+2] = upscaled[row * width + col -2];
                    upscaled[row * width + col+3] = upscaled[row * width + col - 1];
                }
            }

            // Upscale the penultimate column
            else if (col == width - 2 && row % 4 == 0) {
                if (row < height - 4){
                    // upscaled[row * width+col] = downscaled[(row/ 4) *(width / 4) +(width / 4) -1];
                    upscaled[row * width+col] = downscaled[(row/ 4) *(width / 4) +(col -2)/4];
                    upscaled[(row + 1) * width+col] = 3.0f / 4.0f * downscaled[(row/ 4) *(width / 4) +(col -2)/4]+ 1.0f / 4.0f * downscaled[((row+4)/ 4) *(width / 4) +(col -2)/4];
                    upscaled[(row + 2) * width+col] = 2.0f / 4.0f * downscaled[(row/ 4) *(width / 4) +(col -2)/4] + 2.0f / 4.0f * downscaled[((row+4)/ 4) *(width / 4) +(col -2)/4];
                    upscaled[(row + 3) * width+col] = 1.0f / 4.0f * downscaled[(row/ 4) *(width / 4) +(col -2)/4] + 3.0f / 4.0f * downscaled[((row+4)/ 4) *(width / 4) +(col -2)/4];
                    upscaled[(row + 4) * width+col] = downscaled[((row+4)/ 4) *(width / 4) +(col -2)/4];

                }
                else if (row == height - 4){
                    upscaled[(row +1)* width+col] =upscaled[(row -3)*width];
                    upscaled[(row +2)* width+col] =upscaled[(row -2)*width];
                    upscaled[(row +3)* width+col] =upscaled[(row -1)*width];
                }
            
            }


                    
            else if (col == 1 && row % 4 == 0) {
                // Calculate values except the last three elements
                if (row < height - 4){
                    upscaled[row * width+col] = downscaled[(row/ 4) *(width / 4)];
                    upscaled[(row + 1) * width+col] = 3.0f / 4.0f * downscaled[(row/ 4) *(width / 4)]+ 1.0f / 4.0f * downscaled[(row/ 4) *(width / 4) + (width /4)];
                    upscaled[(row + 2) * width+col] = 2.0f / 4.0f * downscaled[(row/ 4) *(width / 4)] + 2.0f / 4.0f * downscaled[(row/ 4) *(width / 4) +(width /4)];
                    upscaled[(row + 3) * width+col] = 1.0f / 4.0f * downscaled[(row/ 4) *(width / 4)] + 3.0f / 4.0f * downscaled[(row/ 4) *(width / 4) +(width /4)];
                    upscaled[(row + 4) * width+col] = downscaled[(row/ 4) *(width / 4) +(width /4)];

                }
                else if (row == height - 4){
                    upscaled[(row +1)* width+col] =upscaled[(row -3)*width];
                    upscaled[(row +2)* width+col] =upscaled[(row -2)*width];
                    upscaled[(row +3)* width+col] =upscaled[(row -1)*width];
                }
            
            }
            
            else if (row == 1 && col % 4 == 0) {
                // Calculate values except the last three elements
                if (col < width - 4) {
                    upscaled[row * width + col] = downscaled[(col/4)];  
                    upscaled[row * width + col +1] = 3.0f / 4.0f * downscaled[(col ) / 4] + 1.0f / 4.0f * downscaled[(col / 4) +1];
                    upscaled[row * width + col + 2] = 2.0f / 4.0f * downscaled[(col) / 4] + 2.0f / 4.0f * downscaled[(col / 4) +1];
                    upscaled[row * width + col + 3] = 1.0f / 4.0f * downscaled[(col) / 4] + 3.0f / 4.0f * downscaled[(col / 4) +1];
                    upscaled[row * width + col + 4] = downscaled[(col/4)+1];
                }
                // Copy the third last element's value from the fourth last value
                else if (col == width - 4) {
                    upscaled[row * width + col+1] = upscaled[row * width + col - 3];
                    upscaled[row * width + col+2] = upscaled[row * width + col -2];
                    upscaled[row * width + col+3] = upscaled[row * width + col - 1];
                }
            }

            // Copy to the last column
            else if (col == width - 1) {
                // if (row < height - 2) {
                //     upscaled[row * width + col] = upscaled[row * width + col - 1];
                // }
                if (row < height - 4){
                    // upscaled[row * width+col] = downscaled[(row/ 4) *(width / 4) +(width / 4) -1];
                    upscaled[row * width+col] = downscaled[(row/ 4) *(width / 4) +(col -3)/4];
                    upscaled[(row + 1) * width+col] = 3.0f / 4.0f * downscaled[(row/ 4) *(width / 4) +(col -3)/4]+ 1.0f / 4.0f * downscaled[((row+4)/ 4) *(width / 4) +(col -3)/4];
                    upscaled[(row + 2) * width+col] = 2.0f / 4.0f * downscaled[(row/ 4) *(width / 4) +(col -3)/4] + 2.0f / 4.0f * downscaled[((row+4)/ 4) *(width / 4) +(col -3)/4];
                    upscaled[(row + 3) * width+col] = 1.0f / 4.0f * downscaled[(row/ 4) *(width / 4) +(col -3)/4] + 3.0f / 4.0f * downscaled[((row+4)/ 4) *(width / 4) +(col -3)/4];
                    upscaled[(row + 4) * width+col] = downscaled[((row+4)/ 4) *(width / 4) +(col -2)/4];

                }
                else if (row == height - 4){
                    upscaled[(row +1)* width+col] =upscaled[(row -3)*width];
                    upscaled[(row +2)* width+col] =upscaled[(row -2)*width];
                    upscaled[(row +3)* width+col] =upscaled[(row -1)*width];
                }        
            }


            // Copy to the last row
            else if (row == height - 1) {
                // upscaled[row * width + col] = upscaled[(row - 1) * width + col];
                if (col < width - 4) {
                    upscaled[row * width + col] = downscaled[(row-3)/4 * (width/4) + col/4 ];  
                    upscaled[row * width + col +1] = 3.0f / 4.0f * downscaled[(row-3)/4 * (width/4) + col/4] + 1.0f / 4.0f * downscaled[(row-3)/4 * (width/4) + col/4 +1];
                    upscaled[row * width + col + 2] = 2.0f / 4.0f * downscaled[(row-3)/4 * (width/4) + col/4] + 2.0f / 4.0f * downscaled[(row-3)/4 * (width/4) + col/4 + 1];
                    upscaled[row * width + col + 3] = 1.0f / 4.0f * downscaled[(row-3)/4 * (width/4) + col/4] + 3.0f / 4.0f * downscaled[(row-3)/4 * (width/4) + col/4 +1];
                    upscaled[row * width + col + 4] = downscaled[(row-3)/4 * (width/4) + col/4 +1];
                    // upscaled[row * width + col + 4] = downscaled[(col/4)+ (width /4) * 3 +1];
                }
                // Copy the third last element's value from the fourth last value
                else if (col == width - 4) {
                    upscaled[row * width + col+1] = upscaled[row * width + col - 3];
                    upscaled[row * width + col+2] = upscaled[row * width + col -2];
                    upscaled[row * width + col+3] = upscaled[row * width + col - 1];
                }        
            }

            // Upscale the main body

                // UpscaleBodyKernel(downscaled, upscaled, width, height);
                // upscaled[row * width + col] = 0;
            UpscaleBody(downscaled, upscaled, static_cast<int>(width / 4), static_cast<int>(height /4));
        }
    }
    return upscaled;   
}
float* calculatePError(float* original, float* upscaled, int width, int height) {
    float* pError = new float[width * height];
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if (row < height && col < width) {
                pError[row * width + col] = original[row * width + col] - upscaled[row * width + col];
            }
        }
    }
    return pError;
}

float* SobelOperator(float* input, int width, int height) {
    float* pEdge = new float[width * height];
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if (row == 0 || col == 0 || row == height - 1 || col == width - 1) {
                pEdge[row * width + col] = 0.0f;
            } else {
                float sobelX = -1.0f * input[(row - 1) * width + (col - 1)] + 0.0f * input[(row - 1) * width + col] + 1.0f * input[(row - 1) * width + (col + 1)]
                             -2.0f * input[row * width + (col - 1)] + 0.0f * input[row * width + col] + 2.0f * input[row * width + (col + 1)]
                             -1.0f * input[(row + 1) * width + (col - 1)] + 0.0f * input[(row + 1) * width + col] + 1.0f * input[(row + 1) * width + (col + 1)];

                float sobelY = 1.0f * input[(row - 1) * width + (col + 1)] + 2.0f * input[row * width + (col + 1)] + 1.0f * input[(row + 1) * width + (col + 1)]
                             -1.0f * input[(row - 1) * width + (col - 1)] - 2.0f * input[row * width + (col - 1)] - 1.0f * input[(row + 1) * width + (col - 1)];

                pEdge[row * width + col] = static_cast<float>(sqrt(pow(sobelX, 2) + pow(sobelY, 2)));
            }
        }
    }
    return pEdge;
}
float CalculateMean(float* pEdge, int width, int height) {
    float mean = 0.0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            mean += pEdge[i * width + j];
        }
    }
    return mean / static_cast<float>(width * height);
}
float* preliminarySharpened( float* pEdge, float* pError, float* upscaleMatrix, int width, int height, float mean, float lightStrength) {
    float* result = new float[width * height];
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if (row < height && col < width) {
                // Apply brightness adjustment to pEdge array 
                pEdge[row * width + col] = pEdge[row * width + col] * lightStrength - mean;

                result[row * width + col] = upscaleMatrix[row * width + col] + pError[row * width + col] + pEdge[row * width + col];
            }
        }
    }
    return result;
}

float* OvershootControl( float* preliminarySharpened, float* original, int width, int height) {
    float* finalSharpened = new float[width * height];
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if (row == 0 || col == 0 || row == height - 1 || col == width - 1) {
                finalSharpened[row * width + col] = preliminarySharpened[row * width + col];
            } else {
                int submatrixSize = 3;  // Size of the submatrix (3x3)
                float maxVal = -1.0f;   // Initialize max value to a small value
                float minVal = 256.0f;  // Initialize min value to a large value 

                // Find the max and min values in the 3x3 submatrix
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        float value = original[(row + i) * width + (col + j)];
                        maxVal = std::max(maxVal, value);
                        minVal = std::min(minVal, value);
                    }
                }

                float oscMax = preliminarySharpened[row * width + col] - maxVal;
                float oscMin = minVal - preliminarySharpened[row * width + col];

                if (preliminarySharpened[row * width + col] > maxVal) {
                    finalSharpened[row * width + col] = std::min(oscMax, 255.0f);
                } else if (preliminarySharpened[row * width + col] < minVal) {
                    finalSharpened[row * width + col] = std::max(oscMin, 0.0f);
                } else {
                    finalSharpened[row * width + col] = std::min(std::max(preliminarySharpened[row * width + col], 0.0f), 255.0f);
                }
            }
        }
    }
    return finalSharpened;
}


void processAndSaveImage(const string& inputImagePath, const string& outputImagePath, double kRescaleFactor) {
    // Read the input image
    Mat inputImage = imread(inputImagePath, IMREAD_GRAYSCALE);

    if (inputImage.empty()) {
        cerr << "Error: Could not read the input image." << endl;
        return;
    }

    // Rescale the input image
    Mat rescaledMat;
    resize(inputImage, rescaledMat, Size(0, 0), kRescaleFactor, kRescaleFactor);

    // Convert the rescaledMat to float
    Mat rescaledFloatMat;
    rescaledMat.convertTo(rescaledFloatMat, CV_32F);

    // Get the width and height of the rescaled image
    int rescaledWidth = rescaledFloatMat.cols;
    int rescaledHeight = rescaledFloatMat.rows;

    // Call the downscaleImageCPU function
    float* downscaleResult = downscaleImageCPU((float*)rescaledFloatMat.data, rescaledWidth, rescaledHeight);

    // Call the upscaleOperationCPU function
    float* upscaleResult = upscaleOperationCPU(downscaleResult, rescaledWidth / 4, rescaledHeight / 4);

    // Convert the result back to Mat
    Mat finalImage(rescaledHeight, rescaledWidth, CV_32F, upscaleResult);

    // Convert the final image to uint8 format for saving
    Mat finalImageUint8;
    finalImage.convertTo(finalImageUint8, CV_8U);

    // Save the final image to the specified path
    imwrite(outputImagePath, finalImageUint8);

    // Free the allocated memory
    delete[] downscaleResult;
    delete[] upscaleResult;
}

int main() {
    // Get user input for kRescaleFactor
    double kRescaleFactor;
    cout << "Enter the rescale factor (VD: 0.75): ";
    cin >> kRescaleFactor;

    // Specify input and output paths
    string inputImagePath = "C:/Users/Admin/Desktop/imageSharpening/treeNew.jpg";
    string outputImagePath = "C:/Users/Admin/Desktop/imageSharpening/finalSharpened.jpg";

    // Call the processing and saving function
    processAndSaveImage(inputImagePath, outputImagePath, kRescaleFactor);

    return 0;
}