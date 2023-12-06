#include <iostream>

int main(int, char**){
    std::cout << "Hello, from imageSharpening!\n";
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

    // // Call the upscaleOperationCPU function
    // float* h_upscaled = upscaleOperationCPU(downscaleResult, rescaledWidth, rescaledHeight);

    // float* h_pError = calculatePError((float*)rescaledFloatMat.data, h_upscaled, rescaledWidth, rescaledHeight);
    // float* h_pEdge =SobelOperator((float*)rescaledFloatMat.data, rescaledWidth, rescaledHeight);

    // float mean = CalculateMean(h_pEdge, rescaledWidth, rescaledHeight);

    // float lightStrength = 0.125f;
    // float* h_preliminary = preliminarySharpened( h_pEdge, h_pError, h_upscaled, rescaledWidth, rescaledWidth, mean, lightStrength);
    // float* h_finalSharpened = OvershootControl( h_preliminary, (float*)rescaledFloatMat.data, rescaledWidth, rescaledHeight);


    // Convert the result back to Mat
    Mat finalImage(rescaledHeight, rescaledWidth, CV_32F, downscaleResult);

    // Convert the final image to uint8 format for saving
    Mat finalImageUint8;
    finalImage.convertTo(finalImageUint8, CV_8U);

    // Save the final image to the specified path
    imwrite(outputImagePath, finalImageUint8);

    // Free the allocated memory
    delete[] downscaleResult;
    // delete[] h_upscaled;
    // delete[] h_pError;
    // delete[] h_pEdge;
    // delete[] h_preliminary;
    // delete[] h_finalSharpened;
}

// int main() {
//     // Get user input for kRescaleFactor
//     double kRescaleFactor;
//     cout << "Enter the rescale factor (VD: 0.75): ";
//     cin >> kRescaleFactor;

//     // Specify input and output paths
//     string inputImagePath = "C:/Users/Admin/Desktop/imageSharpening/treeNew.jpg";
//     string outputImagePath = "C:/Users/Admin/Desktop/imageSharpening/finalSharpened2.jpg";

//     // Call the processing and saving function
//     processAndSaveImage(inputImagePath, outputImagePath, kRescaleFactor);

//     return 0;
// }