#include <iostream>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>



double getPSNR(const cv::Mat& I1, const cv::Mat& I2) {
    cv::Mat s1;
    cv::absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);     // cannot make a square on 8 bits
    s1 = s1.mul(s1);              // |I1 - I2|^2
    cv::Scalar s = cv::sum(s1);    // sum elements per channel
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
    if (sse <= 1e-10) // for small values return zero
        return 0;
    else {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

cv::Scalar getMSSIM(const cv::Mat& i1, const cv::Mat& i2) {
    const double C1 = 6.5025, C2 = 58.5225;

    int d = CV_32F;
    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    cv::Mat I2_2 = I2.mul(I2);        // I2^2
    cv::Mat I1_2 = I1.mul(I1);        // I1^2
    cv::Mat I1_I2 = I1.mul(I2);        // I1 * I2

    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);
    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
    cv::Scalar mssim = cv::mean(ssim_map); // mssim = average of ssim map
    return mssim;
}

double calculateVIFP(const cv::Mat& image1, const cv::Mat& image2) {
    cv::Mat diff;
    cv::absdiff(image1, image2, diff);

    cv::Scalar mean_diff, stddev_diff;
    cv::meanStdDev(diff, mean_diff, stddev_diff);

    double vifp = stddev_diff[0] * stddev_diff[0];
    return vifp;
}


int main() {
    // Load images
    cv::Mat image1 = cv::imread("C:/Users/Admin/Desktop/imageSharpening/aircraft.png");
    cv::Mat image2 = cv::imread("C:/Users/Admin/Desktop/imageSharpening/finalSharpened.png");

    if (image1.empty() || image2.empty()) {
        std::cerr << "Error loading images." << std::endl;
        return -1;
    }

    // Calculate PSNR
    double psnr = getPSNR(image1, image2);
    std::cout << "PSNR: " << psnr << " dB" << std::endl;

    // Calculate SSIM
    cv::Scalar ssim = getMSSIM(image1, image2);
    std::cout << "SSIM: " << ssim.val[0] << std::endl;

    double vifp = calculateVIFP(image1, image2);
    std::cout << "VIFP (Pixel Domain): " << vifp << std::endl;

    return 0;
}


// void processAndSaveImage(const string& inputImagePath, const string& outputImagePath, double kRescaleFactor) {
//     // Read the input image
//     Mat inputImage = imread(inputImagePath, IMREAD_GRAYSCALE);

//     if (inputImage.empty()) {
//         cerr << "Error: Could not read the input image." << endl;
//         return;
//     }

//     // Rescale the input image
//     Mat rescaledMat;
//     resize(inputImage, rescaledMat, Size(0, 0), kRescaleFactor, kRescaleFactor);

//     // Convert the rescaledMat to float
//     Mat rescaledFloatMat;
//     rescaledMat.convertTo(rescaledFloatMat, CV_32F);

//     // Get the width and height of the rescaled image
//     int rescaledWidth = rescaledFloatMat.cols;
//     int rescaledHeight = rescaledFloatMat.rows;

//     // Call the downscaleImageCPU function
//     float* downscaleResult = downscaleImageCPU((float*)rescaledFloatMat.data, rescaledWidth, rescaledHeight);

//     // // Call the upscaleOperationCPU function
//     // float* h_upscaled = upscaleOperationCPU(downscaleResult, rescaledWidth, rescaledHeight);

//     // float* h_pError = calculatePError((float*)rescaledFloatMat.data, h_upscaled, rescaledWidth, rescaledHeight);
//     // float* h_pEdge =SobelOperator((float*)rescaledFloatMat.data, rescaledWidth, rescaledHeight);

//     // float mean = CalculateMean(h_pEdge, rescaledWidth, rescaledHeight);

//     // float lightStrength = 0.125f;
//     // float* h_preliminary = preliminarySharpened( h_pEdge, h_pError, h_upscaled, rescaledWidth, rescaledWidth, mean, lightStrength);
//     // float* h_finalSharpened = OvershootControl( h_preliminary, (float*)rescaledFloatMat.data, rescaledWidth, rescaledHeight);


//     // Convert the result back to Mat
//     Mat finalImage(rescaledHeight, rescaledWidth, CV_32F, downscaleResult);

//     // Convert the final image to uint8 format for saving
//     Mat finalImageUint8;
//     finalImage.convertTo(finalImageUint8, CV_8U);

//     // Save the final image to the specified path
//     imwrite(outputImagePath, finalImageUint8);

//     // Free the allocated memory
//     delete[] downscaleResult;
//     // delete[] h_upscaled;
//     // delete[] h_pError;
//     // delete[] h_pEdge;
//     // delete[] h_preliminary;
//     // delete[] h_finalSharpened;
// }

// // int main() {
// //     // Get user input for kRescaleFactor
// //     double kRescaleFactor;
// //     cout << "Enter the rescale factor (VD: 0.75): ";
// //     cin >> kRescaleFactor;

// //     // Specify input and output paths
// //     string inputImagePath = "C:/Users/Admin/Desktop/imageSharpening/treeNew.jpg";
// //     string outputImagePath = "C:/Users/Admin/Desktop/imageSharpening/finalSharpened2.jpg";

// //     // Call the processing and saving function
// //     processAndSaveImage(inputImagePath, outputImagePath, kRescaleFactor);

// //     return 0;
// // }

