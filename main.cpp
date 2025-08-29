#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // Load an image from the filesystem.
    // The path is relative to the execution directory.
    cv::Mat image = cv::imread("../test.jpg");

    // Verify that the image data has been loaded successfully.
    if (image.empty()) {
        std::cerr << "Error: Image could not be loaded." << std::endl;
        return -1;
    }

    // Create a window for image display.
    cv::namedWindow("Image Display", cv::WINDOW_AUTOSIZE);

    // Render the image in the created window.
    cv::imshow("Image Display", image);

    // Wait indefinitely for a user key press.
    cv::waitKey(0);

    // Release all resources.
    cv::destroyAllWindows();

    return 0;
}