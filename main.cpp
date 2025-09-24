#include <iostream>
#include <opencv2/opencv.hpp>

#include "featureDetection.hpp"
using namespace cv;

int main() {
    Mat img1 = imread("../images/img1.jpeg");
    Mat img2 = imread("../images/img2.jpeg");

    // Using SIFT
    auto siftRes = detectAndMatchSIFT(img1, img2);
    waitKey(0);
    destroyAllWindows();

    Mat stitchedImgSIFT = stitch(img1, img2, siftRes.keypoints1,
                                 siftRes.keypoints2, siftRes.matches, false);
    Mat stitchedImgSIFTFeathering =
        stitch(img1, img2, siftRes.keypoints1, siftRes.keypoints2,
               siftRes.matches, true);

    imshow("Stitched image SIFT", stitchedImgSIFT);
    imshow("Stitched image SIFT - Feathering", stitchedImgSIFTFeathering);

    // Using ORB
    // auto orbRes = detectAndMatchORB(img1, img2);
    // Mat stitchedImgORB = stitch(img1, img2, orbRes.keypoints1,
    //                             orbRes.keypoints2, orbRes.matches, false);
    // imshow("Stitched image ORB", stitchedImgORB);

    // Wait indefinitely for a user key press.
    waitKey(0);

    // Release all resources.
    destroyAllWindows();

    return 0;
}