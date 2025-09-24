#include <chrono>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

struct detectAndMatchResult {
    std::vector<KeyPoint> keypoints1;
    std::vector<KeyPoint> keypoints2;
    std::vector<DMatch> matches;
};

/**
 * Detect and match.
 */
detectAndMatchResult detectAndMatchSIFT(Mat img1, Mat img2);

detectAndMatchResult detectAndMatchORB(Mat img1, Mat img2);

/**
 * Stitch two images together using their keypoints and matches.
 */
Mat stitch(Mat img1, Mat img2, std::vector<KeyPoint> keypoints1,
           std::vector<KeyPoint> keypoints2, std::vector<DMatch> matches,
           bool useFeathering);

/**
 * Detect the keypoints and compute the descriptors given an image using
 * SIFT.
 */
auto detectComputeSIFT(Mat img);

/**
 * Detect the keypoints and compute the descriptors given an image using ORB.
 */
auto detectComputeORB(Mat img);

/**
 * Draw keypoints on an image and show it.
 */
void drawKeypointsAndShow(Mat img, std::vector<KeyPoint> keypoints,
                          String showAs);

/**
 * Computes the matches given some descriptors made with SIFT.
 */
auto computeMatchesSIFT(Mat descriptors1, Mat descriptors2);

/**
 * Computes the matches given some descriptors made with ORB.
 */
auto computeMatchesORB(Mat descriptors1, Mat descriptors2);

void saveDistancesToFile(std::vector<DMatch> matches, std::string fileName);

/**
 * Draw a histogram given some distances and an upper range.
 */
void drawHistogram(Mat distances, float upperRange, std::string name);