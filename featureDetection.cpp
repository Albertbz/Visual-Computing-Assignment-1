#include "featureDetection.hpp"

#include <fstream>
#include <iostream>
#include <opencv2/stitching/detail/blenders.hpp>

using namespace cv;

auto detectComputeSIFT(Mat img) {
    struct result {
        std::vector<KeyPoint> keypoints;
        Mat descriptors;
    };

    Ptr<SIFT> siftPtr = SIFT::create();

    // Detect all keypoints and compute descriptors
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    siftPtr->detectAndCompute(img, noArray(), keypoints, descriptors);

    std::cout << "Amount of keypoints (SIFT): " << keypoints.size()
              << std::endl;
    return result{keypoints, descriptors};
}

auto detectComputeORB(Mat img) {
    struct result {
        std::vector<KeyPoint> keypoints;
        Mat descriptors;
    };

    Ptr<ORB> orbPtr = ORB::create(3000);

    // Detect all keypoints and compute descriptors
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    orbPtr->detectAndCompute(img, noArray(), keypoints, descriptors);

    std::cout << "Amount of keypoints (ORB): " << keypoints.size() << std::endl;
    return result{keypoints, descriptors};
}

auto computeMatchesORB(Mat descriptors1, Mat descriptors2) {
    FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));

    std::vector<std::vector<DMatch>> knnMatches;

    auto startTime = std::chrono::high_resolution_clock::now();

    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

    const float ratioThresh = 1.0f;
    std::vector<DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance <
            ratioThresh * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();

    std::chrono::milliseconds timeDifferenceMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(endTime -
                                                              startTime);
    std::cout << "Time to compute matches: " << timeDifferenceMs.count()
              << std::endl;

    return goodMatches;
}

auto computeMatchesSIFT(Mat descriptors1, Mat descriptors2) {
    Ptr<DescriptorMatcher> matcher =
        DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    std::vector<std::vector<DMatch>> knnMatches;

    auto startTime = std::chrono::high_resolution_clock::now();

    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

    const float ratioThresh = 0.7f;
    std::vector<DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance <
            ratioThresh * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();

    std::chrono::milliseconds timeDifferenceMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(endTime -
                                                              startTime);
    std::cout << "Time to compute matches: " << timeDifferenceMs.count()
              << std::endl;

    return goodMatches;
}

void drawHistogram(Mat distances, float upperRange, std::string name) {
    int histSize = 256;
    float range[] = {0, upperRange};
    const float* histRange[] = {range};

    Mat hist;
    calcHist(&distances, 1, 0, Mat(), hist, 1, &histSize, histRange, true,
             false);

    int histW = 512, histH = 400;
    int binW = cvRound((double)histW / histSize);

    Mat histImage(histH, histW, CV_8UC3, Scalar(0, 0, 0));

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++) {
        line(histImage,
             Point(binW * (i - 1), histH - cvRound(hist.at<float>(i - 1))),
             Point(binW * (i), histH - cvRound(hist.at<float>(i))),
             Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow(name, histImage);
}

void saveDistancesToFile(std::vector<DMatch> matches, std::string fileName) {
    std::ofstream file;
    file.open(fileName);
    for (size_t i = 0; i < matches.size(); i++) {
        file << matches[i].distance << "\n";
    }
    file.close();
}

Mat stitch(Mat img1, Mat img2, std::vector<KeyPoint> keypoints1,
           std::vector<KeyPoint> keypoints2, std::vector<DMatch> matches,
           bool useFeathering) {
    std::vector<Point2f> img1Points;
    std::vector<Point2f> img2Points;

    // Get the points from the keypoints
    for (size_t i = 0; i < matches.size(); i++) {
        img1Points.push_back(keypoints1[matches[i].queryIdx].pt);
        img2Points.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // Test RANSAC with some different thresholds
    for (double thresh : {1.0, 2.0, 3.0, 5.0, 10.0, 20.0}) {
        Mat mask;
        TickMeter tm;
        tm.start();

        Mat H = findHomography(img1Points, img2Points, RANSAC, thresh, mask);

        tm.stop();

        int inlierCount = countNonZero(mask);
        std::cout << "Threshold = " << thresh << " -> inliers: " << inlierCount
                  << " / " << img1Points.size()
                  << ", runtime: " << tm.getTimeMilli() << " ms" << std::endl;
    }

    // Find the homography
    Mat H = findHomography(img1Points, img2Points, RANSAC, 2.0);

    // Find the corners of the first image to use for the warping
    std::vector<Point2f> corners(4);
    corners[0] = Point2f(0, 0);
    corners[1] = Point2f((float)img1.cols, 0);
    corners[2] = Point2f((float)img1.cols, (float)img1.rows);
    corners[3] = Point2f(0, (float)img1.rows);

    std::vector<Point2f> warpedCorners;
    perspectiveTransform(corners, warpedCorners, H);

    // Use the warped corners to find the size of the new image
    Rect bbox = boundingRect(warpedCorners);

    int offsetX = max(-bbox.x, 0);
    int offsetY = max(-bbox.y, 0);
    int width = max(bbox.x + bbox.width, img2.cols) + offsetX;
    int height = max(bbox.y + bbox.height, img2.rows) + offsetY;

    // Translate the homography to be able to stitch them together
    Mat translation =
        (Mat_<double>(3, 3) << 1, 0, offsetX, 0, 1, offsetY, 0, 0, 1);

    Mat translatedH = translation * H;

    // Warp image1
    Mat result;
    warpPerspective(img1, result, translatedH, Size(width, height));

    if (useFeathering) {
        detail::FeatherBlender blender;
        blender.setSharpness(0.02f);

        int canvasWidth = std::max(result.cols, offsetX + img2.cols);
        int canvasHeight = std::max(result.rows, offsetY + img2.rows);
        blender.prepare(Rect(0, 0, canvasWidth, canvasHeight));

        Mat img2S, resultS;
        img2.convertTo(img2S, CV_16S);
        result.convertTo(resultS, CV_16S);

        Mat maskImg2, maskResult;
        cvtColor(img2, maskImg2, COLOR_BGR2GRAY);
        cvtColor(result, maskResult, COLOR_BGR2GRAY);
        threshold(maskImg2, maskImg2, 0, 255, THRESH_BINARY);
        threshold(maskResult, maskResult, 0, 255, THRESH_BINARY);

        blender.feed(resultS, maskResult, Point(0, 0));
        blender.feed(img2S, maskImg2, Point(offsetX, offsetY));

        Mat blendedS, blendMask;
        blender.blend(blendedS, blendMask);

        Mat blended;
        blendedS.convertTo(blended, CV_8U);
        return blended;
    } else {
        // Copy img2 onto img1
        Mat roi(result, Rect(offsetX, offsetY, img2.cols, img2.rows));

        img2.copyTo(roi);
        return result;
    }

    // return result;
}

detectAndMatchResult detectAndMatchSIFT(Mat img1, Mat img2) {
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Image could not be loaded." << std::endl;
    }

    // Detect keypoints and compute descriptors with SIFT
    auto img1Res = detectComputeSIFT(img1);
    auto img2Res = detectComputeSIFT(img2);

    std::vector<DMatch> matches =
        computeMatchesSIFT(img1Res.descriptors, img2Res.descriptors);

    Mat imgMatches;
    drawMatches(img1, img1Res.keypoints, img2, img2Res.keypoints, matches,
                imgMatches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Matches SIFT", imgMatches);
    imwrite("Matches SIFT.png", imgMatches);

    saveDistancesToFile(matches, "Distances SIFT.txt");

    return detectAndMatchResult{img1Res.keypoints, img2Res.keypoints, matches};
}

detectAndMatchResult detectAndMatchORB(Mat img1, Mat img2) {
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Image could not be loaded." << std::endl;
    }

    // Detect keypoints and compute discriptors
    auto img1Res = detectComputeORB(img1);
    auto img2Res = detectComputeORB(img2);

    auto matches = computeMatchesORB(img1Res.descriptors, img2Res.descriptors);

    Mat imgMatches;
    drawMatches(img1, img1Res.keypoints, img2, img2Res.keypoints, matches,
                imgMatches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Matches ORB", imgMatches);

    saveDistancesToFile(matches, "Distances ORB.txt");

    return detectAndMatchResult{img1Res.keypoints, img2Res.keypoints, matches};
}

void drawKeypointsAndShow(Mat img, std::vector<KeyPoint> keypoints,
                          String showAs) {
    Mat imgKeypoints;
    drawKeypoints(img, keypoints, imgKeypoints);
    imshow(showAs, imgKeypoints);
}