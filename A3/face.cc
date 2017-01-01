#include "face.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

using namespace std;

struct FACE::FACEPimpl {
    std::vector<double> pos, neg;
    std::vector<std::vector<double>> allPos, allNeg;
    std::vector<std::pair<cv::Point2i, cv::Point2i>> pts;
};

std::vector<cv::Mat> db_0;
int n_imgs;
int ind;
int total;
cv::PCA pca_0, pca_1, pca_2;
int numPrincipalComponents = 50;
vector<int> mean_;
int width;

/// Constructor
FACE::FACE() {
}

/// Destructor
FACE::~FACE() {
}

/// Start the training.  This resets/initializes the model.
void FACE::startTraining() {
    n_imgs = 0;
    ind = 0;
    db_0.clear();
}


/// Image preprocessing before training / prediction
cv::Mat preprocess(const cv::Mat3b& img1) {
    // cv::blur(img1, img1, cv::Size(3, 3));

    // only take the face
    int r = img1.rows;
    int c = img1.cols;
    float border = 1.0 / 10.0;
    cv::Rect rect(r * border,
                  c * border,
                  r - 2 * r * border,
                  c - 2 * c * border);
    cv::Mat imga = img1(rect).clone();
    return imga.reshape(1, 1);
}

/// Add a new person.
///
/// @param img1:  250x250 pixel image containing a scaled and aligned face
/// @param img2:  250x250 pixel image containing a scaled and aligned face
/// @param same: true if img1 and img2 belong to the same person

void FACE::train(const cv::Mat3b& img1, const cv::Mat3b& img2, bool same) {
    if (n_imgs >= 300)  // speed things up
    {
        return;
    }

    db_0.push_back(preprocess(img1));
    db_0.push_back(preprocess(img2));

    n_imgs += 2;
    total = db_0[0].rows * db_0[0].cols;
}


/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
void FACE::finishTraining() {
    cv::Mat mat_0(total, n_imgs, CV_32FC1);
    mean_.reserve(total);
    width = db_0[0].rows;

    // Calculate mean
    for (unsigned int i = 0; i < db_0.size(); i++) {
        for (int j = 0; j < total; ++j)
        {
            // Compute mean
            mean_[j] += double(db_0[i].at<uchar>(0, j)) / db_0.size();
        }
    }


    // Subtract mean and add matrix for PCA
    for (unsigned int i = 0; i < db_0.size(); i++) {
        for (int j = 0; j < total; ++j)
        {
            db_0[i].at<uchar>(0, j) -= mean_[j];
        }
        db_0[i].reshape(1, total).copyTo(mat_0.col(i));
    }
    pca_0 = cv::PCA(mat_0, cv::Mat(), CV_PCA_DATA_AS_COL, numPrincipalComponents);

    // Export computed mean , eigen vectors & eigen values Matrix
    /*    cv::FileStorage fs("export.dat", cv::FileStorage::WRITE);
        fs << "mean" << pca_0.mean;
        fs << "eigenvectors" << pca_0.eigenvectors;
        fs << "eigenvalues" << pca_0.eigenvalues;
        fs.release();*/

}


/// Verify if img corresponds to the provided name.  The result is a floating point
/// value directly proportional to the probability of being correct.
///
/// @param img1:  250x250 pixel image containing a scaled and aligned face
/// @param img2:  250x250 pixel image containing a scaled and aligned face
/// @return:    similarity score between both images
double FACE::verify(const cv::Mat3b& img1, const cv::Mat3b& img2) {
    cv::Mat img_flat_a = preprocess(img1);
    cv::Mat img_flat_b = preprocess(img2);

    cv::Mat test_a0(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat test_b0(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat compress_a0(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat compress_b0(numPrincipalComponents, 1, CV_32FC1);

    // Subtract mean
    for (int j = 0; j < total; ++j)
    {
        img_flat_a.at<uchar>(0, j) -= mean_[j];
        img_flat_b.at<uchar>(0, j) -= mean_[j];
    }

    img_flat_a.reshape(1, total).copyTo(test_a0);
    img_flat_b.reshape(1, total).copyTo(test_b0);

    pca_0.project(test_a0, compress_a0.col(0));
    pca_0.project(test_b0, compress_b0.col(0));

    // Cosine distance
    //return (1.0 + compress_a0.dot(compress_b0)/(cv::norm(compress_a0, cv::NORM_L2) * cv::norm(compress_b0, cv::NORM_L2)))/2.0;
    return cv::norm(compress_a0 - compress_b0, cv::NORM_L2);
}

