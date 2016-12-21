#include "face.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>
#include <algorithm>

using namespace std;

static const int DESCRIPTOR_SIZE = 25600;

struct FACE::FACEPimpl {
	
	std::vector<double> pos, neg;
	std::vector<std::vector<double>> allPos, allNeg;
	std::vector<std::pair<cv::Point2i,cv::Point2i>> pts;
};

std::vector<cv::Mat> db_0, db_1, db_2;
int n_imgs;
int ind;
int total;
cv::PCA pca_0, pca_1, pca_2;
int numPrincipalComponents = 100;


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
    db_1.clear();
    db_2.clear();
}

/// Add a new person.
///
/// @param img1:  250x250 pixel image containing a scaled and aligned face
/// @param img2:  250x250 pixel image containing a scaled and aligned face
/// @param same: true if img1 and img2 belong to the same person

void FACE::train(const cv::Mat3b& img1, const cv::Mat3b& img2, bool same) {

    int r = img1.rows;
    int c = img1.cols;
    cv::Rect rect(r/5.0, c/5.0, r - 2*r/5.0, c - 2*c/5.0);
    cv::Mat imga = img1(rect).clone();
    cv::Mat imgb = img2(rect).clone();

    n_imgs+=2;
    total = imga.rows * imga.cols;



    cv::Mat channels_a[3];
    cv::Mat channels_b[3];
    cv::split(imga ,channels_a);
    cv::split(imgb ,channels_b);

    db_0.push_back(channels_a[0]);
    db_0.push_back(channels_b[0]);    
    db_1.push_back(channels_a[1]);
    db_1.push_back(channels_b[1]);    
    db_2.push_back(channels_a[2]);
    db_2.push_back(channels_b[2]);

    /*
    std::string s = "tmp/pic" + std::to_string(ind) + ".jpg";
    cv::imwrite(s, imga );
    ind++;
    cv::imwrite(s, imgb );
    ind++;
    */

}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
void FACE::finishTraining() {

    cv::Mat mat_0(total, n_imgs, CV_32FC1);
    cv::Mat mat_1(total, n_imgs, CV_32FC1);
    cv::Mat mat_2(total, n_imgs, CV_32FC1);
    cout << mat_0.size() << endl;
    for(int i = 0; i < db_0.size(); i++) {
        db_0[i].reshape(1, total).copyTo(mat_0.col(i));
        db_1[i].reshape(1, total).copyTo(mat_1.col(i));
        db_2[i].reshape(1, total).copyTo(mat_2.col(i));
    }

    cout << "finish sopying" << endl;
    pca_0 = cv::PCA(mat_0, cv::Mat(), CV_PCA_DATA_AS_COL, numPrincipalComponents);
    //pca_1 = cv::PCA(mat_1, cv::Mat(), CV_PCA_DATA_AS_COL, numPrincipalComponents);
    //pca_2 = cv::PCA(mat_2, cv::Mat(), CV_PCA_DATA_AS_COL, numPrincipalComponents);

    cout << "abaa" << endl;
}

/// Verify if img corresponds to the provided name.  The result is a floating point
/// value directly proportional to the probability of being correct.
///
/// @param img1:  250x250 pixel image containing a scaled and aligned face
/// @param img2:  250x250 pixel image containing a scaled and aligned face
/// @return:    similarity score between both images
double FACE::verify(const cv::Mat3b& img1, const cv::Mat3b& img2) {
	

//	return rand()%256;
//	return -cv::norm(img1-img2);
    int r = img1.rows;
    int c = img1.cols;
    cv::Rect rect(r/5.0, c/5.0, r - 2*r/5.0, c - 2*c/5.0);
    cv::Mat3b imga = img1(rect).clone();
    cv::Mat3b imgb = img2(rect).clone();

    cv::Mat channels_a[3];
    cv::Mat channels_b[3];
    cv::split(imga ,channels_a);
    cv::split(imgb ,channels_b);
	
    cv::Mat test_a0(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat test_a1(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat test_a2(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat test_b0(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat test_b1(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat test_b2(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat test_c0(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat test_c1(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat test_c2(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat compress_a0(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat compress_a1(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat compress_a2(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat compress_b0(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat compress_b1(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat compress_b2(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat compress_c0(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat compress_c1(numPrincipalComponents, 1, CV_32FC1);
    cv::Mat compress_c2(numPrincipalComponents, 1, CV_32FC1);

    channels_a[0].reshape(1, total).copyTo(test_a0);
    channels_a[1].reshape(1, total).copyTo(test_a1);
    channels_a[2].reshape(1, total).copyTo(test_a2);
    channels_b[0].reshape(1, total).copyTo(test_b0);
    channels_b[1].reshape(1, total).copyTo(test_b1);
    channels_b[2].reshape(1, total).copyTo(test_b2);

    pca_0.project(test_a0, compress_a0.col(0));
    //pca_1.project(test_a1, compress_a1.col(0));
    //pca_2.project(test_a2, compress_a2.col(0));
    pca_0.project(test_b0, compress_b0.col(0));
    //pca_1.project(test_b1, compress_b1.col(0));
    //pca_2.project(test_b2, compress_b2.col(0));
    
    return cv::norm(compress_a0 - compress_b0, cv::NORM_L2);
}

