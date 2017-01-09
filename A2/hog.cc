#include "hog.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>

using namespace std;

class HOG::HOGPimpl {
public:

    cv::Mat1f descriptors;
    cv::Mat1f responses;

    cv::SVM svm;
    cv::HOGDescriptor hog;
};


/// Constructor
HOG::HOG()
{
    using namespace cv;
    pimpl = std::shared_ptr<HOGPimpl>(new HOGPimpl());
    pimpl->hog.blockStride = Size(16, 14); // mod 48=2^4*3, 112=2^6*7
    pimpl->hog.nbins = 20;
    pimpl->hog.nlevels = 200;
}

/// Destructor
HOG::~HOG()
{
}

/// Start the training.  This resets/initializes the model.
void HOG::startTraining()
{
}

/// Image preprocessing before training / prediction
cv::Mat3b preprocess(const cv::Mat3b& img) {
    cv::Mat3b img2 = img(cv::Rect((img.cols - 64) / 2, (img.rows - 128) / 2, 64, 128));
    return img2;
}

//int writeout = 1;
/// Add a new training image.
///
/// @param img:  input image
/// @param bool: value which specifies if img represents a person
void HOG::train(const cv::Mat3b& img, bool isPerson)
{
    cv::Mat3b img2 = preprocess(img);
    vector<float> vDescriptor;
    pimpl->hog.compute(img2, vDescriptor);
    cv::Mat1f descriptor(1, vDescriptor.size(), &vDescriptor[0]);

    pimpl->descriptors.push_back(descriptor);
    pimpl->responses.push_back(cv::Mat1f(1, 1, float(isPerson)));
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
void HOG::finishTraining()
{
    cv::SVMParams params;
    params.svm_type = cv::SVM::C_SVC;
    params.kernel_type = cv::SVM::POLY;
    params.nu = 0.5;  // NU_SVC / ONE_CLASS / NU_SVR
    params.p = 0.3;  // EPS_SVR
    params.gamma = 0.30;  // POLY / RBF / SIGMOID
    params.coef0 = 2;  // POLY / SIGMOID
    params.C = 10;  // C_SVC / EPS_SVR / NU_SVR
    params.degree = 3;  // POLY
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 2000, 1e-6);
    pimpl->svm.train( pimpl->descriptors, pimpl->responses, cv::Mat(), cv::Mat(), params );
}

/// Classify an unknown test image.  The result is a floating point
/// value directly proportional to the probability of being a person.
///
/// @param img: unknown test image
/// @return:    probability of human likelihood
double HOG::classify(const cv::Mat3b& img)
{
    cv::Mat3b img2 = preprocess(img);
    vector<float> vDescriptor;
    pimpl->hog.compute(img2, vDescriptor);
    cv::Mat1f descriptor(1, vDescriptor.size(), &vDescriptor[0]);

    return -pimpl->svm.predict(descriptor, true);
}

