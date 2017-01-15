#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include <map>
#include <algorithm>
#include <set>
#include <opencv2/opencv.hpp>

using namespace cv;
cv::Mat1f descriptors;
cv::Mat1f responses;

cv::SVM svm;


cv::Mat1f computeDescriptor(const cv::Mat3b& img, int x, int y) {
    cv::Mat1f vDescriptor;
    // top left
    if (x == 0 || y == 0)
    {
        vDescriptor.push_back(cv::Mat1f(1, 1, 0.f));
        vDescriptor.push_back(cv::Mat1f(1, 1, 0.f));
    } else{
        vDescriptor.push_back(cv::Mat1f(1, 1, img(x-1, y-1)[0]));
        vDescriptor.push_back(cv::Mat1f(1, 1, img(x-1, y-1)[1]));
    }
    // top middle
    if (y == 0) {
        vDescriptor.push_back(cv::Mat1f(1, 1, 0.f));
        vDescriptor.push_back(cv::Mat1f(1, 1, 0.f));
    } else{
        vDescriptor.push_back(cv::Mat1f(1, 1, img(x, y-1)[0]));
        vDescriptor.push_back(cv::Mat1f(1, 1, img(x, y-1)[1]));
    }
    // top right
    if (y == 0 || x + 1 == img.rows) {
        vDescriptor.push_back(cv::Mat1f(1, 1, 0.f));
        vDescriptor.push_back(cv::Mat1f(1, 1, 0.f));
    } else{
        vDescriptor.push_back(cv::Mat1f(1, 1, img(x+1, y-1)[0]));
        vDescriptor.push_back(cv::Mat1f(1, 1, img(x+1, y-1)[1]));
    }
    // middle middle
    vDescriptor.push_back(cv::Mat1f(1, 1, img(x, y)[0]));
    vDescriptor.push_back(cv::Mat1f(1, 1, img(x, y)[1]));
    return vDescriptor.reshape(1, 1);
}

/// Constructor
SkinModel::SkinModel()
{
}

/// Destructor
SkinModel::~SkinModel()
{
}

/// Start the training.  This resets/initializes the model.
void SkinModel::startTraining()
{
}

/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
///

void SkinModel::train(const cv::Mat3b& img, const cv::Mat1b& mask)
{
    // Histogramm für skin und non skin füllen
    using namespace cv;

    cv::cvtColor(img, img, CV_BGR2HSV);

    // Maske vorher morphologisch bearbeiten
    for (int x = 0; x < img.rows; x+=4) {
        for (int y = 0; y < img.cols; y+=4)
        {
            cv::Mat1f descriptor = computeDescriptor(img, x, y);
            descriptors.push_back(descriptor);
            if (mask(x, y) > 250)
            {
                responses.push_back(cv::Mat1f(1, 1, 1.0f));
            }
            else
            {
                responses.push_back(cv::Mat1f(1, 1, 0.0f));
            }
        }
    }


}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
///
/// Implementation hint:
/// e.g normalize w.r.t. the number of training images etc.
void SkinModel::finishTraining()
{
/*    descriptors = descriptors.reshape(1, responses.rows);*/

    // Check dimensions
    std::cout << "descriptors.dims = " << descriptors.dims << "; descriptors.size = [";
    for(int i = 0; i < descriptors.dims; ++i) {
        if(i) std::cout << " X ";
        std::cout << descriptors.size[i];
    }
    std::cout << "] descriptors.channels = " << descriptors.channels() << std::endl;
    std::cout << "responses  .dims = " << responses.dims << "; responses.size = [";
    for(int i = 0; i < responses.dims; ++i) {
        if(i) std::cout << " X ";
        std::cout << responses.size[i];
    }
    std::cout << "] responses.channels = " << responses.channels() << std::endl;

    cv::SVMParams params;
    params.svm_type = cv::SVM::NU_SVR;
    params.kernel_type = cv::SVM::LINEAR;
    params.nu = 0.5;  // NU_SVC / ONE_CLASS / NU_SVR
    params.p = 0.3;  // EPS_SVR
    params.gamma = 0.30;  // POLY / RBF / SIGMOID
    params.coef0 = 2;  // POLY / SIGMOID
    params.C = 0.5;  // C_SVC / EPS_SVR / NU_SVR
    params.degree = 3;  // POLY
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 2000, 1e-6);
    svm.train( descriptors, responses, cv::Mat(), cv::Mat(), params );
}


/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
///
///
///

cv::Mat1b SkinModel::classify(const cv::Mat3b& img)
{
    cv::Mat1b skin = cv::Mat1b::zeros(img.rows, img.cols);
    cv::cvtColor(img, img, CV_BGR2HSV);

    //cv::Mat3b img2 = preprocess(img);
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++)
        {
            cv::Mat1f descriptor = computeDescriptor(img, x, y);
            skin(x, y) = svm.predict(descriptor, true)*256;
        }
    }
    return skin;
}
