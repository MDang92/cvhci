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
    pimpl->hog.blockStride = Size(16, 28); // mod 48=2^4*3, 112=2^6*7

}

//HOG::HOG(int a, int b)
//{
//    using namespace cv;
//   pimpl = std::shared_ptr<HOGPimpl>(new HOGPimpl());
//    pimpl->hog.blockStride = Size(a, b); // mod 48=2^4*3, 112=2^6*7

//8


/// Destructor
HOG::~HOG() 
{
}

/// Start the training.  This resets/initializes the model.
void HOG::startTraining()
{
}
    //int writeout = 1;
/// Add a new training image.
///
/// @param img:  input image
/// @param bool: value which specifies if img represents a person
void HOG::train(const cv::Mat3b& img, bool isPerson)
{
	cv::Mat3b img2 = img(cv::Rect((img.cols-64)/2,(img.rows-128)/2,64,128));

    /*
    if (writeout)
    {
        cv::imwrite("data/train/tmp/1.jpg", img2);
    }
    */
	vector<float> vDescriptor;
	pimpl->hog.compute(img2, vDescriptor);	
    /*if (writeout)
    {
        cout << vDescriptor.size() << endl;
        for (auto v : vDescriptor)
            cout << v << " ";
        cout << endl;
    }*/
    cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);
    
	pimpl->descriptors.push_back(descriptor);
	pimpl->responses.push_back(cv::Mat1f(1,1,float(isPerson)));


    //writeout--;
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
void HOG::finishTraining()
{
	cv::SVMParams params;
	pimpl->svm.train( pimpl->descriptors, pimpl->responses, cv::Mat(), cv::Mat(), params );
}

/// Classify an unknown test image.  The result is a floating point
/// value directly proportional to the probability of being a person.
///
/// @param img: unknown test image
/// @return:    probability of human likelihood
double HOG::classify(const cv::Mat3b& img)
{
	

	cv::Mat3b img2 = img(cv::Rect((img.cols-64)/2,(img.rows-128)/2,64,128));

	vector<float> vDescriptor;
	pimpl->hog.compute(img2, vDescriptor);	
	cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);

	return -pimpl->svm.predict(descriptor, true);
}

