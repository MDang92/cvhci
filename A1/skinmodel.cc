#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include <map>
#include <algorithm>
#include <set>

using namespace std;

int thres;
int prevThres;
double prevDist;
bool colors[25][25];
double rate;

/// Constructor
SkinModel::SkinModel()
{
}

/// Destructor
SkinModel::~SkinModel() 
{
}

/// Start the training.  This resets/initializes the model.
///
/// Implementation hint:
/// Use this function to initialize/clear data structures used for training the skin model.
void SkinModel::startTraining()
{
    //--- IMPLEMENT THIS ---//



}

/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
///

void SkinModel::train(const cv::Mat3b& img, const cv::Mat1b& mask)
{
	//--- IMPLEMENT THIS ---//


    using namespace cv;
    cv::cvtColor(img, img, CV_BGR2HSV);
    normalize(img, img, 0.0, 255.0, NORM_MINMAX);

    int hist[25][25] ={{0}};

    int skinPixels = 0;
    for (int x = 0; x < mask.rows; x++)
        for (int y = 0; y < mask.cols; y++)
        {
            if (mask(x, y) > 127)
            {
                hist[img(x, y)[0]/10][img(x, y)[1]/10] ++;
                skinPixels++;
            }
        }

    double TP = 0;
    double FN = 0;
    double TN = 0;
    double FP = 0;
    int T = 0;
    int N = 0;
    std::map<int, pair<int, int>, std::greater<int> > l;
    for (int x = 0; x < 25; x++)
        for (int y = 0; y < 25; y++)
        {
            l[hist[x][y]] = make_pair(x, y);
        }


    for (auto it=l.begin(); it!=l.end(); ++it)
    {
        if (TP * 4.5 < skinPixels)
        {
            TP += it->first;
            colors[it->second.first][ it->second.second] = true;
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

}


/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
cv::Mat1b SkinModel::classify(const cv::Mat3b& img)
{
    cv::Mat1b skin = cv::Mat1b::zeros(img.rows, img.cols);

	//--- IMPLEMENT THIS ---//
    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {

            if (false)
				skin(row, col) = rand()%256;

			if (false)
				skin(row, col) = img(row,col)[2];

            if (true) {
			
				cv::Vec3b bgr = img(row, col);
				if (bgr[2] > bgr[1] && bgr[1] > bgr[0]) 
					skin(row, col) = 2*(bgr[2] - bgr[0]);

			}
        }
    }

    /*
    cv::Mat1b skinCopy;
    skin.copyTo(skinCopy);

    int k = 3;


    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            skinCopy(row, col) = skin(row, col);
            for (int i = -k/2; i < k/2; i++)
                for (int j = -k/2; j < k/2; j++)
                {
                    if (0 <= row + i && row + i < img.rows && 0 <= col + j && col + j < img.cols)
                    {
                        skinCopy(row, col) = min(skinCopy(row, col), skin(row+i, col+j));
                    }
                }
        }
    }


    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            skin(row, col) = skinCopy(row, col);
            for (int i = -k/2; i < k/2; i++)
                for (int j = -k/2; j < k/2; j++)
                {
                    if (0 <= row + i && row + i < img.rows && 0 <= col + j && col + j < img.cols)
                    {
                        skin(row, col) = max(skin(row, col), skinCopy(row+i, col+j));
                    }
                }
        }
    }


*/
    using  namespace cv;
    cv::cvtColor(img, img, CV_BGR2HSV);
    normalize(img, img, 0.0, 255.0, NORM_MINMAX);
    int hist[25][25] ={{0}};

    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
        {
            if (colors[img(x, y)[0]/10][img(x, y)[1]/10])
            {
                skin(x, y) = 255;
            }
            else
                skin(x, y) = 0;
        }

    cv::Mat1b skinCopy;
    skin.copyTo(skinCopy);

    int k = 3;


    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            skinCopy(row, col) = skin(row, col);
            for (int i = -k/2; i < k/2; i++)
                for (int j = -k/2; j < k/2; j++)
                {
                    if (0 <= row + i && row + i < img.rows && 0 <= col + j && col + j < img.cols)
                    {
                        skinCopy(row, col) = min(skinCopy(row, col), skin(row+i, col+j));
                    }
                }
        }
    }


    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            skin(row, col) = skinCopy(row, col);
            for (int i = -k/2; i < k/2; i++)
                for (int j = -k/2; j < k/2; j++)
                {
                    if (0 <= row + i && row + i < img.rows && 0 <= col + j && col + j < img.cols)
                    {
                        skin(row, col) = max(skin(row, col), skinCopy(row+i, col+j));
                    }
                }
        }
    }


    return skin;
}

