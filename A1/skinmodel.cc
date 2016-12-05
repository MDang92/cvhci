#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include <map>
#include <algorithm>
#include <set>
#include <opencv2/opencv.hpp>

using namespace std;

int skinPixels;
int nonskinPixels ;\
const int binsize=1;
int skinhist[256/binsize][256/binsize];
int nonskinhist[256/binsize][256/binsize];
int ind ;
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
    skinPixels = 0;
    nonskinPixels = 0;
    ind = 0;
    for (int i = 0; i < 255; i++)
        for (int j = 0; j < 255; j++)
        {
            skinhist[i/binsize][j/binsize] =0;
            nonskinhist[i/binsize][j/binsize] =0;
        }


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


    for (int x = 0; x < mask.rows; x++)
        for (int y = 0; y < mask.cols; y++)
        {
            if (mask(x, y) > 250)
            {
                skinhist[img(x, y)[0]/binsize][img(x, y)[1]/binsize] ++;
                skinPixels++;
            }
            else
            {
                nonskinhist[img(x, y)[0]/binsize][img(x, y)[1]/binsize] ++;
                nonskinPixels++;
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
    printf("%d %d\n", skinPixels, nonskinPixels);

    /*
    for (int i = 0; i < 255; i+=binsize)
    {
        for (int j = 0; j < 255; j+=binsize)
        {
            printf("%d ", skinhist[i/binsize][j/binsize]);
        }
        printf("\n");
    }
    */

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


    using  namespace cv;
    cv::Mat3b imgCopy;
    img.copyTo(imgCopy);
    cv::cvtColor(img, img, CV_BGR2HSV);

    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
        {
            double p_x_skin = (skinPixels * 1.0/(skinPixels + nonskinPixels)) * (skinhist[img(x, y)[0]/binsize][img(x, y)[1]/binsize] * 1.0/skinPixels);
            double p_x_nonskin = (nonskinPixels * 1.0/(skinPixels + nonskinPixels)) * (nonskinhist[img(x, y)[0]/binsize][img(x, y)[1]/binsize] * 1.0/nonskinPixels);
            skin(x, y) = p_x_skin > p_x_nonskin? 255 : 0;
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





    std::string s = "testpics/pic" + std::to_string(ind) + ".jpg";

    cv::imwrite(s, skin );
    ind++;
    return skin;
}

