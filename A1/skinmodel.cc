#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include <map>
#include <algorithm>
#include <set>
#include <opencv2/opencv.hpp>

using namespace std;

int skinPixels;
int nonskinPixels;
const int binsize = 1;
const int binsizeSkin = 2;
const int binsizeNonskin = 1;
int skinhist[256 / binsizeSkin][256 / binsizeSkin];
int nonskinhist[256 / binsizeNonskin][256 / binsizeNonskin];

double probSkin[256 / binsizeSkin][256 / binsizeSkin];
double probNonskin[256 / binsizeNonskin][256 / binsizeNonskin];
int ind;

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
    skinPixels = 0;
    nonskinPixels = 0;
    ind = 0;
    for (int i = 0; i < 255; i++) //mögliche Farbwerte
        for (int j = 0; j < 255; j++)
        {
            // Histogramm mit 0 initialisieren
            skinhist[i / binsizeSkin][j / binsizeSkin] = 0;
            nonskinhist[i / binsizeNonskin][j / binsizeNonskin] = 0;
            probSkin[i / binsizeSkin] [j / binsizeSkin] = 0.0;
            probNonskin[i / binsizeNonskin] [j / binsizeNonskin] = 0.0;
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
    // Histogramm für skin und non skin füllen
    using namespace cv;

    cv::cvtColor(img, img, CV_BGR2HSV);

    // Maske vorher morphologisch bearbeiten
    for (int x = 0; x < mask.rows; x++)
        for (int y = 0; y < mask.cols; y++)
        {
            if (mask(x, y) > 250)
            {
                skinhist[img(x, y)[0] / binsizeSkin][img(x, y)[1] / binsizeSkin] ++; //nur H und S werte
                skinPixels++; //counter
            }
            else
            {
                nonskinhist[img(x, y)[0] / binsizeNonskin][img(x, y)[1] / binsizeNonskin] ++;
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
    printf("skinPixels: %d nonskinPixels%d\n", skinPixels, nonskinPixels);
    double p_Skin = 1.0 * skinPixels / (skinPixels + nonskinPixels);
    double p_NonSkin = 1.0 - p_Skin ;
    printf("p_Skin: %f p_NonSkin %f \n", p_Skin, p_NonSkin);

    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            probSkin[i / binsizeSkin] [j / binsizeSkin] += 1.0 * skinhist[i / binsizeSkin] [j / binsizeSkin] / skinPixels;
            //skinhist[i / binsizeSkin] [j / binsizeSkin] = 0;
            //skinPixels = 0;
            probNonskin[i / binsizeNonskin] [j / binsizeNonskin] += 1.0 * nonskinhist[i / binsizeNonskin] [j / binsizeNonskin] / nonskinPixels;
            //nonskinhist[i / binsizeNonskin] [j / binsizeNonskin] = 0;
            //nonskinPixels = 0;
        }
    }
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

    using  namespace cv;
    cv::Mat1b skin = cv::Mat1b::zeros(img.rows, img.cols);

    cv::Mat3b imgCopy;
    img.copyTo(imgCopy);
    cv::cvtColor(img, img, CV_BGR2HSV);

    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
        {
            double p_xSkin = probSkin[img(x, y)[0] / binsizeSkin][img(x, y)[1] / binsizeSkin];
            double p_xNonSkin = probNonskin[img(x, y)[0] / binsizeNonskin][img(x, y)[1] / binsizeNonskin];
            double p_Skin = 0.3; //1.0 * skinPixels / (skinPixels + nonskinPixels);
            double p_NonSkin = 1.0 - p_Skin;
            //skin(x, y) = p_xSkin / (p_xSkin + p_xNonSkin) * 256;
            skin(x, y) = p_xSkin * p_Skin / (p_xSkin * p_Skin + p_xNonSkin * p_NonSkin) * 256;
        }


    cv::Mat1b skinCopy;
    skin.copyTo(skinCopy);

    int k = 10;


    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            skinCopy(row, col) = skin(row, col);
            for (int i = -k / 2; i < k / 2; i++)
                for (int j = -k / 2; j < k / 2; j++)
                {
                    if (0 <= row + i && row + i < img.rows && 0 <= col + j && col + j < img.cols)
                    {
                        skinCopy(row, col) = min(skinCopy(row, col), skin(row + i, col + j));
                    }
                }
        }
    }


    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            skin(row, col) = skinCopy(row, col);
            for (int i = -k / 2; i < k / 2; i++)
                for (int j = -k / 2; j < k / 2; j++)
                {
                    if (0 <= row + i && row + i < img.rows && 0 <= col + j && col + j < img.cols)
                    {
                        skin(row, col) = max(skin(row, col), skinCopy(row + i, col + j));
                    }
                }
        }
    }
    blur(skin, skin, Size(5, 5));

    std::string s = "testpics/pic" + std::to_string(ind) + ".jpg";

    cv::imwrite(s, skin );
    ind++;
    return skin;
}
