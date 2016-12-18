///
///  Assignment 2
///  Pedestrian Detection
///
///  Group Number:
///  Authors:
///
#define _OPENCV_FLANN_HPP_
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <fstream>
#include <iostream>

#include "hog.h"
#include "ROC.h"
using namespace std;



int main(int argc, char* argv[]) {

	cv::setNumThreads(0);
	
	/// parse command line options
	boost::program_options::variables_map pom;
	{
		namespace po = boost::program_options;
		po::options_description pod(string("Allowed options for ")+argv[0]);
		pod.add_options() 
			("help,h", "produce this help message")
			("gui,g", "Enable the GUI")
			("out,o", po::value<string>(), "Path where to write the results")
			("path", po::value<string>()->default_value("data"), "Path where to read input data");

		po::positional_options_description pop;
		pop.add("path", -1);

		po::store(po::command_line_parser( argc, argv ).options(pod).positional(pop).run(), pom);
		po::notify(pom);

		if (pom.count("help")) {
			cout << "Usage:" << endl <<  pod << "\n";
			return 0;
		}
	}
	
	/// get image filenames
	string path = pom["path"].as<string>();
	vector<pair<string,bool>> trainImgs, testImgs;
	{
		namespace fs = boost::filesystem; 
		for (fs::directory_iterator it(fs::path(path+"/train/p")); it!=fs::directory_iterator(); it++)
			if (is_regular_file(*it))
				trainImgs.push_back({it->path().filename().string(),1});
		for (fs::directory_iterator it(fs::path(path+"/train/n")); it!=fs::directory_iterator(); it++)
			if (is_regular_file(*it))
				trainImgs.push_back({it->path().filename().string(),0});
		for (fs::directory_iterator it(fs::path(path+"/test/p")); it!=fs::directory_iterator(); it++)
			if (is_regular_file(*it))
				testImgs.push_back({it->path().filename().string(),1});
		for (fs::directory_iterator it(fs::path(path+"/test/n")); it!=fs::directory_iterator(); it++)
			if (is_regular_file(*it))
				testImgs.push_back({it->path().filename().string(),0});

    } 
    
    int a, b;
    int bestA, bestB;
    double bestScore = 0;
    a = b = bestA = bestB =1;// a mod 48, b mod 112;
    ROC<double> roc; //musste vorgezogen werden

    ofstream myfile;
    myfile.open ("example2.txt");

    for (int iteration1 = 0; b < 113; b++) {
        a = 1;
    for (int iteration = 0; a < 49 ; a++) {
        cout << "iteration " << iteration << endl;

            random_shuffle(trainImgs.begin(), trainImgs.end());
            random_shuffle(testImgs.begin(), testImgs.end());

        /// create person classification model instance
        HOG model(a, b);
        cout << "a: " << a << ", b: " << b << "," << endl;
        /// train model with all images in the train folder
        //cout << "Start Training" << endl;
            model.startTraining();

            for (auto &f:trainImgs) {
            //cout << "Training on Image " << path+"/train/"+"np"[f.second]+"/"+f.first << endl;
                    cv::Mat3b img = cv::imread(path+"/train/"+"np"[f.second]+"/"+f.first,-1);
                    model.train( img, f.second );
            }

        //cout << "Finish Training" << endl;
            model.finishTraining();

        /// test model with all images in the test folder,
           // ROC<double> roc; //wurde vorgezogen
            for (auto &f:testImgs) {
                    cv::Mat3b img = cv::imread(path+"/test/"+"np"[f.second]+"/"+f.first,-1);
                    double hyp = model.classify(img);
                    roc.add(f.second, hyp);
            //cout << "Testing Image " << f.second << " " << hyp << " " << path+"/test/"+"np"[f.second]+"/"+f.first << endl;
            }

            /// After testing, update statistics and show results
            roc.update();
        if (bestScore < roc.F1) {
            bestScore = roc.F1;
            bestA = a;
            bestB = b;
        }
        cout <<" F1 score: " << roc.F1 << endl;
        myfile << "a: " << a << ", b: " << b << "," << "F1 score: " << roc.F1 << "\n";
    }
    }
    cout <<" best F1 score: " << bestScore << endl;
    myfile << "best a: " << bestA << ", best b: " << bestB << "," << "best F1 score: " << bestScore << "\n";
        myfile.close();

	/// Display final result if desired
	if (pom.count("gui")) {
		cv::imshow("ROC", roc.draw());
		cv::waitKey(0);
	}

	/// Ouput a summary of the data if required
	if (pom.count("out")) {
		
		string p = pom["out"].as<string>();
		
		/// GRAPH format with one FPR and TPR coordinates per line
		ofstream graph(p+"/graph.txt");
		for (auto &dot : roc.graph)
			graph << dot.first << " " << dot.second << endl;
		
		/// Single output of the F1 score
		ofstream score(p+"/score.txt");
		score << roc.F1 << endl;
		/// Ouput of the obtained ROC figure
		cv::imwrite(p+"/ROC.png", roc.draw());
	}
}

