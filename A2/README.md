## General info

See [nu vs epsilon](http://stats.stackexchange.com/a/167545/25741)


## SVM tests

Parameters: [docs](http://docs.opencv.org/2.4/modules/ml/doc/support_vector_machines.html#svm-params-params) (v3 is NOT on the server)

* `cv::SVM::NU_SVR`, `nu=0.10`: 0.337434 online
* `cv::SVM::NU_SVR`, `nu=0.25`, `C = 10`: 0.341099 online
* `cv::SVM::NU_SVR`, `nu=0.50`, `C = 10`: 0.664657 online
* `cv::SVM::NU_SVR`, `nu=0.50`, `C =  5`: 0.337434 online
* `cv::SVM::NU_SVR`, `nu=0.75`, `C = 10`: 0.337434 online
* `cv::SVM::NU_SVR`, `nu=0.50`: 0.337434 online
* `cv::SVM::NU_SVR`, `nu=0.50`, `kernel_type = CvSVM::RBF`: 0.337434 online
* `cv::SVM::EPS_SVR`, `p=0.0`, `kernel_type = CvSVM::RBF`: Command terminated by signal (6: SIGABRT)
* `cv::SVM::EPS_SVR`, `p=0.2`, `kernel_type = CvSVM::RBF`: 0.337434 online
* `cv::SVM::EPS_SVR`, `p=0.3`, `kernel_type = CvSVM::RBF`: 0.337434 online
* `cv::SVM::EPS_SVR`, `p=0.5`, `kernel_type = CvSVM::RBF`: Command terminated by signal (6: SIGABRT)
* `cv::SVM::EPS_SVR`, `p=1.0`, `kernel_type = CvSVM::RBF`: Command terminated by signal (6: SIGABRT)
* `cv::SVM::EPS_SVR`, `p=1000.0`, `kernel_type = CvSVM::RBF`: Command terminated by signal (6: SIGABRT)
* `cv::SVM::C_SVC`, `gamma=0.10`, `C=1`, `kernel_type = CvSVM::RBF`: 0.877569
* `cv::SVM::C_SVC`, `gamma=0.15`, `C=1`, `kernel_type = CvSVM::RBF`: 0.891566
* `cv::SVM::C_SVC`, `gamma=0.19`, `C=1`, `kernel_type = CvSVM::RBF`: 0.870796
* `cv::SVM::C_SVC`, `gamma=0.20`, `C=1`, `kernel_type = CvSVM::RBF`: 0.899548
* `cv::SVM::C_SVC`, `gamma=0.20`, `C=1`, `kernel_type = CvSVM::RBF`, `CV_TERMCRIT_ITER=500`: 0.95713
* `cv::SVM::C_SVC`, `gamma=0.20`, `C=10`, `kernel_type = CvSVM::RBF`: 0.90383
* `cv::SVM::C_SVC`, `gamma=0.20`, `C=100`, `kernel_type = CvSVM::RBF`: 0.90383
* `cv::SVM::C_SVC`, `gamma=0.20`, `C=100`, `kernel_type = CvSVM::SIGMOID`, `CV_TERMCRIT_ITER=2000`: 0.342176
* `cv::SVM::C_SVC`, `gamma=0.10`, `degree = 3`, `kernel_type = CvSVM::POLY`, `CV_TERMCRIT_ITER=2000`: 0.967798
* `cv::SVM::C_SVC`, `gamma=0.18`, `degree = 3`, `kernel_type = CvSVM::POLY`, `CV_TERMCRIT_ITER=2000`: 0.967798
* `cv::SVM::C_SVC`, `gamma=0.20`, `degree = 3`, `kernel_type = CvSVM::POLY`, `CV_TERMCRIT_ITER=2000`: 0.967798
* `cv::SVM::C_SVC`, `gamma=0.30`, `degree = 3`, `kernel_type = CvSVM::POLY`, `CV_TERMCRIT_ITER=2000`: 0.967798
* `cv::SVM::C_SVC`, `gamma=0.30`, `degree = 3`, `coef0 = 1`, `kernel_type = CvSVM::POLY`, `CV_TERMCRIT_ITER=2000`: 0.97302
* `cv::SVM::C_SVC`, `gamma=0.30`, `degree = 3`, `coef0 = 2`, `kernel_type = CvSVM::POLY`, `CV_TERMCRIT_ITER=2000`: 0.974805
* `cv::SVM::C_SVC`, `gamma=0.30`, `degree = 3`, `coef0 = 2.5`, `kernel_type = CvSVM::POLY`, `CV_TERMCRIT_ITER=2000`: 0.973958
* `cv::SVM::C_SVC`, `gamma=0.30`, `degree = 3`, `coef0 = 3`, `kernel_type = CvSVM::POLY`, `CV_TERMCRIT_ITER=2000`: 0.973958
* `cv::SVM::C_SVC`, `gamma=0.30`, `degree = 3`, `coef0 = 4`, `kernel_type = CvSVM::POLY`, `CV_TERMCRIT_ITER=2000`: 0.972222
* `cv::SVM::C_SVC`, `gamma=0.20`, `degree = 4`, `kernel_type = CvSVM::POLY`, `CV_TERMCRIT_ITER=2000`: 0.965819
* `cv::SVM::C_SVC`, `gamma=0.30`, `C=1`, `kernel_type = CvSVM::RBF`: 0.89578
* `cv::SVM::C_SVC`, `gamma=0.30`, `C=100`, `kernel_type = CvSVM::RBF`: 0.90383
* `cv::SVM::C_SVC`, `gamma=0.50`, `C=1`, `kernel_type = CvSVM::RBF`: 0.893274
* `cv::SVM::C_SVC`, `gamma=0.80`, `C=1`, `kernel_type = CvSVM::RBF`: 0.865209
* `cv::SVM::C_SVC`, `gamma=0.90`, `C=1`, `kernel_type = CvSVM::RBF`: 0.837291
* `cv::SVM::C_SVC`, `gamma=1.10`, `C=1`, `kernel_type = CvSVM::RBF`: 0.827832
* `cv::SVM::C_SVC`, `kernel_type = CvSVM::LINEAR`: 0.856889


## Speed

* `cv::SVM::C_SVC`, `kernel_type = CvSVM::LINEAR`, `bins=10`: 0.857143 (10.374s)
* `cv::SVM::C_SVC`, `kernel_type = CvSVM::LINEAR`, `bins=9`: 0.857143 (9.52s)
