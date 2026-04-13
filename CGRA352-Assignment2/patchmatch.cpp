#include "patchmatch.hpp"

void PatchMatch::output_image() {
	cv::Mat img = cv::imread("Target.jpg");
	cv::imshow("kjdoaks", img);
	cv::waitKey(0);
}

void PatchMatch::init() {

}