#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "patchmatch.hpp"

void main() {
	cv::Mat source = cv::imread("Source.jpg");
	cv::Mat target = cv::imread("Target.jpg");
	PatchMatch* p = new PatchMatch();
	p->init(source, target);
	p->random_search();
	p->output_info(target);
}