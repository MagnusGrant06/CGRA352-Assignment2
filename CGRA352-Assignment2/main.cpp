#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "patchmatch.hpp"
#include "imagequilting.hpp"

void main() {
	cv::Mat source = cv::imread("Source.jpg");
	cv::Mat target = cv::imread("Target.jpg");
	ImageQuilting* iq = new ImageQuilting();
	iq->run_quilting();
	//PatchMatch* p = new PatchMatch();
	//p->init(source, target);
	//p->iterate(4);
	//p->output_info(source);
	//p->reconstruct_image();
}