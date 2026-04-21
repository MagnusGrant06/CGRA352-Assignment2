#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include "imagequilting.hpp"
#include "imagereshuffling.hpp"

void main() {
	cv::Mat source = cv::imread("Source.jpg");
	cv::Mat target = cv::imread("Target.jpg");
	
	cv::Mat shuffling_source = cv::imread("ReshuffleSource.jpg");
	cv::Mat shuffling_mask = cv::imread("ReshuffleMask.jpg", cv::IMREAD_GRAYSCALE);
	ImageReshuffling ir;
	ir.init(shuffling_source, shuffling_mask);
	ir.create_pyramid_nnfs();
	//ImageQuilting iq;
	//iq.run_quilting();
//	PatchMatch p;
//	p.init(source, target);
//	p.iterate(4);
//	p.output_info(source);
//	cv::imshow("dhasijda,",p.reconstruct_image());
//	cv::waitKey(0);
}