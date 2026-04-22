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

	cv::Mat inpainting_source = cv::imread("ToInpaint.jpg");
	cv::Mat inpainting_mask = cv::imread("ToInpaint_Mask.jpg", cv::IMREAD_GRAYSCALE);

	PatchMatch p(7);
	p.init(source, target);
	p.iterate(4);
	p.output_info(source);

	ImageQuilting iq;
	iq.run_quilting();

	ImageReshuffling ir;
	ir.init(shuffling_source, shuffling_mask);
	ir.create_pyramid_nnfs();

	ImageReshuffling inpainting;
	inpainting.init_for_inpainting(inpainting_source, inpainting_mask);
	inpainting.create_pyramid_nnfs();
}