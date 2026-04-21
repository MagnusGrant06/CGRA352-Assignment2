#include "imagereshuffling.hpp"

void ImageReshuffling::init(cv::Mat source_img, cv::Mat mask) {
	target = source_img.clone();
	source = source_img.clone();
	initial_nnf = cv::Mat::zeros(source.size(), CV_32SC2);

	for (int r = 0; r < source.rows; r++) {
		for (int c = 0; c < source.cols; c++) {
			if (mask.at<uchar>(r, c) == 0) initial_nnf.at<cv::Vec2i>(r, c) = cv::Vec2i(0, 270);
			else {
				cv::Vec3b temp_pix = target.at<cv::Vec3b>(r, c - 270);
				target.at<cv::Vec3b>(r, c - 270) = source.at<cv::Vec3b>(r, c);
				target.at<cv::Vec3b>(r, c) = temp_pix;

				initial_nnf.at<cv::Vec2i>(r, c) = cv::Vec2i(0, -270);
			}
		}
	}

	cv::buildPyramid(source, source_pyr, pyramid_depth);
	cv::buildPyramid(target, target_pyr, pyramid_depth);
	nnf_pyr.resize(5);
	//cv::imshow("source",source);
	//cv::imshow("target", target);

	//cv::waitKey(0);
}

void ImageReshuffling::create_pyramid_nnfs() {
	PatchMatch p;
	p.init_with_nnf(source_pyr[pyramid_depth], target_pyr[pyramid_depth],initial_nnf);
	p.iterate(patchmatch_iter);
	nnf_pyr[pyramid_depth] = p.get_nnf();
	target_pyr[pyramid_depth] = p.reconstruct_image();

	if (pyramid_depth > 0) {
		nnf_pyr[pyramid_depth - 1] = upsample_nnf(nnf_pyr[pyramid_depth], target_pyr[pyramid_depth - 1]);
	}

	for (int k = pyramid_depth - 1; k >= 0; k--) {
		p.init_with_nnf(source_pyr[k], target_pyr[k], nnf_pyr[k]);
		for (int i = 0; i < patchmatch_iter; i++) {
			p.iterate(patchmatch_iter);
			nnf_pyr[k] = p.get_nnf();
			target_pyr[k] = p.reconstruct_image();
		}
		cv::imshow("djasodk", PatchMatch::nnf2img(nnf_pyr[k], target_pyr[k]));
		//cv::imshow("djasodk", target_pyr[k]);
		cv::waitKey(0);
		if (k > 0) {
			nnf_pyr[k - 1] = upsample_nnf(nnf_pyr[k], target_pyr[k - 1]);
		}
	}

}

cv::Mat ImageReshuffling::upsample_nnf(cv::Mat nnf, cv::Mat next) {
	cv::Mat output_nnf;
	cv::resize(nnf, output_nnf, next.size(), 0, 0, cv::INTER_NEAREST);
	output_nnf *= 2;
	return output_nnf;
}