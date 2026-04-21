#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "patchmatch.hpp"
class ImageReshuffling {
private:
	cv::Mat source, target;
	cv::Mat initial_nnf;

	std::vector<cv::Mat> source_pyr, target_pyr, nnf_pyr;

	int patch_size = 7;
	int patchmatch_iter = 3;
	int pyramid_depth = 4;
	int correction_iter = 4;
public:

	void init(cv::Mat source, cv::Mat target);
	void create_pyramid_nnfs();
	cv::Mat upsample_nnf(cv::Mat nnf, cv::Mat next);
};