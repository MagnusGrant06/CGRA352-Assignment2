#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class PatchMatch {
private:
	cv::Mat nnf;
	cv::Mat pixel_cost;

	int patch_size = 7;

public:

	void init(cv::Mat source, cv::Mat target);

	void improveNNF(cv::Mat source, cv::Mat target, cv::Point patch_coord, cv::Point source_patch_coord, cv::Mat& nnf, cv::Mat& cost_mat);

	cv::Mat nnf2img(cv::Mat nnf, cv::Mat s);

	void output_info(cv::Mat source);
};