#include <iostream>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class PatchMatch {
private:
	cv::Mat nnf;
	cv::Mat pixel_cost;

	cv::Mat padded_source;
	cv::Mat padded_target;

	int patch_size = 7;

	std::mt19937 random;

public:

	PatchMatch();

	void init(cv::Mat source, cv::Mat target);

	void init_with_nnf(cv::Mat source, cv::Mat target, cv::Mat nnf);

	void improveNNF(cv::Mat source, cv::Mat target, cv::Point patch_coord, cv::Point source_patch_coord, cv::Mat& nnf, cv::Mat& cost_mat);

	void random_search(int i, int j);

	void propogate(int i, int j, int negative);

	void iterate(int iteration_num);

	cv::Mat reconstruct_image();

	static cv::Mat nnf2img(cv::Mat nnf, cv::Mat s);

	void output_info(cv::Mat source);

	cv::Mat get_nnf();
};