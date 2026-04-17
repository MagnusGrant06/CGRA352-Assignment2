#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class ImageQuilting {
private:
	cv::Mat source_img = cv::imread("TextureSample.jpg");
	int patch_size = 100;
	int overlap = 20;

	int start_row = 30;
	int start_col = 30;

	int iterations = 5;

	int temp = 0;
	
	cv::Mat synthesis = source_img(
		cv::Range(start_row, start_row + patch_size),
		cv::Range(start_col, start_col + patch_size)
	);
	
	cv::Mat canvas;
public:
	void run_quilting();

	float calc_overlap_cost(cv::Mat left, cv::Mat right, int overlap_size);

	void h_quilt(cv::Mat left_image, cv::Mat quilt, int overlap_size);
};