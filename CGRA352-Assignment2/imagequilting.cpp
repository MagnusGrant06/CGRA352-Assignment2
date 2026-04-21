#include "imagequilting.hpp"

void ImageQuilting::run_quilting() {
	//create canvas correct size of 5x patch then place first, fixed image
	canvas = cv::Mat(patch_size, patch_size*iterations, CV_8UC3, cv::Scalar(0,0,0));
	synthesis.copyTo(canvas(cv::Rect(0, 0, synthesis.rows, synthesis.cols)));

	for (int i = 0; i < iterations; i++) {

		cv::Mat best_patch;
		float best_cost = FLT_MAX;

		for (int r = 0; r < source_img.rows - patch_size; r++) {
			for (int c = 0; c < source_img.cols - patch_size; c++) {
				cv::Mat current_patch = source_img(cv::Range(r, r + patch_size), cv::Range(c, c + patch_size));
				float current_cost = calc_overlap_cost(synthesis, current_patch, overlap);
				if (current_cost < best_cost) {
					best_cost = current_cost;
					best_patch = current_patch;
				}
			}
		}
		h_quilt(synthesis, best_patch, overlap);

	}
	cv::imshow("doajdojsa", canvas);
	cv::waitKey(0);

}

void ImageQuilting::h_quilt(cv::Mat left, cv::Mat right, int overlap_size) {
	assert(left.rows == right.rows);

	cv::Mat left_overlap = left(cv::Range(0, left.rows), cv::Range(left.cols - overlap_size,left.cols));
	cv::Mat right_overlap = right(cv::Range(0, right.rows), cv::Range(0, overlap_size));

	cv::Mat mask = create_mask(left_overlap, right_overlap, overlap_size);

	synthesis = right;
	right.copyTo(canvas(cv::Rect((patch_size-overlap)*patch_num,0,patch_size,patch_size)), mask);
	patch_num++;
}

cv::Mat ImageQuilting::create_mask(cv::Mat left_overlap, cv::Mat right_overlap, int overlap_size) {
	cv::Mat error(patch_size, overlap_size, CV_32FC1);
	//compute pixel wise difference in overlapping area

	cv::Mat acc_error(patch_size, overlap_size, CV_32FC1);
	std::vector<int> trace(patch_size, -1);

	//create error matrix
	for (int r = 0; r < patch_size; r++) {
		for (int c = 0; c < overlap_size; c++) {
			float diff = cv::norm(left_overlap.at<cv::Vec3b>(r, c), right_overlap.at<cv::Vec3b>(r, c), cv::NORM_L2SQR);
			error.at<float>(r, c) = diff;
		}
	}

	acc_error = error.clone();

	//calculate cumalative error for neighboring pixels
	for (int r = 1; r < error.rows; r++) {
		for (int c = 1; c < error.cols - 1; c++) {
			float left_up = error.at<float>(r - 1, c - 1);
			float up = error.at<float>(r - 1, c);
			float right_up = error.at<float>(r - 1, c + 1);

			float min = std::min(left_up, std::min(up, right_up));

			acc_error.at<float>(r, c) += min;
		}
	}

	//get minimum col in starting row
	cv::Point temp_point;
	cv::minMaxLoc(acc_error.row(acc_error.rows - 1), nullptr, nullptr, &temp_point, nullptr);
	int starting_col = temp_point.x;

	//go through rows and find minimum neighbor to current row
	trace[acc_error.rows - 1] = starting_col;
	for (int r = acc_error.rows - 2; r > -1; r--) {
		int col = trace[r + 1];

		float left_up = FLT_MAX;
		float right_up = FLT_MAX;

		if (col != 0) left_up = acc_error.at<float>(r, col - 1);   //bounds checking
		if (col != acc_error.cols - 1) right_up = acc_error.at<float>(r, col + 1);   //bounds checking
		float up = acc_error.at<float>(r, col);

		float min = std::min(left_up, std::min(up, right_up));
		if (min == left_up) {
			trace[r] = col - 1;
		}
		else if (min == up) {
			trace[r] = col;
		}
		else if (min == right_up) {
			trace[r] = col + 1;
		}

	}

	cv::Mat mask(patch_size, patch_size, CV_8U, cv::Scalar(0));

	//create mask, 0 if left image and 255 if right image, for use in copyTo
	for (int r = 0; r < mask.rows; r++) {
		for (int c = trace[r]; c < mask.cols; c++) {
			if (c > trace[r]) {
				mask.at<uchar>(r, c) = 255;
			}
		}
	}

	return mask;
}

//helper method to calculate the error of two overlapping sections
float ImageQuilting::calc_overlap_cost(cv::Mat left, cv::Mat right, int overlap_size) {
	cv::Mat left_overlap = left(cv::Range(0,left.rows), cv::Range(left.cols - overlap_size, left.cols));
	cv::Mat right_overlap = right(cv::Range(0, right.rows), cv::Range(0, overlap_size));

	return cv::norm(left_overlap, right_overlap, cv::NORM_L2SQR);

}