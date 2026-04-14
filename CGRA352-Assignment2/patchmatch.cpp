#include <random>
#include "patchmatch.hpp"

void PatchMatch::output_info(cv::Mat source) {
	cv::imshow("yeah",nnf2img(nnf, source));
	cv::waitKey(0);

}

void PatchMatch::init(cv::Mat source, cv::Mat target) {

	cv::Mat temp_nnf(target.rows, target.cols, CV_32SC2);
	cv::Mat temp_pixel_cost(target.rows, target.cols, CV_32FC1);

	nnf = temp_nnf;
	pixel_cost = temp_pixel_cost;

	//create padded images to avoid going out of bounds when calculating patches
	cv::Mat padded_source; 
	cv::copyMakeBorder(source, padded_source, 0, patch_size, 0, patch_size, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	cv::Mat padded_target;
	cv::copyMakeBorder(target, padded_target, 0, patch_size, 0, patch_size, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	std::mt19937 random(std::random_device{}());
	std::uniform_int_distribution<int> random_col(0, source.cols);
	std::uniform_int_distribution<int> random_row(0, source.rows);

	//assumes source and target are the same size
	for (int i = 0; i < source.rows; i++) {
		for (int j = 0; j < source.cols; j++) {
			cv::Vec2i rand_source(random_col(random), random_row(random));

			cv::Rect target_patch(j, i, patch_size, patch_size);
			cv::Rect source_patch(rand_source[0], rand_source[1], patch_size, patch_size);
			
			//calculate offset
			cv::Vec2i& nnf_offset = nnf.at<cv::Vec2i>(i, j);
			nnf_offset = cv::Vec2i(rand_source[0] - i, rand_source[1]- j);

			//calculate cost
			int current_cost = cv::norm(padded_target(target_patch), padded_source(source_patch));
			pixel_cost.at<float>(i, j) = current_cost;
		}
	}
}

void PatchMatch::random_search(cv::Mat source, cv::Mat target) {

	std::mt19937 random(std::random_device{}());

	for (int i = 0; i < target.rows; i++) {
		for (int j = 0; j < target.cols; j++) {
			int row_radius = target.rows;
			int col_radius = target.cols;

			while (row_radius <= 1 || col_radius <= 1) {
				std::uniform_int_distribution<int> random_col(0, col_radius);
				std::uniform_int_distribution<int> random_row(0, row_radius);

				cv::Point rand_point(std::max(std::min(random_row(random),target.rows),0),
					std::max(std::min(random_col(random),target.cols),0));

				improveNNF(source, target, cv::Point(i, j), rand_point, nnf, pixel_cost);

				row_radius /= 2;
				col_radius /= 2;

			}
			
		}
	}
}

//simple method to compare patches, assumes they are padded
void PatchMatch::improveNNF(cv::Mat source, cv::Mat target, cv::Point patch_coord, cv::Point source_patch_coord, cv::Mat& nnf, cv::Mat& cost_mat) {

	cv::Rect source_patch(source_patch_coord, cv::Point2i(patch_size, patch_size));
	cv::Rect target_patch(patch_coord, cv::Point2i(patch_size, patch_size));

	float proposed_cost = cv::norm(source(source_patch), target(target_patch), cv::NORM_L2SQR);

	if (proposed_cost < cost_mat.at<float>(patch_coord.x, patch_coord.y)) {
		cost_mat.at<float>(patch_coord.x, patch_coord.y) = proposed_cost;
		nnf.at<cv::Vec2i>(patch_coord.x, patch_coord.y) = cv::Vec2i(source_patch_coord.x - patch_coord.x, source_patch_coord.y - source_patch.y);
	}
}



//code from assignment brief for debugging
cv::Mat PatchMatch::nnf2img(cv::Mat nnf, cv::Mat s) {
	cv::Mat nnf_img(nnf.rows, nnf.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < nnf.rows; ++i) {
		for (int j = 0; j < nnf.cols; ++j) {
			cv::Vec2i p = nnf.at<cv::Vec2i>(i, j);
			if (p[0] < 0 || p[1] < 0 || p[0] >= s.rows || p[1] >= s.cols) {
				/* coordinate is outside, insert error of choice */
				//std::cout << "uhh error"  << p[0] << p[1] << std::endl;
			}
			int r = int(p[1] * 255.0 / s.cols); // cols -> red
			int g = int(p[0] * 255.0 / s.rows); // rows -> green
			int b = 255 - cv::max(r, g); // blue
			nnf_img.at<cv::Vec3b>(i, j) = cv::Vec3b(b, g, r);
		}
	}
	return nnf_img;
}