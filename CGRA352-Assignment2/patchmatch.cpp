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

	cv::copyMakeBorder(source, padded_source, 0, patch_size, 0, patch_size, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	cv::copyMakeBorder(target, padded_target, 0, patch_size, 0, patch_size, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	std::mt19937 random(std::random_device{}());
	std::uniform_int_distribution<int> random_col(0, source.cols-patch_size);
	std::uniform_int_distribution<int> random_row(0, source.rows-patch_size);

	//assumes source and target are the same size
	for (int i = 0; i < padded_source.rows - patch_size; i++) {
		for (int j = 0; j < padded_source.cols - patch_size; j++) {
			cv::Vec2i rand_source(random_col(random), random_row(random));

			cv::Rect target_patch(j, i, patch_size, patch_size);
			cv::Rect source_patch(rand_source[0], rand_source[1], patch_size, patch_size);
			
			//calculate offset
			cv::Vec2i& nnf_offset = nnf.at<cv::Vec2i>(i, j);
			nnf_offset = cv::Vec2i(rand_source[0] - j, rand_source[1]- i);

			//calculate cost
			int current_cost = cv::norm(padded_target(target_patch), padded_source(source_patch), cv::NORM_L2SQR);
			pixel_cost.at<float>(i, j) = current_cost;
		}
	}
}

void PatchMatch::propogate(int i, int j, int negative) {
	cv::Vec2i offset = nnf.at<cv::Vec2i>(i, j);

	cv::Vec2i up_offset = nnf.at<cv::Vec2i>(i - 1*negative, j) + cv::Vec2i(1*negative, 0);
	cv::Point up_source(
		std::max(std::min(j + up_offset[0], padded_source.cols - patch_size), 0),
		std::max(std::min(i + up_offset[1], padded_source.rows - patch_size), 0));
	improveNNF(padded_source, padded_target, cv::Point(j, i), up_source, nnf, pixel_cost);

	cv::Vec2i left_offset = nnf.at<cv::Vec2i>(i,j-1*negative) + cv::Vec2i(0,1*negative);
	cv::Point left_source(
		std::max(std::min(j + left_offset[0],padded_source.cols - patch_size),0),
		std::max(std::min(i + left_offset[1], padded_source.rows - patch_size),0));
	improveNNF(padded_source, padded_target, cv::Point(j, i), left_source, nnf, pixel_cost);

}

void PatchMatch::random_search(int i, int j) {


	std::mt19937 random(std::random_device{}());
	int row_radius = padded_target.rows;
	int col_radius = padded_target.cols;
	while (row_radius >= 1 || col_radius >= 1) {
		if (row_radius < patch_size || col_radius < patch_size) break;
		int source_x = j + nnf.at<cv::Vec2i>(i, j)[0];
		int source_y = i + nnf.at<cv::Vec2i>(i, j)[1];
		std::uniform_int_distribution<int> random_col(-col_radius, col_radius);
		std::uniform_int_distribution<int> random_row(-row_radius, row_radius);

		cv::Point rand_point(
			std::max(std::min(source_x + random_col(random),padded_target.cols-patch_size),0),
			std::max(std::min(source_y + random_row(random),padded_target.rows-patch_size),0));

		improveNNF(padded_source, padded_target, cv::Point(j, i), rand_point, nnf, pixel_cost);

		row_radius /= 2;
		col_radius /= 2;
			}
			
}

void PatchMatch::iterate(int iteration_num) {

	for (int iteration = 0; iteration < iteration_num; iteration++) {
		if (iteration % 2 != 0) {
			for (int i = padded_source.rows - patch_size-2; i > 1; i--) {
				for (int j = padded_source.cols - patch_size-2; j > 1; j--) {
					propogate(i, j, -1);
					random_search(i, j);
				}
			}
		}
		else {
			for (int i = 1; i < padded_source.rows - patch_size; i++) {
				for (int j = 1; j < padded_source.cols - patch_size; j++) {
					propogate(i, j, 1);
					random_search(i, j);
				}
			}	
		}
	}
}

void PatchMatch::reconstruct_image() {

	for (int i = 1; i < padded_source.rows - patch_size-2; i++) {
		for (int j = 1; j < padded_source.cols - patch_size-2; j++) {
			int offset_x = std::clamp(i + nnf.at<cv::Vec2i>(i, j)[0],0,padded_source.rows-patch_size);
			int offset_y = std::clamp(j + nnf.at<cv::Vec2i>(i, j)[1], 0, padded_source.cols - patch_size);
			padded_target.at<cv::Vec3b>(i, j) = padded_source.at<cv::Vec3b>(offset_x, offset_y);
		}
	}

	cv::imshow("ojads", padded_target);
	cv::waitKey(0);
}

//simple method to compare patches, assumes they are padded
void PatchMatch::improveNNF(cv::Mat source, cv::Mat target, cv::Point patch_coord, cv::Point source_patch_coord, cv::Mat& nnf, cv::Mat& cost_mat) {

	cv::Rect source_patch(source_patch_coord.x,source_patch_coord.y, patch_size, patch_size);

	cv::Rect target_patch(patch_coord.x, patch_coord.y, patch_size, patch_size);

	float proposed_cost = cv::norm(padded_source(source_patch), padded_target(target_patch), cv::NORM_L2SQR);

	if (proposed_cost < cost_mat.at<float>(patch_coord.y, patch_coord.x)) {
		cost_mat.at<float>(patch_coord.y, patch_coord.x) = proposed_cost;
		nnf.at<cv::Vec2i>(patch_coord.y, patch_coord.x) = cv::Vec2i(source_patch_coord.x - patch_coord.x, source_patch_coord.y - patch_coord.y);
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