#include "imagereshuffling.hpp"

//initialize method for image swapping, taking the source image and a mask 
void ImageReshuffling::init(cv::Mat source_img, cv::Mat mask) {
	target = source_img.clone();
	source = source_img;
	initial_nnf = cv::Mat::zeros(source.size(), CV_32SC2);
	patch_size = 7;

	for (int r = 0; r < source.rows; r++) {
		for (int c = 0; c < source.cols; c++) {
			if (mask.at<uchar>(r, c) == 0) initial_nnf.at<cv::Vec2i>(r, c) = cv::Vec2i(0, 0); //if not mask area, offset is 0
			else {
				cv::Vec3b temp_pix = target.at<cv::Vec3b>(r, c - 270);
				target.at<cv::Vec3b>(r, c - 270) = source.at<cv::Vec3b>(r, c);      //swap pixels
				target.at<cv::Vec3b>(r, c) = temp_pix;  //at original position and -270 to the left

				initial_nnf.at<cv::Vec2i>(r, c) = cv::Vec2i(0,  -270);   //set nnf offset to -270 for masked area
			}
		}
	}

	//create guassian pyaramids
	cv::buildPyramid(source, source_pyr, pyramid_depth);
	cv::buildPyramid(target, target_pyr, pyramid_depth);
	nnf_pyr.resize(5);
}

//special initialization method for the inpainting, 
// as we want random initialization for the mask
void ImageReshuffling::init_for_inpainting(cv::Mat source_img, cv::Mat mask) {
	target = source_img.clone();
	source = source_img.clone();
	initial_nnf = cv::Mat::zeros(source.size(), CV_32SC2);

	patch_size = 15;
	inpainting = true;

	for (int r = 0; r < source.rows; r++) {
		for (int c = 0; c < source.cols; c++) {
			if (mask.at<uchar>(r, c) == 0) initial_nnf.at<cv::Vec2i>(r, c) = cv::Vec2i(0, 0);
			else {
				int rand_r;
				int rand_c;
				do {
					rand_r = rand() % source.rows;
					rand_c = rand() % source.cols;
				} while (mask.at<uchar>(rand_r, rand_c) != 0);   //ensure random point is outside the mask to avoid corruption
				
				target.at<cv::Vec3b>(r, c) = source.at<cv::Vec3b>(rand_r,rand_c);
				initial_nnf.at<cv::Vec2i>(r, c) = cv::Vec2i(rand_r - r, rand_c - c);
			}
		}
	}

	cv::buildPyramid(source, source_pyr, pyramid_depth);
	cv::buildPyramid(target, target_pyr, pyramid_depth);
	cv::buildPyramid(mask, mask_pyr, pyramid_depth);   //need a mask pyramid for the inpainting to pass reconstructed target upwards
	nnf_pyr.resize(5);
}

//use patchmatch to reconstruct target image with source image at each step of the pyramid
void ImageReshuffling::create_pyramid_nnfs() {
    PatchMatch p(patch_size);
    
    //downsample initial nnf to coarsest level first
    cv::Mat coarse_nnf = downsample_nnf(initial_nnf, source_pyr[pyramid_depth]);
    
	//initial iteration to form coarsest layer of each pyramid
    p.init_with_nnf(source_pyr[pyramid_depth], target_pyr[pyramid_depth], coarse_nnf);
    p.iterate(patchmatch_iter);
    nnf_pyr[pyramid_depth] = p.get_nnf();
    target_pyr[pyramid_depth] = p.reconstruct_image();
	
    for (int k = pyramid_depth; k >= 0; k--) {
		p.init_with_nnf(source_pyr[k], target_pyr[k], nnf_pyr[k]); //reinitialize patchmatch object with new nnf and cost matrix
        for (int i = 0; i < patchmatch_iter; i++) {
            p.iterate(patchmatch_iter);

			//set pyramid layers to new nnf and reconstructed image
            nnf_pyr[k] = p.get_nnf();
            target_pyr[k] = p.reconstruct_image();
        }

		cv::imshow("odjjo", p.reconstruct_image());
		cv::imshow("aioshdihka", PatchMatch::nnf2img(nnf_pyr[k], target_pyr[k]));
		cv::waitKey(0);

        if (k > 0) {
            nnf_pyr[k - 1] = upsample_nnf(nnf_pyr[k], target_pyr[k - 1]);  //upsample nnf so its correct size for next layer
			//if inpainting, feed reconstructed target mask area back into pyramid to compound accuracy of patchmatch
			if (inpainting) {
				cv::Mat temp;
				cv::pyrUp(target_pyr[k], temp, target_pyr[k - 1].size());
				temp.copyTo(target_pyr[k - 1], mask_pyr[k - 1]);
			}
			
        }
    }
}

cv::Mat ImageReshuffling::upsample_nnf(cv::Mat nnf, cv::Mat next) {
	cv::Mat output_nnf;
	cv::resize(nnf, output_nnf, next.size(), 0, 0, cv::INTER_NEAREST);
	output_nnf *= 2;
	return output_nnf;
}

cv::Mat ImageReshuffling::downsample_nnf(cv::Mat nnf, cv::Mat target_level) {
	cv::Mat output_nnf;
	cv::resize(nnf, output_nnf, target_level.size(), 0, 0, cv::INTER_NEAREST);
	output_nnf /= 2;
	return output_nnf;
}
