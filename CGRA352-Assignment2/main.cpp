#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "patchmatch.hpp"

void main() {
	PatchMatch* p = new PatchMatch();
	p->output_image();
}