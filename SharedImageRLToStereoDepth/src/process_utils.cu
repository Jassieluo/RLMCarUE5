#include "process_utils.h"

__global__ void uchar2float_kernel(
	uint8_t* src, int src_line_size, int src_width,
	int src_height, float* dst, int dst_width,
	int dst_height, uint8_t const_value_st, int edge) {
	int position = blockDim.x * blockIdx.x + threadIdx.x;
	if (position >= edge) return;

	int dx = position % dst_width;
	int dy = position / dst_width;
	float c0, c1, c2;

	c0 = float(*(src + dy * src_line_size + dx * 3));
	c1 = float(*(src + dy * src_line_size + dx * 3 + 1));
	c2 = float(*(src + dy * src_line_size + dx * 3 + 2));

	// bgr -> rgb
	float temp = c2;
	c2 = c0;
	c0 = temp;

	// normalization
	c0 /= 255.0f;
	c1 /= 255.0f;
	c2 /= 255.0f;
	// rgbrgbrgb -> rrrgggbbb
	int area = dst_height * dst_width;
	float* pdst_c0 = dst + dy * dst_width + dx;
	float* pdst_c1 = pdst_c0 + area;
	float* pdst_c2 = pdst_c1 + area;
	*pdst_c0 = c0;
	*pdst_c1 = c1;
	*pdst_c2 = c2;
}

void preprocess_no_resize(
	uint8_t* src, const int& src_width, const int& src_height,
	float* dst, const int& dst_width, const int& dst_height,
	cudaStream_t stream) {

	int jobs = dst_height * dst_width;
	int threads = 256;
	int blocks = ceil(jobs / (float)threads);
	uchar2float_kernel << <blocks, threads, 0, stream >> > (
		src, src_width * 3, src_width,
		src_height, dst, dst_width,
		dst_height, 128, jobs);
}