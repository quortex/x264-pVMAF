#pragma once

#include "common/common.h"
#include <immintrin.h>    // AVX/AVX2
#include "pvmaf_ml_model.h"
#include "../config.h"

#if (HAVE_MMX)

#define XMM                               __m128i
#define YMM                               __m256i
#define XMMf                              __m128
#define YMMf                              __m256
#define movdqu_ld_ymm(m)                  _mm256_loadu_si256((const YMM*)(m))                 // Load  32 bytes from unaligned memory

#endif

#define MAX_ARR_SIZE                      2263680


void calculate_blurriness(pixel* src, int pitch, int width, int height, int step, float* blur);
void calculate_si(pixel* src, int pitch, int width, int height, int step, float* si);
void calculate_si_blurriness(pixel* src, int pitch, int width, int height, int step, float* blur, float* si);
void ModelInference(float* NormalizedFeature, float* intLayer[], const float* weight[], const float* bias[], const int* layer_dim, int layers_count);
void NormalizeInput(float* feature, float* mean, float* std, float* norm, int input_dim);
void ClipInput(float* feature, float* clipped_feature, float* clip, int input_dim);
void compute_pvmaf(float *feature, float *pvmaf);
int64_t SAD_AVX2(pixel* sour, pixel* dist, int pitch, int width, int height);

x264_frame_t* get_previous_frame(x264_t *h);
x264_frame_t* get_closest_neighbour_frame(x264_t *h);
x264_frame_t* get_closest_previous_frame(x264_t *h);
x264_frame_t* get_closest_future_frame(x264_t *h);