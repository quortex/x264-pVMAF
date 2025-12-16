#pragma once

#include "common/common.h"
#include "pvmaf_ml_model.h"
#include "../config.h"

#if (HAVE_AVX2)

#include <immintrin.h>    // AVX/AVX2
#define XMM                               __m128i
#define YMM                               __m256i
#define XMMf                              __m128
#define YMMf                              __m256
#define movdqu_ld_ymm(m)                  _mm256_loadu_si256((const YMM*)(m))                 // Load  32 bytes from unaligned memory

#endif

#define MAX_ARR_SIZE                      2263680


void calculate_blurriness_c(pixel* src, int pitch, int width, int height, int step, float* blur);
void calculate_si_c(pixel* src, int pitch, int width, int height, int step, float* si);
void ModelInference_c(float* NormalizedFeature, float* intLayer[], const float* weight[], const float* bias[], const int* layer_dim, int layers_count);
void NormalizeInput_c(float* feature, float* mean, float* std, float* norm, int input_dim);
void ClipInput_c(float* feature, float* clipped_feature, float* clip, int input_dim);
int64_t SAD_pvmaf_c(pixel* sour, pixel* dist, int pitch, int width, int height);

#if (HAVE_AVX2)

void calculate_blurriness_avx2(pixel* src, int pitch, int width, int height, int step, float* blur);
void calculate_si_avx2(pixel* src, int pitch, int width, int height, int step, float* si);
void ModelInference_avx2(float* NormalizedFeature, float* intLayer[], const float* weight[], const float* bias[], const int* layer_dim, int layers_count);
void NormalizeInput_avx2(float* feature, float* mean, float* std, float* norm, int input_dim);
void ClipInput_avx2(float* feature, float* clipped_feature, float* clip, int input_dim);
int64_t SAD_pvmaf_avx2(pixel* sour, pixel* dist, int pitch, int width, int height);

#endif

#if(HAVE_AARCH64)

// TO DO

#endif

void compute_pvmaf(x264_t *h, float *feature, float *pvmaf);

x264_frame_t* get_previous_frame(x264_t *h);
x264_frame_t* get_closest_neighbour_frame(x264_t *h);
x264_frame_t* get_closest_previous_frame(x264_t *h);
x264_frame_t* get_closest_future_frame(x264_t *h);

void x264_pvmaf_init( uint32_t cpu, x264_pvmaf_functions_t *vqmf );
