#include "pvmaf.h"
#include <math.h>

void x264_pvmaf_init( uint32_t cpu, x264_pvmaf_functions_t *vqmf )
{
  vqmf->calculate_blurriness = calculate_blurriness_c;
  vqmf->calculate_si = calculate_si_c;
  vqmf->ModelInference = ModelInference_c;
  vqmf->NormalizeInput = NormalizeInput_c;
  vqmf->ClipInput = ClipInput_c;
  vqmf->SAD_pvmaf = SAD_pvmaf_c;
#if (HAVE_AVX2)
  if (cpu & X264_CPU_AVX2)
  {
    vqmf->calculate_blurriness = calculate_blurriness_avx2;
    vqmf->calculate_si = calculate_si_avx2;
    vqmf->ModelInference = ModelInference_avx2;
    vqmf->NormalizeInput = NormalizeInput_avx2;
    vqmf->ClipInput = ClipInput_avx2;
    vqmf->SAD_pvmaf = SAD_pvmaf_avx2;
  }
#endif
/*
TO DO...

#if (HAVE_AARCH64)
  if (cpu & X264_CPU_NEON)
  {
    vqmf->calculate_blurriness = calculate_blurriness_neon;
    vqmf->calculate_si = calculate_si_neon;
    vqmf->ModelInference = ModelInference_neon;
    vqmf->NormalizeInput = NormalizeInput_neon;
    vqmf->ClipInput = ClipInput_neon;
    vqmf->SAD_pvmaf = SAD_pvmaf_neon;
  }
#endif
*/
}

x264_frame_t* get_closest_previous_frame(x264_t *h)
{
    int curr_frame_num = h->fenc->i_frame;
    int diff = INT_MAX;
    x264_frame_t* prev_frame = NULL;

    for (int i = 0; i < h->frames.i_max_ref0; i++)
    {
        if (h->frames.reference[i])
        {
            if((curr_frame_num > h->frames.reference[i]->i_frame) && (curr_frame_num - h->frames.reference[i]->i_frame < diff))
            {
              prev_frame = h->frames.reference[i];
              diff = curr_frame_num - prev_frame->i_frame ;
              if (diff == 1)
              {
                return prev_frame;
              }
            }
        }
    }
    return prev_frame;
}

x264_frame_t* get_closest_future_frame(x264_t *h)
{
    int curr_frame_num = h->fenc->i_frame;
    int diff = INT_MAX;
    x264_frame_t* future_frame = NULL;

    for (int i = 0; i < h->frames.i_max_ref0; i++)
    {
        if (h->frames.reference[i])
        {
            if((curr_frame_num < h->frames.reference[i]->i_frame) && (h->frames.reference[i]->i_frame - curr_frame_num < diff))
            {
              future_frame = h->frames.reference[i];
              diff = h->frames.reference[i]->i_frame - curr_frame_num ;
              if (diff == 1)
              {
                return future_frame;
              }
            }
        }
    }
    return future_frame;
}

int64_t SAD_pvmaf_c(pixel* src, pixel* dst, int pitch, int width, int height)
{
  int i,j;
  pixel *sour, *dist;
  int64_t sad;
  sad = 0;
  sour = src;
  dist = dst;
  for (j = 0; j < height; j++){
    for (i = 0; i < width; i++){
      sad += abs(sour[i]-dist[i]);
    }
    sour += pitch;
    dist += pitch;
  }
  return sad;
}


void NormalizeInput_c(float* feature, float* mean, float* std, float* norm, int input_dim)
{
  for (int i = 0; i < input_dim; i++)
      norm[i] = ((float)feature[i] - mean[i]) * std[i];
}


void ClipInput_c(float* feature, float* clipped_feature, float* clip, int input_dim)
{
  for (int i = 0; i < input_dim; i++)
      clipped_feature[i] = (feature[i] > clip[i]) ? clip[i] : feature[i];
}



void ModelInference_c(float* NormalizedFeature, float* intLayer[], const float* weight[], const float* bias[], const int* layer_dim, int layers_count)
{
  int i,j;
  for (i = 0; i < layer_dim[1]; i++)
  {
      intLayer[0][i] = bias[0][i];
      for (j = 0; j < layer_dim[0]; j++)
        intLayer[0][i]  += NormalizedFeature[j] * weight[0][i * layer_dim[0] + j];

      // ReLU
      if (intLayer[0][i] < 0)
          intLayer[0][i] = 0;
      // else if (intLayer[0][i] > 1)
      //     intLayer[0][i] = 1;
  }

  // Compute Hidden layers
  for (int k = 2; k < layers_count; k++)
  {
      for (i = 0; i < layer_dim[k]; i++)
      {
          intLayer[k-1][i] = bias[k-1][i];
          for (j = 0; j < layer_dim[k-1]; j++)
            intLayer[k-1][i]  += intLayer[k-2][j] * weight[k-1][i * layer_dim[k-1] + j];

          // ReLU
          if (intLayer[k-1][i] < 0)
            intLayer[k-1][i] = 0;
          // else if (intLayer[k-1][i] > 1)
          //   intLayer[k-1][i] = 1;
      }
  }
}

void calculate_si_c(pixel* src, int pitch, int width, int height, int step, float* si)
{
  int i,j,k;
  const int size = (width-2) * (height-2)/(step);
  const int shiftBy = 4;
  int16_t* hyp = (int16_t*) malloc((MAX_ARR_SIZE/step) * sizeof(int16_t));
  uint32_t mean_sum = 0;
  src += pitch;

  uint32_t mean_;
  uint32_t var_sum = 0;
  k=0;
  int16_t sob_h, sob_v;

  for (j = 1; j < height - 1; j+=step) {
    for (i = 1; i < width - 1; i++) {
      /* Sobel mask of size 3x3  */
      sob_h =  (int16_t)(- src[-pitch + (i-1)] /*+ 0*/  + src[-pitch + (i+1)]
                        - (src[i - 1] << 1)    /*- 0*/  + (src[i + 1] << 1)
                        -  src[ pitch + (i-1)] /*+ 0*/  + src[ pitch + (i+1)] );

      sob_v =  (int16_t)(+ src[-pitch + (i-1)] + (src[-pitch + (i)] << 1) + src[-pitch + (i+1)]
                        // +            0         +             0             +          0
                        - src[ pitch + (i-1)] - (src[ pitch + (i)] << 1) - src[ pitch + (i+1)] );
      hyp[k] = (uint16_t) ((abs(sob_h) + abs(sob_v))<<shiftBy);
      mean_sum += (uint32_t) hyp[k];
      k+=1;
    }
    src += pitch*step;
  }

  mean_ = mean_sum / size;
  for (i = 0; i < size ; i++){
    var_sum +=  (abs(hyp[i] - mean_));
  }
  var_sum = var_sum >> shiftBy;
  free(hyp);
  *si = ((float)var_sum / (size-1));
}


void calculate_blurriness_c(pixel* src, int pitch, int width, int height, int step, float* blur)
{
  int i, j, k;
  int size = (width-2) * (height-2) / (step);
  int32_t mean_sum = 0, var_sum=0;
  int16_t* hyp = (int16_t*) malloc((MAX_ARR_SIZE/step) * sizeof(int16_t));
  src += pitch;
  float mean_;
  k = 0;
  assert(hyp != NULL);
  int16_t lap;

  for (j = 1; j < height - 1; j+=step) {
    for (i = 1; i < width - 1; i++) {
      /* Laplacian mask of size 3x3 */
      lap =  (int16_t)(src[-pitch + i] + src[i - 1] - (src[i]<<2) + src[i + 1] + src[ pitch + i]);
      hyp[k] = lap;
      mean_sum += (int32_t)lap;
      k += 1;
    }
    src += pitch*step;
  }

  mean_ = (float)mean_sum/size;
  for (i = 0; i < size ; i++)
  {
    var_sum += (int32_t)abs(hyp[i] - (int16_t) mean_);
  }
  
  free(hyp);
  *blur = (float)var_sum/(size-1);
}

#if (HAVE_AVX2)
int64_t SAD_pvmaf_avx2(pixel* src, pixel* dst, int pitch, int width, int height)
{
  int i,j;
  pixel *sour, *dist;
  int64_t sad;
  sad = 0;
  sour = src;
  dist = dst;
  YMM r00, r01;
  YMM vec_sum1 = _mm256_setzero_si256();
  for (j = 0; j < height; j++)
  {
    for (i = 0; i < width; i+=32)
    {
      r00 = movdqu_ld_ymm(&sour[i]);
      r01 = movdqu_ld_ymm(&dist[i]);
      vec_sum1 = _mm256_add_epi64(vec_sum1, _mm256_sad_epu8(r00,r01));
    }
    sour += pitch;
    dist += pitch;
  }
  int64_t result[4];
  _mm256_storeu_si256((__m256i*)result, vec_sum1);
  sad = (int64_t) (result[0] + result[1] + result[2] + result[3]);
  return sad;
}

void NormalizeInput_avx2(float* feature, float* mean, float* std, float* norm, int input_dim)
{
  YMMf f_mean, f_std, f_feature;

  for (int i = 0; i < input_dim; i += 8)
  {
      f_feature     = _mm256_loadu_ps   ((&feature[i]));
      f_mean        = _mm256_loadu_ps   (&mean[i]);
      f_std         = _mm256_loadu_ps   (&std[i]);
      _mm256_storeu_ps(&norm[i], _mm256_mul_ps(_mm256_sub_ps(f_feature, f_mean), f_std));
  }
}

void ClipInput_avx2(float* feature, float* clipped_feature, float* clip, int input_dim)
{
  YMMf feature_clips, f_feature;
  for (int i = 0; i < input_dim; i += 8)
  {
    f_feature            = _mm256_loadu_ps   (&feature[i]);
    feature_clips        = _mm256_loadu_ps   (&clip[i]);
    _mm256_storeu_ps(&clipped_feature[i], _mm256_min_ps(f_feature, feature_clips));
  }
}

void ModelInference_avx2(float* NormalizedFeature, float* intLayer[], const float* weight[], const float* bias[], const int* layer_dim, int layers_count)
{
  int i,j;
  YMMf sum;
  XMMf v_x;
  for (i = 0; i < layer_dim[1]; i++)
  {
      sum   = _mm256_setzero_ps();
      for (j = 0; j < layer_dim[0]; j += 8)
      {
          sum = _mm256_fmadd_ps(_mm256_loadu_ps(&NormalizedFeature[j]), _mm256_loadu_ps(weight[0]+(i*layer_dim[0]+j)), sum);
      }
      v_x = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
      v_x = _mm_add_ps(v_x, _mm_movehl_ps  (v_x, v_x));
      v_x = _mm_add_ps(v_x, _mm_shuffle_ps (v_x, v_x, 1));
      intLayer[0][i] = X264_MAX(_mm_cvtss_f32(v_x) + bias[0][i], 0);
      // intLayer[0][i] = X264_MIN(intLayer[0][i], 1);
  }
  // Hidden layer
  for (int k = 2; k < layers_count; k++)
  {
      for (i = 0; i < layer_dim[k]; i++)
      {
          sum   = _mm256_setzero_ps();
          for (j = 0; j < layer_dim[k-1]; j += 8)
          {
              sum = _mm256_fmadd_ps(_mm256_loadu_ps(&intLayer[k-2][j]), _mm256_loadu_ps(weight[k-1]+(i*layer_dim[k-1]+j)), sum);
          }
          v_x = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
          v_x = _mm_add_ps(v_x, _mm_movehl_ps  (v_x, v_x));
          v_x = _mm_add_ps(v_x, _mm_shuffle_ps (v_x, v_x, 1));
          intLayer[k-1][i] = X264_MAX(_mm_cvtss_f32(v_x) + bias[k-1][i], 0);
          // intLayer[k-1][i] = X264_MIN(intLayer[k-1][i], 1);
      }
  }
}

void calculate_si_avx2(pixel* src, int pitch, int width, int height, int step, float* si)
{
  int i,j,k;
  const int size = (width-2) * (height-2)/(step);
  const int shiftBy = 4;
  int16_t* hyp = (int16_t*) malloc((MAX_ARR_SIZE/step) * sizeof(int16_t));
  uint32_t mean_sum = 0;
  src += pitch;

  uint32_t mean_;
  uint32_t var_sum = 0;
  k=0;

  int size8x = size - (size%8);

  __m128i ones = _mm_set1_epi64x(1);
  __m128i vShiftBy = _mm_set1_epi64x(shiftBy);
  for (j = 1; j < height - 1; j+=step) {
    __m256i r1, r2, r3, r4, rt1, rt2, sum_ymm_h, sum_ymm_v;
    for (i = 1; i < width - 1; i+=16) {
      /* Sobel Horizontal mask of size 3x3 */
      r1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src[-pitch+i-1]));
      rt1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src[       i-1]));
      r3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src[ pitch+i-1]));
      r2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src[-pitch+i+1]));
      rt2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src[       i+1]));
      r4 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src[ pitch+i+1]));

      sum_ymm_h = _mm256_add_epi16(_mm256_sll_epi16(rt2, ones), _mm256_add_epi16(r2, r4));
      sum_ymm_h = _mm256_sub_epi16(sum_ymm_h, _mm256_add_epi16(_mm256_sll_epi16(rt1, ones), _mm256_add_epi16(r1, r3)));
      sum_ymm_h = _mm256_sll_epi16(_mm256_abs_epi16(sum_ymm_h),vShiftBy);

      /* Sobel Horizontal mask of size 3x3 */
      rt1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src[-pitch+i  ]));
      rt2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src[ pitch+i]));

      sum_ymm_v = _mm256_add_epi16(_mm256_sll_epi16(rt1, ones), _mm256_add_epi16(r1, r2));
      sum_ymm_v = _mm256_sub_epi16(sum_ymm_v, _mm256_add_epi16(_mm256_sll_epi16(rt2, ones), _mm256_add_epi16(r3, r4)));
      sum_ymm_v = _mm256_sll_epi16(_mm256_abs_epi16(sum_ymm_v),vShiftBy);

      /* Store the result of laplacian */

      _mm256_storeu_si256(( __m256i*)&hyp[k], _mm256_add_epi16(sum_ymm_h, sum_ymm_v));
      k += 16;
    }
    /* Account for boundary conditions */
    k -= 2;
    src += pitch*step;
  }

  // /* Assign few of the elements of this array to 0 */
  hyp[k] = 0;
  hyp[k+1] = 0;

  __m256i sum_ymm = _mm256_setzero_si256();
  for ( i = 0; i < size8x; i+=8) {
    sum_ymm = _mm256_add_epi32(sum_ymm, _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)&hyp[i])));
  }
  uint32_t sum_[8];
  _mm256_storeu_si256(( __m256i*)&sum_, sum_ymm);
  mean_sum = (sum_[0]+sum_[1]+sum_[2]+sum_[3]+sum_[4]+sum_[5]+sum_[6]+sum_[7]);
  // Add remaining elements if size is not a multiple of 8
  for ( ; i < size ; i++) {
    mean_sum += (uint32_t)hyp[i];
  }

  mean_ = (uint32_t)mean_sum/size;

  __m256i sum_ymm_s = _mm256_setzero_si256();
  __m256i mean_ymm = _mm256_set1_epi32(mean_);
  // __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); // To compute absolute
  __m256i elements_ymm;

  for ( i = 0; i < size8x; i+=8) {
    elements_ymm = (_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)&hyp[i])));
    sum_ymm_s = _mm256_add_epi32(sum_ymm_s, _mm256_abs_epi32(_mm256_sub_epi32(elements_ymm,mean_ymm)));
  }
  uint32_t sum_f[8];
  _mm256_storeu_si256(( __m256i*)&sum_f[0], sum_ymm_s);
  var_sum = (sum_f[0]+sum_f[1]+sum_f[2]+sum_f[3]+sum_f[4]+sum_f[5]+sum_f[6]+sum_f[7]);

  // Add remaining elements if size is not a multiple of 8
  for (; i < size ; i++) {
    var_sum += abs((uint32_t)hyp[i] - mean_);
  }
  var_sum = var_sum >> shiftBy;
  free(hyp);
  *si = ((float)var_sum / (size-1));
}

void calculate_blurriness_avx2(pixel* src, int pitch, int width, int height, int step, float* blur)
{
  int i, j, k;
  int size = (width-2) * (height-2) / (step);
  int32_t mean_sum = 0, var_sum=0;
  int16_t* hyp = (int16_t*) malloc((MAX_ARR_SIZE/step) * sizeof(int16_t));
  src += pitch;
  float mean_;
  k = 0;
  assert(hyp != NULL);
  int size8x = size - (size%8);

  __m128i two_ = _mm_set1_epi64x(2);
  for (j = 1; j < height - 1; j+=step) {
    __m256i ru, rl, rc, rr, rd, sum_ymm;
    for (i = 1; i < width - 1; i+=16) {
      /* Apply Laplacian mask of size 3x3 */
      ru = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src[-pitch+i  ]));
      rl = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src[       i-1]));
      rc = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src[       i  ]));
      rr = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src[       i+1]));
      rd = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src[ pitch+i  ]));
      sum_ymm = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(ru,  rl), rr), rd);
      sum_ymm = _mm256_sub_epi16(sum_ymm, _mm256_sll_epi16(rc, two_));

      /* Store the result of convolution */

      _mm256_storeu_si256(( __m256i*)&hyp[k], sum_ymm);
      k += 16;
    }
    /* Account for boundary condition */
    k -= 2;
    src += pitch*step;
  }

  // /* Assign few of the elements of this array to 0 */
  hyp[k] = 0;
  hyp[k+1] = 0;

  __m256i sum_ymm = _mm256_setzero_si256();
  for ( i = 0; i < size8x; i+=8) {
    sum_ymm = _mm256_add_epi32(sum_ymm, _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)&hyp[i])));
  }
  int32_t sum_[8];
  _mm256_storeu_si256(( __m256i*)&sum_, sum_ymm);
  mean_sum = (sum_[0]+sum_[1]+sum_[2]+sum_[3]+sum_[4]+sum_[5]+sum_[6]+sum_[7]);
  // Add remaining elements if size is not a multiple of 8
  for ( ; i < size ; i++) {
    mean_sum += hyp[i];
  }

  mean_ = (float)mean_sum/size;

  __m256i mean_ymm = _mm256_set1_epi32((int32_t)mean_);
  sum_ymm = _mm256_setzero_si256();
  for ( i = 0; i < size8x; i+=8) {
    sum_ymm = _mm256_add_epi32(sum_ymm, _mm256_abs_epi32(_mm256_sub_epi32(_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)&hyp[i])),mean_ymm)));
  }
  _mm256_storeu_si256(( __m256i*)&sum_, sum_ymm);
  var_sum = (sum_[0]+sum_[1]+sum_[2]+sum_[3]+sum_[4]+sum_[5]+sum_[6]+sum_[7]);

  // Add remaining elements if size is not a multiple of 8
  for (; i < size ; i++) {
    var_sum += (int32_t)abs(hyp[i] - (int16_t) mean_);
  }

  free(hyp);
  *blur = (float)var_sum/(size-1);
}

#endif

#if(HAVE_AARCH64)
// TO DO
#endif

// Compute pVMAF

void compute_pvmaf(x264_t *h, float *feature, float *pvmaf)
{
  int32_t i;

  // Clip feature values
  float ClippedFeature[input_dim_pvmaf];
  h->pvmaf_f.ClipInput(feature, ClippedFeature, feature_clip_pvmaf, layer_dim_pvmaf[0]);

  // Normalize the features
  float NormalizedFeature[input_dim_pvmaf];
  h->pvmaf_f.NormalizeInput(ClippedFeature, feature_mean_pvmaf, feature_std_pvmaf, NormalizedFeature, layer_dim_pvmaf[0]);

//   // Dimension verification
  assert(input_dim_pvmaf % 8 == 0);
  assert(mlp_dim1_pvmaf % 8 == 0);
  assert(mlp_dim2_pvmaf % 8 == 0);

  // Neural Network Init
  float IntLayer[layers_count_pvmaf-1][max_dim_size_pvmaf];
  float* intLayer[layers_count_pvmaf-1];
  const float* weights[] = {dense0_weight_pvmaf, dense1_weight_pvmaf, dense2_weight_pvmaf};
  const float* bias[]   = {dense0_bias_pvmaf, dense1_bias_pvmaf, dense2_bias_pvmaf};
  for (i=0; i<layers_count_pvmaf-1; i++)
    intLayer[i] = IntLayer[i];

  // Neural Network Inference
  h->pvmaf_f.ModelInference(NormalizedFeature, intLayer, weights, bias, layer_dim_pvmaf, layers_count_pvmaf);

  *pvmaf = intLayer[layers_count_pvmaf-2][0];
}