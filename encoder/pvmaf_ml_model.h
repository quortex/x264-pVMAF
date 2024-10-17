#pragma once
#include <stdint.h>

// MLP model: input_dim_pvmaf x mlp_dim_pvmaf x mlp_last_pvmaf x 1
#define input_dim_pvmaf 24       // number of MLP model inputs
#define mlp_dim1_pvmaf 16 // number of nodes in network layer 1
#define mlp_dim2_pvmaf 16 // number of nodes in network layer 2
#define output_dim_pvmaf 1       // number of MLP model outputs
#define max_dim_size_pvmaf 24       // Dim max
#define layers_count_pvmaf 4       // number of MLP model outputs
extern const uint32_t feature_stats_pvmaf[3 * input_dim_pvmaf];
extern const uint32_t model_weights_pvmaf[((input_dim_pvmaf+1) * mlp_dim1_pvmaf) +((mlp_dim1_pvmaf+1) * mlp_dim2_pvmaf) +((mlp_dim2_pvmaf+1) * output_dim_pvmaf)];

extern const int layer_dim_pvmaf[layers_count_pvmaf];
#define dense0_weight_pvmaf  ((float *) model_weights_pvmaf)
#define dense0_bias_pvmaf  ((float *) dense0_weight_pvmaf+ input_dim_pvmaf * mlp_dim1_pvmaf)
#define dense1_weight_pvmaf  ((float *) dense0_bias_pvmaf+ mlp_dim1_pvmaf)
#define dense1_bias_pvmaf  ((float *) dense1_weight_pvmaf+ mlp_dim1_pvmaf * mlp_dim2_pvmaf)
#define dense2_weight_pvmaf  ((float *) dense1_bias_pvmaf+ mlp_dim2_pvmaf)
#define dense2_bias_pvmaf  ((float *) dense2_weight_pvmaf+ mlp_dim2_pvmaf * output_dim_pvmaf)

#define feature_mean_pvmaf  ((float *) feature_stats_pvmaf)                                      // mean of features:        input_dim_pvmaf x 1
#define feature_std_pvmaf   ((float *) feature_mean_pvmaf + input_dim_pvmaf)         // std of features:         input_dim_pvmaf x 1
#define feature_clip_pvmaf   ((float *) feature_std_pvmaf + input_dim_pvmaf)     // clipping values of features:         input_dim_pvmaf x 2

typedef enum eFeature_pvmaf
{
frame_qp_pvmaf, // frame_qp
frame_size_pvmaf, // frame_size
i_frame_pvmaf, // i_frame
p_frame_pvmaf, // p_frame
b_frame_pvmaf, // b_frame
num_intra_macroblocks_pvmaf, // num_intra_macroblocks
num_inter_macroblocks_pvmaf, // num_inter_macroblocks
num_skip_macroblocks_pvmaf, // num_skip_macroblocks
global_motion_abs_pvmaf, // global_motion_abs
pa_spatial_information2_pvmaf, // pa_spatial_information2
spatial_information2_pvmaf, // spatial_information2
spatial_information_ratio_pvmaf, // spatial_information_ratio
pa_blurriness2_pvmaf, // pa_blurriness2
blurriness2_pvmaf, // blurriness2
blurriness_ratio_pvmaf, // blurriness_ratio
motion_approximation_pvmaf, // motion_approximation
offset_pvmaf, // offset
psnr_y_codec_pvmaf, // psnr_y_codec
i_ME_cost_intra_pvmaf, // i_ME_cost_intra
i_ME_cost_inter_pvmaf, // i_ME_cost_inter
i_ME_cost_B_frame_pvmaf, // i_ME_cost_B_frame
DUMMY_0_pvmaf, // DUMMY_0
DUMMY_1_pvmaf, // DUMMY_1
DUMMY_2_pvmaf, // DUMMY_2
} eFeature_pvmaf;
