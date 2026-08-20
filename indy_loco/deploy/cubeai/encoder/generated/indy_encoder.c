/**
  ******************************************************************************
  * @file    indy_encoder.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-08-19T20:30:12-0400
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "indy_encoder.h"
#include "indy_encoder_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_indy_encoder
 
#undef AI_INDY_ENCODER_MODEL_SIGNATURE
#define AI_INDY_ENCODER_MODEL_SIGNATURE     "0xd1136eae31a868985a39017781b835ad"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-08-19T20:30:12-0400"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_INDY_ENCODER_N_BATCHES
#define AI_INDY_ENCODER_N_BATCHES         (1)

static ai_ptr g_indy_encoder_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_indy_encoder_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  features_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 9600, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  features_Transpose_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9600, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_Transpose_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Sub_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Pow_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Sqrt_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Add_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_Transpose_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _convolutions_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3328, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _Slice_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _Add_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _activation_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _convolutions_1_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3456, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _Slice_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _Add_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _activation_1_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _convolutions_2_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3712, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _Slice_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _Add_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _activation_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _convolutions_3_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4224, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _Slice_3_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  _Add_3_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  _activation_3_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 3200, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Constant_output_0_3D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  spatial_1_normalization_weight_3D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  spatial_1_normalization_bias_3D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  _convolutions_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  _convolutions_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  _convolutions_1_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  _convolutions_1_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  _convolutions_2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  _convolutions_2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  _convolutions_3_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  _convolutions_3_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  _spatial_spatial_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  _convolutions_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Add_1_output_0_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Add_1_output_0_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _Add_2_output_0_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Add_2_output_0_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _Add_3_output_0_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Add_3_output_0_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _Add_output_0_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Add_output_0_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _Slice_1_output_0_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Slice_1_output_0_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _Slice_2_output_0_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Slice_2_output_0_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _Slice_3_output_0_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Slice_3_output_0_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _Slice_output_0_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Slice_output_0_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _activation_1_Relu_output_0_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_activation_1_Relu_output_0_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _activation_2_Relu_output_0_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_activation_2_Relu_output_0_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _activation_3_Relu_output_0_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_activation_3_Relu_output_0_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _activation_Relu_output_0_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_activation_Relu_output_0_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _convolutions_0_Conv_output_0_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_convolutions_0_Conv_output_0_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _convolutions_0_Conv_output_0_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 52), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_convolutions_0_Conv_output_0_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _convolutions_0_Conv_output_0_scratch0, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 3), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_convolutions_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _convolutions_0_Conv_output_0_weights, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_convolutions_0_Conv_output_0_weights_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _convolutions_1_Conv_output_0_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_convolutions_1_Conv_output_0_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _convolutions_1_Conv_output_0_output, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 54), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_convolutions_1_Conv_output_0_output_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _convolutions_1_Conv_output_0_weights, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_convolutions_1_Conv_output_0_weights_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _convolutions_2_Conv_output_0_bias, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_convolutions_2_Conv_output_0_bias_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _convolutions_2_Conv_output_0_output, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 58), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_convolutions_2_Conv_output_0_output_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _convolutions_2_Conv_output_0_weights, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_convolutions_2_Conv_output_0_weights_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _convolutions_3_Conv_output_0_bias, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_convolutions_3_Conv_output_0_bias_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _convolutions_3_Conv_output_0_output, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 66), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_convolutions_3_Conv_output_0_output_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _convolutions_3_Conv_output_0_weights, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_convolutions_3_Conv_output_0_weights_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_0_Conv_output_0_bias, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_spatial_spatial_0_Conv_output_0_bias_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_0_Conv_output_0_output, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_spatial_spatial_0_Conv_output_0_output_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_0_Conv_output_0_scratch0, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &_spatial_spatial_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_0_Conv_output_0_weights, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 192, 1, 1, 64), AI_STRIDE_INIT(4, 4, 768, 49152, 49152),
  1, &_spatial_spatial_0_Conv_output_0_weights_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_Transpose_1_output_0_output, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_spatial_spatial_1_Transpose_1_output_0_output_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_Transpose_output_0_output, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 64), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_Transpose_output_0_output_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Add_1_output_0_output, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 64), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_normalization_Add_1_output_0_output_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Constant_output_0_3D, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_spatial_spatial_1_normalization_Constant_output_0_3D_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Div_output_0_output, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 64), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_normalization_Div_output_0_output_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Mul_output_0_output, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 64), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_normalization_Mul_output_0_output_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Pow_output_0_output, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 64), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_normalization_Pow_output_0_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_bias, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_bias_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_1_output_0_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_normalization_ReduceMean_1_output_0_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_bias, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_normalization_ReduceMean_output_0_Mul_bias_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_output, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_normalization_ReduceMean_output_0_Mul_output_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_scale, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_normalization_ReduceMean_output_0_Mul_scale_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_output_0_output, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_normalization_ReduceMean_output_0_output_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Sqrt_output_0_output, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_normalization_Sqrt_output_0_output_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Sub_output_0_output, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 64), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &_spatial_spatial_1_normalization_Sub_output_0_output_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  _spatial_spatial_2_Relu_output_0_output, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_spatial_spatial_2_Relu_output_0_output_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  features_Transpose_output, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 50), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &features_Transpose_output_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  features_output, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 192), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &features_output_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  spatial_1_normalization_bias_3D, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 64), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &spatial_1_normalization_bias_3D_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  spatial_1_normalization_weight_3D, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 64), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &spatial_1_normalization_weight_3D_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  _activation_3_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_3_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_3_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _activation_3_Relu_output_0_layer, 47,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_activation_3_Relu_output_0_chain,
  NULL, &_activation_3_Relu_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_3_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Slice_3_output_0_output, &_activation_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_3_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_3_output_0_layer, 46,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Add_3_output_0_chain,
  NULL, &_activation_3_Relu_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)


AI_STATIC_CONST ai_u8 _Slice_3_output_0_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_3_output_0_axes, AI_ARRAY_FORMAT_U8,
    _Slice_3_output_0_axes_data, _Slice_3_output_0_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _Slice_3_output_0_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_3_output_0_starts, AI_ARRAY_FORMAT_S16,
    _Slice_3_output_0_starts_data, _Slice_3_output_0_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _Slice_3_output_0_ends_data[] = { 50 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_3_output_0_ends, AI_ARRAY_FORMAT_S16,
    _Slice_3_output_0_ends_data, _Slice_3_output_0_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Slice_3_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_convolutions_3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Slice_3_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Slice_3_output_0_layer, 45,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &_Slice_3_output_0_chain,
  NULL, &_Add_3_output_0_layer, AI_STATIC, 
  .axes = &_Slice_3_output_0_axes, 
  .starts = &_Slice_3_output_0_starts, 
  .ends = &_Slice_3_output_0_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _convolutions_3_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_convolutions_3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_convolutions_3_Conv_output_0_weights, &_convolutions_3_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _convolutions_3_Conv_output_0_layer, 40,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_convolutions_3_Conv_output_0_chain,
  NULL, &_Slice_3_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 8), 
  .filter_pad = AI_SHAPE_INIT(4, 16, 0, 16, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _activation_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _activation_2_Relu_output_0_layer, 39,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_activation_2_Relu_output_0_chain,
  NULL, &_convolutions_3_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Slice_2_output_0_output, &_activation_1_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_2_output_0_layer, 38,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Add_2_output_0_chain,
  NULL, &_activation_2_Relu_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)


AI_STATIC_CONST ai_u8 _Slice_2_output_0_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_2_output_0_axes, AI_ARRAY_FORMAT_U8,
    _Slice_2_output_0_axes_data, _Slice_2_output_0_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _Slice_2_output_0_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_2_output_0_starts, AI_ARRAY_FORMAT_S16,
    _Slice_2_output_0_starts_data, _Slice_2_output_0_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _Slice_2_output_0_ends_data[] = { 50 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_2_output_0_ends, AI_ARRAY_FORMAT_S16,
    _Slice_2_output_0_ends_data, _Slice_2_output_0_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Slice_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_convolutions_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Slice_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Slice_2_output_0_layer, 37,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &_Slice_2_output_0_chain,
  NULL, &_Add_2_output_0_layer, AI_STATIC, 
  .axes = &_Slice_2_output_0_axes, 
  .starts = &_Slice_2_output_0_starts, 
  .ends = &_Slice_2_output_0_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _convolutions_2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_1_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_convolutions_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_convolutions_2_Conv_output_0_weights, &_convolutions_2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _convolutions_2_Conv_output_0_layer, 32,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_convolutions_2_Conv_output_0_chain,
  NULL, &_Slice_2_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 4), 
  .filter_pad = AI_SHAPE_INIT(4, 8, 0, 8, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _activation_1_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_1_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _activation_1_Relu_output_0_layer, 31,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_activation_1_Relu_output_0_chain,
  NULL, &_convolutions_2_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Slice_1_output_0_output, &_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_1_output_0_layer, 30,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Add_1_output_0_chain,
  NULL, &_activation_1_Relu_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)


AI_STATIC_CONST ai_u8 _Slice_1_output_0_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_1_output_0_axes, AI_ARRAY_FORMAT_U8,
    _Slice_1_output_0_axes_data, _Slice_1_output_0_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _Slice_1_output_0_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_1_output_0_starts, AI_ARRAY_FORMAT_S16,
    _Slice_1_output_0_starts_data, _Slice_1_output_0_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _Slice_1_output_0_ends_data[] = { 50 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_1_output_0_ends, AI_ARRAY_FORMAT_S16,
    _Slice_1_output_0_ends_data, _Slice_1_output_0_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Slice_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_convolutions_1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Slice_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Slice_1_output_0_layer, 29,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &_Slice_1_output_0_chain,
  NULL, &_Add_1_output_0_layer, AI_STATIC, 
  .axes = &_Slice_1_output_0_axes, 
  .starts = &_Slice_1_output_0_starts, 
  .ends = &_Slice_1_output_0_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _convolutions_1_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_convolutions_1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_convolutions_1_Conv_output_0_weights, &_convolutions_1_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _convolutions_1_Conv_output_0_layer, 24,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_convolutions_1_Conv_output_0_chain,
  NULL, &_Slice_1_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 2), 
  .filter_pad = AI_SHAPE_INIT(4, 4, 0, 4, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _activation_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_activation_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _activation_Relu_output_0_layer, 23,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_activation_Relu_output_0_chain,
  NULL, &_convolutions_1_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Slice_output_0_output, &_spatial_spatial_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_output_0_layer, 22,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Add_output_0_chain,
  NULL, &_activation_Relu_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)


AI_STATIC_CONST ai_u8 _Slice_output_0_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_output_0_axes, AI_ARRAY_FORMAT_U8,
    _Slice_output_0_axes_data, _Slice_output_0_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _Slice_output_0_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_output_0_starts, AI_ARRAY_FORMAT_S16,
    _Slice_output_0_starts_data, _Slice_output_0_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _Slice_output_0_ends_data[] = { 50 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_output_0_ends, AI_ARRAY_FORMAT_S16,
    _Slice_output_0_ends_data, _Slice_output_0_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Slice_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_convolutions_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Slice_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Slice_output_0_layer, 21,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &_Slice_output_0_chain,
  NULL, &_Add_output_0_layer, AI_STATIC, 
  .axes = &_Slice_output_0_axes, 
  .starts = &_Slice_output_0_starts, 
  .ends = &_Slice_output_0_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _convolutions_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_convolutions_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_convolutions_0_Conv_output_0_weights, &_convolutions_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_convolutions_0_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _convolutions_0_Conv_output_0_layer, 16,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_convolutions_0_Conv_output_0_chain,
  NULL, &_Slice_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 2, 0, 2, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_Transpose_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_2_Relu_output_0_layer, 15,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_spatial_spatial_2_Relu_output_0_chain,
  NULL, &_convolutions_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_1_Transpose_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_Transpose_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_1_Transpose_1_output_0_layer, 14,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_spatial_spatial_1_Transpose_1_output_0_chain,
  NULL, &_spatial_spatial_2_Relu_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Add_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_spatial_spatial_1_normalization_Mul_output_0_output, &spatial_1_normalization_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Add_1_output_0_layer, 13,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_spatial_spatial_1_normalization_Add_1_output_0_chain,
  NULL, &_spatial_spatial_1_Transpose_1_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_spatial_spatial_1_normalization_Div_output_0_output, &spatial_1_normalization_weight_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Mul_output_0_layer, 12,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_spatial_spatial_1_normalization_Mul_output_0_chain,
  NULL, &_spatial_spatial_1_normalization_Add_1_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_spatial_spatial_1_normalization_Sub_output_0_output, &_spatial_spatial_1_normalization_Sqrt_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Div_output_0_layer, 11,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_spatial_spatial_1_normalization_Div_output_0_chain,
  NULL, &_spatial_spatial_1_normalization_Mul_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Sqrt_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_Sqrt_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Sqrt_output_0_layer, 10,
  NL_TYPE, 0x0, NULL,
  nl, forward_sqrt,
  &_spatial_spatial_1_normalization_Sqrt_output_0_chain,
  NULL, &_spatial_spatial_1_normalization_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_ReduceMean_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_spatial_spatial_1_normalization_ReduceMean_output_0_Mul_scale, &_spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_layer, 9,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_chain,
  NULL, &_spatial_spatial_1_normalization_Sqrt_output_0_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float _spatial_spatial_1_normalization_ReduceMean_1_output_0_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    _spatial_spatial_1_normalization_ReduceMean_1_output_0_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    _spatial_spatial_1_normalization_ReduceMean_1_output_0_neutral_value_data, _spatial_spatial_1_normalization_ReduceMean_1_output_0_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_Pow_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_ReduceMean_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_1_output_0_layer, 9,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &_spatial_spatial_1_normalization_ReduceMean_1_output_0_chain,
  NULL, &_spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &_spatial_spatial_1_normalization_ReduceMean_1_output_0_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Pow_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_spatial_spatial_1_normalization_Sub_output_0_output, &_spatial_spatial_1_normalization_Constant_output_0_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_Pow_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Pow_output_0_layer, 6,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_spatial_spatial_1_normalization_Pow_output_0_chain,
  NULL, &_spatial_spatial_1_normalization_ReduceMean_1_output_0_layer, AI_STATIC, 
  .operation = ai_pow, 
  .buffer_operation = ai_pow_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Sub_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_spatial_spatial_1_Transpose_output_0_output, &_spatial_spatial_1_normalization_ReduceMean_output_0_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_1_normalization_Sub_output_0_layer, 4,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_spatial_spatial_1_normalization_Sub_output_0_chain,
  NULL, &_spatial_spatial_1_normalization_Pow_output_0_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_ReduceMean_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_ReduceMean_output_0_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_spatial_spatial_1_normalization_ReduceMean_output_0_Mul_scale, &_spatial_spatial_1_normalization_ReduceMean_output_0_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_layer, 3,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_spatial_spatial_1_normalization_ReduceMean_output_0_Mul_chain,
  NULL, &_spatial_spatial_1_normalization_Sub_output_0_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float _spatial_spatial_1_normalization_ReduceMean_output_0_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    _spatial_spatial_1_normalization_ReduceMean_output_0_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    _spatial_spatial_1_normalization_ReduceMean_output_0_neutral_value_data, _spatial_spatial_1_normalization_ReduceMean_output_0_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_normalization_ReduceMean_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_1_normalization_ReduceMean_output_0_layer, 3,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &_spatial_spatial_1_normalization_ReduceMean_output_0_chain,
  NULL, &_spatial_spatial_1_normalization_ReduceMean_output_0_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &_spatial_spatial_1_normalization_ReduceMean_output_0_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_1_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_1_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_1_Transpose_output_0_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_spatial_spatial_1_Transpose_output_0_chain,
  NULL, &_spatial_spatial_1_normalization_ReduceMean_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _spatial_spatial_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &features_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_spatial_spatial_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_spatial_spatial_0_Conv_output_0_weights, &_spatial_spatial_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_spatial_spatial_0_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _spatial_spatial_0_Conv_output_0_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_spatial_spatial_0_Conv_output_0_chain,
  NULL, &_spatial_spatial_1_Transpose_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  features_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &features_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &features_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  features_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &features_Transpose_chain,
  NULL, &_spatial_spatial_0_Conv_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 248156, 1, 1),
    248156, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 76800, 1, 1),
    76800, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_INDY_ENCODER_IN_NUM, &features_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_INDY_ENCODER_OUT_NUM, &_activation_3_Relu_output_0_output),
  &features_Transpose_layer, 0x7f57e424, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 248156, 1, 1),
      248156, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 76800, 1, 1),
      76800, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_INDY_ENCODER_IN_NUM, &features_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_INDY_ENCODER_OUT_NUM, &_activation_3_Relu_output_0_output),
  &features_Transpose_layer, 0x7f57e424, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool indy_encoder_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_indy_encoder_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    features_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    features_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    features_Transpose_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 38400);
    features_Transpose_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 38400);
    _spatial_spatial_0_Conv_output_0_scratch0_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _spatial_spatial_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _spatial_spatial_0_Conv_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 768);
    _spatial_spatial_0_Conv_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 768);
    _spatial_spatial_1_Transpose_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 13568);
    _spatial_spatial_1_Transpose_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 13568);
    _spatial_spatial_1_normalization_ReduceMean_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _spatial_spatial_1_normalization_ReduceMean_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 200);
    _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 200);
    _spatial_spatial_1_normalization_Sub_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 400);
    _spatial_spatial_1_normalization_Sub_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 400);
    _spatial_spatial_1_normalization_Pow_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 13200);
    _spatial_spatial_1_normalization_Pow_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 13200);
    _spatial_spatial_1_normalization_ReduceMean_1_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _spatial_spatial_1_normalization_ReduceMean_1_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 200);
    _spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 200);
    _spatial_spatial_1_normalization_Sqrt_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _spatial_spatial_1_normalization_Sqrt_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _spatial_spatial_1_normalization_Div_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 13200);
    _spatial_spatial_1_normalization_Div_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 13200);
    _spatial_spatial_1_normalization_Mul_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _spatial_spatial_1_normalization_Mul_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _spatial_spatial_1_normalization_Add_1_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _spatial_spatial_1_normalization_Add_1_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _spatial_spatial_1_Transpose_1_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _spatial_spatial_1_Transpose_1_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _spatial_spatial_2_Relu_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _spatial_spatial_2_Relu_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _convolutions_0_Conv_output_0_scratch0_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _convolutions_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _convolutions_0_Conv_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 25600);
    _convolutions_0_Conv_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 25600);
    _Slice_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _Slice_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _Add_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 25600);
    _Add_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 25600);
    _activation_Relu_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _activation_Relu_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _convolutions_1_Conv_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _convolutions_1_Conv_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _Slice_1_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 26624);
    _Slice_1_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 26624);
    _Add_1_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _Add_1_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _activation_1_Relu_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _activation_1_Relu_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _convolutions_2_Conv_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _convolutions_2_Conv_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _Slice_2_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 27648);
    _Slice_2_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 27648);
    _Add_2_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _Add_2_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _activation_2_Relu_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _activation_2_Relu_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _convolutions_3_Conv_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _convolutions_3_Conv_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _Slice_3_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 29696);
    _Slice_3_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 29696);
    _Add_3_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _Add_3_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 12800);
    _activation_3_Relu_output_0_output_array.data = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    _activation_3_Relu_output_0_output_array.data_start = AI_PTR(g_indy_encoder_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool indy_encoder_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_indy_encoder_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _spatial_spatial_1_normalization_Constant_output_0_3D_array.format |= AI_FMT_FLAG_CONST;
    _spatial_spatial_1_normalization_Constant_output_0_3D_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 0);
    _spatial_spatial_1_normalization_Constant_output_0_3D_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 0);
    spatial_1_normalization_weight_3D_array.format |= AI_FMT_FLAG_CONST;
    spatial_1_normalization_weight_3D_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 4);
    spatial_1_normalization_weight_3D_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 4);
    spatial_1_normalization_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    spatial_1_normalization_bias_3D_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 260);
    spatial_1_normalization_bias_3D_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 260);
    _spatial_spatial_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _spatial_spatial_0_Conv_output_0_weights_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 516);
    _spatial_spatial_0_Conv_output_0_weights_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 516);
    _spatial_spatial_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _spatial_spatial_0_Conv_output_0_bias_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 49668);
    _spatial_spatial_0_Conv_output_0_bias_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 49668);
    _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_scale_array.format |= AI_FMT_FLAG_CONST;
    _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_scale_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 49924);
    _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_scale_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 49924);
    _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_bias_array.format |= AI_FMT_FLAG_CONST;
    _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_bias_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 50124);
    _spatial_spatial_1_normalization_ReduceMean_output_0_Mul_bias_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 50124);
    _spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_bias_array.format |= AI_FMT_FLAG_CONST;
    _spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_bias_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 50324);
    _spatial_spatial_1_normalization_ReduceMean_1_output_0_Mul_bias_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 50324);
    _convolutions_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _convolutions_0_Conv_output_0_weights_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 50524);
    _convolutions_0_Conv_output_0_weights_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 50524);
    _convolutions_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _convolutions_0_Conv_output_0_bias_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 99676);
    _convolutions_0_Conv_output_0_bias_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 99676);
    _convolutions_1_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _convolutions_1_Conv_output_0_weights_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 99932);
    _convolutions_1_Conv_output_0_weights_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 99932);
    _convolutions_1_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _convolutions_1_Conv_output_0_bias_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 149084);
    _convolutions_1_Conv_output_0_bias_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 149084);
    _convolutions_2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _convolutions_2_Conv_output_0_weights_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 149340);
    _convolutions_2_Conv_output_0_weights_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 149340);
    _convolutions_2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _convolutions_2_Conv_output_0_bias_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 198492);
    _convolutions_2_Conv_output_0_bias_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 198492);
    _convolutions_3_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _convolutions_3_Conv_output_0_weights_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 198748);
    _convolutions_3_Conv_output_0_weights_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 198748);
    _convolutions_3_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _convolutions_3_Conv_output_0_bias_array.data = AI_PTR(g_indy_encoder_weights_map[0] + 247900);
    _convolutions_3_Conv_output_0_bias_array.data_start = AI_PTR(g_indy_encoder_weights_map[0] + 247900);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_indy_encoder_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_INDY_ENCODER_MODEL_NAME,
      .model_signature   = AI_INDY_ENCODER_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 3542560,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x7f57e424,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_indy_encoder_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_INDY_ENCODER_MODEL_NAME,
      .model_signature   = AI_INDY_ENCODER_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 3542560,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x7f57e424,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_indy_encoder_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_indy_encoder_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_indy_encoder_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_indy_encoder_create(network, AI_INDY_ENCODER_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_indy_encoder_data_params_get(&params) != true) {
    err = ai_indy_encoder_get_error(*network);
    return err;
  }
#if defined(AI_INDY_ENCODER_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_INDY_ENCODER_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_indy_encoder_init(*network, &params) != true) {
    err = ai_indy_encoder_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_indy_encoder_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_indy_encoder_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_indy_encoder_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_indy_encoder_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= indy_encoder_configure_weights(net_ctx, params);
  ok &= indy_encoder_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_indy_encoder_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_indy_encoder_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_INDY_ENCODER_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

