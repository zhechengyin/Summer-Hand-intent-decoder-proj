/**
  ******************************************************************************
  * @file    indy_gru_head.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-08-19T20:30:44-0400
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


#include "indy_gru_head.h"
#include "indy_gru_head_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_indy_gru_head
 
#undef AI_INDY_GRU_HEAD_MODEL_SIGNATURE
#define AI_INDY_GRU_HEAD_MODEL_SIGNATURE     "0x6e6ec000cc3661698f9f99344f0cdb45"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-08-19T20:30:44-0400"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_INDY_GRU_HEAD_N_BATCHES
#define AI_INDY_GRU_HEAD_N_BATCHES         (1)

static ai_ptr g_indy_gru_head_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_indy_gru_head_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  encoded_sequence_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 3200, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  gru_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3200, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  velocity_norm_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 100, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  gru_kernel_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  gru_recurrent_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  gru_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  velocity_norm_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  velocity_norm_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  gru_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  encoded_sequence_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &encoded_sequence_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  gru_bias, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 192, 2, 1, 1), AI_STRIDE_INIT(4, 4, 768, 1536, 1536),
  1, &gru_bias_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  gru_kernel, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 64, 192, 1, 1), AI_STRIDE_INIT(4, 4, 256, 49152, 49152),
  1, &gru_kernel_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  gru_output0, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 50), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gru_output0_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  gru_recurrent, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 64, 192, 1, 1), AI_STRIDE_INIT(4, 4, 256, 49152, 49152),
  1, &gru_recurrent_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  gru_scratch0, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1536, 1536),
  1, &gru_scratch0_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  velocity_norm_bias, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &velocity_norm_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  velocity_norm_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 50), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &velocity_norm_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  velocity_norm_weights, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 64, 2, 1, 1), AI_STRIDE_INIT(4, 4, 256, 512, 512),
  1, &velocity_norm_weights_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  velocity_norm_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gru_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &velocity_norm_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &velocity_norm_weights, &velocity_norm_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  velocity_norm_layer, 2,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &velocity_norm_chain,
  NULL, &velocity_norm_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gru_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &encoded_sequence_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gru_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 6, &gru_kernel, &gru_recurrent, NULL, NULL, &gru_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gru_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gru_layer, 1,
  GRU_TYPE, 0x0, NULL,
  gru, forward_gru,
  &gru_chain,
  NULL, &velocity_norm_layer, AI_STATIC, 
  .n_units = 64, 
  .activation_nl = nl_func_tanh_array_f32, 
  .go_backwards = false, 
  .reverse_seq = false, 
  .return_state = false, 
  .reset_after = true, 
  .recurrent_nl = nl_func_sigmoid_array_f32, 
  .state = AI_HANDLE_PTR(NULL), 
  .init = AI_LAYER_FUNC(NULL), 
  .destroy = AI_LAYER_FUNC(NULL), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 100360, 1, 1),
    100360, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 27136, 1, 1),
    27136, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_INDY_GRU_HEAD_IN_NUM, &encoded_sequence_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_INDY_GRU_HEAD_OUT_NUM, &velocity_norm_output),
  &gru_layer, 0xb0a9ff0b, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 100360, 1, 1),
      100360, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 27136, 1, 1),
      27136, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_INDY_GRU_HEAD_IN_NUM, &encoded_sequence_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_INDY_GRU_HEAD_OUT_NUM, &velocity_norm_output),
  &gru_layer, 0xb0a9ff0b, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool indy_gru_head_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_indy_gru_head_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    encoded_sequence_output_array.data = AI_PTR(g_indy_gru_head_activations_map[0] + 0);
    encoded_sequence_output_array.data_start = AI_PTR(g_indy_gru_head_activations_map[0] + 0);
    gru_scratch0_array.data = AI_PTR(g_indy_gru_head_activations_map[0] + 12800);
    gru_scratch0_array.data_start = AI_PTR(g_indy_gru_head_activations_map[0] + 12800);
    gru_output0_array.data = AI_PTR(g_indy_gru_head_activations_map[0] + 14336);
    gru_output0_array.data_start = AI_PTR(g_indy_gru_head_activations_map[0] + 14336);
    velocity_norm_output_array.data = AI_PTR(g_indy_gru_head_activations_map[0] + 0);
    velocity_norm_output_array.data_start = AI_PTR(g_indy_gru_head_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool indy_gru_head_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_indy_gru_head_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    gru_kernel_array.format |= AI_FMT_FLAG_CONST;
    gru_kernel_array.data = AI_PTR(g_indy_gru_head_weights_map[0] + 0);
    gru_kernel_array.data_start = AI_PTR(g_indy_gru_head_weights_map[0] + 0);
    gru_recurrent_array.format |= AI_FMT_FLAG_CONST;
    gru_recurrent_array.data = AI_PTR(g_indy_gru_head_weights_map[0] + 49152);
    gru_recurrent_array.data_start = AI_PTR(g_indy_gru_head_weights_map[0] + 49152);
    gru_bias_array.format |= AI_FMT_FLAG_CONST;
    gru_bias_array.data = AI_PTR(g_indy_gru_head_weights_map[0] + 98304);
    gru_bias_array.data_start = AI_PTR(g_indy_gru_head_weights_map[0] + 98304);
    velocity_norm_weights_array.format |= AI_FMT_FLAG_CONST;
    velocity_norm_weights_array.data = AI_PTR(g_indy_gru_head_weights_map[0] + 99840);
    velocity_norm_weights_array.data_start = AI_PTR(g_indy_gru_head_weights_map[0] + 99840);
    velocity_norm_bias_array.format |= AI_FMT_FLAG_CONST;
    velocity_norm_bias_array.data = AI_PTR(g_indy_gru_head_weights_map[0] + 100352);
    velocity_norm_bias_array.data_start = AI_PTR(g_indy_gru_head_weights_map[0] + 100352);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_indy_gru_head_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_INDY_GRU_HEAD_MODEL_NAME,
      .model_signature   = AI_INDY_GRU_HEAD_MODEL_SIGNATURE,
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
      
      .n_macc            = 1241700,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xb0a9ff0b,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_indy_gru_head_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_INDY_GRU_HEAD_MODEL_NAME,
      .model_signature   = AI_INDY_GRU_HEAD_MODEL_SIGNATURE,
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
      
      .n_macc            = 1241700,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xb0a9ff0b,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_indy_gru_head_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_indy_gru_head_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_indy_gru_head_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_indy_gru_head_create(network, AI_INDY_GRU_HEAD_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_indy_gru_head_data_params_get(&params) != true) {
    err = ai_indy_gru_head_get_error(*network);
    return err;
  }
#if defined(AI_INDY_GRU_HEAD_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_INDY_GRU_HEAD_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_indy_gru_head_init(*network, &params) != true) {
    err = ai_indy_gru_head_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_indy_gru_head_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_indy_gru_head_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_indy_gru_head_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_indy_gru_head_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= indy_gru_head_configure_weights(net_ctx, params);
  ok &= indy_gru_head_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_indy_gru_head_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_indy_gru_head_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_INDY_GRU_HEAD_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

