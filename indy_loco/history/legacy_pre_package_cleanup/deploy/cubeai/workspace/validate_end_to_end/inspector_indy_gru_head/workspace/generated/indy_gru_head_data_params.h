/**
  ******************************************************************************
  * @file    indy_gru_head_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-08-19T20:30:53-0400
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef INDY_GRU_HEAD_DATA_PARAMS_H
#define INDY_GRU_HEAD_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_INDY_GRU_HEAD_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_indy_gru_head_data_weights_params[1]))
*/

#define AI_INDY_GRU_HEAD_DATA_CONFIG               (NULL)


#define AI_INDY_GRU_HEAD_DATA_ACTIVATIONS_SIZES \
  { 27136, }
#define AI_INDY_GRU_HEAD_DATA_ACTIVATIONS_SIZE     (27136)
#define AI_INDY_GRU_HEAD_DATA_ACTIVATIONS_COUNT    (1)
#define AI_INDY_GRU_HEAD_DATA_ACTIVATION_1_SIZE    (27136)



#define AI_INDY_GRU_HEAD_DATA_WEIGHTS_SIZES \
  { 100360, }
#define AI_INDY_GRU_HEAD_DATA_WEIGHTS_SIZE         (100360)
#define AI_INDY_GRU_HEAD_DATA_WEIGHTS_COUNT        (1)
#define AI_INDY_GRU_HEAD_DATA_WEIGHT_1_SIZE        (100360)



#define AI_INDY_GRU_HEAD_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_indy_gru_head_activations_table[1])

extern ai_handle g_indy_gru_head_activations_table[1 + 2];



#define AI_INDY_GRU_HEAD_DATA_WEIGHTS_TABLE_GET() \
  (&g_indy_gru_head_weights_table[1])

extern ai_handle g_indy_gru_head_weights_table[1 + 2];


#endif    /* INDY_GRU_HEAD_DATA_PARAMS_H */
