/**
  ******************************************************************************
  * @file    indy_encoder_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-08-19T20:29:50-0400
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

#ifndef INDY_ENCODER_DATA_PARAMS_H
#define INDY_ENCODER_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_INDY_ENCODER_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_indy_encoder_data_weights_params[1]))
*/

#define AI_INDY_ENCODER_DATA_CONFIG               (NULL)


#define AI_INDY_ENCODER_DATA_ACTIVATIONS_SIZES \
  { 76800, }
#define AI_INDY_ENCODER_DATA_ACTIVATIONS_SIZE     (76800)
#define AI_INDY_ENCODER_DATA_ACTIVATIONS_COUNT    (1)
#define AI_INDY_ENCODER_DATA_ACTIVATION_1_SIZE    (76800)



#define AI_INDY_ENCODER_DATA_WEIGHTS_SIZES \
  { 248156, }
#define AI_INDY_ENCODER_DATA_WEIGHTS_SIZE         (248156)
#define AI_INDY_ENCODER_DATA_WEIGHTS_COUNT        (1)
#define AI_INDY_ENCODER_DATA_WEIGHT_1_SIZE        (248156)



#define AI_INDY_ENCODER_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_indy_encoder_activations_table[1])

extern ai_handle g_indy_encoder_activations_table[1 + 2];



#define AI_INDY_ENCODER_DATA_WEIGHTS_TABLE_GET() \
  (&g_indy_encoder_weights_table[1])

extern ai_handle g_indy_encoder_weights_table[1 + 2];


#endif    /* INDY_ENCODER_DATA_PARAMS_H */
