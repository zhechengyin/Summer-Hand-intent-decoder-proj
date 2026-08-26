/**
  ******************************************************************************
  * @file    indy_phase6_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-08-19T20:11:15-0400
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

#ifndef INDY_PHASE6_DATA_PARAMS_H
#define INDY_PHASE6_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_INDY_PHASE6_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_indy_phase6_data_weights_params[1]))
*/

#define AI_INDY_PHASE6_DATA_CONFIG               (NULL)


#define AI_INDY_PHASE6_DATA_ACTIVATIONS_SIZES \
  { 76800, }
#define AI_INDY_PHASE6_DATA_ACTIVATIONS_SIZE     (76800)
#define AI_INDY_PHASE6_DATA_ACTIVATIONS_COUNT    (1)
#define AI_INDY_PHASE6_DATA_ACTIVATION_1_SIZE    (76800)



#define AI_INDY_PHASE6_DATA_WEIGHTS_SIZES \
  { 348772, }
#define AI_INDY_PHASE6_DATA_WEIGHTS_SIZE         (348772)
#define AI_INDY_PHASE6_DATA_WEIGHTS_COUNT        (1)
#define AI_INDY_PHASE6_DATA_WEIGHT_1_SIZE        (348772)



#define AI_INDY_PHASE6_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_indy_phase6_activations_table[1])

extern ai_handle g_indy_phase6_activations_table[1 + 2];



#define AI_INDY_PHASE6_DATA_WEIGHTS_TABLE_GET() \
  (&g_indy_phase6_weights_table[1])

extern ai_handle g_indy_phase6_weights_table[1 + 2];


#endif    /* INDY_PHASE6_DATA_PARAMS_H */
