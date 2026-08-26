---  
title: ONNX toolbox support  
---  
# Overview   
This document lists the layers (or operators) which can be imported and converted. Supported operators allow to address a large range of classical topologies targeting a Mobile or IoT resource-constrained runtime environment: SqueezeNet, MobileNet V1 or V2, Inception, SSD MobileNet v1,..  
   
> Purpose of this document is to list the operators and their associated constraints or limitations, please refer to the original documentation for details on a given layer.  
   
[ONNX](<https://onnx.ai/>) is an open format built to represent machine learning models. A part of a subset of operators from Opset 7, 8, 9, 10 up to 20 of ONNX 1.15 is supported.  
   
*This file was automatically generated.*  
   
- ST Edge AI Core version : 2.2    
- 132 operators found     

## Summary table  
   
Following table contains the list of the operators that can be imported, if the constraints or limitations are met.   
   
- supported optional fused activation (or non-linearity): gelu, linear, relu, quantized_relu, relu_n1_to_1, relu_0_to_1, leaky_relu, relu6, elu, selu, sigmoid, hard_sigmoid, hard_swish, exponential, tanh, softmax, softplus, softsign, abs, acos, acosh, asin, asinh, atan, atanh, ceil, clip, cos, cosh, erf, flexerf, exp, floor, identity, log, logistic, neg, logical_not, prelu, probit, reciprocal, relu_generic, relu_thresholded, round, sign, sin, sinh, softmax_zero, sqrt, swish, tan  
   
| operator  | data types  | constraints/limitations  |  
| :-------- | :---------- | :----------------------- |  
| [Abs](#abs) | float32 | [common](#common-constraints) |   
| [Acos](#acos) | float32 | [common](#common-constraints) |   
| [Acosh](#acosh) | float32 | [common](#common-constraints) |   
| [Add](#add) | float32 | [common](#common-constraints) |   
| [And](#and) | bool | [common](#common-constraints) |   
| [ArgMax](#argmax) | float32, int32 | [common](#common-constraints) |   
| [ArgMin](#argmin) | float32, int32 | [common](#common-constraints) |   
| [ArrayFeatureExtractor](#arrayfeatureextractor) | float32 | [common](#common-constraints), [specific](#arrayfeatureextractor) |   
| [Asin](#asin) | float32 | [common](#common-constraints) |   
| [Asinh](#asinh) | float32 | [common](#common-constraints) |   
| [Atan](#atan) | float32 | [common](#common-constraints) |   
| [Atanh](#atanh) | float32 | [common](#common-constraints) |   
| [AveragePool](#averagepool) | float32 | [common](#common-constraints), [specific](#averagepool) |   
| [BatchNormalization](#batchnormalization) | float32 | [common](#common-constraints), [specific](#batchnormalization) |   
| [BitShift](#bitshift) | float32 | [common](#common-constraints) |   
| [Cast](#cast) | bool, int8, uint8, float32 | [common](#common-constraints), [specific](#cast) |   
| [Ceil](#ceil) | float32 | [common](#common-constraints) |   
| [Clip](#clip) | float32 | [common](#common-constraints) |   
| [Concat](#concat) | float32 | [common](#common-constraints), [specific](#concat) |   
| [Constant](#constant) | float32 | [common](#common-constraints) |   
| [ConstantOfShape](#constantofshape) | float32 | [common](#common-constraints) |   
| [Conv](#conv) | float32 | [common](#common-constraints), [specific](#conv) |   
| [ConvTranspose](#convtranspose) | float32 | [common](#common-constraints), [specific](#convtranspose) |   
| [Cos](#cos) | float32 | [common](#common-constraints) |   
| [Cosh](#cosh) | float32 | [common](#common-constraints) |   
| [DepthToSpace](#depthtospace) | float32, int8, uint8 | [common](#common-constraints), [specific](#depthtospace) |   
| [DequantizeLinear](#dequantizelinear) | float32, int8, uint8 | [common](#common-constraints), [specific](#dequantizelinear) |   
| [Div](#div) | float32 | [common](#common-constraints) |   
| [Dropout](#dropout) | float32 | [common](#common-constraints), [specific](#dropout) |   
| [Einsum](#einsum) | float32 | [common](#common-constraints), [specific](#einsum) |   
| [Elu](#elu) | float32 | [common](#common-constraints) |   
| [Equal](#equal) | float32, bool | [common](#common-constraints) |   
| [Erf](#erf) | float32 | [common](#common-constraints) |   
| [Exp](#exp) | float32 | [common](#common-constraints) |   
| [Expand](#expand) | float32, int8, uint8 | [common](#common-constraints), [specific](#expand) |   
| [Flatten](#flatten) | float32 | [common](#common-constraints), [specific](#flatten) |   
| [Floor](#floor) | float32 | [common](#common-constraints) |   
| [Gather](#gather) | float32 | [common](#common-constraints), [specific](#gather) |   
| [GatherElements](#gatherelements) | float32 | [common](#common-constraints), [specific](#gatherelements) |   
| [GatherND](#gathernd) | float32 | [common](#common-constraints), [specific](#gathernd) |   
| [Gelu](#gelu) | float32 | [common](#common-constraints) |   
| [Gemm](#gemm) | float32 | [common](#common-constraints) |   
| [GlobalAveragePool](#globalaveragepool) | float32 | [common](#common-constraints) |   
| [GlobalMaxPool](#globalmaxpool) | float32 | [common](#common-constraints) |   
| [Greater](#greater) | float32, bool | [common](#common-constraints) |   
| [GreaterOrEqual](#greaterorequal) | float32, bool | [common](#common-constraints) |   
| [GRU](#gru) | float32 | [common](#common-constraints), [specific](#gru) |   
| [Hardmax](#hardmax) | float32 | [common](#common-constraints), [specific](#hardmax) |   
| [HardSigmoid](#hardsigmoid) | float32 | [common](#common-constraints), [specific](#hardsigmoid) |   
| [HardSwish](#hardswish) | float32 | [common](#common-constraints) |   
| [Identity](#identity) | float32 | [common](#common-constraints) |   
| [InstanceNormalization](#instancenormalization) | float32 | [common](#common-constraints) |   
| [LabelEncoder](#labelencoder) | float32 | [common](#common-constraints), [specific](#labelencoder) |   
| [LeakyRelu](#leakyrelu) | float32 | [common](#common-constraints) |   
| [Less](#less) | float32, bool | [common](#common-constraints) |   
| [LessOrEqual](#lessorequal) | float32, bool | [common](#common-constraints) |   
| [LinearClassifier](#linearclassifier) | float32 | [common](#common-constraints), [specific](#linearclassifier) |   
| [Log](#log) | float32 | [common](#common-constraints) |   
| [LogSoftmax](#logsoftmax) | float32 | [common](#common-constraints), [specific](#logsoftmax) |   
| [LpNormalization](#lpnormalization) | float32 | [common](#common-constraints), [specific](#lpnormalization) |   
| [LRN](#lrn) | float32 | [common](#common-constraints) |   
| [LSTM](#lstm) | float32 | [common](#common-constraints), [specific](#lstm) |   
| [MatMul](#matmul) | float32 | [common](#common-constraints), [specific](#matmul) |   
| [Max](#max) | float32 | [common](#common-constraints) |   
| [MaxPool](#maxpool) | float32 | [common](#common-constraints), [specific](#maxpool) |   
| [Mean](#mean) | float32 | [common](#common-constraints) |   
| [Min](#min) | float32 | [common](#common-constraints) |   
| [Mod](#mod) | float32 | [common](#common-constraints) |   
| [Mul](#mul) | float32 | [common](#common-constraints) |   
| [Neg](#neg) | float32 | [common](#common-constraints) |   
| [Normalizer](#normalizer) | float32 | [common](#common-constraints), [specific](#normalizer) |   
| [Not](#not) | float32 | [common](#common-constraints) |   
| [Or](#or) | bool | [common](#common-constraints) |   
| [Pad](#pad) | float32 | [common](#common-constraints), [specific](#pad) |   
| [Pow](#pow) | float32 | [common](#common-constraints) |   
| [PRelu](#prelu) | float32 | [common](#common-constraints), [specific](#prelu) |   
| [QLinearAdd](#qlinearadd) | uint8, int8 | [common](#common-constraints) |   
| [QLinearAveragePool](#qlinearaveragepool) | uint8, int8 | [common](#common-constraints), [specific](#qlinearaveragepool) |   
| [QLinearConcat](#qlinearconcat) | uint8, int8 | [common](#common-constraints), [specific](#qlinearconcat) |   
| [QLinearConv](#qlinearconv) | int8, uint8 | [common](#common-constraints), [specific](#qlinearconv) |   
| [QLinearGlobalAveragePool](#qlinearglobalaveragepool) | uint8, int8 | [common](#common-constraints) |   
| [QLinearMatMul](#qlinearmatmul) | int8, uint8 | [common](#common-constraints) |   
| [QLinearMul](#qlinearmul) | uint8, int8 | [common](#common-constraints) |   
| [QuantizeLinear](#quantizelinear) | float32, int8, uint8 | [common](#common-constraints), [specific](#quantizelinear) |   
| [Range](#range) | float32 | [common](#common-constraints) |   
| [Reciprocal](#reciprocal) | float32 | [common](#common-constraints) |   
| [ReduceL1](#reducel1) | float32 | [common](#common-constraints) |   
| [ReduceL2](#reducel2) | float32 | [common](#common-constraints) |   
| [ReduceLogSumExp](#reducelogsumexp) | float32 | [common](#common-constraints) |   
| [ReduceMax](#reducemax) | float32 | [common](#common-constraints) |   
| [ReduceMean](#reducemean) | float32 | [common](#common-constraints) |   
| [ReduceMin](#reducemin) | float32 | [common](#common-constraints) |   
| [ReduceProd](#reduceprod) | float32 | [common](#common-constraints) |   
| [ReduceSum](#reducesum) | float32 | [common](#common-constraints) |   
| [ReduceSumSquare](#reducesumsquare) | float32 | [common](#common-constraints) |   
| [Relu](#relu) | float32 | [common](#common-constraints) |   
| [Reshape](#reshape) | float32 | [common](#common-constraints), [specific](#reshape) |   
| [Resize](#resize) | float32 | [common](#common-constraints), [specific](#resize) |   
| [Round](#round) | float32 | [common](#common-constraints) |   
| [Scaler](#scaler) | float32, float | [common](#common-constraints), [specific](#scaler) |   
| [ScatterND](#scatternd) | float32 | [common](#common-constraints), [specific](#scatternd) |   
| [Selu](#selu) | float32 | [common](#common-constraints) |   
| [Shape](#shape) | float32, int8, uint8, int32 | [common](#common-constraints), [specific](#shape) |   
| [Sigmoid](#sigmoid) | float32 | [common](#common-constraints) |   
| [Sign](#sign) | float32 | [common](#common-constraints) |   
| [Sin](#sin) | float32 | [common](#common-constraints) |   
| [Sinh](#sinh) | float32 | [common](#common-constraints) |   
| [Slice](#slice) | float32, int8, uint8 | [common](#common-constraints) |   
| [Softmax](#softmax) | float32 | [common](#common-constraints), [specific](#softmax) |   
| [Softplus](#softplus) | float32 | [common](#common-constraints) |   
| [Softsign](#softsign) | float32 | [common](#common-constraints) |   
| [SpaceToDepth](#spacetodepth) | float32, int8, uint8 | [common](#common-constraints) |   
| [Split](#split) | float32, int8, uint8 | [common](#common-constraints), [specific](#split) |   
| [Sqrt](#sqrt) | float32 | [common](#common-constraints) |   
| [Squeeze](#squeeze) | float32 | [common](#common-constraints) |   
| [Sub](#sub) | float32 | [common](#common-constraints) |   
| [Sum](#sum) | float32 | [common](#common-constraints) |   
| [SVMClassifier](#svmclassifier) | float32 | [common](#common-constraints), [specific](#svmclassifier) |   
| [SVMRegressor](#svmregressor) | float32 | [common](#common-constraints), [specific](#svmregressor) |   
| [Tan](#tan) | float32 | [common](#common-constraints) |   
| [Tanh](#tanh) | float32 | [common](#common-constraints) |   
| [ThresholdedRelu](#thresholdedrelu) | float32 | [common](#common-constraints) |   
| [Tile](#tile) | float32, int8, uint8 | [common](#common-constraints), [specific](#tile) |   
| [TopK](#topk) | float32, int8, uint8 | [common](#common-constraints) |   
| [Transpose](#transpose) | float32, int8, uint8 | [common](#common-constraints), [specific](#transpose) |   
| [TreeEnsembleClassifier](#treeensembleclassifier) | float32 | [common](#common-constraints), [specific](#treeensembleclassifier) |   
| [TreeEnsembleRegressor](#treeensembleregressor) | float32 | [common](#common-constraints), [specific](#treeensembleregressor) |   
| [Unsqueeze](#unsqueeze) | float32 | [common](#common-constraints) |   
| [Upsample](#upsample) | float32 | [common](#common-constraints), [specific](#upsample) |   
| [Where](#where) | float32, int8, uint8, int16, uint16, int32, uint32, bool | [common](#common-constraints) |   
| [Xor](#xor) | bool | [common](#common-constraints) |   
| [ZipMap](#zipmap) | float32 | [common](#common-constraints), [specific](#zipmap) |   

## Common constraints  
   
- input and output tensors must be **not dynamic**.   
   - variable-length batch dimension (i.e. `(None,)`) is considered as equal to 1  
   - must not be greater than 6D  
   - dimension must be in the range `[0, 65536[`  
   - batch dimension is partially supported in the axis/axes parameters  
- data type for the weights/activations tensors must be:  
   - float32, int8, uint8  
   - only int32 for the bias tensor is considered  
   - for some operators, bool and binary types are also supported  
- 1D operator is mapped on the respective 2D operator by adding a singleton dimension on the input:  (12,3) -> (12, 1, 3)  

# Operators  
   

## Abs  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Acos  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Acosh  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Add  
   
Performs element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: float32  

## And  
   
Performs boolean element-wise operation  
   
- category: eltwise operator  
- input data types: bool  
- output data types: bool  

## ArgMax  
   
Computes the indices of the max elements of the input tensor's element along the provided axis.  
   
- category: generic layer  
- input data types: float32  
- output data types: int32  

## ArgMin  
   
Computes the indices of the min elements of the input tensor's element along the provided axis.  
   
- category: generic layer  
- input data types: float32  
- output data types: int32  

## ArrayFeatureExtractor  
   
Select elements of the input tensor besed of the indices passed  
   
- category: onnx.ml  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- used to support for example the skit-learn algo:  
  - Nu-Support Vector Classification (sklearn.svm.NuSVC)  
  - C-Support Vector Classification. (sklearn.svm.SVC)  
  - A random forest classifier (sklearn.ensemble.RandomForestClassifier)  
  - ...  

## Asin  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Asinh  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Atan  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Atanh  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## AveragePool  
   
Downsamples the input  
   
- category: pooling layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- arbitrary strides, provided that they are smaller than the input size  
- arbitrary pool sizes, provided that they are smaller than the input size  
- The attribute dilations different from the default value is not supported  

## BatchNormalization  
   
Performs the normalization of the input  
   
- category: normalization layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- Only one output (Y) is supported  

## BitShift  
   
Performs element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: float32  

## Cast  
   
Cast elements of the input tensor to the specified output tensor data  
   
- category: conversion layer  
- input data types: bool, int8, uint8, float32  
- output data types: bool, int8, uint8, float32  
   
Specific constraints/recommendations:  
   
- The attribute saturate different from the default value is not supported  

## Ceil  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Clip  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Concat  
   
Performs concatenation of a list of inputs  
   
- category: merge operator  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- concatenating on the batch dimension is not supported  

## Constant  
   
- input data types: float32  
- output data types: float32  

## ConstantOfShape  
   
Generates a tensor with given value and shape  
   
- category: constant layer  
- input data types: float32  
- output data types: float32  

## Conv  
   
Performs convolution operation  
   
- category: convolutional layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- arbitrary strides, provided that they are smaller than the input size  
- arbitrary filter kernel sizes, provided that they are smaller than the input size  

## ConvTranspose  
   
Transposed convolutional layer  
   
- category: convolutional layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- arbitrary strides, provided that they are smaller than the input size  
- arbitrary filter kernel sizes, provided that they are smaller than the input size  

## Cos  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Cosh  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## DepthToSpace  
   
Permutes the dimensions of the input according to a given pattern  
   
- category: reshaping layer  
- input data types: float32, int8, uint8  
- output data types: float32, int8, uint8  
   
Specific constraints/recommendations:  
   
- transposing the batch dimension is not supported  

## DequantizeLinear  
   
Computes element-wise data conversion low precision to full precision, based on the scale/zeropoint parameters  
   
- category: conversion layer  
- input data types: float32, int8, uint8  
- output data types: float32, int8, uint8  
   
Specific constraints/recommendations:  
   
- The attribute block_size different from the default value is not supported  

## Div  
   
Performs element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: float32  

## Dropout  
   
Applies Dropout to the input  
   
- category: regularization layers  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- operator is dropped during the conversion  

## Einsum  
   
Einsum (Einstein Summation) operator  
   
- category: einsum operator  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- related ONNX operator: Einsum   
- requires 3 operators: 2 inputs and 1 output  
- only 4-D operators are supported  

## Elu  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Equal  
   
Performs logical element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: bool  

## Erf  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Exp  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Expand  
   
Constructs a tensor by tiling the input tensor  
   
- category: reshaping layer  
- input data types: float32, int8, uint8  
- output data types: float32, int8, uint8  
   
Specific constraints/recommendations:  
   
- tiling on batch-dimension is not supported  

## Flatten  
   
Flattens the non-batch input dimensions to a vector  
   
- category: Reshaping operation  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- Flatten on the batch dimension is not supported  
- operator is dropped during the conversion  

## Floor  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Gather  
   
Gathers values along a specified axis  
   
- category: activation function  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- Gather is not supported with indices dimensions > 2 (Batch is not considered), axis > 3and axis = 0, batch_dims attribute is not handled  

## GatherElements  
   
Gathers Elements  along a specified axis  
   
- category: activation function  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- GatherElements is not supported with indices dimensions > 2 (Batch is not considered), axis > 3and axis = 0, batch_dims attribute is not handled  

## GatherND  
   
Gathers slices from input tensor into an output tensor with shape specified by indices  
   
- category: activation function  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- Batch_dims attribute is not handled. 4D or more indices (BATCH is not included) are not implementedLast dimension of indices > 4 with 2D inputs tensor case is not handled  

## Gelu  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Gemm  
   
General Matrix multiplication: Compute Y = alpha*A*B + beta*C  
   
- category: core layer  
- input data types: float32  
- output data types: float32  

## GlobalAveragePool  
   
Downsamples the input  
   
- category: pooling layer  
- input data types: float32  
- output data types: float32  

## GlobalMaxPool  
   
Downsamples the input  
   
- category: pooling layer  
- input data types: float32  
- output data types: float32  

## Greater  
   
Performs logical element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: bool  

## GreaterOrEqual  
   
Performs logical element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: bool  

## GRU  
   
Gated Recurrent Unit  
   
- category: recurrent layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- stateless and stateful (batch=1 only) mode support  
- fused activation: gelu, linear, relu, quantized_relu, relu_n1_to_1, relu_0_to_1, leaky_relu, relu6, elu, selu, sigmoid, hard_sigmoid, hard_swish, exponential, tanh, softmax, softplus, softsign  
- fused recurrent activation: gelu, linear, relu, quantized_relu, relu_n1_to_1, relu_0_to_1, leaky_relu, relu6, elu, selu, sigmoid, hard_sigmoid, hard_swish, exponential, tanh, softmax, softplus, softsign  
- `return_state` not supported  

## Hardmax  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- The value 1 is supported as the default value of the axis attribute, not the value -1  

## HardSigmoid  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- It is supported only for 1D tensor and only on the channel dimension  

## HardSwish  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Identity  
   
This layer should be used as a placeholder when no operation is to be performed.The layer just returns its inputs argument as output.  
   
- category: Identity layer  
- input data types: float32  
- output data types: float32  

## InstanceNormalization  
   
Apply instance normalization  
   
- category: normalization function  
- input data types: float32  
- output data types: float32  

## LabelEncoder  
   
Maps each element in the input tensor to another value.  
   
- category: LabelEncoder  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- Warning: This operator is not supported for C code generation, it is only used during iForest generation. Support is not complete.  
- Attributes keys_floats, keys_strings, values_int64s, values_strings are not supported  

## LeakyRelu  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Less  
   
Performs logical element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: bool  

## LessOrEqual  
   
Performs logical element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: bool  

## LinearClassifier  
   
LinearClassifier layer  
   
- category: onnx.ml  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- used to support for example the skit-learn algo:  
  - Linear Support Vector Classification. (sklearn.svm.LinearSVC)  
  - Logistic Regression (aka logit, MaxEnt) classifier. (sklearn.linear_model.LogisticRegression)  
  - Classifier using Ridge regression. (sklearn.linear_model.RidgeClassifier)  
  - Linear Discriminant Analysis. (sklearn.discriminant_analysis.LinearDiscriminantAnalysis)  

## Log  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## LogSoftmax  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- It is supported only for 1D tensor and only on the channel dimension  
- The value 1 is supported as the default value of the axis attribute, not the value -1  

## LpNormalization  
   
Apply Lp-normalization along the provided axis  
   
- category: normalization function  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- order of the normalization (1 and 2 supported)  

## LRN  
   
Apply Local Response Normalization over local input regions  
   
- category: normalization function  
- input data types: float32  
- output data types: float32  

## LSTM  
   
Computes a multi-layer long short-term memory (LSTM) RNN to an input sequence (batch=1, timesteps, features)  
   
- category: recurrent layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- stateless mode support only  
- fused activation: sigmoid  
- fused recurrent activation: sigmoid  
- `return_state` not supported  
- Only 1 input is allowed for LSTM; the others should be constant and placed into the initializers  
- layout=1 attribute is not supported  

## MatMul  
   
General Matrix multiplication: Compute Y = alpha*A*B + beta*C  
   
- category: core layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- Only up to 3D matrix multiplication is supported  

## Max  
   
Computes the maximum (element-wise) a list of inputs  
   
- category: eltwise operator  
- input data types: float32  
- output data types: float32  

## MaxPool  
   
Downsamples the input  
   
- category: pooling layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- arbitrary strides, provided that they are smaller than the input size  
- arbitrary pool sizes, provided that they are smaller than the input size  
- The attribute dilations different from the default value is not supported  
- The attribute storage_order different from the default value is not supported  

## Mean  
   
Performs element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: float32  

## Min  
   
Computes the minimum (element-wise) a list of inputs  
   
- category: eltwise operator  
- input data types: float32  
- output data types: float32  

## Mod  
   
Performs element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: float32  

## Mul  
   
Performs element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: float32  

## Neg  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Normalizer  
   
Normalize the input.  
   
- category: onnx.ml  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- used to support for example the skit-learn algo:  
  - Logistic Regression (aka logit, MaxEnt) classifier. (sklearn.linear_model.LogisticRegression)  
  - Normalize samples individually to unit norm. (sklearn.preprocessing.Normalizer)  
- Note that the **implementation of the max norm** is aligned with sklearn implementation, not the onnx implementation. The difference is that sklearn is using absolute values to get a positive divisor when computing the norm. Onnx implementation does not use absolute values.  

## Not  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Or  
   
Performs boolean element-wise operation  
   
- category: eltwise operator  
- input data types: bool  
- output data types: bool  

## Pad  
   
Pads an input tensor  
   
- category: Reshaping layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- Axes are not supported for Pad operator  
- wrap value for mode attribute is not supported  

## Pow  
   
Performs element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: float32  

## PRelu  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- Only 1 input is allowed for PRelu; the slope should be constant and placed into the initializers  

## QLinearAdd  
   
Performs element-wise binary addition on 8 bit data types(with Numpy-style broadcasting support).  
   
- category: eltwise operator  
- input data types: uint8, int8  
- output data types: uint8, int8  

## QLinearAveragePool  
   
Downsamples an input tensor X and applies average pooling across the tensor according to kernel sizes, stride sizes, and pad lengths  
   
- category: pooling layer  
- input data types: uint8, int8  
- output data types: uint8, int8  
   
Specific constraints/recommendations:  
   
- arbitrary strides, provided that they are smaller than the input size  
- arbitrary pool sizes, provided that they are smaller than the input size  

## QLinearConcat  
   
Concatenate a list of tensors into a single tensor  
   
- category: merge operator  
- input data types: uint8, int8  
- output data types: uint8, int8  
   
Specific constraints/recommendations:  
   
- concatenating on the batch dimension is not supported  

## QLinearConv  
   
Consumes a quantized input tensor  
   
- category: convolutional layer  
- input data types: int8, uint8  
- output data types: int8, uint8  
   
Specific constraints/recommendations:  
   
- arbitrary strides, provided that they are smaller than the input size  
- arbitrary filter kernel sizes, provided that they are smaller than the input size  

## QLinearGlobalAveragePool  
   
Downsamples an input tensor X and applies Average pooling acrossthe values in the same channel  
   
- category: pooling layer  
- input data types: uint8, int8  
- output data types: uint8, int8  

## QLinearMatMul  
   
Matrix product consuming two quantized input tensors  
   
- category: core layer  
- input data types: int8, uint8  
- output data types: int8, uint8  

## QLinearMul  
   
Performs element-wise binary multiplication on 8 bit data types(with Numpy-style broadcasting support).  
   
- category: eltwise operator  
- input data types: uint8, int8  
- output data types: uint8, int8  

## QuantizeLinear  
   
Computes element-wise data conversion full precision to low precision, based on the scale/zeropoint parameters  
   
- category: conversion layer  
- input data types: float32, int8, uint8  
- output data types: float32, int8, uint8  
   
Specific constraints/recommendations:  
   
- The attribute block_size different from the default value is not supported  
- The attribute saturate different from the default value is not supported  
- The attribute output_dtype different from the default value is not supported  

## Range  
   
Generate a tensor containing a sequence of numbers that begin at start and extends by increments of delta up to limit (exclusive).  
   
- category: generic layer  
- input data types: float32  
- output data types: float32  

## Reciprocal  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## ReduceL1  
   
- category: reduction operation  
- input data types: float32  
- output data types: float32  

## ReduceL2  
   
- category: reduction operation  
- input data types: float32  
- output data types: float32  

## ReduceLogSumExp  
   
- category: reduction operation  
- input data types: float32  
- output data types: float32  

## ReduceMax  
   
Computes the Max of the input tensor's element along the provided axes  
   
- category: reduction operation  
- input data types: float32  
- output data types: float32  

## ReduceMean  
   
Computes the Mean of the input tensor's element along the provided axes  
   
- category: reduction operation  
- input data types: float32  
- output data types: float32  

## ReduceMin  
   
Computes the Min of the input tensor's element along the provided axes  
   
- category: reduction operation  
- input data types: float32  
- output data types: float32  

## ReduceProd  
   
Computes the Product of the input tensor's element along the provided axes  
   
- category: reduction operation  
- input data types: float32  
- output data types: float32  

## ReduceSum  
   
Computes the Sum of the input tensor's element along the provided axes  
   
- category: reduction operation  
- input data types: float32  
- output data types: float32  

## ReduceSumSquare  
   
Computes the Sum Square of the input tensor's element along the provided axes  
   
- category: reduction operation  
- input data types: float32  
- output data types: float32  

## Relu  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Reshape  
   
Reshapes a tensor  
   
- category: Reshaping operation  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- The attribute allowzero different from the default value is not supported  

## Resize  
   
Resize the input tensor: it calculates every value in the output tensor as a weighted average of neighborhood  
   
- category: resizing operation  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- Cubic resize is not supported  
- half_pixel_symmetric value for coordinate_tansformation_mode is not supported  
- The attribute antialias different from the default value is not supported  
- axes attribute is not supported  
- The attribute keep_aspect_ratio_policy different from the default value is not supported  

## Round  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Scaler  
   
Compute the Scale and Offset of input tensor  
   
- category: onnx.ml  
- input data types: float32  
- output data types: float  
   
Specific constraints/recommendations:  
   
- used to support for example the skit-learn algo:  
  - Gradient Boosting for classification. (sklearn.ensemble.GradientBoostingClassifier)  
  - Transform features by scaling each feature to a given range. (sklearn.preprocessing.MinMaxScaler)  
  - Standardize features by removing the mean and scaling to unit variance. (sklearn.preprocessing.StandardScaler)  
  - Scale features using statistics that are robust to outliers. (sklearn.preprocessing.RobustScaler)  
  - Scale each feature by its maximum absolute value. (sklearn.preprocessing.MaxAbsScaler)  

## ScatterND  
   
Scatters updates into a tensor of shape equal to shape attribute according indices (TFLite),The output of the ScatterND layer is produced by creating a copy of the input data, and then updating its values to values specified by updates at specific index positions specified by indices (ONNX)  
   
- category: activation function  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- 5D or more indices (BATCH is not included) are not implemented, 5D or more data (BATCH is not included) are not implemented  

## Selu  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Shape  
   
Returns a tensor containing the shape of the input tensor  
   
- category: Reshaping operation  
- input data types: float32, int8, uint8  
- output data types: int32  
   
Specific constraints/recommendations:  
   
- Shape layer is not supported as the only layer in a network;Shape layer only supported with know shapes in input; when batch size is undetermined, it is set to 1  
- end attribute is not supported  
- The attribute start different from the default value is not supported  

## Sigmoid  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Sign  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Sin  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Sinh  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Slice  
   
Crops the input  
   
- category: reshaping layer  
- input data types: float32, int8, uint8  
- output data types: float32, int8, uint8  

## Softmax  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- It is supported only for 1D tensor and only on the channel dimension  
- The value 1 is supported as the default value of the axis attribute, not the value -1  

## Softplus  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Softsign  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## SpaceToDepth  
   
Rearranges blocks of spatial data into depth  
   
- category: reshaping layer  
- input data types: float32, int8, uint8  
- output data types: float32, int8, uint8  

## Split  
   
Splits a tensor into a list of sub tensors  
   
- category: split operator  
- input data types: float32, int8, uint8  
- output data types: float32, int8, uint8  
   
Specific constraints/recommendations:  
   
- Only supported if the number of splits is equal to the size of the splitting dimension  

## Sqrt  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Squeeze  
   
Reshapes a tensor  
   
- category: Reshaping operation  
- input data types: float32  
- output data types: float32  

## Sub  
   
Performs element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: float32  

## Sum  
   
Performs element-wise operation  
   
- category: eltwise operator  
- input data types: float32  
- output data types: float32  

## SVMClassifier  
   
SVMClassifier layer  
   
- category: onnx.ml  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- used to support for example the skit-learn algo:  
  - C-Support Vector Classification. (sklearn.svm.SVC)  
  - Nu-Support Vector Classification (sklearn.svm.NuSVC)  

## SVMRegressor  
   
Support Vector Machine regression prediction and one-class SVM anomaly detection  
   
- category: onnx.ml  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- kernel_type argument supports one of the following 'LINEAR','POLY','RBF','SIGMOID'.  
- used to support for example the skit-learn algo:  
  - Unsupervised Outlier Detection. (sklearn.svm.OneClassSVM)  

## Tan  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Tanh  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## ThresholdedRelu  
   
Applies an activation function to the input tensor  
   
- category: activation layer  
- input data types: float32  
- output data types: float32  

## Tile  
   
Constructs a tensor by tiling the input tensor  
   
- category: reshaping layer  
- input data types: float32, int8, uint8  
- output data types: float32, int8, uint8  
   
Specific constraints/recommendations:  
   
- tiling on batch-dimension is not supported  

## TopK  
   
Retrieve the top-K largest or smallest elements along a specified axis  
   
- category: topK operator  
- input data types: float32, int8, uint8  
- output data types: float32, int8, uint8  

## Transpose  
   
Permutes the dimensions of the input according to a given pattern  
   
- category: reshaping layer  
- input data types: float32, int8, uint8  
- output data types: float32, int8, uint8  
   
Specific constraints/recommendations:  
   
- transposing the batch dimension is not supported  
- perm attribute must have a length equal to the rank of the input  

## TreeEnsembleClassifier  
   
Tree ensemble classifier. Returns the top class for each of N inputs.  
   
- category: onnx.ml  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- This layer supports compression levels.
	- none: disable compresion and optimization
	- lossless (default): enable optimization only
	- low, medium, high:  enable optimization and weight compression
  
- used to support for example the skit-learn algo:  
  - A random forest classifier (sklearn.ensemble.RandomForestClassifier)  
  - An extra-trees classifier (sklearn.ensemble.ExtraTreesClassifier)  
  - Gradient Boosting for classification. (sklearn.ensemble.GradientBoostingClassifier)  
  - Histogram-based Gradient Boosting Classification Tree. (sklearn.ensemble.HistGradientBoostingClassifier)  
  - A decision tree classifier. (sklearn.tree.DecisionTreeClassifier)  
- For detailed information, see [Machine Learning support  (ONNX-ML operators) \[ONNX-ML\]][STAI_CORE_ONNX_ML] article  

## TreeEnsembleRegressor  
   
Tree Ensemble regressor. Returns the regressed values for each input in N.  
   
- category: onnx.ml  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- Warning: This operator is not supported for C code generation, it is only used during iForest generation. Not all arguments are supported.  
- used to support for example the skit-learn algo:  
  - A Bagging classifier (sklearn.ensemble.BaggingClassifier)  
  - An AdaBoost classifier (sklearn.ensemble.AdaBoostClassifier)  
  - A Bagging regressor (sklearn.ensemble.BaggingRegressor)  
  - A random forest regressor (sklearn.ensemble.RandomForestRegressor)  
  - An extra-trees regressor (sklearn.ensemble.ExtraTreesRegressor)  
  - Gradient Boosting for regression (sklearn.ensemble.GradientBoostingRegressor)  
  - Histogram-based Gradient Boosting Regression Tree (sklearn.ensemble.HistGradientBoostingRegressor)  
  - Isolation Forest Algorithm (sklearn.ensemble.IsolationForest)  
- For detailed information, see [Machine Learning support  (ONNX-ML operators) \[ONNX-ML\]][STAI_CORE_ONNX_ML] article  

## Unsqueeze  
   
Reshapes a tensor  
   
- category: Reshaping operation  
- input data types: float32  
- output data types: float32  

## Upsample  
   
Resize the input tensor: it calculates every value in the output tensor as a weighted average of neighborhood  
   
- category: resizing operation  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- Cubic resize is not supported  
- half_pixel_symmetric value for coordinate_tansformation_mode is not supported  
- The attribute antialias different from the default value is not supported  
- axes attribute is not supported  
- The attribute keep_aspect_ratio_policy different from the default value is not supported  

## Where  
   
Where layer  
   
- category: generic layer  
- input data types: float32, int8, uint8, int16, uint16, int32, uint32, bool  
- output data types: float32, int8, uint8, int16, uint16, int32, uint32, bool  

## Xor  
   
Performs boolean element-wise operation  
   
- category: eltwise operator  
- input data types: bool  
- output data types: bool  

## ZipMap  
   
Creates a map from the input and the attributes.  
   
- category: onnx.ml  
- input data types: float32  
- output data types: float32  
   
Specific constraints/recommendations:  
   
- used to support for example the skit-learn algo:  
  - A decision tree classifier. (sklearn.tree.DecisionTreeClassifier)  
  - ...  
