#pragma once
#include <Eigen/Dense>
#include <NvInferRuntime.h>
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <vector>
static constexpr int TILE_W = 32;
static constexpr int TILE_H = 32;
__global__ void letterbox_kernel_shared(
    const uchar* __restrict__ input_bgr,
    int in_w,
    int in_h,
    float* __restrict__ output_nchw,
    int out_w,
    int out_h,
    float scale,
    int pad_t,
    int pad_l,
    float norm,
    bool swap_rb
);
__global__ void letterbox_kernel_pitched(
    const unsigned char* __restrict__ d_input_bgr,
    size_t pitch,
    int src_w,
    int src_h,
    float* __restrict__ d_nchw,
    int OUT_W,
    int OUT_H,
    float scale,
    int pad_t,
    int pad_l,
    float norm,
    bool swap_rb
);
