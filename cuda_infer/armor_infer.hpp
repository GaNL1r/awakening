// armor_cuda_infer.hpp
#pragma once

#include <Eigen/Dense>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <vector>

namespace armor_cuda_infer {

class CudaInfer {
public:
    CudaInfer();
    ~CudaInfer() noexcept;

    void init(int max_src_w, int max_src_h, int input_w, int input_h);
    void release();
    bool isInitialized() const {
        return d_input_bgr_ && d_nchw_ && d_input_bgr_pitched_;
    }
    void getOutEnoughMem(int img_w, int img_h);
    void rellocMem();

    float* preprocess(
        const unsigned char* input_bgr_host,
        int img_w,
        int img_h,
        float norm,
        bool swap_rb,
        Eigen::Matrix3f& tf_matrix,
        cudaStream_t stream
    );
    float* preprocess_pitched(
        const unsigned char* input_bgr_host,
        int img_w,
        int img_h,
        int host_step,
        float norm,
        bool swap_rb,
        Eigen::Matrix3f& tf_matrix,
        cudaStream_t stream
    );
    float* preprocess_gpu(
        const unsigned char* input_bgr_device,
        int img_w,
        int img_h,
        float norm,
        bool swap_rb,
        Eigen::Matrix3f& tf_matrix,
        cudaStream_t stream
    );
    float* preprocess_pitched_gpu(
        const unsigned char* input_bgr_device,
        int img_w,
        int img_h,
        int host_step,
        float norm,
        bool swap_rb,
        Eigen::Matrix3f& tf_matrix,
        cudaStream_t stream
    );
    cv::Mat tensorToMat(float* d_nchw, int W, int H, float norm, cudaStream_t stream) const;

private:
    CudaInfer(const CudaInfer&) = delete;
    CudaInfer& operator=(const CudaInfer&) = delete;
    unsigned char* d_input_bgr_ = nullptr;
    float* d_nchw_ = nullptr;
    unsigned char* d_input_bgr_pitched_ = nullptr;
    size_t input_pitch_bytes_ = 0;
    int input_w_;
    int input_h_;
    int max_src_w_, max_src_h_;
};
} // namespace armor_cuda_infer