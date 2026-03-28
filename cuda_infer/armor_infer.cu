// armor_cuda_infer.cu
#include "armor_infer.hpp"
#include "letter_box.hpp"
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <opencv2/core/hal/interface.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf( \
                stderr, \
                "CUDA error at %s:%d: %s\n", \
                __FILE__, \
                __LINE__, \
                cudaGetErrorString(err) \
            ); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
namespace armor_cuda_infer {
__global__ void nchw_float_to_hwc_uchar4(
    const float* __restrict__ src,
    uchar4* __restrict__ dst,
    int W,
    int H,
    float norm
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    const int idx = y * W + x;
    const int plane = W * H;

    float r = __ldg(src + idx + plane * 0);
    float g = __ldg(src + idx + plane * 1);
    float b = __ldg(src + idx + plane * 2);

    r = fminf(fmaxf(r / norm, 0.f), 255.f);
    g = fminf(fmaxf(g / norm, 0.f), 255.f);
    b = fminf(fmaxf(b / norm, 0.f), 255.f);

    dst[idx] = make_uchar4((unsigned char)b, (unsigned char)g, (unsigned char)r, 255);
}

cv::Mat CudaInfer::tensorToMat(float* d_nchw, int W, int H, float norm, cudaStream_t stream) const {
    static uchar4* d_hwc = nullptr;
    static size_t cap = 0;

    const size_t need = W * H * sizeof(uchar4);
    if (cap < need) {
        if (d_hwc)
            cudaFree(d_hwc);
        cudaMalloc(&d_hwc, need);
        cap = need;
    }

    const dim3 block(TILE_W, TILE_H);
    const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    nchw_float_to_hwc_uchar4<<<grid, block, 0, stream>>>(d_nchw, d_hwc, W, H, norm);

    cv::Mat img(H, W, CV_8UC4);

    cudaMemcpyAsync(img.data, d_hwc, need, cudaMemcpyDeviceToHost, stream);

    // cudaStreamSynchronize(stream);
    return img;
}

CudaInfer::CudaInfer() = default;
CudaInfer::~CudaInfer() {
    release();
}

void CudaInfer::init(int max_src_w, int max_src_h, int input_w, int input_h) {
    input_w_ = input_w;
    input_h_ = input_h;
    max_src_h_ = max_src_h;
    max_src_w_ = max_src_w;
    rellocMem();
}
void CudaInfer::rellocMem() {
    CUDA_CHECK(cudaMalloc(&d_input_bgr_, max_src_w_ * max_src_h_ * 3 * sizeof(unsigned char)));
    CUDA_CHECK(cudaMallocPitch(
        &d_input_bgr_pitched_,
        &input_pitch_bytes_,
        max_src_w_ * 3 * sizeof(unsigned char),
        max_src_h_
    ));
    CUDA_CHECK(cudaMalloc(&d_nchw_, input_w_ * input_h_ * 3 * sizeof(float)));
    printf("Relloc memory for CudaInfer\n");
}
void CudaInfer::getOutEnoughMem(int img_w, int img_h) {
    if (img_w > max_src_w_ || img_h > max_src_h_) {
        if (img_w > max_src_w_) {
            max_src_w_ = img_w;
        }
        if (img_h > max_src_h_) {
            max_src_h_ = img_h;
        }
        rellocMem();
    }
}

void CudaInfer::release() {
    if (d_input_bgr_)
        cudaFree(d_input_bgr_), d_input_bgr_ = nullptr;
    if (d_input_bgr_pitched_)
        cudaFree(d_input_bgr_pitched_), d_input_bgr_pitched_ = nullptr;
    if (d_nchw_)
        cudaFree(d_nchw_), d_nchw_ = nullptr;
}

float* CudaInfer::preprocess(
    const unsigned char* input_bgr_host,
    int img_w,
    int img_h,
    float norm,
    bool swap_rb,
    Eigen::Matrix3f& tf_matrix,
    cudaStream_t stream
) {
    if (!isInitialized()) {
        throw std::runtime_error("CudaInfer not initialized properly.");
    }

    if (!input_bgr_host || !d_input_bgr_ || !d_nchw_) {
        fprintf(stderr, "[Error] Null pointer in preprocess input\n");
        return nullptr;
    }
    getOutEnoughMem(img_w, img_h);
    float scale = fminf(input_w_ / (float)img_w, input_h_ / (float)img_h);
    int rw = round(img_w * scale), rh = round(img_h * scale);
    int pad_l = (input_w_ - rw) / 2, pad_t = (input_h_ - rh) / 2;

    tf_matrix << 1.f / scale, 0, -pad_l / scale, 0, 1.f / scale, -pad_t / scale, 0, 0, 1;

    size_t img_size = img_w * img_h * 3;
    CUDA_CHECK(
        cudaMemcpyAsync(d_input_bgr_, input_bgr_host, img_size, cudaMemcpyHostToDevice, stream)
    );

    dim3 threads(TILE_W, TILE_H);
    dim3 blocks((input_w_ + TILE_W - 1) / TILE_W, (input_h_ + TILE_H - 1) / TILE_H);

    letterbox_kernel_shared<<<blocks, threads, 0, stream>>>(
        d_input_bgr_,
        img_w,
        img_h,
        d_nchw_,
        input_w_,
        input_h_,
        scale,
        pad_t,
        pad_l,
        norm,
        swap_rb
    );

    CUDA_CHECK(cudaGetLastError());
    return d_nchw_;
}
float* CudaInfer::preprocess_gpu(
    const unsigned char* input_bgr_device,
    int img_w,
    int img_h,
    float norm,
    bool swap_rb,
    Eigen::Matrix3f& tf_matrix,
    cudaStream_t stream
) {
    if (!isInitialized()) {
        throw std::runtime_error("CudaInfer not initialized properly.");
    }

    if (!input_bgr_device || !d_nchw_) {
        fprintf(stderr, "[Error] Null pointer in preprocess input\n");
        return nullptr;
    }
    getOutEnoughMem(img_w, img_h);
    float scale = fminf(input_w_ / (float)img_w, input_h_ / (float)img_h);
    int rw = round(img_w * scale), rh = round(img_h * scale);
    int pad_l = (input_w_ - rw) / 2, pad_t = (input_h_ - rh) / 2;

    tf_matrix << 1.f / scale, 0, -pad_l / scale, 0, 1.f / scale, -pad_t / scale, 0, 0, 1;

    size_t img_size = img_w * img_h * 3;

    dim3 threads(TILE_W, TILE_H);
    dim3 blocks((input_w_ + TILE_W - 1) / TILE_W, (input_h_ + TILE_H - 1) / TILE_H);

    letterbox_kernel_shared<<<blocks, threads, 0, stream>>>(
        input_bgr_device,
        img_w,
        img_h,
        d_nchw_,
        input_w_,
        input_h_,
        scale,
        pad_t,
        pad_l,
        norm,
        swap_rb
    );

    CUDA_CHECK(cudaGetLastError());
    return d_nchw_;
}
float* CudaInfer::preprocess_pitched(
    const unsigned char* input_bgr_host,
    int img_w,
    int img_h,
    int host_step,
    float norm,
    bool swap_rb,
    Eigen::Matrix3f& tf_matrix,
    cudaStream_t stream
) {
    if (!isInitialized()) {
        throw std::runtime_error("CudaInfer not initialized properly.");
    }

    if (!input_bgr_host || !d_nchw_) {
        fprintf(stderr, "[Error] Null pointer in preprocess input\n");
        return nullptr;
    }
    getOutEnoughMem(img_w, img_h);
    float scale = fminf((float)input_w_ / img_w, (float)input_h_ / img_h);
    int rw = round(img_w * scale);
    int rh = round(img_h * scale);
    int pad_l = (input_w_ - rw) / 2;
    int pad_t = (input_h_ - rh) / 2;
    tf_matrix << 1.f / scale, 0, -pad_l / scale, 0, 1.f / scale, -pad_t / scale, 0, 0, 1;
    CUDA_CHECK(cudaMemcpy2DAsync(
        d_input_bgr_pitched_,
        input_pitch_bytes_,
        input_bgr_host,
        host_step,
        img_w * 3,
        img_h,
        cudaMemcpyHostToDevice,
        stream
    ));
    dim3 threads(TILE_W, TILE_H);
    dim3 blocks((input_w_ + TILE_W - 1) / TILE_W, (input_h_ + TILE_H - 1) / TILE_H);

    letterbox_kernel_pitched<<<blocks, threads, 0, stream>>>(
        d_input_bgr_pitched_,
        input_pitch_bytes_,
        img_w,
        img_h,
        d_nchw_,
        input_w_,
        input_h_,
        scale,
        pad_t,
        pad_l,
        norm,
        swap_rb
    );

    CUDA_CHECK(cudaGetLastError());
    return d_nchw_;
}
float* CudaInfer::preprocess_pitched_gpu(
    const unsigned char* input_bgr_device,
    int img_w,
    int img_h,
    int input_step,
    float norm,
    bool swap_rb,
    Eigen::Matrix3f& tf_matrix,
    cudaStream_t stream
) {
    if (!isInitialized()) {
        throw std::runtime_error("CudaInfer not initialized properly.");
    }

    if (!input_bgr_device || !d_nchw_) {
        fprintf(stderr, "[Error] Null pointer in preprocess_pitched_gpu\n");
        return nullptr;
    }
    getOutEnoughMem(img_w, img_h);
    float scale = fminf(static_cast<float>(input_w_) / img_w, static_cast<float>(input_h_) / img_h);

    int rw = static_cast<int>(roundf(img_w * scale));
    int rh = static_cast<int>(roundf(img_h * scale));

    int pad_l = (input_w_ - rw) / 2;
    int pad_t = (input_h_ - rh) / 2;

    tf_matrix << 1.f / scale, 0.f, -pad_l / scale, 0.f, 1.f / scale, -pad_t / scale, 0.f, 0.f, 1.f;

    dim3 threads(TILE_W, TILE_H);
    dim3 blocks((input_w_ + TILE_W - 1) / TILE_W, (input_h_ + TILE_H - 1) / TILE_H);

    letterbox_kernel_pitched<<<blocks, threads, 0, stream>>>(
        input_bgr_device,
        input_step,
        img_w,
        img_h,
        d_nchw_,
        input_w_,
        input_h_,
        scale,
        pad_t,
        pad_l,
        norm,
        swap_rb
    );

    CUDA_CHECK(cudaGetLastError());

    return d_nchw_;
}

} // namespace armor_cuda_infer