#include "letter_box.hpp"
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
) {
    // global x/y
    int x = blockIdx.x * TILE_W + threadIdx.x;
    int y = blockIdx.y * TILE_H + threadIdx.y;
    if (x >= out_w || y >= out_h)
        return;

    // 共享内存 + halo
    __shared__ uchar4 smem[TILE_H + 1][TILE_W + 1];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_smem = (TILE_W + 1) * (TILE_H + 1);
    int threads_per_block = blockDim.x * blockDim.y;
    int iter = (total_smem + threads_per_block - 1) / threads_per_block;

    float inv_scale = 1.0f / scale;
    float block_start_x = blockIdx.x * TILE_W - pad_l;
    float block_start_y = blockIdx.y * TILE_H - pad_t;

    // load shared memory
    for (int i = 0; i < iter; i++) {
        int idx = tid + i * threads_per_block;
        if (idx < total_smem) {
            int sx = idx % (TILE_W + 1);
            int sy = idx / (TILE_W + 1);

            float in_x = (block_start_x + sx) * inv_scale;
            float in_y = (block_start_y + sy) * inv_scale;

            int ix = floorf(in_x);
            int iy = floorf(in_y);

            uchar4 p = make_uchar4(114, 114, 114, 0); // padding BGR
            if (ix >= 0 && iy >= 0 && ix < in_w && iy < in_h) {
                int offset = (iy * in_w + ix) * 3;
                p.x = input_bgr[offset]; // b
                p.y = input_bgr[offset + 1]; // g
                p.z = input_bgr[offset + 2]; // r
            }

            smem[sy][sx] = p;
        }
    }
    __syncthreads();

    // 双线性插值
    float in_x = (x - pad_l) * inv_scale;
    float in_y = (y - pad_t) * inv_scale;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float dx = in_x - floorf(in_x);
    float dy = in_y - floorf(in_y);
    float dx1 = 1.0f - dx, dy1 = 1.0f - dy;

    uchar4 p00 = smem[ty][tx];
    uchar4 p01 = smem[ty][tx + 1];
    uchar4 p10 = smem[ty + 1][tx];
    uchar4 p11 = smem[ty + 1][tx + 1];

    float out_r = dx1 * dy1 * p00.z + dx * dy1 * p01.z + dx1 * dy * p10.z + dx * dy * p11.z;
    float out_g = dx1 * dy1 * p00.y + dx * dy1 * p01.y + dx1 * dy * p10.y + dx * dy * p11.y;
    float out_b = dx1 * dy1 * p00.x + dx * dy1 * p01.x + dx1 * dy * p10.x + dx * dy * p11.x;

    int out_idx = y * out_w + x;
    if (swap_rb) {
        output_nchw[out_idx + 0 * out_w * out_h] = out_r * norm;
        output_nchw[out_idx + 1 * out_w * out_h] = out_g * norm;
        output_nchw[out_idx + 2 * out_w * out_h] = out_b * norm;
    } else {
        output_nchw[out_idx + 0 * out_w * out_h] = out_b * norm;
        output_nchw[out_idx + 1 * out_w * out_h] = out_g * norm;
        output_nchw[out_idx + 2 * out_w * out_h] = out_r * norm;
    }
}

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
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= OUT_W || oy >= OUT_H)
        return;

    float fx = (ox - pad_l) / scale;
    float fy = (oy - pad_t) / scale;

    int out_idx = oy * OUT_W + ox;
    int plane = OUT_W * OUT_H;

    // clamp coordinates
    fx = fmaxf(0.f, fminf(fx, src_w - 2.f));
    fy = fmaxf(0.f, fminf(fy, src_h - 2.f));

    int x0 = floorf(fx), y0 = floorf(fy);
    int x1 = x0 + 1, y1 = y0 + 1;

    float dx = fx - x0, dy = fy - y0;
    float dx1 = 1.f - dx, dy1 = 1.f - dy;

    // row pointers
    const uchar3* row0 = (const uchar3*)((const char*)d_input_bgr + y0 * pitch);
    const uchar3* row1 = (const uchar3*)((const char*)d_input_bgr + y1 * pitch);

    uchar3 p00 = row0[x0];
    uchar3 p01 = row0[x1];
    uchar3 p10 = row1[x0];
    uchar3 p11 = row1[x1];

    // bilinear interpolation
    float r = dx1 * dy1 * p00.z + dx * dy1 * p01.z + dx1 * dy * p10.z + dx * dy * p11.z;
    float g = dx1 * dy1 * p00.y + dx * dy1 * p01.y + dx1 * dy * p10.y + dx * dy * p11.y;
    float b = dx1 * dy1 * p00.x + dx * dy1 * p01.x + dx1 * dy * p10.x + dx * dy * p11.x;

    if (swap_rb) {
        d_nchw[out_idx + 0 * plane] = r * norm;
        d_nchw[out_idx + 1 * plane] = g * norm;
        d_nchw[out_idx + 2 * plane] = b * norm;
    } else {
        d_nchw[out_idx + 0 * plane] = b * norm;
        d_nchw[out_idx + 1 * plane] = g * norm;
        d_nchw[out_idx + 2 * plane] = r * norm;
    }
}
