#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>

#include <cstdio>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

__global__ void DiagGaussForwardCUDAKernel(uint32_t P,
                                           uint32_t M,
                                           const float * __restrict__ xyz,
                                           const float * __restrict__ cov3d,
                                           float * diag) {
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= P * M)
        return;

    const uint32_t p = t / M;
    xyz += t * 3;
    cov3d += 6 * p;
    diag += t;

    float x = xyz[0], y = xyz[1], z = xyz[2];

    diag[0] = x * x * cov3d[0] + y * y * cov3d[3] + z * z * cov3d[5] + 2 * x * y * cov3d[1] + 2 * x * z * cov3d[2] + 2 * y * z * cov3d[4];
}

__global__ void DiagGaussBackCUDAKernel(uint32_t P,
                                        uint32_t M,
                                        const float * __restrict__ xyz,
                                        const float * __restrict__ dL_ddiag,
                                        float * dL_dcov3d) {
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= P * M)
        return;

    const uint32_t p = t / M;
    xyz += t * 3;
    dL_dcov3d += 6 * p;

    float dL = dL_ddiag[t], x = xyz[0], y = xyz[1], z = xyz[2];

    atomicAdd(dL_dcov3d + 0, x * x * dL);
    atomicAdd(dL_dcov3d + 1, 2 * x * y * dL);
    atomicAdd(dL_dcov3d + 2, 2 * x * z * dL);
    atomicAdd(dL_dcov3d + 3, y * y * dL);
    atomicAdd(dL_dcov3d + 4, 2 * y * z * dL);
    atomicAdd(dL_dcov3d + 5, z * z * dL);
}

void DiagGaussForward(at::Tensor xyz,
                      at::Tensor cov3d,
                      const uint32_t P,
                      const uint32_t M,
                      at::Tensor diag) {
    CHECK_CUDA(xyz);
    CHECK_CUDA(cov3d);
    CHECK_CUDA(diag);

    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(cov3d);
    CHECK_CONTIGUOUS(diag);

    CHECK_IS_FLOATING(xyz);
    CHECK_IS_FLOATING(cov3d);
    CHECK_IS_FLOATING(diag);

    static constexpr uint32_t N_THREADS = 128;

    DiagGaussForwardCUDAKernel<<<div_round_up(P * M, N_THREADS), N_THREADS>>>(
        P,
        M,
        xyz.data_ptr<float>(),
        cov3d.data_ptr<float>(),
        diag.data_ptr<float>());
}

void DiagGaussBackward(at::Tensor xyz,
                       at::Tensor dL_ddiag,
                       const uint32_t P,
                       const uint32_t M,
                       at::Tensor dL_dcov3d) {
    CHECK_CUDA(xyz);
    CHECK_CUDA(dL_ddiag);
    CHECK_CUDA(dL_dcov3d);

    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(dL_ddiag);
    CHECK_CONTIGUOUS(dL_dcov3d);

    CHECK_IS_FLOATING(xyz);
    CHECK_IS_FLOATING(dL_ddiag);
    CHECK_IS_FLOATING(dL_dcov3d);

    static constexpr uint32_t N_THREADS = 128;
    dL_dcov3d.fill_(0);

    DiagGaussBackCUDAKernel<<<div_round_up(P * M, N_THREADS), N_THREADS>>>(
        P,
        M,
        xyz.data_ptr<float>(),
        dL_ddiag.data_ptr<float>(),
        dL_dcov3d.data_ptr<float>());
}