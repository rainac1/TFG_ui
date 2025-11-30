# pragma once

#include <stdint.h>
#include <torch/torch.h>

void DiagGaussForward(at::Tensor xyz, at::Tensor cov3d, const uint32_t P, const uint32_t M, at::Tensor diag);
void DiagGaussBackward(at::Tensor xyz, at::Tensor dL_ddiag, const uint32_t P, const uint32_t M, at::Tensor dL_dcov3d);