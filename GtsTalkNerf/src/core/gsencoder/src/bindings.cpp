#include <torch/extension.h>

#include "freqencoder.h"
#include "compute_cov3d.h"
#include "compute_gauss.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("freq_encode_forward", &freq_encode_forward, "freq encode forward (CUDA)");
    m.def("freq_encode_backward", &freq_encode_backward, "freq encode backward (CUDA)");
    m.def("compute_cov3d_forward", &computeCov3DForward, "computeCov3D forward (CUDA)");
    m.def("compute_cov3d_backward", &computeCov3DBackward, "computeCov3D backward (CUDA)");
    m.def("compute_gauss_forward", &DiagGaussForward, "computeDiag forward (CUDA)");
    m.def("compute_gauss_backward", &DiagGaussBackward, "computeDiag backward (CUDA)");
}