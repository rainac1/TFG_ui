import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _gsencoder as _backend
except ImportError:
    from .backend import _backend


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


class _ComputeGauss(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # force float32 for better precision
    def forward(ctx, xyzs, cov3Ds):
        P, M, _ = xyzs.shape
        xyzs = xyzs.contiguous()
        cov3Ds = cov3Ds.contiguous()
        diags = torch.zeros(P, M, dtype=xyzs.dtype, device=xyzs.device)
        _backend.compute_gauss_forward(xyzs, cov3Ds, P, M, diags)
        # save variables for backward
        ctx.save_for_backward(xyzs)
        # ctx.save_for_backward(xyzs)

        return diags

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, dL_ddiags):
        # get saved variables from forward
        (xyzs, ) = ctx.saved_tensors
        P, M = dL_ddiags.shape
        dL_ddiags = dL_ddiags.contiguous()
        dL_dcov3ds = torch.zeros(P, 6, dtype=dL_ddiags.dtype, device=dL_ddiags.device)
        _backend.compute_gauss_backward(xyzs, dL_ddiags, P, M, dL_dcov3ds)

        return None, dL_dcov3ds


class _ComputeCov3D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # force float32 for better precision
    def forward(ctx, scales, uquats):
        cov3Ds = _backend.compute_cov3d_forward(scales, uquats)
        # save variables for backward
        ctx.save_for_backward(scales, uquats)

        return cov3Ds

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, dL_dcov3Ds):
        # get saved variables from forward
        scales, uquats = ctx.saved_tensors

        (dL_dscales, dL_duquats) = _backend.compute_cov3d_backward(scales, uquats, dL_dcov3Ds)

        return dL_dscales, dL_duquats


trunc_exp = _trunc_exp.apply
compute_gauss = _ComputeGauss.apply
compute_cov3d = _ComputeCov3D.apply
