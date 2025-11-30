import os
import glob
from tqdm import tqdm
import random
import torch.utils.tensorboard as tensorboardX
import matplotlib.backends.backend_agg as plt_backend_agg
import matplotlib.pyplot as plt
import numpy as np

import time

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rich.console import Console

from packaging import version as pver
import imageio
import lpips


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


# copied from pytorch3d
def _angle_from_tan(
        axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str = 'XYZ') -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    # if len(convention) != 3:
    #     raise ValueError("Convention must have 3 letters.")
    # if convention[1] in (convention[0], convention[2]):
    #     raise ValueError(f"Invalid convention {convention}.")
    # for letter in convention:
    #     if letter not in ("X", "Y", "Z"):
    #         raise ValueError(f"Invalid letter {letter} in convention string.")
    # if matrix.size(-1) != 3 or matrix.size(-2) != 3:
    #     raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


@torch.cuda.amp.autocast(enabled=False)
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


@torch.cuda.amp.autocast(enabled=False)
def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str = 'XYZ') -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    # print(euler_angles, euler_angles.dtype)

    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]

    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


@torch.cuda.amp.autocast(enabled=False)
def get_bg_coords(H, W, device):
    X = torch.arange(H, device=device) / (H - 1) * 2 - 1  # in [-1, 1]
    Y = torch.arange(W, device=device) / (W - 1) * 2 - 1  # in [-1, 1]
    xs, ys = custom_meshgrid(X, Y)
    bg_coords = torch.cat([xs.reshape(-1, 1), ys.reshape(-1, 1)], dim=-1).unsqueeze(0)  # [1, H*W, 2], in [-1, 1]
    return bg_coords


@torch.cuda.amp.autocast(enabled=False)
def get_rays(cams, H, W, N=-1, patch_size=1, rect=None, mask=None):
    """ get rays
    args:
        cams: [B, 3, 7], K, R, T  world2cam
        H, W, N: int
    returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """
    device = cams.device
    B = cams.shape[0]

    if rect is not None:
        xmin, xmax, ymin, ymax = rect
        N = (xmax - xmin) * (ymax - ymin)
    if mask is not None:
        N = mask.sum()

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device),
                           torch.linspace(0, H - 1, H, device=device))  # float
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}
    if N > 0:
        N = min(N, H * W)

        if patch_size > 1:

            # random sample left-top cores.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randperm(H, device=device)[:num_patch]
            inds_y = torch.randperm(W, device=device)[:num_patch]
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.reshape(-1, 2)  # [N, 2]
            inds[..., 0][inds[..., 0] >= H] = H - 1
            inds[..., 1][inds[..., 1] >= W] = W - 1
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

            inds = inds.expand([B, N])

        # only get rays in the specified rect
        elif rect is not None:
            mask = torch.zeros(H, W, dtype=torch.bool, device=device)
            xmin, xmax, ymin, ymax = rect
            mask[xmin:xmax, ymin:ymax] = 1
            inds = torch.where(mask.view(-1))[0]  # [nzn]
            inds = inds.unsqueeze(0)  # [1, N]
        elif mask is not None:
            inds = torch.where(mask.view(-1))[0]  # [nzn]
            inds = inds.unsqueeze(0)  # [1, N]
        else:
            # inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = torch.randperm(H * W, device=device)[:N]
            inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    results['i'] = i
    results['j'] = j
    results['inds'] = inds

    K, R, T = torch.split(cams, [3, 3, 1], dim=-1)
    grids_h = torch.stack([i, j, torch.ones_like(i)], -1)
    inv_K = torch.inverse(K)
    rays_o = -torch.bmm(T.transpose(1, 2), R)  # (B, 1, 3)
    rays_d = F.normalize(grids_h @ inv_K.transpose(1, 2) @ R, dim=-1)  # (B, h*w, 3)

    results['rays_o'] = rays_o.expand_as(rays_d)
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0).squeeze()
        x = x.detach().cpu().numpy()

    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')

    x = x.astype(np.float32)

    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    @staticmethod
    def prepare_inputs(*inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range in [0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='vgg', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item()  # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'


class LMDMeter:
    def __init__(self, backend='dlib', region='mouth'):
        self.backend = backend
        self.region = region  # mouth or face

        if self.backend == 'dlib':
            import dlib

            # load checkpoint manually
            self.predictor_path = './shape_predictor_68_face_landmarks.dat'
            if not os.path.exists(self.predictor_path):
                raise FileNotFoundError('Please download dlib checkpoint from'
                                        ' http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')

            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.predictor_path)

        else:

            import face_alignment

            try:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            except AttributeError:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        self.V = 0
        self.N = 0

    def get_landmarks(self, img):
        if self.backend == 'dlib':
            dets = self.detector(img, 1)
            shape = self.predictor(img, dets[0])
            # ref: https://github.com/PyImageSearch/imutils/blob/master/imutils/face_utils/helpers.py
            lms = np.zeros((68, 2), dtype=np.int32)
            for i in range(0, 68):
                lms[i, 0] = shape.part(i).x
                lms[i, 1] = shape.part(i).y

        else:
            lms = self.predictor.get_landmarks(img)[0]

        # self.vis_landmarks(img, lms)
        lms = lms.astype(np.float32)

        return lms

    def vis_landmarks(self, img, lms):
        plt.imshow(img)
        plt.plot(lms[48:68, 0], lms[48:68, 1], marker='o', markersize=1, linestyle='-', lw=2)
        plt.show()

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.detach().cpu().numpy()
            inp = (inp * 255).astype(np.uint8)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        # assert B == 1
        preds, truths = self.prepare_inputs(preds[0], truths[0])  # [H, W, 3] numpy array

        # get lms
        lms_pred = self.get_landmarks(preds)
        lms_truth = self.get_landmarks(truths)

        if self.region == 'mouth':
            lms_pred = lms_pred[48:68]
            lms_truth = lms_truth[48:68]

        # avarage
        lms_pred = lms_pred - lms_pred.mean(0)
        lms_truth = lms_truth - lms_truth.mean(0)

        # distance
        dist = np.sqrt(((lms_pred - lms_truth) ** 2).sum(1)).mean(0)

        self.V += dist
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LMD ({self.backend})"), self.measure(), global_step)

    def report(self):
        return f'LMD ({self.backend}) = {self.measure():.6f}'


def add_lms_bound(img, lms, cam, aabb):
    cam = cam.reshape(3, 7).cpu()  # 3,7
    K, R, T = torch.split(cam, [3, 3, 1], dim=-1)
    lms = lms[0].T.cpu()  # 3,68
    aabb = torch.cartesian_prod(aabb[::3], aabb[1::3], aabb[2::3]).cpu().T  # 3,8
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax = fig.gca()
    ax.imshow(img)
    ax.axis('off')
    lms = K @ (R @ lms + T)
    aabb = K @ (R @ aabb + T)
    lms = lms[:2] / lms[2:]
    # aabb=aabb[:2]/aabb[2:]
    # ax.scatter(*lms,c='red')
    # ax.plot(*aabb[:,[0, 1, 3, 2, 6, 4, 5, 7, 3]],'b')
    # ax.plot(*aabb[:,[2, 0, 4]],'b')
    # ax.plot(*aabb[:,[7, 6]],'b')
    # ax.plot(*aabb[:,[1, 5]],'b')
    canvas = plt_backend_agg.FigureCanvasAgg(fig)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    plt.close(fig)
    return image_hwc


class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 ema_update_interval=1000,  # update ema per $ training steps.
                 lr_scheduler=None,  # scheduler
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=2,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 ):

        self.writer = None
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.ema_update_interval = ema_update_interval
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.flip_finetune_lips = self.opt.finetune_lips
        self.flip_finetune_eyes = self.opt.finetune_eyes
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            from torch_ema import ExponentialMovingAverage
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # optionally use LPIPS loss for patch-based training
        if self.opt.patch_size > 1 or self.opt.finetune_lips:
            import lpips
            self.criterion_lpips = lpips.LPIPS(net='vgg').to(self.device)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log('[INFO] Trainer: {} | {} | {} | {} | {}'.format(
            self.name, self.time_stamp, self.device, "fp16" if self.fp16 else "fp32", self.workspace))
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    # ------------------------------

    def train_step(self, data):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        aud = data['aud']  # [B, 64]
        lmks = data['lmk']  # [B, 68, 3]
        R = data['R']  # [B, 3, 3]
        face_mask = data['face_mask']  # [B, N]
        index = data['index']  # [B]
        rgb = data['images']  # [B, N, 3]
        # B, N, C = rgb.shape

        if self.opt.color_space == 'linear':
            rgb[..., :3] = srgb_to_linear(rgb[..., :3])

        bg_color = data['bg_color']

        outputs = self.model.render(
            rays_o, rays_d, lmks, R, aud, index=index, bg_color=bg_color, perturb=True,
            force_all_rays=False if (self.opt.patch_size <= 1 and not self.opt.train_camera) else True, **vars(self.opt)
        )

        pred_rgb = outputs['image']

        # MSE loss
        loss = self.criterion(pred_rgb, rgb).mean(-1)  # [B, N, 3] --> [B, N]

        # Eye loss
        if self.opt.finetune_eyes:
            l_wink = data['l_wink']  # [B]
            r_wink = data['r_wink']  # [B]
            le_mask = data['le_mask']  # [B,N,1]
            re_mask = data['re_mask']  # [B,N,1]
            loss = loss + torch.exp(1 - l_wink) * self.criterion(pred_rgb[le_mask], rgb[le_mask]).mean() + \
                   torch.exp(1 - r_wink) * self.criterion(pred_rgb[re_mask], rgb[re_mask]).mean()
        # camera optim regularization
        # if self.opt.train_camera:
        #     cam_reg = self.model.camera_dR[index].abs().mean() + self.model.camera_dT[index].abs().mean()
        #     loss = loss + 1e-2 * cam_reg

        # lips finetune
        if self.opt.finetune_lips:
            xmin, xmax, ymin, ymax = data['rect']
            rgb = rgb.view(-1, xmax - xmin, ymax - ymin, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb = pred_rgb.view(-1, xmax - xmin, ymax - ymin, 3).permute(0, 3, 1, 2).contiguous()

            # torch_vis_2d(rgb[0])
            # torch_vis_2d(pred_rgb[0])

            # LPIPS loss
            loss = loss + 0.01 * self.criterion_lpips(pred_rgb, rgb)

        # flip every step... if finetune lips
        if self.flip_finetune_lips:
            self.opt.finetune_lips = not self.opt.finetune_lips
        if self.flip_finetune_eyes:
            self.opt.finetune_eyes = not self.opt.finetune_eyes

        # patch-based rendering
        if self.opt.patch_size > 1 and (self.flip_finetune_eyes == self.opt.finetune_eyes) and (self.flip_finetune_lips == self.opt.finetune_lips):
            rgb = rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb = pred_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()

            # torch_vis_2d(rgb[0])
            # torch_vis_2d(pred_rgb[0])

            # LPIPS loss ?
            loss = loss + 0.001 * self.criterion_lpips(pred_rgb, rgb).sum()

        pred_alphas = outputs['weights_sum'].view(1, -1)
        alphas = data['alphas'].view(1, -1)
        loss = loss + 0.5 * self.criterion(pred_alphas, alphas)

        loss = loss.mean()

        # weights_sum loss
        # entropy to encourage weights_sum to be 0 or 1.
        # alphas = outputs['weights_sum'].clamp(1e-5, 1 - 1e-5)
        # loss_ws = - alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)
        # loss = loss + 1e-4 * loss_ws.mean()

        # ambient loss (regions out of face should be static)
        ambient = outputs['ambient']  # [N], abs sum
        loss_amb = (ambient * (~alphas.view(-1))).mean()

        # gradually increase it
        lambda_amb = min(self.global_step / self.opt.iters, 1.0) * self.opt.lambda_amb
        # lambda_amb = self.opt.lambda_amb
        loss = loss + lambda_amb * loss_amb

        return pred_rgb, rgb, loss

    def eval_step(self, data):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        aud = data['aud']  # [B, 64]
        lmks = data['lmk']  # [B, 68, 3]
        R = data['R']  # [B, 3, 3]

        images = data['images']  # [B, N, 3/4]
        index = data['index']  # [B]
        H, W = data['H'], data['W']

        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])
        
        if not self.opt.finetune_eyes and not self.opt.finetune_lips:
            images = images.reshape(B, H, W, 3)
        elif self.opt.finetune_lips:
            xmin, xmax, ymin, ymax = data['rect']
            images = images.reshape(B, xmax - xmin, ymax - ymin, 3)

        # eval with fixed background color
        # bg_color = 1
        bg_color = data['bg_color']

        outputs = self.model.render(rays_o, rays_d, lmks, R, aud, index=index, bg_color=bg_color,
                                    perturb=False, **vars(self.opt))


        if not self.opt.finetune_eyes and not self.opt.finetune_lips:
            pred_rgb = outputs['image'].reshape(B, H, W, 3)
            pred_depth = outputs['depth'].reshape(B, H, W)
        elif self.opt.finetune_lips:
            pred_rgb = outputs['image'].reshape(B, xmax - xmin, ymax - ymin, 3)
            pred_depth = outputs['depth'].reshape(B, xmax - xmin, ymax - ymin)
        else:
            pred_rgb = outputs['image']
            pred_depth = outputs['depth']

        loss = self.criterion(pred_rgb, images).mean()

        return pred_rgb, pred_depth, images, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        aud = data['aud']  # [B, 64]
        lmks = data['lmk']  # [B, 68, 3]
        R = data['R']  # [B, 3, 3]
        index = data['index']
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)
        else:
            bg_color = data['bg_color']

        outputs = self.model.render(rays_o, rays_d, lmks, R, aud, index=index, bg_color=bg_color,
                                    perturb=perturb, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth

    # ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.cams)

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_image=False):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm(total=len(loader) * loader.batch_size,
                    bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        all_preds = []

        with torch.no_grad():

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)

                path = os.path.join(save_path, f'{name}_{i:04d}_rgb.png')
                path_depth = os.path.join(save_path, f'{name}_{i:04d}_depth.png')

                # self.log(f"[INFO] saving test image to {path}")

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_image:
                    imageio.imwrite(path, pred)
                    imageio.imwrite(path_depth, pred_depth)

                all_preds.append(add_lms_bound(pred, data['lmk'], data['cams'], self.model.aabb_train))

                pbar.update(loader.batch_size)

        # write video
        imageio.mimwrite(os.path.join(save_path, f'{name}.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm(total=len(loader) * loader.batch_size,
                        bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.ema is not None and self.global_step % self.ema_update_interval == 0:
                self.ema.update()

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description("loss={:.4f} ({:.4f}), lr={:.6f}".format(
                        loss_val, total_loss / self.local_step, self.optimizer.param_groups[0]['lr']))
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm(total=len(loader) * loader.batch_size,
                        bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss = self.eval_step(data)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation',
                                                   f'{name}_{self.local_step:04d}_depth.png')
                    # save_path_gt = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_gt.png')

                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred_depth = preds_depth[0].detach().cpu().numpy()

                    cv2.imwrite(save_path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, (pred_depth * 255).astype(np.uint8))
                    # cv2.imwrite(save_path_gt, cv2.cvtColor((linear_to_srgb(truths[0].detach().cpu().numpy()) * 255
                    #                                         ).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {'epoch': self.epoch, 'global_step': self.global_step, 'stats': self.stats,
                 'mean_count': self.model.mean_count, 'mean_density': self.model.mean_density}

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                # always save new as best... (since metric cannot really reflect performance...)
                if True:

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid
                    # to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded bare model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if 'mean_count' in checkpoint_dict:
            self.model.mean_count = checkpoint_dict['mean_count']
        if 'mean_density' in checkpoint_dict:
            self.model.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
