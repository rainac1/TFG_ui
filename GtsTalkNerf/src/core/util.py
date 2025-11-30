# UTF-8 encoded
import lpips
import matplotlib.backends.backend_agg as plt_backend_agg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from itertools import accumulate
from pathlib import Path
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch.utils.data import Dataset, Subset
from typing import Sequence, Union

lpips_model = None


#https://github.com/pytorch/pytorch/blob/v2.6.0/torch/utils/data/dataset.py#L217
class StackDataset(Dataset):
    r"""Dataset as a stacking of multiple datasets.

    This class is useful to assemble different parts of complex input data, given as datasets.

    Example:
        >>> # xdoctest: +SKIP
        >>> images = ImageDataset()
        >>> texts = TextDataset()
        >>> tuple_stack = StackDataset(images, texts)
        >>> tuple_stack[0] == (images[0], texts[0])
        >>> dict_stack = StackDataset(image=images, text=texts)
        >>> dict_stack[0] == {'image': images[0], 'text': texts[0]}

    Args:
        *args (Dataset): Datasets for stacking returned as tuple.
        **kwargs (Dataset): Datasets for stacking returned as dict.
    """

    datasets: Union[tuple, dict]

    def __init__(self, *args: Dataset, **kwargs: Dataset) -> None:
        if args:
            if kwargs:
                raise ValueError(
                    "Supported either ``tuple``- (via ``args``) or"
                    "``dict``- (via ``kwargs``) like input/output, but both types are given."
                )
            self._length = len(args[0])  # type: ignore[arg-type]
            if any(self._length != len(dataset) for dataset in args):  # type: ignore[arg-type]
                raise ValueError("Size mismatch between datasets")
            self.datasets = args
        elif kwargs:
            tmp = list(kwargs.values())
            self._length = len(tmp[0])  # type: ignore[arg-type]
            if any(self._length != len(dataset) for dataset in tmp):  # type: ignore[arg-type]
                raise ValueError("Size mismatch between datasets")
            self.datasets = kwargs
        else:
            raise ValueError("At least one dataset should be passed")

    def __getitem__(self, index):
        if isinstance(self.datasets, dict):
            return {k: dataset[index] for k, dataset in self.datasets.items()}
        return tuple(dataset[index] for dataset in self.datasets)

    def __getitems__(self, indices: list):
        # add batched sampling support when parent datasets supports it.
        if isinstance(self.datasets, dict):
            dict_batch = [{} for _ in indices]
            for k, dataset in self.datasets.items():
                if callable(getattr(dataset, "__getitems__", None)):
                    items = dataset.__getitems__(indices)  # type: ignore[attr-defined]
                    if len(items) != len(indices):
                        raise ValueError(
                            "Nested dataset's output size mismatch."
                            f" Expected {len(indices)}, got {len(items)}"
                        )
                    for data, d_sample in zip(items, dict_batch):
                        d_sample[k] = data
                else:
                    for idx, d_sample in zip(indices, dict_batch):
                        d_sample[k] = dataset[idx]
            return dict_batch

        # tuple data
        list_batch = [[] for _ in indices]
        for dataset in self.datasets:
            if callable(getattr(dataset, "__getitems__", None)):
                items = dataset.__getitems__(indices)  # type: ignore[attr-defined]
                if len(items) != len(indices):
                    raise ValueError(
                        "Nested dataset's output size mismatch."
                        f" Expected {len(indices)}, got {len(items)}"
                    )
                for data, t_sample in zip(items, list_batch):
                    t_sample.append(data)
            else:
                for idx, t_sample in zip(indices, list_batch):
                    t_sample.append(dataset[idx])
        tuple_batch = [tuple(sample) for sample in list_batch]
        return tuple_batch

    def __len__(self):
        return self._length


def str2bool(s: str):
    return s.lower() in ("true", "t", "1", "yes", "y")


def render_to_rgb(figure, close=True):
    canvas = plt_backend_agg.FigureCanvasAgg(figure)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    if close:
        plt.close(figure)
    return image_hwc


def calcSSIM(x, y, data_range=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy().clip(0, 1.)
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    assert x.ndim == y.ndim == 3
    channel_axis = 0 if x.shape[0] in [1, 3] else 2
    if data_range is None:
        data_range = 1.
    return structural_similarity(x, y, data_range=data_range, channel_axis=channel_axis)


def calcPSNR(x, y, data_range=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy().clip(0, 1.)
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    assert x.ndim == y.ndim == 3
    if data_range is None:
        data_range = 1.
    return peak_signal_noise_ratio(x, y, data_range=data_range)


@torch.no_grad()
def calcLPIPS(x, y):
    if x.ndim == 3:
        x = x[None].clamp(0, 1.)
    if y.ndim == 3:
        y = y[None]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.ndim == 4 and y.ndim == 4
    global lpips_model
    if lpips_model is None:
        lpips_model = lpips.LPIPS()
    lpips_model = lpips_model.to(x.device)
    return lpips_model(x, y, normalize=True).item()


# class VGGLoss(torch.nn.Module):
#     def __init__(self):
#         super(VGGLoss, self).__init__()
#         self.vgg_net = lpips.LPIPS(net='vgg')
#
#     def forward(self, x, y, normalize=True):
#         # default input is [0,1], if not, set normalize to False
#         val = self.vgg_net(x, y, normalize=normalize)
#         return val.mean()
#
#
# class LPIPSLoss(torch.nn.Module):
#     def __init__(self):
#         super(LPIPSLoss, self).__init__()
#         self.lpips_model = lpips.LPIPS(net="alex")
#
#     def forward(self, x, y, normalize=True):
#         # default input is [0,1], if not, set normalize to False
#         val = self.lpips_model(x, y, normalize=normalize)
#         return val.mean()
