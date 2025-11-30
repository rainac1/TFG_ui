import cv2
import glob
import numpy as np
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from einops import rearrange
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboardX
import time
from rich.console import Console
import imageio
import lpips

from ops_dcnv3.modules import DCNv3


class DCN(nn.Module):
    def __init__(self, c_in, c_out, ks=3, stride=2, mode='down'):
        super().__init__()
        if mode == 'down':
            self.conv = nn.Conv2d(c_in, c_out, ks, stride=stride, padding=1, bias=False)
        elif mode == 'up':
            self.conv = nn.Sequential(nn.Upsample(scale_factor=stride),
                                      nn.Conv2d(c_in, c_out, ks, 1, 1))
        self.dcn = DCNv3(c_out)

    def forward(self, x):
        x = self.conv(x)
        sh = x.shape
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.dcn(F.relu(x, True))
        return rearrange(x, 'b h w c -> b c h w', h=sh[2]).contiguous()


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        drop = nn.Dropout(0.5)
        self.downs = []
        self.downs += [DCN(3, 64, 3)]
        self.downs += [nn.Sequential(nn.ReLU(True), DCN(64, 64, 3))]
        self.downs += [nn.Sequential(nn.ReLU(True), DCN(64, 64, 3))]
        self.downs += [nn.Sequential(nn.ReLU(True), nn.Conv2d(64, 128, 3, 2, 1, bias=False), drop)]
        self.downs += [nn.Sequential(nn.ReLU(True), nn.Conv2d(128, 128, 3, 2, 1, bias=False), drop)]
        self.downs = nn.ModuleList(self.downs)
        self.ups = []
        self.ups += [nn.Sequential(nn.ReLU(True), nn.Upsample(scale_factor=2), nn.Conv2d(128, 128, 3, 1, 1), drop)]
        self.ups += [nn.Sequential(nn.ReLU(True), nn.Upsample(scale_factor=2), nn.Conv2d(256, 64, 3, 1, 1), drop)]
        self.ups += [nn.Sequential(nn.ReLU(True), DCN(128, 64, 3, mode='up'))]
        self.ups += [nn.Sequential(nn.ReLU(True), DCN(128, 64, 3, mode='up'))]
        self.ups += [nn.Sequential(nn.ReLU(True), DCN(128, 64, 3, mode='up'))]
        self.ups = nn.ModuleList(self.ups)
        self.out_conv = nn.Sequential(nn.ReLU(True), nn.Conv2d(64, 3, 1))

    def forward(self, x):
        out_lst = []
        for down in self.downs:
            x = down(x)
            out_lst.append(x)
        out_lst.pop()
        h = None
        for up in self.ups:
            if h is None:
                x = up(x)
            else:
                x = up(torch.cat([x, h], 1))
            if out_lst:
                h = out_lst.pop()
        return torch.tanh(self.out_conv(x))


class UNetDataset:
    def __init__(self, opt, device, batch_size=1, mode='train', downscale=1):
        super().__init__()
        self.opt = opt
        self.device = device
        self.batch_size = batch_size
        self.mode = mode  # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload  # 0 = disk, 1 = cpu, 2 = gpu
        self.fp16 = opt.fp16

        self.training = self.mode in ['train', 'all', 'trainval', 'val']

        # read images
        frames = sorted(glob.glob(os.path.join(self.root_path, 'nerf/results/*rgb.png')))

        self.H, self.W = cv2.imread(frames[0]).shape[:2]
        self.H = self.H // downscale
        self.W = self.W // downscale

        total_length = len(frames)
        split_length = int(total_length * 0.8)
        if mode == 'all':
            slices = slice(total_length)
        elif mode == 'trainval':
            slices = slice(total_length)
        elif mode == 'train':
            slices = slice(split_length)
            if self.opt.part:
                slices = slice(0, split_length, 10)  # 1/10 frames
            elif self.opt.part2:
                slices = slice(min(split_length, 375))  # first 15s
        elif mode == 'val':
            slices = slice(split_length, min(total_length, split_length + 100), 10)  # first 10/100 frames for val
        else:  # mode == 'test'
            slices = slice(split_length, total_length)

        frames = frames[slices]

        print(f'[INFO] load {len(frames)} {mode} frames.')

        self.heads = []
        self.images = []

        for index in tqdm.trange(len(frames), desc=f'Loading {mode} data'):
            f_path = frames[index]
            image = os.path.join(self.root_path, 'frame_imgs', re.sub('[^0-9]', '', f_path)[-4:] + '.jpg')
            assert os.path.exists(f_path), f'[ERROR] {f_path} NOT FOUND!'

            if self.preload > 0:
                head = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
                head = cv2.cvtColor(head, cv2.COLOR_BGR2RGB)
                head = head.astype(np.float32) / 127.5 - 1  # [H, W, 3/4]
                self.heads.append(head)
                if self.training:
                    image = cv2.imread(image, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = image.astype(np.float32) / 127.5 - 1  # [H, W, 3/4]
                    self.images.append(image)
            else:
                self.heads.append(f_path)
                self.images.append(image)

        if self.preload > 0:
            self.heads = np.stack(self.heads, 0)
            self.heads = torch.from_numpy(self.heads).permute(0, 3, 1, 2).contiguous()  # [N, C, H, W]
            if self.training:
                self.images = np.stack(self.images, 0)
                self.images = torch.from_numpy(self.images).permute(0, 3, 1, 2).contiguous()  # [N, C, H, W]
        else:
            self.heads = np.array(self.heads)
            self.images = np.array(self.images)

        if self.preload > 1:
            self.heads = self.heads.to(torch.half).to(self.device)
            if self.training:
                self.images = self.images.to(torch.half).to(self.device)

    def mirror_index(self, index):
        size = self.heads.shape[0]
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def collate(self, index):
        results = {}

        # head pose and bg image may mirror (replay --> <-- --> <--).
        index = [self.mirror_index(_index) for _index in index]  # a list of length b

        head = self.heads[index]  # [B, 3/4, H, W]
        if self.preload == 0:
            head = [cv2.imread(_head, cv2.IMREAD_UNCHANGED) for _head in head]
            head = [cv2.cvtColor(_head, cv2.COLOR_BGR2RGB) for _head in head]
            head = np.stack(head, 0).astype(np.float32) / 127.5 - 1  # [B, H, W, 3]
            head = torch.from_numpy(head).permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        head = head.to(self.device)
        results['x'] = head

        if self.training:
            image = self.images[index]
            if self.preload == 0:
                image = [cv2.imread(_image, cv2.IMREAD_UNCHANGED) for _image in image]
                image = [cv2.cvtColor(_image, cv2.COLOR_BGR2RGB) for _image in image]
                image = np.stack(image, 0).astype(np.float32) / 127.5 - 1  # [B, H, W, 3]
                image = torch.from_numpy(image).permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            image = image.to(self.device)
            results['y'] = image

        return results

    def dataloader(self):
        size = self.heads.shape[0]

        loader = DataLoader(list(range(size)), batch_size=self.batch_size, collate_fn=self.collate,
                            shuffle=self.training, num_workers=0)
        loader._data = self  # an ugly fix... we need poses in trainer.

        return loader


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
                inp = inp.detach().cpu().numpy() * 0.5 + 0.5
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2, (-1, -2, -3)))  # [B]

        self.V += psnr.sum()
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
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)
        v = self.fn(truths, preds, normalize=False).sum().item()
        self.V += v
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'


class UNetTrainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 ema_update_interval=1000,  # update ema per $ training steps.
                 lr_scheduler=None,  # scheduler
                 metrics=None,
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
        if metrics is None:
            metrics = []
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
        pred_rgb = self.model(data['x'])
        rgb = data['y']
        # MSE loss
        loss = self.criterion(pred_rgb, rgb).mean(-1)  # [B, 3, H, W]
        # LPIPS loss
        loss = loss + 0.001 * self.criterion_lpips(pred_rgb, rgb).sum()
        return pred_rgb, rgb, loss

    def eval_step(self, data):
        pred_rgb = self.model(data['x'])
        rgb = data['y']
        # MSE loss
        loss = self.criterion(pred_rgb, rgb).mean(-1)  # [B, 3, H, W]
        # LPIPS loss
        loss = loss + 0.001 * self.criterion_lpips(pred_rgb, rgb).sum()
        return pred_rgb, rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data):
        pred_rgb = self.model(data['x'])
        return pred_rgb

    # ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

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

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        all_preds = []

        with torch.no_grad():

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds = self.test_step(data)

                path = os.path.join(save_path, f'{name}_{i:04d}_rgb.png')

                # self.log(f"[INFO] saving test image to {path}")

                pred = np.moveaxis(preds[0].detach().cpu().numpy(), 0, -1) * 0.5 + 0.5
                pred = (pred * 255).astype(np.uint8)

                if write_image:
                    imageio.imwrite(path, pred)

                all_preds.append(pred)

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
            bar_format = '{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format=bar_format)

        self.local_step = 0

        for data in loader:
            self.local_step += loader.batch_size
            self.global_step += loader.batch_size

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
            bar_format = '{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format=bar_format)

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += loader.batch_size

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss = self.eval_step(data)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    # save_path_gt = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_gt.png')

                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = np.moveaxis(preds[0].detach().cpu().numpy(), 0, -1) * 0.5 + 0.5

                    cv2.imwrite(save_path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
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

        state = {'epoch': self.epoch, 'global_step': self.global_step, 'stats': self.stats}

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

    # noinspection PyBroadException
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
