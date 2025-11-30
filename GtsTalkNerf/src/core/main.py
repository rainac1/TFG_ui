import torch
import argparse

from nerf_lmk.provider import NeRFDataset
from nerf_lmk.utils import *
from nerf_lmk.network import NeRFNetwork

# torch.autograd.set_detect_anomaly(True)
# Close tf32 features. Fix low numerical accuracy on rtx30xx gpu.
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except AttributeError as e:
    print('Info. This pytorch version is not support with tf32.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray")
    parser.add_argument('--test', action='store_true', help="test mode (load model and test dataset)")
    parser.add_argument('--test_train', action='store_true', help="test mode (load model and train dataset)")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    # training options
    parser.add_argument('--iters', type=int, default=200_000, help="training iters")
    parser.add_argument('--lr', type=float, default=5e-3, help="initial learning rate")
    parser.add_argument('--lr_net', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096,
                        help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=16,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    # parser.add_argument('--num_steps', type=int, default=16,
    #                     help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    # parser.add_argument('--upsample_steps', type=int, default=0,
    #                     help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    # parser.add_argument('--max_ray_batch', type=int, default=4096,
    #                     help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    # network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    parser.add_argument('--lambda_amb', type=float, default=0.1, help="lambda for ambient loss")

    parser.add_argument('--bg_img', type=str, default='', help="background image")
    # parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
    # parser.add_argument('--fix_eye', type=float, default=-1,
    #                     help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")

    # dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0,
                        help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3,"
                                                               " if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--dt_gamma', type=float, default=1 / 256, help="dt_gamma (>=0) for adaptive ray marching."
                        " set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10,
                        help="threshold for density grid to be occupied (sigma)")
    parser.add_argument('--patch_size', type=int, default=64, help="[experimental] render patches in training,"
                        " so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    parser.add_argument('--aud', type=str, default='',
                        help="audio source (empty will load the default, else should be a path to a npy file)")
    parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
    parser.add_argument('--finetune_eyes', action='store_true', help="use eye mask to fine tune eye region")

    parser.add_argument('--head_ckpt', type=str, default='', help="head model")

    parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=10000,
                        help="number of individual codes, should be larger than training dataset size")

    parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
    parser.add_argument('--smooth_path', action='store_true',
                        help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=3, help="smoothing window size")

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True

    if opt.test:
        opt.smooth_path = True
        opt.smooth_eye = True

    opt.cuda_ray = True
    # assert opt.cuda_ray, "Only support CUDA ray mode."

    if opt.patch_size > 1:
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    if opt.finetune_lips or opt.finetune_eyes:
        # do not update density grid in finetune stage
        opt.update_extra_interval = 1e9

    # print(opt)

    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opt.path)
    model = NeRFNetwork(opt)
    # print(model)

    criterion = torch.nn.SmoothL1Loss(reduction="none")

    if opt.test:
        # metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        metrics = [PSNRMeter(), LPIPSMeter(device=device), LMDMeter(backend='fan')]

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16,
                          metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.test_train:
            test_set = NeRFDataset(opt, device=device, mode='train')
            # a manual fix to test on the training dataset
            test_set.training = False
            test_set.num_rays = -1
            test_loader = test_set.dataloader()
        else:
            test_loader = NeRFDataset(opt, device=device, mode='test').dataloader()

        # test and save video (fast)
        trainer.test(test_loader)

    else:
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr, opt.lr_net), betas=(0.9, 0.99), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, mode='train').dataloader()

        assert len(train_loader) < opt.ind_num, f"[ERROR] dataset too many frames: {len(train_loader)},"\
                                                " please increase --ind_num to this number!"

        # temp fix: for update_extra_states
        model.init_state(train_loader._data.lmk0, train_loader._data.R0, train_loader._data.a0, train_loader._data.aabb)

        # decay to 0.1 * init_lr at last iter step
        if opt.finetune_lips or opt.finetune_eyes:
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda it: 0.05 ** (it / opt.iters))
        else:
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda it: 0.1 ** (it / opt.iters))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]

        eval_interval = max(1, int(5000 / len(train_loader)))
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                          criterion=criterion, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt,
                          eval_interval=eval_interval)

        valid_loader = NeRFDataset(opt, device=device, mode='val').dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        print(f'[INFO] max_epoch = {max_epoch}')
        trainer.train(train_loader, valid_loader, max_epoch)

        # free some mem
        del valid_loader, train_loader
        torch.cuda.empty_cache()

        # also test
        opt.smooth_path = True
        test_set = NeRFDataset(opt, device=device, mode='all')
        test_set.training = False
        test_set.num_rays = -1
        trainer.test(test_set.dataloader(), write_image=opt.finetune_lips)
        