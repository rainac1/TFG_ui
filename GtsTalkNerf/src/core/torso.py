import torch
import argparse

from unet import *
# torch.autograd.set_detect_anomaly(True)
# Close tf32 features. Fix low numerical accuracy on rtx30xx gpu.
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
except AttributeError as e:
    print('Info. This pytorch version is not support with tf32.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16")
    parser.add_argument('--test', action='store_true', help="test mode (load model and test dataset)")
    parser.add_argument('--test_train', action='store_true', help="test mode (load model and train dataset)")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--seed', type=int, default=0)

    # training options
    parser.add_argument('--iters', type=int, default=100_000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    # dataset options
    parser.add_argument('--preload', type=int, default=0,
                        help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    opt = parser.parse_args()
    opt.workspace = os.path.join(opt.path, 'unet')

    if opt.O:
        opt.fp16 = True

    # print(opt)

    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opt.path)
    model = UNet()
    # print(model)

    criterion = torch.nn.SmoothL1Loss()

    if opt.test:
        # metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]

        trainer = UNetTrainer('unet', opt, model, device=device, workspace=opt.workspace, criterion=criterion,
                              fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.test_train:
            test_set = UNetDataset(opt, device=device, batch_size=1, mode='train')
            # a manual fix to test on the training dataset
            test_set.training = False
            test_set.num_rays = -1
            test_loader = test_set.dataloader()
        else:
            test_loader = UNetDataset(opt, device=device, batch_size=1, mode='test').dataloader()

        # test and save video (fast)
        trainer.test(test_loader)

    else:
        optimizer = lambda model: torch.optim.AdamW(model.parameters(), opt.lr, betas=(0.9, 0.99), eps=1e-15)

        train_loader = UNetDataset(opt, device=device, batch_size=opt.batch_size, mode='train').dataloader()

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda it: 0.1 ** (
                it * train_loader.batch_size / opt.iters))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]

        eval_interval = max(1, int(5000 / len(train_loader) / train_loader.batch_size))
        trainer = UNetTrainer('unet', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                              criterion=criterion, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler,
                              scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt,
                              eval_interval=eval_interval)

        valid_loader = UNetDataset(opt, device=device, batch_size=1, mode='val').dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader) / train_loader.batch_size).astype(np.int32)
        print(f'[INFO] max_epoch = {max_epoch}')
        trainer.train(train_loader, valid_loader, max_epoch)

        # free some mem
        del valid_loader, train_loader
        torch.cuda.empty_cache()

        # also test
        test_set = UNetDataset(opt, device=device, batch_size=1, mode='all')
        test_set.training = False
        trainer.test(test_set.dataloader())
