import argparse
import os

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils, models
from tqdm import tqdm

from dataset import TorsoDataset
from networks.generator import StyleUNet
from augmentation import augment, AdaptiveAugment


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer
        for param in self.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss().to(device)
        
    def forward(self, source, target):
        loss = 0 
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)
        return loss


def train(args, loader, generator, g_optim):
    torch.manual_seed(0)
    loader = sample_data(loader)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.set_float32_matmul_precision('high')

    vgg = VGGLoss(args.device)

    pbar = tqdm(range(args.iter), initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, args.device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        inp, tgt, pose = next(loader)
        inp = inp.to(args.device)
        tgt = tgt.to(args.device)
        pose = pose.to(args.device)
        # latent = torch.randn(inp.shape[0], args.style_dim).to(args.device)

        if args.augment:
            tgt, aug_mat = augment(tgt, ada_aug_p, (None, None), use_affine=False, use_color=True)
            inp, _ = augment(inp, ada_aug_p, aug_mat, use_affine=False, use_color=True)

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(tgt.new_rand(tgt.shape[0]))

        out = generator(inp, pose)

        l1_loss = F.smooth_l1_loss(out, tgt) * args.l1_loss_w
        vgg_loss = vgg(out, tgt) * args.vgg_loss_w

        generator.zero_grad()
        (l1_loss + vgg_loss).backward()
        g_optim.step()

        l1_loss_val = l1_loss.mean().item()
        vgg_loss_val = vgg_loss.mean().item()

        pbar.set_description(f"l1: {l1_loss_val:.4f}; vgg: {vgg_loss_val:.4f} ")

        if i % 1000 == 0:  # and i != args.start_iter:
            with torch.no_grad():
                sample = generator(inp, pose).clamp(min=-1, max=1)
                utils.save_image(
                    torch.cat([sample, tgt], dim=3),
                    f"{args.path}/logs/sample/{args.savename}_{str(i).zfill(6)}.png",
                    nrow=int(args.batch ** 0.5),
                    normalize=True,
                    value_range=(-1, 1)
                )

        if i % 5000 == 0 and i != args.start_iter:
            torch.save(
                {
                    "g": generator.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "args": args
                },
                f"{args.path}/logs/checkpoint/{args.savename}_{str(i).zfill(6)}.pt",
            )

    # Save the final model
    torch.save(
        {
            "g": generator.state_dict(),
            "g_optim": g_optim.state_dict(),
            "args": args
        },
        f"{args.path}/logs/checkpoint/{args.savename}_final.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")
    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--iter", type=int, default=30000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=4, help="batch sizes for each gpus")
    parser.add_argument("--n_sample", type=int, default=4, help="number of the samples generated during training")
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the applying path length regularization")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
    parser.add_argument("--augment_p", type=float, default=0, help="probability of applying augmentation. 0 = use adaptive augmentation")
    parser.add_argument("--ada_target", type=float, default=0.6, help="target augmentation probability for adaptive augmentation")
    parser.add_argument("--ada_length", type=int, default=500 * 1000, help="target duraing to reach augmentation probability for adaptive augmentation")
    parser.add_argument("--ada_every", type=int, default=256, help="probability update interval of the adaptive augmentation")
    args = parser.parse_args()
    os.makedirs(f"{args.path}/logs/checkpoint", exist_ok=True)
    os.makedirs(f"{args.path}/logs/sample", exist_ok=True)

    args.l1_loss_w = 5
    args.vgg_loss_w = 0.03
    args.use_concat = False

    args.savename = 'ch'
    args.input_size = 128
    args.output_size = 512
    args.style_dim = 64
    args.n_mlp = 4
    args.device = "cuda"
    generator = StyleUNet(args.input_size, args.output_size, args.style_dim, args.n_mlp, args.channel_multiplier).to(args.device)

    transform_inp = transforms.Compose([transforms.ToTensor(), transforms.Resize(args.input_size), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    transform_out = transforms.Compose([transforms.ToTensor(), transforms.Resize(args.output_size), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    dataset = TorsoDataset(args.path, transform_inp, transform_out)

    args.start_iter = 0
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    g_optim = optim.Adam(generator.parameters(), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0].split('_')[-1])
            generator.load_state_dict(ckpt["g"], strict=False)
            g_optim.load_state_dict(ckpt["g_optim"])
        except:
            pass

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    train(args, loader, generator, g_optim)
