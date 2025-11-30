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


def test(args, loader, generator):
    torch.manual_seed(0)
    length = len(loader)
    loader = iter(loader)
    # loader = sample_data(loader)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.set_float32_matmul_precision('high')

    # vgg = VGGLoss(args.device)

    pbar = tqdm(range(length), initial=0, dynamic_ncols=True, smoothing=0.01)
    imid = 0

    for idx in pbar:
        i = idx #+ args.start_iter

        if i > length:
            print("Done!")
            break

        inp, tgt, pose = next(loader)
        inp = inp.to(args.device)
        # tgt = tgt.to(args.device)
        pose = pose.to(args.device)

        out = generator(inp, pose).clamp(min=-1, max=1)
        for o in out:
            utils.save_image(
                o,
                f"{args.path}/logs/res2/{str(imid).zfill(4)}.png",
                normalize=True,
                value_range=(-1, 1)
            )
            imid += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")
    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--batch", type=int, default=4, help="batch sizes for each gpus")
    parser.add_argument("--ckpt", type=str, default="logs/checkpoint/ch_025000.pt", help="path to the checkpoints to resume training")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    args = parser.parse_args()
    os.makedirs(f"{args.path}/logs/checkpoint", exist_ok=True)
    os.makedirs(f"{args.path}/logs/sample", exist_ok=True)

    args.l1_loss_w = 5
    args.vgg_loss_w = 0.03
    args.use_concat = False

    args.savename = 'ch'
    args.input_size = 512
    args.output_size = 512
    args.style_dim = 64
    args.n_mlp = 4
    args.device = "cuda"
    generator = StyleUNet(args.input_size, args.output_size, args.style_dim, args.n_mlp, args.channel_multiplier).to(args.device)

    transform_inp = transforms.Compose([transforms.ToTensor(), transforms.Resize(args.input_size), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    transform_out = transforms.Compose([transforms.ToTensor(), transforms.Resize(args.output_size), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    dataset = TorsoDataset(args.path, transform_inp, transform_out)
    args.start_iter = 0

    if args.ckpt is not None:
        args.ckpt = os.path.join(args.path, args.ckpt)
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0].split('_')[-1])
            generator.load_state_dict(ckpt["g"], strict=False)
        except:
            pass

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, drop_last=False)

    test(args, loader, generator)
