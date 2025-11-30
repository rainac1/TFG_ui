import argparse
import imageio.v2 as imageio
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time
import torch
import torch.optim as optim
import torchaudio
from collections import defaultdict
from glob import glob
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

import util
from trans_aud import AudioFace, FormantFeatureExporter


def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    workspace = Path(args.workspace)
    wavfile = workspace / f'{workspace.stem}.wav'

    frames = {'train': json.load(open(workspace / 'transforms_train.json'))['frames'],
              'val': json.load(open(workspace / 'transforms_val.json'))['frames'],
              'test': json.load(open(workspace / 'transforms_test.json'))['frames']}
    total_frame = len(frames['train']) + len(frames['val']) + len(frames['test'])

    lmk0 = torch.tensor(json.load(open(workspace / 'transforms_train.json'))['base_landmark']).float()
    if lmk0.shape[-2:] == (68, 3):
        lmk0 = torch.movedim(lmk0, -1, -2).contiguous()

    ffe = FormantFeatureExporter(wavfile, sr=16000)
    fmt_frq, fmt_bw = ffe.formant()
    formant = torch.from_numpy(np.concatenate([fmt_frq, fmt_bw], 0)).float()  # 6, 2L
    formant = torch.nn.functional.interpolate(formant[None], size=total_frame, mode='linear')[0].t().contiguous()
    wav, sample_rate = torchaudio.load(str(wavfile))
    if sample_rate != 16000:
        wav = torchaudio.functional.resample(wav, sample_rate, 16000)
    if wav.size(0) >= 2:
        wav = wav.mean(0, keepdim=True)
    wav = wav[0].float()

    data["train"]["wav"] = wav[:wav.size(-1)//16*14]
    data["test"]["wav"] = wav[wav.size(-1)//16*14:wav.size(-1)//16*15]
    data["val"]["wav"] = wav[wav.size(-1)//16*15:]
    data["total"]["wav"] = wav

    data["train"]['formant'] = formant[:len(frames['train'])]
    data["test"]['formant'] = formant[len(frames['train']):-len(frames['val'])]
    data["val"]['formant'] = formant[-len(frames['val']):]
    data["total"]['formant'] = formant

    for type in ["train", "test", "val"]:
        torchaudio.save(str(workspace / f"{workspace.stem}_{type}.wav"), data[type]["wav"][None], 16000)
        data[type]["lmk0"] = lmk0.reshape(1, 3, 68)
        lms = torch.tensor([frame['landmarks'] for frame in frames[type]]).float()
        if lms.shape[-2:] == (68, 3):
            lms = torch.movedim(lms, -1, -2).contiguous()  # L, 3, 68
        data[type]['lms'] = lms

    data["total"]["lmk0"] = lmk0.reshape(1, 3, 68)
    lms = torch.tensor([frame['landmarks'] for frame in sum(frames.values(), [])]).float()
    if lms.shape[-2:] == (68, 3):
        lms = torch.movedim(lms, -1, -2).contiguous()  # L, 3, 68
    data["total"]['lms'] = lms

    return data['train'], data['val'], data['test'], data["total"]


def pair_data(data, seq_len, is_train=True):
    wav, lmk0, formant = data["wav"], data["lmk0"], data["formant"]
    frame_num = formant.shape[0]
    pid = torch.tensor([0])
    wav_length = (640 * seq_len - 320, 640 * seq_len + 720)
    len_data = frame_num // seq_len
    dataset = {
        'wav': [wav[i * 640 * seq_len: i * 640 * seq_len + np.random.randint(*wav_length)] for i in
                range(len_data)],
        'formant': [formant[i * seq_len: i * seq_len + seq_len] for i in range(len_data)],
    }
    last_wav_num = wav[len_data * 640 * seq_len:].shape[0]
    if (last_wav_num // 80 - 1) // 8 >= seq_len // 4 or not is_train:
        dataset['wav'] += [wav[len_data * 640 * seq_len:]]
        dataset['formant'] += [formant[len_data * seq_len:]]
    if 0 < (last_wav_num // 80 - 1) // 8 < seq_len // 4 and is_train:
        print("Warning, the last group of audio is too short")

    dataset['lmk0'] = [lmk0] * len(dataset['formant'])
    dataset['pid'] = [pid] * len(dataset['formant'])
    if is_train:
        lms = data["lms"]
        dataset['lms'] = [lms[i * seq_len: i * seq_len + seq_len] for i in range(len_data)]
        if (last_wav_num // 80 - 1) // 8 >= seq_len // 4:
            dataset['lms'] += [lms[len_data * seq_len:]]

    return util.StackDataset(**dataset)


def train(args):
    device = args.device
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    workspace = Path(args.workspace, 'logs/stage1')
    pretrained = None
    if args.pretrained:
        if os.path.isdir(args.pretrained):
            pretrained = sorted(glob(f'{args.pretrained}/*.tar'))[-1]
        elif os.path.isfile(args.pretrained) and args.pretrained.endswith('.tar'):
            pretrained = args.pretrained

    # dataset
    train_data, valid_data, test_data, total_data = read_data(args)
    traindata = pair_data(train_data, args.max_length)
    valdata = pair_data(valid_data, args.max_length)

    # model
    if pretrained:
        print(f"Loading {pretrained}")
        weight = torch.load(pretrained)
        args.max_length = weight['config']['max_length']
        args.feat_dim = weight['config']['emb_dim']
        audioface = AudioFace(**weight['config']).to(device)
        audioface.load_state_dict(weight['model'])
    else:
        audioface = AudioFace(args.feat_dim, max_length=args.max_length).to(device)

    # optimizer
    params = audioface.parameters()
    if args.optimtype.lower() == "sgd":
        optimizer = optim.SGD(params=params, lr=args.lr, momentum=0.9)
    elif args.optimtype.lower() == "adam":
        optimizer = optim.Adam(params=params, lr=args.lr, eps=1e-15)
    elif args.optimtype.lower() == "adamw":
        optimizer = optim.AdamW(params=params, lr=args.lr, eps=1e-15)
    else:
        raise ValueError(f"Unknown Optimizer {args.optimtype.lower()}")
    if args.decay > 0:
        ema = ExponentialMovingAverage(params, decay=args.decay)
    else:
        ema = None

    # train
    current_time = time.strftime("%Y-%m-%dT%H_%M", time.localtime())
    log_dir = workspace / f'{args.optimtype.lower()}_{current_time}'
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True)
    summarywriter = SummaryWriter(log_dir=f'{log_dir}')
    traindataloader = DataLoader(traindata,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=0 if os.name == 'nt' else args.nworkers)
    valdataloader = DataLoader(valdata,
                               batch_size=1,
                               shuffle=True,
                               num_workers=0 if os.name == 'nt' else args.nworkers)
    total = args.epoch * len(traindataloader)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda it: (1 - it / total)**2)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda it: 0.01**(it / total))
    step = 0
    pbar = tqdm(total=total)
    for i in range(1, args.epoch + 1):
        audioface.train()
        losses = 0
        for data in traindataloader:
            for k, v in data.items():  # wav, lms, lmk0, formant, pid
                data[k] = v.to(device)
            loss, _ = audioface(teacher_forcing=args.teacher_force, **data)
            summarywriter.add_scalar('Loss', loss, step)
            summarywriter.add_scalar('LR', scheduler.get_last_lr()[0], step)
            losses += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            scheduler.step()
            pbar.update()
        if ema is not None:
            ema.update()

        if i % 10 == 0:
            torch.save(dict(model=audioface.state_dict(), config=audioface.config),
                       workspace / f'{i:03d}.tar')
        # val
        audioface.eval()
        if ema is not None:
            ema.store()
            ema.copy_to()
        with torch.inference_mode():
            data = next(iter(valdataloader))
            for k, v in data.items():
                data[k] = v.to(device)

            loss, _ = audioface(teacher_forcing=False, **data)
            summarywriter.add_scalar('ValLoss', loss, i)
        tqdm.write(f"epoch: {i}, {args.optimtype.lower()}_loss: {losses / len(traindataloader)}, Val_loss: {loss.item()}")
        if ema is not None:
            ema.restore()


def test(args):
    device = args.device
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    workspace = Path(args.workspace, 'logs/stage1')
    # model
    model_weight = torch.load(sorted(workspace.glob("*.tar"), reverse=True)[0], map_location=device)
    audioface = AudioFace(**model_weight.pop('config')).to(device)
    audioface.load_state_dict(model_weight.pop('model'))
    audioface.eval()
    audioface.requires_grad_(False)
    # dataset
    train_data, valid_data, test_data, total_data = read_data(args)
    testdata = pair_data(test_data, args.max_length)
    testdataloader = DataLoader(testdata,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0 if os.name == 'nt' else args.nworkers)
    totaldata = pair_data(total_data, args.max_length, False)
    totaldataloader = DataLoader(totaldata,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0 if os.name == 'nt' else args.nworkers)
    # save audio feature
    audfeats = []
    with torch.inference_mode():
        for data in totaldataloader:
            wav = data['wav'].to(device)
            formant = data['formant'].to(device)
            audfeats.append(audioface.audenc(wav, formant)[0])
    audfeats = torch.cat(audfeats, dim=0)
    # print(audfeats.shape)
    np.save(os.path.join(args.workspace, 'aud_af.npy'), audfeats.cpu().numpy())
    del wav, formant, audfeats
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    # forsave
    savefolder = workspace / 'lmk_pred'
    savefolder.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5.12, 5.12))
    ax = fig.gca()
    ax.axis('off')
    # run
    curdir = os.path.abspath(os.curdir)
    with torch.inference_mode():
        save_lmk_pred = defaultdict(list)
        save_lmk_gt = defaultdict(list)
        save_wav = defaultdict(list)
        for data in testdataloader:
            for k, v in data.items():  # wav, lms, lmk0, formant, pid
                data[k] = v.to(device)
            pid = data['pid'][0, 0].item()
            lms_pred = audioface.inference_forward(**data)  # 1, L, 3, 68 and 1, L, D
            save_lmk_pred[pid].append(lms_pred[0].detach())
            save_lmk_gt[pid].append(data['lms'][0])
            save_wav[pid].append(data['wav'])
        command = 'ffmpeg -i {0}.wav -i {0}.avi -c:a aac -c:v h264_nvenc -movflags +faststart -y {0}.mp4 -loglevel warning'
        os.chdir(str(savefolder))
        for pid in save_lmk_pred.keys():
            loss = 0
            anime = imageio.get_writer(f"pred.avi", fps=25)
            torchaudio.save(f"pred.wav", torch.cat(save_wav[pid], 1).cpu(), 16000)
            xl, yl, _ = save_lmk_gt[pid][0].amin([0, 2]).cpu().numpy()
            xr, yr, _ = save_lmk_gt[pid][0].amax([0, 2]).cpu().numpy()
            ax.set_xlim(xl * 1.1, xr * 1.1)
            ax.set_ylim(yl * 1.1, yr * 1.1)
            for sid in range(len(save_lmk_pred[pid])):
                lmk_pred = save_lmk_pred[pid][sid].cpu()  # L, 3, 68
                lmk_gt = save_lmk_gt[pid][sid].cpu()  # L, 3, 68
                loss += torch.nn.functional.smooth_l1_loss(lmk_pred, lmk_gt)
                for pred, gt in zip(lmk_pred.numpy(), lmk_gt.numpy()):
                    ax.plot(*gt[:2], 'ro')
                    ax.plot(*pred[:2], 'bo')
                    anime.append_data(util.render_to_rgb(fig, close=False))
                    ax._children = []
            anime.close()
            os.system(command.format(f'pred'))
        os.chdir(curdir)
        print(loss)


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    # Close tf32 features. Fix low numerical accuracy on rtx30xx gpu.
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except AttributeError as e:
        print('Info. This pytorch version is not support with tf32.')
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default='data/voca/pretrained')
    parser.add_argument("-w", "--workspace", type=str, default='.')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('-d', '--feat_dim', type=int, default=64)
    parser.add_argument('-e', '--epoch', type=int, default=30)
    parser.add_argument('-l', '--lr', type=float, default=1e-4)
    parser.add_argument('-m', '--max_length', type=int, default=201)
    parser.add_argument('-n', '--nworkers', type=int, default=4)
    parser.add_argument('-o', '--optimtype', type=str, default='adamw')
    parser.add_argument('-t', '--teacher_force', type=util.str2bool, default=False)
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--decay', type=float, default=0.9)

    args = parser.parse_args()
    print(args.decay, args.lr)
    if args.train:
        train(args)
    if args.test:
        test(args)
    elif not args.train:
        raise RuntimeError("No running mode set!")
