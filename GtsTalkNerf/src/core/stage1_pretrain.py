import argparse
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
import time
import torch
import torch.optim as optim
import torchaudio
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import util
from trans_aud import AudioFace, FormantFeatureExporter


def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []
    wav_path = Path(args.wav_path)
    lmk_path = Path(args.landmark_npy_path)
    base_lmks = pickle.load(open(args.bases_pkl_path, 'rb'), encoding='latin1')
    for wavfile in wav_path.glob('*.wav'):
        ffe = FormantFeatureExporter(wavfile, sr=16000)
        wav, sample_rate = torchaudio.load(f'{wavfile}')
        if sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, sample_rate, 16000)
        if wav.size(0) >= 2:
            wav = wav.mean(0, keepdim=True)
        wav = wav[0].float()
        key = wavfile.stem
        data[key]["wav"] = wav
        data[key]["name"] = key
        subject_id = key[:key.rfind('_')]
        lmk0 = torch.from_numpy(base_lmks[subject_id]).float()
        if lmk0.shape[-2:] == (68, 3):
            lmk0 = torch.movedim(lmk0, -1, -2).contiguous()
        data[key]["lmk0"] = lmk0.reshape(1, 3, 68)
        lmkfile = lmk_path / f'{key}.npy'  # 60fps
        if not lmkfile.exists():
            del data[key]
        else:
            lms = torch.from_numpy(np.load(f'{lmkfile}')).float()
            if lms.shape[-2:] == (68, 3):
                lms = torch.movedim(lms, -1, -2).contiguous()  # L, 3, 68
            data[key]['lms'] = lms
            fmt_frq, fmt_bw = ffe.formant()
            formant = torch.from_numpy(np.concatenate([fmt_frq, fmt_bw], 0)).float()  # 6, L
            formant = torch.nn.functional.interpolate(formant[None], size=lms.shape[0], mode='linear')[0]
            data[key]['formant'] = formant.t().contiguous()

    subjects_dict = {
        "train": ["FaceTalk_170728_03272_TA", "FaceTalk_170904_00128_TA", "FaceTalk_170725_00137_TA",
                  "FaceTalk_170915_00223_TA", "FaceTalk_170811_03274_TA", "FaceTalk_170913_03279_TA",
                  "FaceTalk_170904_03276_TA", "FaceTalk_170912_03278_TA"],
        "val": ["FaceTalk_170811_03275_TA", "FaceTalk_170908_03277_TA"],
        "test": ["FaceTalk_170809_00138_TA", "FaceTalk_170731_00024_TA"]
    }
    splits = {'train': range(1, 41), 'val': range(21, 41), 'test': range(21, 41)}
    for k, v in data.items():
        subject_id = k[:k.rfind('_')]
        sentence_id = int(k[-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits['test']:
            test_data.append(v)
    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict


def pair_data(data, seq_len):
    newdata = defaultdict(list)
    names = []
    for d in data:
        wav, name, lmk0, lms, formant = d["wav"], d["name"], d["lmk0"], d["lms"], d["formant"]
        frame_num = lms.size(0)
        name = name[:name.rfind('_')]
        if name not in names:
            names.append(name)
        pid = torch.tensor([names.index(name) + 1])
        wav_length = (640 * seq_len - 320, 640 * seq_len + 720)
        len_data = frame_num // seq_len
        dataset = {
            'wav': [wav[i * 640 * seq_len: i * 640 * seq_len + np.random.randint(*wav_length)] for i in
                    range(len_data)],
            'lms': [lms[i * seq_len: i * seq_len + seq_len] for i in range(len_data)],
            'formant': [formant[i * seq_len: i * seq_len + seq_len] for i in range(len_data)],
        }
        last_wav_num = wav[len_data * 640 * seq_len:].shape[0]
        if min((last_wav_num // 80 - 1) // 8, lms[len_data * seq_len:].shape[0]) >= seq_len // 4:
            dataset['wav'] += [wav[len_data * 640 * seq_len:]]
            dataset['lms'] += [lms[len_data * seq_len:]]
            dataset['formant'] += [formant[len_data * seq_len:]]
        if dataset['lms']:
            dataset['lmk0'] = [lmk0] * len(dataset['lms'])
            dataset['pid'] = [pid] * len(dataset['lms'])
            for k, v in dataset.items():
                newdata[k].extend(v)
        else:
            print(f"data {name} is too short. {wav.shape}, {lms.shape}")
    # return torch.utils.data.StackDataset(*(newdata.values()))
    return util.StackDataset(**newdata)


def train(args):
    device = args.device
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    workspace = Path(args.workspace)

    # dataset
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    traindata = pair_data(train_data, args.max_length)
    valdata = pair_data(valid_data, args.max_length)

    # model
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
        tqdm.write(f"epoch: {i}, {args.optimtype.lower()}_loss: {losses / len(traindataloader)}")

        if i % 10 == 0:
            torch.save(dict(model=audioface.state_dict(), config=audioface.config),
                       workspace / f'{i:03d}.tar')
        # val
        audioface.eval()
        with torch.inference_mode():
            data = next(iter(valdataloader))
            for k, v in data.items():
                data[k] = v.to(device)

            loss, _ = audioface(teacher_forcing=False, **data)
            summarywriter.add_scalar('ValLoss', loss, i)


def test(args):
    device = args.device
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    workspace = Path(args.workspace)
    # model
    model_weight = torch.load(sorted(workspace.glob("*.tar"), reverse=True)[0], map_location=device)
    audioface = AudioFace(**model_weight.pop('config')).to(device)
    audioface.load_state_dict(model_weight.pop('model'))
    audioface.eval()
    audioface.requires_grad_(False)
    # dataset
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    testdata = pair_data(test_data, args.max_length)
    testdataloader = DataLoader(testdata,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0 if os.name == 'nt' else args.nworkers)
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
            lms_pred = audioface.inference_forward(**data)  # 1, L, 3, 68
            save_lmk_pred[pid].append(lms_pred[0].detach())
            save_lmk_gt[pid].append(data['lms'][0])
            save_wav[pid].append(data['wav'])
        command = 'ffmpeg -i {0}.wav -i {0}.avi -c:a aac -c:v h264_nvenc -movflags +faststart -y {0}.mp4 -loglevel warning'
        os.chdir(str(savefolder))
        for pid in save_lmk_pred.keys():
            loss = 0
            anime = imageio.get_writer(f"{pid}_pred.avi", fps=25)
            torchaudio.save(f"{pid}_pred.wav", torch.cat(save_wav[pid], 1).cpu(), 16000)
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
            os.system(command.format(f'{pid}_pred'))
        os.chdir(curdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark_npy_path", type=str, default="data/voca/dataset/landmarks")
    parser.add_argument('--bases_pkl_path', type=str, default='data/voca/dataset/bases.pkl')
    parser.add_argument("--wav_path", type=str, default='data/voca/dataset/wav')
    parser.add_argument("-w", "--workspace", type=str, default='data/voca/pretrained')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('-d', '--feat_dim', type=int, default=64)
    parser.add_argument('-e', '--epoch', type=int, default=30)
    parser.add_argument('-l', '--lr', type=float, default=5e-5)
    parser.add_argument('-m', '--max_length', type=int, default=201)
    parser.add_argument('-n', '--nworkers', type=int, default=4)
    parser.add_argument('-o', '--optimtype', type=str, default='adamw')
    parser.add_argument('-t', '--teacher_force', type=util.str2bool, default=True)
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")

    args = parser.parse_args()
    if args.train:
        train(args)
    if args.test:
        test(args)
    elif not args.train:
        raise RuntimeError("No running mode set!")
