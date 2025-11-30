import os
import argparse
import pickle
import numpy as np
import torch.nn.functional
import trimesh
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import resample
from tqdm import tqdm


def generate_vertices_npy(args, face_vert_mmap, data2array_verts, lmk_faces, lmk_bary_coords):
    total_num = sum([len(v) for v in data2array_verts.values()])
    with tqdm(total=total_num) as pbar:
        for sub in data2array_verts.keys():
            for seq in data2array_verts[sub].keys():
                vertices_npy_name = sub + "_" + seq
                vertices_npy = []
                landmark_npy = []
                for frame, array_idx in data2array_verts[sub][seq].items():
                    vertice = face_vert_mmap[array_idx]
                    vertices_npy.append(vertice)
                    landmark_npy.append(np.einsum('lfi,lf->li', vertice[lmk_faces], lmk_bary_coords))
                # vertices_npy = np.array(vertices_npy).reshape(-1, args.vertices_dim)
                landmark_npy = np.array(landmark_npy)
                new_landmark_npy = resample(landmark_npy, round(landmark_npy.shape[0] * 25 / 60))
                # np.save(os.path.join(args.vertices_npy_path, vertices_npy_name), vertices_npy)
                np.save(os.path.join(args.landmark_npy_path, vertices_npy_name), new_landmark_npy)
                pbar.update(1)


def generate_wav(args, raw_audio):
    total_num = sum([len(v) for v in raw_audio.values()])
    with tqdm(total=total_num) as pbar:
        for sub in raw_audio.keys():
            for seq in raw_audio[sub].keys():
                wav_name = sub + "_" + seq
                olddata = raw_audio[sub][seq]['audio']
                newdata = resample(olddata, int(round(len(olddata) * 16000 / raw_audio[sub][seq]['sample_rate'])))
                wavfile.write(os.path.join(args.wav_path, wav_name + '.wav'), 16000, np.round(newdata).astype(np.int16))
                pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertices_dim", type=int, default=5023 * 3)
    parser.add_argument("--verts_path", type=str, default="data/voca/data_verts.npy")
    parser.add_argument("--raw_audio_path", type=str, default='data/voca/raw_audio_fixed.pkl')
    parser.add_argument('--templates_pkl_path', type=str, default='data/voca/templates.pkl')
    parser.add_argument("--data2array_verts_path", type=str, default='data/voca/subj_seq_to_idx.pkl')

    # parser.add_argument("--vertices_npy_path", type=str, default="data/voca/dataset/data_verts")
    parser.add_argument("--landmark_npy_path", type=str, default="data/voca/dataset/landmarks")
    parser.add_argument('--bases_pkl_path', type=str, default='data/voca/dataset/bases.pkl')
    parser.add_argument("--wav_path", type=str, default='data/voca/dataset/wav')

    parser.add_argument('--landmark_embedding_npy_path', type=str, default='data/FLAME2020/landmark_embedding.npy')
    parser.add_argument('--template_obj_path', type=str, default='data/FLAME2020/head_template_color.obj')

    args = parser.parse_args()

    face_vert_mmap = np.load(args.verts_path, mmap_mode='r+')
    raw_audio = pickle.load(open(args.raw_audio_path, 'rb'), encoding='latin1')
    templates = pickle.load(open(args.templates_pkl_path, 'rb'), encoding='latin1')
    data2array_verts = pickle.load(open(args.data2array_verts_path, 'rb'))

    # Path(args.vertices_npy_path).mkdir(exist_ok=True, parents=True)
    Path(args.landmark_npy_path).mkdir(exist_ok=True, parents=True)
    Path(args.wav_path).mkdir(exist_ok=True, parents=True)
    faces = trimesh.load(args.template_obj_path).faces
    landmark_embedding = np.load(args.landmark_embedding_npy_path, allow_pickle=True, encoding='latin1')[()]
    lmk_faces = faces[landmark_embedding['full_lmk_faces_idx'].reshape(-1)]
    lmk_bary_coords = landmark_embedding['full_lmk_bary_coords'].reshape(-1, 3)
    base_landmarks = {}
    for k, v in templates.items():
        base_landmarks[k] = np.einsum('lfi,lf->li', v[lmk_faces], lmk_bary_coords)

    with open(args.bases_pkl_path, 'wb') as f:
        pickle.dump(base_landmarks, f)

    generate_vertices_npy(args, face_vert_mmap, data2array_verts, lmk_faces, lmk_bary_coords)
    generate_wav(args, raw_audio)
