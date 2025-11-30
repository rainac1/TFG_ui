# GtsTalkNeRF
**Decoupled Two-Stage Talking Head Generation via Gaussian-Landmark-Based Neural Radiance Fields**

[Boyao Ma](boyaoma@bit.edu.cn), Yuanping Cao, Lei Zhang

Beijing Institute of Technology

### Abstract
>Talking head generation based on neural radiance fields (NeRF) has gained prominence, primarily owing to its implicit 3D representation capability within neural networks. However, most NeRF-based methods often intertwine audio-to-video conversion in a joint training process, resulting in challenges such as inadequate lip synchronization, limited learning efficiency, large memory requirement and lack of editability. In response to these issues, this paper introduces a fully decoupled NeRF-based method for generating talking head. This method separates the audio-to-video conversion into two stages through the use of facial landmarks.
Notably, the Transformer network is used to establish the cross-modal connection between audio and landmarks effectively and generate landmarks conforming to the distribution of training data. We also explore formant features of the audio as additional conditions to guide landmark generation.
Then, these landmarks are combined with Gaussian relative position coding to refine the sampling points on the rays, thereby constructing a dynamic neural radiation field conditioned on these landmarks and audio features for rendering the generated head.
This decoupled setup enhances both the fidelity and flexibility of mapping audio to video with two independent small-scale networks. Additionally, it supports the generation of the torso part from the head-only image with improved StyleUnet, further enhancing the realism of the generated talking head. 
The experimental results demonstrate that our method excels in producing lifelike talking head, 和 the lightweight neural network models also exhibit superior speed and learning efficiency with less memory requirement.

## Data prepare

Save these code as a bash file and run it, you may need to have FLAME and VOCA accounts:
```bash
#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# username and password input
echo -e "\nIf you do not have an account you can register at https://flame.is.tue.mpg.de/ following the installation instruction."
read -p "Username (FLAME):" username
read -p "Password (FLAME):" password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading FLAME..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -O './FLAME2020.zip' --no-check-certificate -c
unzip FLAME2020.zip -d data/FLAME2020/
mv data/FLAME2020/Readme.pdf data/FLAME2020/Readme_FLAME.pdf
rm -rf FLAME2020.zip

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=TextureSpace.zip&resume=1' -O './TextureSpace.zip' --no-check-certificate -c
unzip TextureSpace.zip -d data/FLAME2020/
rm -rf TextureSpace.zip

echo -e "\nDownloading pretrained weight..."
mkdir -p data/pretrained/face_parsing
wget -O data/pretrained/mica.tar "https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c0f96de/?dl=1"
wget -O data/face_parsing/79999_iter.pth "https://keeper.mpdl.mpg.de/f/a3c400dc55b84b10a7d1/?dl=1"

# https://github.com/deepinsight/insightface/issues/1896
# Insightface has problems with hosting the models
echo -e "\nDownloading insightface models..."
mkdir -p ~/.insightface/models/
if [ ! -d ~/.insightface/models/antelopev2 ]; then
  wget -O ~/.insightface/models/antelopev2.zip "https://keeper.mpdl.mpg.de/f/2d58b7fed5a74cb5be83/?dl=1"
  unzip ~/.insightface/models/antelopev2.zip -d ~/.insightface/models/antelopev2
fi
if [ ! -d ~/.insightface/models/buffalo_l ]; then
  wget -O ~/.insightface/models/buffalo_l.zip "https://keeper.mpdl.mpg.de/f/8faabd353cfc457fa5c5/?dl=1"
  unzip ~/.insightface/models/buffalo_l.zip -d ~/.insightface/models/buffalo_l
fi

echo -e "\nIf you do not have an account you can register at https://voca.is.tue.mpg.de/ following the installation instruction."
read -p "Username (VOCA):" username
read -p "Password (VOCA):" password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading VOCA trainingdata..."
mkdir -p data/voca/
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=voca&sfile=trainingdata.zip&resume=1' -O './trainingdata.zip' --no-check-certificate -c
unzip trainingdata.zip -d data/voca/
rm -rf trainingdata.zip
rm data/voca/init_expression_basis.npy data/voca/processed_audio_deepspeech.pkl data/voca/readme.pdf
```

## Environment

### Python environment
Using the environment definition file:
```bash
# cuda >= 12.1, cudnn >= 9.2.1
conda env create -f environment.yml
pip install chumpy==0.70 lpips sounddevice==0.4.7 dearpygui albumentations==1.3.1 face-alignment insightface==0.7.2 mediapipe==0.10.10 --no-deps
```
or install by yourself:
```bash
# note to install python < 3.11
conda install numpy=1.23 numba scikit-image scikit-learn tqdm matplotlib tensorboard ninja rich cython onnx prettytable==3.5.0 loguru fsspec attrs onnxruntime==1.17.1 -y
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install iopath fvcore pytorch3d librosa torch-ema trimesh einops imageio-ffmpeg easydict -c conda-forge -c pytorch3d -y

# conda search onnxruntime=1.17.1=*cuda opencv=4.9.0=headless* -c conda-forge
conda install qudida=0.0.4 opencv=4.9.0=headless_py310h18fe71b_15 onnxruntime=1.17.1=py310hf79c3c9_201_cuda -c conda-forge -y
# or pip install qudida==0.0.4 opencv-contrib-python-headless==4.9.0.80 onnxruntime-gpu==1.17.1 --no-deps

pip install chumpy==0.70 lpips sounddevice==0.4.7 dearpygui albumentations==1.3.1 face-alignment insightface==0.7.2 mediapipe==0.10.10 --no-deps
conda uninstall ffmpeg --force
sudo apt-get install ffmpeg
```
### Get glm
```bash
wget https://github.com/g-truc/glm/releases/download/1.0.1/glm-1.0.1-light.zip
unzip glm-1.0.1-light.zip -d gsencoder/glm
rm glm-1.0.1-light.zip
```

### Build styleunet
```bash
cd styleunet/networks/stylegan2_ops
python setup.py install
```

## Traing

```bash
python process_voca_data.py
python stage1_pretrain.py --train --test
cd process_data
python process.py -f ?.mp4
cd ..
python stage1_finetune.py --train --test -w results/?
python main.py results/?/ --workspace results/?/logs/stage2 -O --iters 50000
python main.py results/?/ --workspace results/?/logs/stage2 -O --iters 20000 --finetune_eyes
python main.py results/?/ --workspace results/?/logs/stage2 -O --iters 40000 --finetune_lips
python styleunet/stage.py results/?
```

## Acknowledgement & License
The code is partially borrowed from [StyleAvatar](https://github.com/LizhenWangT/StyleAvatar), [metrical-tracker](https://github.com/Zielon/metrical-tracker) and [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF). And many thanks to the volunteers participated in data collection. Our License can be found in [LICENSE](./LICENSE). Thanks to the authors of these open source libraries.
