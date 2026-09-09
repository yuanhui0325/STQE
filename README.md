### STQE

Official inference release of STQE for compressed point cloud quality enhancement.

### Installation

We recommend creating a dedicated Conda environment:

```bash
conda create -n STQE python=3.10.12 -y
conda activate STQE
```

We recommend creating a dedicated Conda environment:

conda create -n STQE python=3.10.12 -y
conda activate STQE

Install the main dependencies:

pip install torch==2.6.0
pip install numpy==1.23.5
pip install h5py==3.13.0
pip install open3d==0.19.0
pip install plyfile==1.1
pip install sewar==0.4.6
pip install scipy==1.15.2
pip install tqdm==4.67.1

Depending on your CUDA configuration, please install the corresponding PyTorch build following the official PyTorch installation instructions.

### Repository Structure

The public release is organized as follows:

STQE/
├── main.py
├── data.py
├── util.py
├── stqe_runtime.py
├── stqe_core.so
│
└── pretrained/
    ├── stqe_y.stqe
    ├── stqe_u.stqe
    └── stqe_v.stqe

### Pretrained Models

Three pretrained models are used to process the three YUV components independently:

pretrained/stqe_y.stqe
pretrained/stqe_u.stqe
pretrained/stqe_v.stqe

### Training and Inference
For inference, set '--eval' to True.

python main_mix_release.py
