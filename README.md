# Markerless Camera-to-Robot Pose Estimation via Self-supervised Sim-to-Real Transfer

PyTorch implementation of CtRNet https://arxiv.org/abs/2302.14332


### Dependencies
Recommend set up the environment using Anaconda.
Code is developed and tested on Ubuntu 20.04.
- Python(3.8)
- Numpy(1.22.4)
- PyTorch(1.10.0)
- torchvision(0.11.1)
- pytorch3d(0.6.2)
- Kornia(0.6.3)
- Transforms3d(0.3.1)

More details see "environment.yml".

## Dataset

1. [DREAM dataset](https://github.com/NVlabs/DREAM/blob/master/data/DOWNLOAD.sh)
2. [Baxter dataset](https://drive.google.com/file/d/12bCv6GBuh-FdvLGKjlUx2jPN-DBRUqUn/view?usp=share_link)

## Weights

Weights for Panda and Baxter can be found [here](https://drive.google.com/file/d/1OAamxl3_cMLdlpksNo0p-8K20fSMukbI/view?usp=share_link).

## Notes

This repo is work-in-progress. Please stay tuned.