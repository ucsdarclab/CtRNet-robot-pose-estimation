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

More details see `environment.yml`.

## Usage
- See `inference_single_frame.ipynb` for example single frame inference.
- We provide ROS node for CtRNet, which subscribes image and joint state topics and publishes robot pose.
```
python ros_node/panda_pose.py
```

## Dataset

1. [DREAM dataset](https://github.com/NVlabs/DREAM/blob/master/data/DOWNLOAD.sh)
2. [Baxter dataset](https://drive.google.com/file/d/12bCv6GBuh-FdvLGKjlUx2jPN-DBRUqUn/view?usp=share_link)

## Weights

Weights for Panda and Baxter can be found [here](https://drive.google.com/file/d/1OAamxl3_cMLdlpksNo0p-8K20fSMukbI/view?usp=share_link).


## Videos
Using CtRNet for visual servoing experiment with moving camera.



https://user-images.githubusercontent.com/22925036/230662315-5ca62a10-e6b5-4eee-8abc-c3bf2b0fbaae.mp4



https://user-images.githubusercontent.com/22925036/230662322-89b90a32-ca7b-4f64-a6ad-f9b807ddc08d.mp4

