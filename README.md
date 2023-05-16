# Self-Supervised Marine Organism Detection from Underwater Images
We release the code of Self-Supervised Marine Organism Detection from Underwater Images in our papers:
### Introduction
We design an Self-Supervised Marine Organism Detection framework for underwater images to detect organisms. We propose a set of underwater image augmentation strategies to enhance the quality of representation we learned, besides, we also propose a novel Underwater Attention module to explore effective underwater representation for marine organism detection. This code is based on the Simsiam and mmdetection codebase (v2.13.0).
![Image text](https://github.com/lenka844/SSLMarineOrgnismDET/blob/main/fig.png)
### Requirements
-Linux or macOS (Windows is in experimental support)
-Python 3.8+
-PyTorch 1.9+
-CUDA 10.2+ (If you build PyTorch from source, CUDA 10.0 is also compatible)
-GCC 5+
-MMCV
### Datasets
HabCam, TRASH ICRA-2019, UIEBD, RUIE, URPC2021.
-Contact dataset provider, and download the datasets and annotations.
-Put all images and annotation files to ./data folder.
### Running
-To train a SSL Method on Pretext dataset, you can run the script:
```
CUDA_VISIBLE_DEVICES=0,1 python DDP_simsiam_ccrop_pretrain.py configs/small/underwater/simsiam_ccrop_pretrain.py
```
-You can also set the variables (CONFIG_FILE, GPU_NUM) in pretrain.sh and configs/small/underwater/simsiam_ccrop_pretrain.py, and then run the script:
```
bash pretrain.sh
```
-After pretext task training, you can run the folllowing code to transfer the weight file to downstream task：
```
python self-weight_converter_pre.py
```
-Besides, the downstream task is training on the mmdetection version 2.13.0.
### Models
We will provide the models and results later.
### Acknowledgement
We really appreciate the contributors of following codebases.

[open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)

[facebookresearch/simsiam](https://github.com/facebookresearch/simsiam)
