# 3DGNN for RGB-D segmentation
This is the Pytorch implementation of [3D Graph Neural Networks for RGBD Semantic Segmentation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qi_3D_Graph_Neural_ICCV_2017_paper.pdf): 

### Data Preparation
1. Download NYU_Depth_V2 dataset from [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and select scenes and save as `./datasets/data/nyu_depth_v2_labeled.mat`
2. Transfer depth images to hha by yourself from [here](https://github.com/charlesCXK/Depth2HHA) and save in `./datasets/data/hha/`. 

### Emviroment
Required CUDA (8.0) + pytorch 0.4.1


