# 3DGNN for RGB-D segmentation
This is the Pytorch implementation of [3D Graph Neural Networks for RGBD Semantic Segmentation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qi_3D_Graph_Neural_ICCV_2017_paper.pdf): 

![](https://github.com/xjqicuhk/3DGNN/blob/master/overallpipeline.png)

### Data Preparation
1. Download the prepared training data (prepared hdf5 data) (https://mycuhk-my.sharepoint.com/:u:/g/personal/1155051740_link_cuhk_edu_hk/EVGJ_xXvtNVCh7spid94AmQB_byhW49i-VH_vqx8oZbrZQ?e=COhKwr).
2. Download the testing data  (https://mycuhk-my.sharepoint.com/:u:/g/personal/1155051740_link_cuhk_edu_hk/EVdjeNQqnINOj359HN8WXDgBsouAqSoZC1lRgkSbPNo2hA?e=e0w2sO).
3. Download the original provided data (https://mycuhk-my.sharepoint.com/:u:/g/personal/1155051740_link_cuhk_edu_hk/EZuJHYVcULRNkQ3qm34ugIoBg-69Vprq2POiaat4u5ZLXQ?e=QmWXec).
4. You need transfer depth images to hha by yourself and save in `datasets/data/hha/`. [Tools](https://github.com/charlesCXK/Depth2HHA)

### Emviroment
Required CUDA (8.0) + pytorch 0.4.1


