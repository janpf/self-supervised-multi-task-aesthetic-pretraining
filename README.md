# Self-Supervised Multi-Task Pretraining Improves Image Aesthetic Assessment

In this repository you find the code and data for the paper ["Self-Supervised Multi-Task Pretraining Improves Image Aesthetic Assessment"](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Pfister_Self-Supervised_Multi-Task_Pretraining_Improves_Image_Aesthetic_Assessment_CVPRW_2021_paper.html) published at [NTIRE21](https://data.vision.ee.ethz.ch/cvl/ntire21/):

> Neural networks for Image Aesthetic Assessment are usually initialized with weights of pretrained ImageNet models and then trained using a labeled image aesthetics dataset. We argue that the ImageNet classification task is not well-suited for pretraining, since content based classification is designed to make the model invariant to features that strongly influence the image's aesthetics, e.g. style-based features such as brightness or contrast.
We propose to use self-supervised aesthetic-aware pretext tasks that let the network learn aesthetically relevant features, based on the observation that distorting aesthetic images with image filters usually reduces their appeal. To ensure that images are not accidentally improved when filters are applied, we introduce a large dataset comprised of highly aesthetic images as the starting point for the distortions. The network is then trained to rank less distorted images higher than their more distorted counterparts. To exploit effects of multiple different objectives, we also embed this task into a multi-task setting by adding either a self-supervised classification or regression task. In our experiments, we show that our pretraining improves performance over the ImageNet initialization and reduces the number of epochs until convergence by up to 47%. Additionally, we can match the performance of an ImageNet-initialized model while reducing the labeled training data by 20%. We make our code, data, and pretrained models available.

## Underlying code
The code in this repository is based on / uses:
- github.com/kentsyx/Neural-IMage-Assessment
- github.com/spijkervet/SimCLR
- github.com/gidariss/FeatureLearningRotNet

## Files
https://oc.informatik.uni-wuerzburg.de/s/6L5CNQsbp2yMpwC

PW: `oosCmnMQ10`

## Citation
```bib
@InProceedings{Pfister_2021_CVPR,
    author    = {Pfister, Jan and Kobs, Konstantin and Hotho, Andreas},
    title     = {Self-Supervised Multi-Task Pretraining Improves Image Aesthetic Assessment},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {816-825}
}
```
