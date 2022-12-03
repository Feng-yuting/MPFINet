# MPFINet
@Article{rs14236118,
AUTHOR = {Feng, Yuting and Jin, Xin and Jiang, Qian and Wang, Quanli and Liu, Lin and Yao, Shaowen},
TITLE = {MPFINet: A Multilevel Parallel Feature Injection Network for Panchromatic and Multispectral Image Fusion},
JOURNAL = {Remote Sensing},
VOLUME = {14},
YEAR = {2022},
NUMBER = {23},
ARTICLE-NUMBER = {6118},
URL = {https://www.mdpi.com/2072-4292/14/23/6118},
ISSN = {2072-4292},
ABSTRACT = {The fusion of a high-spatial-resolution panchromatic (PAN) image and a corresponding low-resolution multispectral (MS) image can yield a high-resolution multispectral (HRMS) image, which is also known as pansharpening. Most previous methods based on convolutional neural networks (CNNs) have achieved remarkable results. However, information of different scales has not been fully mined and utilized, and still produces spectral and spatial distortion. In this work, we propose a multilevel parallel feature injection network that contains three scale levels and two parallel branches. In the feature extraction branch, a multi-scale perception dynamic convolution dense block is proposed to adaptively extract the spatial and spectral information. Then, the sufficient multilevel features are injected into the image reconstruction branch, and an attention fusion module based on the spectral dimension is designed in order to fuse shallow contextual features and deep semantic features. In the image reconstruction branch, cascaded transformer blocks are employed to capture the similarities among the spectral bands of the MS image. Extensive experiments are conducted on the QuickBird and WorldView-3 datasets to demonstrate that MPFINet achieves significant improvement over several state-of-the-art methods on both spatial and spectral quality assessments.},
DOI = {10.3390/rs14236118}
}
