---
layout: post
title: "Image Pyramids and Its Applications in Deep Learning"
excerpt: "Traditional Computer Vision Techniques Boosted Deep Learning"
modified: 2018-08-04T14:17:25-04:00
categories: blog
tags: [Computer Vision, Deep Learning]
comments: true
share: true
---

### Introduction

Image pyramids were basically a series of images with different resolutions stacking together [1]. The original image was repeatedly downsampled to a low resolution image using kernels, scuh as Gaussian kernel and Laplacian kernel. These images stack together from high resolution images at the bottom and low resolution images at the top, forming a "pyramid". Each image in the image pyramids is called an "octave". 


:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](/images/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/img0.png)  |  ![](/images/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/img1.png)|  ![](/images/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/img2.png) |  ![](/images/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/img3.png)  

One of the advantages of such image pyramids is that the object search could be faster using a coarse-to-fine strategy. For example, if you are looking for the hippo's eye in the image using translational searching, you will have to go through the whole high resolution image. However, if you first try to look for the hippo's eye in the low resolution image, determine its rough location, and do fine search in the corresponding location in the high resolution image, the number of translations you did would be significantly smaller than going through the high resolution image. While the search is much faster, the storage cost only increases by 1/3 at most if the dimension reduction is 1/2 at each dimension.

<br />

In some situations, we also need to resize the low resolution images to the size of high resolution image. This is mainly used for [image blending](https://docs.opencv.org/3.4.2/dc/dff/tutorial_py_pyramids.html). [Google DeepDream](https://github.com/leimao/DeepDream) also used image pyramids of equal image sizes to calculate the gradients for generating the patterns, probably for the smoothing purpose I guess (Honestly when I first implemented Google DeepDream I didn't quite understand why they did this "complicated" step).

:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](/images/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/img0.png)  |  ![](/images/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/img1_resized.png)|  ![](/images/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/img2_resized.png) |  ![](/images/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/img3_resized.png)  

CMU has a basic introduction to image pyramids, which could be downloaded [here](/downloads/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/SamplingPyramids.pdf). 


### Image Pyramids in Deep Learning

The concept of image pyramids have also been employed in deep learning for feature extractions.


#### Spatial Pyramid Pooling (SPP)

Kaiming He et al. used the following pooling strategy resulting a fixed length feature vector, independent of the size of input tensor, in the neural network [2]. Baiscally given a tensor input, three max pooling was done in parallel, resulting 4 x 4, 2 x 2, and 1 x 1 tensors. The tensors were then linearized and concatenated into one vector (which may contain multiple channels). This vector contains rich spatial information that increases the computer vision task accuracy.

<div class = "titled-image">
<figure>
    <img src = "{{ site.url }}/images/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/spatial_pyramid_pooling.png" width = "95%">
    <figcaption>Spatial Pyramid Pooling</figcaption>
</figure>
</div>

Obviously, the concept of this method is extremely close to image pyramids in the traditional computer vision.

<br />

The implementation of such spatial pyramid pooling in TensorFlow could be found [here](https://github.com/tensorflow/tensorflow/issues/6011).


#### Pyramid Pooling

One shotcoming of Kaiming He's method is that the spatial pyramid pooling could only be used before a fully connected neural network because the sizes of outputs from parallel max poolings are different. You would not be able to use that method between convolutional layers.

<br />

Hengshuang Zhao used a method similar to Google DeepDream's method by resizing the pooled tensor to the original size by bilinear interpolation in PSPNet (I believe Google DeepDream algorithm came earlier than their publications). Therefore the outputs from parallel poolings could be stacked together.

<div class = "titled-image">
<figure>
    <img src = "{{ site.url }}/images/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/pyramid_pooling-module.png" width = "95%">
    <figcaption>Pyramid Pooling</figcaption>
</figure>
</div>

Bilinear interpolation is differentiable. In TensorFlow, we could use either 
[tf.image.resize_bilinear](https://www.tensorflow.org/api_docs/python/tf/image/resize_bilinear) or [
tf.image.resize_images](https://www.tensorflow.org/api_docs/python/tf/image/resize_images) to do bilinear image upsampling.


#### Atrous Spatial Pyramid Pooling (ASPP)

Liang-Chieh Chen used atrous convolution to do spatial pyramid pooling in their DeepLab series [4,5,6]. Atrous convolution is nothing special but an extension of oridinary convolution. In other words, ordinary convolution is a special case of atrous convolution. In atrous convolution, when rate = 1, it becomes ordinary convolution. I don't know why people invented "atrous" for such convolution, but I used to call it [dilated convolution](https://github.com/vdumoulin/conv_arithmetic).


<div class = "titled-image">
<figure>
    <img src = "{{ site.url }}/images/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/atrous_convolution.png" width = "95%">
    <figcaption>Atrous Convolution</figcaption>
</figure>
</div>

The name of "atrous spatial pyramid pooling" is actually illusive because there is no "pooling" in the method. In stead of doing parallel pooling, parallel atrous convolutions were conducted using "same" padding, resulting outputs of same size.

<div class = "titled-image">
<figure>
    <img src = "{{ site.url }}/images/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/atrous_spatial_pyramid_pooling.png" width = "95%">
    <figcaption>Atrous Spatial Pyramid Pooling</figcaption>
</figure>
</div>

Similarly, the resulting outpus were stacked together as input for the following layers.


<div class = "titled-image">
<figure>
    <img src = "{{ site.url }}/images/blog/2018-08-04-Image-Pyramids-In-Deep-Learning/deeplabv3.png" width = "95%">
    <figcaption>DeepLab v3+ Network Architecture</figcaption>
</figure>
</div>

It is actually easy to implement atrous spatial pyramid pooling in TensorFlow. In [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d) and [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d), we could set the rate in the "dilation_rate" argument. Because DeepLab is a Google product, they even added [tf.nn.atrous_conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d) lately. However it is nothing fancy but a wrapper for tf.nn.conv2d. Finally, remember to set the "padding" argument to "same". 


### References

[1] E.H. Andelson and C.H. Anderson and J.R. Bergen and P.J. Burt and J.M. Ogden. "[Pyramid methods in image processing](http://persci.mit.edu/pub_pdfs/RCA84.pdf)". 1984.

[2] K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. ECCV, 2014.

[3] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia. Pyramid scene parsing network. CVPR, 2017.


[4] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. TPAMI, 2017.

[5] L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam. Rethinking atrous convolution for semantic image segmentation. arXiv:1706.05587, 2017.

[6] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, H. Adam. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. arXiv:1802.02611, 2018.