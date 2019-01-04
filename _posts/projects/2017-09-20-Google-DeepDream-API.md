---
layout: post
title: "Google DeepDream API"
excerpt: "Doing Digital Art in Google DeepDream API"
modified: 2018-1-26T14:17:25-04:00
categories: project
tags: [deep learning, digital art]
comments: true
share: true
image:
  teaser: /images/projects/2017-09-20-Google-DeepDream-API/deepdream_teaser.png
---

### Introduction

The introduction of Google DeepDream and its application in digital art can be found in my article ["Google DeepDream in Python"](https://leimao.github.io/article/Google-DeepDream-Python/).

### Description

This API could add pattern features learned by Google DeepDream to your paintings. 

<br />

The development of this API is still in progress. More functions will be added in the future.

### Requirements

* Python 3.6

### Dependencies

* tensorflow 1.3
* numpy
* PIL
* os
* sys
* zipfile
* six
* argparse

### Usage

Here I am using Georges Seurat's "Sunday Afternoon On The Island Of La Grande Jatte" as an example to illustrate the usage of this Google DeepDream API.

#### List the available layers and the number of channels

To check the available layer names and channel numbers for the deepdream program. 

<br />

**Input**

{% highlight shell %}
python deepdream_api.py -l
{% endhighlight %}

**Output**
{% highlight shell %}
import/conv2d0_pre_relu/conv 64
import/conv2d1_pre_relu/conv 64
import/conv2d2_pre_relu/conv 192
import/mixed3a_1x1_pre_relu/conv 64
import/mixed3a_3x3_bottleneck_pre_relu/conv 96
import/mixed3a_3x3_pre_relu/conv 128
import/mixed3a_5x5_bottleneck_pre_relu/conv 16
import/mixed3a_5x5_pre_relu/conv 32
import/mixed3a_pool_reduce_pre_relu/conv 32
import/mixed3b_1x1_pre_relu/conv 128
import/mixed3b_3x3_bottleneck_pre_relu/conv 128
import/mixed3b_3x3_pre_relu/conv 192
import/mixed3b_5x5_bottleneck_pre_relu/conv 32
...
{% endhighlight %}
In the output, the layer name is on the left, the number of channels in the layer is on the right.


#### Preview the feature pattern of the neural network

To preview the feature pattern learned in a certain channel of a certain layer in the neural network. This is helpful for the user to select layers and channels used for image modification. While not cessary, you may also preview all the feature patterns learned in the neural network [here](http://storage.googleapis.com/deepdream/visualz/tensorflow_inception/index.html). It should be noted that the high frequencies of the patterns might have been suppressed by Laplacian pyramid decomposition in those images.

<br />

**Input**

<br />

-p layer_name channel_number, --preview layer_name channel_number

{% highlight shell %}
python deepdream_api.py -p mixed4b_1x1_pre_relu 70 pattern.jpeg
{% endhighlight %}

**Output**

<div class = "titled-image">
<figure>
    <img src = "{{ site.url }}/images/projects/2017-09-20-Google-DeepDream-API/pattern.jpeg">
    <figcaption>Feature Pattern Learned in the Neural Layers</figcaption>
</figure>
</div>

#### Preview the feature pattern of the neural network

To preview the feature pattern learned in a certain channel of a certain layer in the neural network with Laplacian Pyramid Gradient Normalization. High frequency patterns were suppressed by Laplacian Pyramid Gradient Normalization.

<br />

It should be noted that the feature No.70 in "mixed4b_1x1_pre_relu" layer I generated in this example is almost the same to the [one](http://storage.googleapis.com/deepdream/visualz/tensorflow_inception/mixed4b_1x1_pre_relu.html) archieved in Google's server. 

<br />

**Input**

<br />

-pl layer_name channel_number, --preview layer_name channel_number

{% highlight shell %}
python deepdream_api.py -p mixed4b_1x1_pre_relu 70 pattern_lap.jpeg
{% endhighlight %}

**Output**

<div class = "titled-image">
<figure>
    <img src = "{{ site.url }}/images/projects/2017-09-20-Google-DeepDream-API/pattern_lap.jpeg">
    <figcaption>Feature Pattern Learned in the Neural Layers</figcaption>
</figure>
</div>

#### Render the image with the features from the neural network

Apply feature pattern learned in a certain channel of a certain layer in the neural network to the image that the user provided.

<br />

**Input**

<br />

-r image_path layer_name channel_number, --render image_path layer_name channel_number

{% highlight shell %}
python deepdream_api.py -r inputs/sunday_afternoon.jpg mixed4b_1x1_pre_relu 70 sunday_afternoon_deepdream.jpeg
{% endhighlight %}

<div class = "titled-image">
<figure>
    <img src = "{{ site.url }}/images/projects/2017-09-20-Google-DeepDream-API/sunday_afternoon.jpg">
</figure>
</div>

**Output**

<div class = "titled-image">
<figure>
    <img src = "{{ site.url }}/images/projects/2017-09-20-Google-DeepDream-API/sunday_afternoon_deepdream.jpeg">
</figure>
</div>

#### Render the image with the features from the neural network

Apply feature pattern learned in a certain channel of a certain layer in the neural network to the image that the user provided.

<br />

**Input**

<br />

-rl image_path layer_name channel_number, --render image_path layer_name channel_number

{% highlight shell %}
python deepdream_api.py -r inputs/sunday_afternoon.jpg mixed4b_1x1_pre_relu 70 sunday_afternoon_deepdream_lap.jpeg
{% endhighlight %}

<div class = "titled-image">
<figure>
    <img src = "{{ site.url }}/images/projects/2017-09-20-Google-DeepDream-API/sunday_afternoon.jpg">
</figure>
</div>

**Output**

<div class = "titled-image">
<figure>
    <img src = "{{ site.url }}/images/projects/2017-09-20-Google-DeepDream-API/sunday_afternoon_deepdream_lap.jpeg">
</figure>
</div>


### More Functions

I also tried to add the features from a guide image to customize the pattern. However, it seems that my understanding of the optimzation objective is not correct (see my post on [StackFlow](https://stackoverflow.com/questions/46324533/controlling-dreams-in-tensorflow)). Therefore, I will hold this until I totally resolve my confusions.


### Thought Updates

#### 1/26/2018

I was looking at the Caffe source code, which contains the customization of patterns, this morning. I think I understand what they were really doing and will update the program once I got some casual time.

```python
end = 'inception_3b/output'
h, w = guide.shape[:2]
src, dst = net.blobs['data'], net.blobs[end]
src.reshape(1,3,h,w)
src.data[0] = preprocess(net, guide)
net.forward(end=end)
guide_features = dst.data[0].copy()


def objective_guide(dst):
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

_=deepdream(net, img, end=end, objective=objective_guide)
```

They first exacted and saved the neural network output of guided image at certain layer, and the dimension of this output tensor was, say, ```[64,20,20]```. Then they extracted the neural network output of another image which they wished to modify at the same layer, and the dimension of this output tensor was, say, ```[64,30,30]```. Then ``ch = 64``. After reshaping, ``x`` becomes ``[64,400]``, and ``y`` becomes ``[64,900]``. Therefore, ``A`` is a matrix of size ``[400,900]``. ``A.argmax(1)`` returns an array of size ``[400]``, each element in the array ranges from ``0`` to ``900``. ``y[:,A.argmax(1)]`` is therefore a tensor of shape ``[64,400]``. This will be the gradient to ``x[64,400]`` (If I understand it correctly, in Caffe, dst.diff specifies gradient of the last layer). Caffe could specify the gradient of the last layer directly, I need to figure out how to do it in TensorFlow or thinking of a loss function to optimize using TensorFlow autograd.

<br />

It's really annoying that I do not have time to do these interesting things. So many things bothered me.

### References

* [Google Official DeepDream Tutorial in Caffe](https://github.com/google/deepdream/blob/master/dream.ipynb)
* [Google Official DeepDream Tutorial in TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb)
* [Google Research Blog](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
* [Siraj Raval's Video Tutorial](https://www.youtube.com/watch?v=MrBzgvUNr4w)
* [Siraj Raval's Code](https://github.com/llSourcell/deep_dream_challenge/blob/master/deep_dream.py)
* [Concept of Octave](https://www.quora.com/What-does-octave-in-SIFT-means-Is-it-different-resolutions-of-the-image)
* [SIFT](http://aishack.in/tutorials/sift-scale-invariant-feature-transform-scale-space/)

### Link to GitHub

[DeepDream](https://github.com/leimao/DeepDream)