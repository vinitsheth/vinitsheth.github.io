---
layout: post
title: "Batch Normalization Explained"
excerpt: "Fill More Holes in Deep Learning"
modified: 2018-09-07T14:17:25-04:00
categories: blog
tags: [Deep Learning]
comments: true
share: true
---

### Introduction


Recently I was working on a collaborative deep learning project trying to reproduce a model from the publication, but I found the model was overfit significantly. My dear colleague examined my code and pointed out that there might be some problems in my Tensorflow batch normalization implementation. After checking the possible correct implementation, I realized that probably I did not fully understand batch normalization. In this blog post, I am going to review batch normalization again on its mathematical definition and intuitions.


### Motivation of Batch Normalization

I am not going to explain why batch normalization works well in real practice, since Andrew Ng has a very good [video](https://www.youtube.com/watch?v=nUUqwaxLnWs) explaining that.

### Mathematical Definition

#### Training Phase

Given inputs $x$ over a minibatch of size $m$, $B = \\{x_1, x_2, ..., x_m\\}$, by applying transformation of your inputs using some learned parameters $\gamma$ and $\beta$, the outputs could be expressed as $B' = \\{y_1, y_2, ..., y_m\\}$, where $y_i = {\text{BN}}_{\gamma, \beta} (x_i)$.


<br />

More concretely, we first calculate the mean and the variance of the samples from the minibatch.

$$
\mu_B = \frac{1}{m} \sum_{1}^{m} x_i \\

\sigma_B^2 = \frac{1}{m} \sum_{1}^{m} (x_i - \mu_B)^2
$$

Then we normalize the samples to zero means and unit variance. $\epsilon$ is for numerical stability in case the denominator becomes zero by chance.

$$

\hat{x_i} = \frac{x_i-\mu_B}{\sqrt{\sigma_B^2 + \epsilon}}

$$

Finally, a little bit suprising, there is a scaling and shifting step. $\gamma$ and $\beta$ are learnable parameters.


$$
y_i = \gamma \hat{x_i} + \beta \equiv {\text{BN}}_{\gamma, \beta} (x_i)

$$


#### Test Phase


During test phase, specifically when you only have one test sample, doing batch normalization as the one in the train phase does not make sense, because your outputs at each layer of the network will be exactly zero. To overcome this, people invented "running mean" and "running variance", which are updated in real time during training.

<br />

More concretely, at training time step t, the running mean $\mu_B'\[t\]$ and running variance $\sigma_B^{\prime 2}\[t\]$ are calculated as follows:

$$
\mu_B'[t] = \mu_B'[t] \times \text{momentum} + \mu_B[t] \times (1 - \text{momentum})\\

\sigma_B^{\prime 2}[t] = \sigma_B^{\prime 2}[t] \times \text{momentum} + \sigma_B^2[t] \times (1 - \text{momentum}) 
$$

Here momentum is sometimes also called decay.

### Caveates

#### Value of Momentum

Suprisingly, momentum is a very import parameter for model validation performance. In TensorFlow, it suggests how to set momentum correctly.

<br />

decay: Decay for the moving average. Reasonable values for decay are close to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc. Lower decay value (recommend trying decay=0.9) if model experiences reasonably good training performance but poor validation and/or test performance. Try zero_debias_moving_mean=True for improved stability.

<br />

Intuitively, when momentum = 0, the running means and variances are always the means and variance of the last minibatch. This running means and variance could be highly biased and thus the training performance and validation performance differs significantly. When momentum = 1.0, the running means and variances are always the means and variance of the first minibatch, which could also be highly biased. So the momentum value should not be extremely close to 0 or 1.0. In addition, because we want to "average" over as many sample as possible and the samples in the past minibatches are important which should be given more weights, the momentum value should be a large number close to 1.0. The momentum could be thought as a weight factor for the past and the present information! Therefore, taken together, a value of multiple nines are recommended for momentum.

#### Specify Training Mode and Test Mode 

In TensorFlow you will have to specify training mode and test mode when you are running your model in different stages. 

<br />


There is also a very special setting in TensorFlow if you want to train a model that has a batch norm layer. Unfortunately, for some of my previous codes, I did not have this settings, which means that during test stage, the samples were not probably normalized "correctly" as expected. Zero will be output from batch norm layer if test samples are tested one by one, which will significantly affect the testing performance!

```
x_norm = tf.layers.batch_normalization(x, training=training)

# ...

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
```



































