---
layout: post
title: "Logics Behind Tying the Weights of Output Layer to Word Embeddings in the Language Models"
excerpt: "Understand Tying the Weights"
modified: 2018-02-18T14:17:25-04:00
categories: blog
tags: [Deep Learning, Natural Language Processing]
comments: true
share: true
---

### Introduction


"Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
https://arxiv.org/abs/1608.05859

"Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
https://arxiv.org/abs/1611.01462

Basically, if the length of word embedding is the same to the number of hidden layer, for the output softmax layer, we use the the word embedding as the weights.


Example, if we have a corpus of three words (A, B, C), and we use word embedding length of four. 

Suppose at certain time of the model, we have the following embedding matrix and hidden layer output.

Word Embedding: $[[1.2, 2.0, -1.2, -2.4],[-3.2, -4.4, 5.8, 4.1], [1.1, 2.6, 3.6, -3.3]]$

Hidden Layer Output: $[1.0, 1.0, 1.0, 1.0]$

To calculate the model ouput, instead of using another layer of trainable weights, we use the word embedding as the weights.

Model Output Before Softmax: $[-0.4, 2.3, 4]$

Model Output After Softmax: $[0.010, 0.153, 0.837]$

So the probability of choosing word A is 0.010, choosing word B is 0.153, and choosing word C is 0.837, respectively.

This sounds crazy. What is the rationale behind this?