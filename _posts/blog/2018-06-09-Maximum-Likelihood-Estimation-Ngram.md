---
layout: post
title: "Maximum Likelihood Estimation of N-Gram Model Parameters"
excerpt: "Mathematical Proof of the Maximum Likelihood Estimation of N-Gram Model Parameters"
modified: 2018-06-09T14:17:25-04:00
categories: blog
tags: [Probability, Natural Language Processing]
comments: true
share: true
---

### Introduction

A language model is a probability distribution over sequences of words, namely:

$$p(w_1, w_2, w_3, ..., w_n)$$

According to the chain rule,

$$
  p(w_1, w_2, w_3, ..., w_n) = p(w_1)p(w_2|w_1)p(w_3|w_2,w_1)...p(w_n|w_{n-1},w_{n-2},...,w_1)
$$

However, the parameters for this language model are $p(w_1)$, $p(w_2\|w_1)$, ..., $p(w_n\|w_{n-1},...,w_1)$, which are usually too computationally expensive to calculate especially for the conditional probability with many conditioning words, even with a small dataset.

<br />

To approximate $p(w_1, w_2, w_3, ..., w_n)$, we could use N-Gram models to approximate the language model, namely:

#### N-Gram Model

$$
p(w_1, w_2, w_3, ..., w_n) \approx p(w_1)p(w_2|w_1)p(w_3|w_2,w_1)...p(w_n|w_{n-1},w_{n-2},...,w_{n-N})
$$

In particular, we usually use unigram model, bigram model and trigram model in language modelings.

#### Unigram Model

$$
p(w_1, w_2, w_3, ..., w_n) \approx p(w_1)p(w_2)p(w_3)...p(w_n)
$$

#### Bigram Model

$$
p(w_1, w_2, w_3, ..., w_n) \approx p(w_1)p(w_2|w_1)p(w_3|w_2)...p(w_n|w_{n-1})
$$


#### Trigram Model

$$
p(w_1, w_2, w_3, ..., w_n) \approx p(w_1)p(w_2|w_1)p(w_3|w_2,w_1)...p(w_n|w_{n-1},w_{n-2})
$$

With the N-Gram model approximations, calculating $p(w_n\|w_{n-1},w_{n-2},...,w_{n-N})$ is usually not too computationally expensive.


### Maximum Likelihood Estimation of N-Gram Model Parameters

To estimate $p(w_n\|w_{n-1},w_{n-2},...,w_{n-N})$, an intuitive way is to do  maximum likelihood estimation (MLE).

<br />

Maximum likelihood esitmation estimates the model parameters such that the probability is maximized.

<br />

In our case, the parameters are $p(w_n\|w_{n-1},w_{n-2},...,w_{n-N})$, and the probability we maximizes is $p(w_1)p(w_2\|w_1)p(w_3\|w_2,w_1)...p(w_n\|w_{n-1},w_{n-2},...,w_{n-N})$

<br />

In practice, we simply count the occurrance of word patterns to calculate the maximum likelihood estimation of $p(w_n\|w_{n-1},w_{n-2},...,w_{n-N})$.

#### Unigram Model

$$p(w_i) = \frac{c(w_i)}{\sum_{w}^{} c(w)}$$

#### Bigram Model

$$p(w_i|w_{i-1}) = \frac{c(w_{i-1},w_i)}{\sum_{w}^{} c(w_{i-1},w)}$$

#### Trigram Model

$$p(w_i|w_{i-1},w_{i-2}) = \frac{c(w_{i-2},w_{i-1},w_i)}{\sum_{w}^{} c(w_{i-2},w_{i-1},w)}$$


Now the question becomes why these formulas are the maximum likelihood estimations. Most of the books and online tutorials only gives these formulas without showing the formal mathematical proof. 

<br />

Here I am going to rigorously show that these are actually the formulas of maximum likelihood estimation.


### Mathematical Derivation of Maximum Likelihood Estimation of N-Gram Model Parameters

#### Unigram Model

Let us warm up with unigram model. 

<br />

We have a collection of unique words, $w_1, w_2, ..., w_n$. 

<br />

For any given sequence of words $\mathbf{w}$ of length $N$ ($\mathbf{w} = (w_1, w_2, w_1, w_5, w_7, w_2)$ for example), we have

$$
\begin{aligned}
p(\mathbf{w}) 
& = p(w_1)^{c(w_1)} p(w_2)^{c(w_2)} p(w_3)^{c(w_3)}...p(w_n)^{c(w_n)}\\
& = \prod_{i=1}^{n}p(w_i)^{c(w_i)}
\end{aligned}
$$

where $c(w_i)$ is the count of word $w_i$ in the sentence.

<br />

We take the log of $p(\mathbf{w})$, we then have:

$$
\begin{aligned}
\log{p(\mathbf{w})}
& = c(w_1)\log{p(w_1)} + c(w_2)\log{p(w_2)} + c(w_3)\log{p(w_3)} + ... + c(w_n)\log{p(w_n)}\\
& = \sum_{i=1}^{n}c(w_i)\log{p(w_i)}
\end{aligned}
$$

To maximize $p(\mathbf{w})$, equivalently we have the following optimization problem:

<br />

Maximize $\log{p(\mathbf{w})}$, subject to $\forall i \in [1 \dotsc N]$, $\sum_{i =  1}^{n} p(w_i) = 1$.

<br />


Equivalently, we introduce auxilliary optimization function using Lagrange multiplier ($\sum_{i=1}^{n}p(w_i)-1 = 0$):


$$
\mathcal{L} = \sum_{i=1}^{n}c(w_i)\log{p(w_i)} + \lambda(\sum_{i=1}^{n}p(w_i)-1)
$$

For any $p(w_j)$, we take the derivatives of $\mathcal{L}$ respective to $p(w_j)$:

$$\frac{\partial \mathcal{L}}{\partial p(w_j)} = \frac{c(w_j)}{p(w_j)} + \lambda = 0$$

$$p(w_j) = -\frac{c(w_j)}{\lambda}$$


Because $\sum_{i=1}^{n}p(w_i) = 1$, we have:

$$\sum_{i=1}^{n}p(w_i) = \sum_{i=1}^{n} -\frac{c(w_i)}{\lambda} = \frac{\sum_{i=1}^{n} c(w_i)}{-\lambda} = 1$$

$$\lambda = - \sum_{i=1}^{n} c(w_i)$$

Because $p(w_j) = -c(w_j)/{\lambda}$, therefore

$$p(w_j) = \frac{c(w_j)}{\sum_{i=1}^{n} c(w_i)}$$

This concludes the proof.


#### Bigram Model


Now let us move on to bigram model to see what is different.

<br />

We have a collection of unique words, $w_1, w_2, ..., w_n$. 

<br />

For the conditional probabilities, we have $n \times n$ possibilities.

<br />

For any given sequence of words $\mathbf{w}$ of length $N$ ($\mathbf{w} = (w_1, w_2, w_1, w_5, w_7, w_2)$ for example), we have

$$
\begin{aligned}
p(\mathbf{w}) 
& = \prod_{i=1}^{n} p(w_i)^{s(w_i)} \prod_{i=1}^{n} \prod_{j=1}^{n} p(w_j|w_i)^{c(w_i, w_j)}
\end{aligned}
$$

where $c(w_i, w_j)$ is the count of word sequence $w_i, w_j$ in the sentence and 


$$
s(w_i) = \begin{cases}
    1, & \text{if $w_i$ is the first word}\\
    0, & \text{otherwise}
    \end{cases}
$$


We take the log of $p(\mathbf{w})$, we then have:

$$
\begin{aligned}
\log{p(\mathbf{w})}
& = \sum_{i=1}^{n}s(w_i)\log{p(w_i)} + \sum_{i=1}^{n}\sum_{j=1}^{n}c(w_i,w_j)\log{p(w_j|w_i)}
\end{aligned}
$$

To maximize $p(\mathbf{w})$, equivalently we have the following optimization problem:

<br />

Maximize $\log{p(\mathbf{w})}$, subject to $\forall i \in [1 \dotsc N]$, $\sum_{j =  1}^{n} p(w_j\|w_i) = 1$.

<br />


Equivalently, we introduce auxilliary optimization function using Lagrange multiplier ($\sum_{j =  1}^{n} p(w_j\|w_i)-1 = 0$):


$$
\mathcal{L} = \sum_{i=1}^{n}s(w_i)\log{p(w_i)} + \sum_{i=1}^{n}\sum_{j=1}^{n}c(w_i,w_j)\log{p(w_j|w_i)} +  \sum_{i=1}^{n} \lambda_i \bigg( \big(\sum_{j = 1}^{n} p(w_j|w_i) \big) - 1 \bigg)
$$

For any $p(w_k\|w_i)$, we take the derivatives of $\mathcal{L}$ respective to $p(w_k\|w_i)$:

$$\frac{\partial \mathcal{L}}{\partial p(w_k|w_i)} = \frac{c(w_i, w_k)}{p(w_k|w_i)} + \lambda_i = 0$$

$$p(w_k|w_i) = -\frac{c(w_i, w_k)}{\lambda_i}$$

Because $\sum_{j =  1}^{n} p(w_j\|w_i) = 1$, we have:

$$\sum_{j =  1}^{n} p(w_j|w_i) = \sum_{j =  1}^{n} -\frac{c(w_i, w_j)}{\lambda_i} = \frac{\sum_{j =  1}^{n} c(w_i, w_j)}{-\lambda_i} = 1$$


$$\lambda_i = -\sum_{j =  1}^{n} c(w_i, w_j)$$

Because $p(w_k\|w_i) = -c(w_i, w_k)/\lambda_i$, therefore

$$p(w_k|w_i) = \frac{c(w_i, w_k)}{\sum_{j =  1}^{n} c(w_i, w_j)}$$

This concludes the proof.

#### N-Gram Model

Without generality, the maximum likelihood estimation of n-gram model parameters could also be proved in the same way.



### Conclusion

Mathematics is important for (statistical) machine learning.