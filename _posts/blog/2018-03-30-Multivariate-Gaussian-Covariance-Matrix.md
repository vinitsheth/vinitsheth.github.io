---
layout: post
title: "Multivariate Gaussian and Covariance Matrix"
excerpt: "Fill Up Some Probability Holes"
modified: 2018-03-30T14:17:25-04:00
categories: blog
tags: [Probability, Linear Algebra]
comments: true
share: true
---

### Introduction

In ordinary probability theory courses, the course instructor would usually not emphasize the concepts and properties of the multivariate Gaussian distribution. However, multivariate Gaussian distribution is actually quite widely used in many different models. So here we are going to fill some holes in the probability theory we learned.

### Multivariate Gaussian Distribution

If a random vector variable $x$ follows a multivariate Gaussian distribution with mean $\mu$ and covariance matrix $\Sigma$, its probability density function (pdf) is given by:

$$p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp{(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu))}$$

We write this as $x \sim \mathcal{N}(\mu, \Sigma)$.

<br />

To understand the multivariate Gaussian distribution properly, we need to first understand the covariance matrix.

### Covariance Matrix

Covariance is actually the critical part in multivariate Gaussian distribution. We will first look at some of the properties of the covariance matrix and try to proof them. 

The two major properties of the covariance matrix are:

* Covariance matrix is positive semi-definite.
* Covariance matrix in multivariate Gaussian distribution is positive definite.


A symmetric matrix $M$ is said to be positive semi-definite if $y^TMy$ is always non-negative for any vector $y$.

<br />

Similarly, a symmetric matrix $M$ is said to be positive definite if $y^TMy$ is always positive for any non-zero vector $y$.

#### Covariance matrix is positive semi-definite

We will first see covariance matrix is positive semi-definite.

<br />

By definition, given a random vector $x$, the covariance matrix is:

$$\Sigma = E[(x-\mu)(x-\mu)^T]$$

For any vector $v$, we have:

$$v^T \Sigma v = v^T E[(x-\mu)(x-\mu)^T] v = E[((x-\mu)^T v)^T ((x-\mu)^T v)] = E[((x-\mu)^T v)^2] \geq 0$$


Therefore, covariance matrix is positive semi-definite.

#### Covariance matrix in multivariate Gaussian distribution is positive definite

Now we need to see why the covariance matrix in multivariate Gaussian distribution is positive definite.

<br />

Notice from the pdf of the multivariate Gaussian distribution that the covariance matrix $\Sigma$ must be invertible, otherwise the pdf does not exist.

<br />

Because $\Sigma$ is intertible, it must be full rank, and linear system $\Sigma x = 0$ only has single solution $x = 0$.

<br />

From lemma 2, because $\Sigma$ is symmetric, we know that $\Sigma$ could be decomposed as $\Sigma = B^T B$.

<br />

Therefore, for any non-zero vector $y$, $y^T \Sigma y = 0$ if and only if $By = 0$. Because $y^T \Sigma y = y^T B^T B y = (By)^T (By)$. If $By = 0$, $\Sigma y = B^T By = 0$. Because $\Sigma$ is full rank, and $y \neq 0$, there is no solution for linear system $y^T \Sigma y = 0$. Therefore, $y^T \Sigma y > 0$ and $\Sigma$ is positive definite. 

#### Significance


From lemma 3, we know that $\Sigma^{-1}$ is also positive definite.

<br />

We have proved all the way to show that $\Sigma^{-1}$ is positive definite, why would we need to know this? 

<br />

In the pdf of multivariate Gaussian, $(x-\mu)^T \Sigma^{-1} (x-\mu)$ is always greater than 0, and $\exp{(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu))}$ is always less than 1.

### Multivariate Gaussian Distribution Properties

We now could move to learn some Gaussian distribution properties.

#### Gaussian Properties

1. The sum of independent Gaussian random variables is Gaussian.
2. The marginal of a joint Gaussian distribution is Gaussian.
3. The conditional of a joint Gaussian distribution is Gaussian.

The proof of these properties are rather complicated. The reference to these proofs are provided in the reference 2.

#### Example Facts

Suppose the real, scalar random variables $X$, $Y$, and $Z$ are jointly Gaussian. We could infer that:

1. $X$, $Y$, and $Z$ independently are Gaussians because of the Gaussian property 2.
2. $2X+Y-Z$ is Gaussian because of the Gaussian property 1 and example fact 1.
3. $X\|Y$ is Gaussian because of the Gaussian property 3.
4. $X\|Y,Z$ is Gaussian.

Example fact 4 might not be obvious. Here we could show using the Gaussian property 3 and conditional probability theorem without formal proof.

<br />

Because $p(X\|Y,Z) = p(X\|Y) \times p(Y\|Z)$, $p(X\|Y)$ and $p(Y\|Z)$ are the pdf of Gaussian distribution we know from the Gaussian property 3. The product of two Gaussian pdf is a pdf of a new Gaussian ([Demo](https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html) and [formal proof](http://www.tina-vision.net/docs/memos/2003-003.pdf)). 


### Lemmas

These are the lemmas that I proved to support part of the statements in this blog post.

#### Lemma 1

**A symmetric matrix is positive definite if and only if its eigenvalues are all positive.**

<br />

Proof:

<br />

Any square symmetric matrix $M$ could be eigendecomposed ([Wikipedia](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Real_symmetric_matrices)). $M = P^{-1}DP = P^TDP$ (For symmetric matrix, $P$ is orthonormal, $P^{-1} = P^T$) where $D$ is the diagnal eigenvalue matrix. $y^TMy = y^TP^TDPy = (Py)^TD(Py)$. Because $D$ is diagnal, $(Py)^T(Py)$ is non-negative, therefore, $M$ is positive definite if and only if all the diagnal elements, i.e., the eigenvalues of $M$, are positive.

#### Lemma 2

**A symmetric matrix $A$ could be decomposed as $A = B^T B$ where $B$ is also a square matrix**

<br />

Proof:

<br />

$A = P^TDP = P^T (D^{1/2})^T D^{1/2} P = (D^{1/2} P)^T (D^{1/2} P) = B^T B$, where $B = D^{1/2} P$

#### Lemma 3

**If matrix $K$ is positive definite, then $K^{-1}$ is also positive definite.**

<br />

Proof:

<br />

From lemma 1, we know that if $K$ is positive definite, all the eigenvalues of $K$ is positive. Therefore, the determinant of $K$ is positive, and $K$ must be invertible.

<br />

For any vector non-zero $y$, we could always express $y$ as $y = Kx$ because $K$ is invertible. Given $K$ is positive definite and $v^T K v > 0$ for any non-zero vector $v$. 

<br />

Then we have:

$$y^T K^{-1} y = (Kx)^T K^{-1} Kx = x^T K^T (K^{-1} K) x = x^T K^T x = (x^T K x)^T > 0$$

Therefore, $K^{-1}$ is also positive definite.


### References

The references below provide a lot of useful properties and facts without showing some of the detailed self-contained subtle proofs I provided above.

<br />

[1] [The Multivariate Gaussian Distribution](/downloads/blog/2018-03-30-Multivariate-Gaussian-Covariance-Matrix/multivariate_gaussian_stanford.pdf)

[2] [More on Multivariate Gaussians](/downloads/blog/2018-03-30-Multivariate-Gaussian-Covariance-Matrix/multivariate_gaussian_more_stanford.pdf)

[3] [Products and Convolutions of Gaussian Probability Density Functions](/downloads/blog/2018-03-30-Multivariate-Gaussian-Covariance-Matrix/product_of_gaussian_pdf.pdf)
