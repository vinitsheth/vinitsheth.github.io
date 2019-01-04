---
layout: post
title: "LogSumExp and Its Numeric Stability"
excerpt: "Never Jump Out of Log Domain"
modified: 2018-05-04T14:17:25-04:00
categories: blog
tags: [Numeric Stability]
comments: true
share: true
---

### Introduction

When we do large number or small number multiplications, it is easy to overflow or underflow in computer programs so that we would not get our expected result. To avoid these situations, people tend to do all the multiplications in log domain.

<br />

For example, $a = e^{1000}$ and $b = e^{2000}$, and we would like to know $ab$. Instead of calculating the explosive $ab$, we have $\log a = 1000$, $\log b = 2000$, and $\log ab =$ $\log a + \log b =$ $1000 + 2000  = 3000$. Similarly, when $a = e^{-1000}$ and $b = e^{-2000}$, we have $\log a = -1000$, $\log b = -2000$, and $\log ab = $ $\log a + \log b =$ $-1000 - 2000 = -3000$.

<br />

We would not jump out of the log domain unless it is the final step. However, how should we calculate the log of the sum of exponentials (logsumexp) without leaving the log domain? More formally, given $\log x_1, \log x_2, ..., \log x_n$, how to calculate $\log \sum_{i = 1}^{n} x_i$?

### Incorrect Approach

$$\log \sum_{i = 1}^{n} x_i = \log \sum_{i = 1}^{n} e^{\log x_i}$$

This is not the correct approach. If one of $\log x_i$ is extremely large, you would be just calculating $$\log \infty$$ in the computer program. On the otherhand, if all the $\log x_i$ are extremely small, you would be just calculating $\log 0$ in the computer program. In either case, there would be a problem.

### Correct Approach

$$\log \sum_{i = 1}^{n} x_i = \log \sum_{n = 1}^{N} e^{\log x_i} = \log (e^{a} \times \sum_{n = 1}^{N} e^{\log x_i - a}) = a + \log \sum_{n = 1}^{N} e^{\log x_i - a}$$

where $a = max(\log x_1, \log  x_2, ..., \log x_n)$.

<br />

If at least one of the $\log x_i$ is extremely large, the value of $e^{\log x_i - a}$ would not be greater than 1.0. If all the $\log x_i$ are extremely small, there would be at least one $e^{\log x_i - a}$ that is equal to 1.0, and the rest of $e^{\log x_i - a}$ might be very close or equal to 0. Taken together, $\sum_{n = 1}^{N} e^{\log x_i - a}$ is always in a range of [1, N].

<br />

Therefore, when it comes to computer program, instead of using $\log \sum_{i = 1}^{n} e^{\log x_i}$, we use $a + \log \sum_{n = 1}^{N} e^{\log x_i - a}$ to calculate the logsumexp without jumping out of the log domain.

### Miscellaneous

The overflow and underflow problems are common in many problems such as machine learning, hidden Markov model and other probabilistic models. One should be careful handling these problems when implementing something from scratch.

<br />

SciPy has the [``logsumexp``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.logsumexp.html) function that are ready for use.

